from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ------------------------------------------

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        # key, query value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        # output projection
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        # not realy a bias, more of mask, but following naming
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_emb)
        # calculate query, key, value for all heads in batch and move head forward to the batch 
        # nh is a 'number of heads', hs is 'head size', and C is 'number of chanels'  = nh * hs
        # eg. in GPT-2(124M) nh = 12, hs = 64, so nh*hs = 768 channels in Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim = 2)
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)#(B, T, nh, hs)
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)#(B, T, nh, hs)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)#(B, T, nh, hs)
        # attention (materializes the large (T,T)   matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v #(B, T, nh, hs) x (B, T, nh, hs) -> (B, T, nh, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y
        


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12 #number of layers
    n_head: int = 12 #number of heads
    n_emb: int = 768 #embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb), #weights of token embedding
            wpe = nn.Embedding(config.block_size, config.n_emb),#weights of possition embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),#hidden layer
            ln_f = nn.LayerNorm(config.n_emb),#final layer norm
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)#final classifier

        #weigth sharing scheme
        self.transformer.wte.weigth = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence to length {T}, block size is {self.config.block_size}'
        # forward the token and possition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_emb)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_emb)
        x = tok_emb + pos_emb
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        # INVESTIGATE: uncovered strange behaviour after applying normalixzation tensor.mean() close to 0.5 when expected close to 0, tensor.std() closeto 8 when expected close to 1
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights for pretrained model: %s' % model_type)

        # n_layer, n_head, n_emb determined from model_type
        config_args = {
            'gpt2':dict(n_layer=12, n_head=12, n_emb=768), # 124M params
            'gpt2-medium':dict(n_layer=24, n_head=16, n_emb=1024), # 350M params
            'gpt2-large':dict(n_layer=36, n_head=20, n_emb=1280), # 774M params
            'gpt2-xl':dict(n_layer=48, n_head=25, n_emb=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoint
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoint
        # create from scratch initialized miniGPT model
        config = GPTConfig(**config_args)
        model = GPT(config) 
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer

        # init huggingface/transformers model

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are alligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf =  [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore this
        sd_keys_hf =  [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same just mask / buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoint use a "Conv1D" module, but we only want to use vanila Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys : {len(sd_keys_hf)} != {len(sd_keys)}' 
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanila copy over parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
# ----------------------------------------------
import tiktoken

class DataLoaderLight:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt','r') as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded tokens : {len(tokens)}')
        print(f'1 epoch = {len(tokens) // (B*T)} batches')

        #state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B,T)#inputs
        y = (buf[1:]).view(B,T)#target
        # advance position in a tensor
        self.current_position += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ----------------------------------------------
import time 

num_return_sequences = 5
max_length = 30
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'usnig: {device}')

torch.manual_seed(1337)

train_loader = DataLoaderLight(B=4, T=1024)

torch.set_float32_matmul_precision('high')

#get logits
model = GPT(GPTConfig())
model.to(device)

#optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        #mport code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    delta_t = (t1 - t0) * 1000 # time diff  in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'step {i}, loss : {loss.item()}, time_delta: {delta_t}, tokens per second: {tokens_per_sec}')

print(loss)
import sys; sys.exit(0)
#print('didnt crash')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode('Hello, I`m a language model,')
tokens = torch.tensor(tokens,dtype= torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to('mps')

# generate! right now x is (B, T) where B = 5, T = 8
# set seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at a last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here become (5, 50) , topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probs
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the coresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print('>', decode)
