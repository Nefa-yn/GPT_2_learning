from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math
import inspect
from hellaswag import render_example, iterate_examples

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

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v #(B, T, nh, hs) x (B, T, nh, hs) -> (B, T, nh, hs)
        # Flash Attention implementation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        #start will all candidate parameters (that require grad)
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        #create optim groups. Any parameter that is 2D will be weigth decayed, otherwise no
        #i.e. all weight tensors is matmuls + embeddings decay. All biases and layer norms dont. 
        decay_params = [p for n, p in  param_dict.items() if p.dim() >= 2 ]
        nodecay_params = [p for n, p in  param_dict.items() if p.dim() < 2 ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"number decayed parameter tensor: {len(decay_params)}, with {num_decay_params} parameters ")
        print(f"number non decayed parameter tensor: {len(nodecay_params)}, with {num_nodecay_params} parameters ")
        # create AdamW optimizer and use the fused version if it is avalible
        fused_avalible = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avalible and 'mps' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas=(0.9, 0.95), eps = 1e-8, fused= use_fused)
        return optimizer
    
# ----------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype= torch.long)
    return ptt

class DataLoaderLight:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train' , 'val'}

        #get the shards filename
         # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# ----------------------------------------------

# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ----------------------------------------------
import time 
import os
from torch.distributed import init_process_group, destroy_process_group



ddp = int(os.environ.get('RANK', -1)) != -1# is this a ddp run?
if ddp:
    #use DDP atm demands CUDA, we set device apropriately acording to rank
    assert torch.cuda.is_available(), "For ddp CUDA is required"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0# this proces will do logging, checkpointing, etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'usnig: {device}')

torch.manual_seed(1337)
enc = tiktoken.get_encoding("gpt2")


total_batch_size = 524288
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0 , 'make sure total batch size devisible by B * T * ddp_world_size'
grad_acum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'Total batch size is: {total_batch_size}')
    print(f'Calculated gradient accumulation steps: {grad_acum_steps}')

train_loader = DataLoaderLight(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='train')
val_loader = DataLoaderLight(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='val')


torch.set_float32_matmul_precision('high')

#create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# TODO Not working on MPS thing to research
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# model = torch.compile(model)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model #always contains raw unwrapped model


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # if it > lr_decay_iters, return min lr
    if it > max_steps:
        return min_lr
    # in between use cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coef starts at 1 goes to 0
    return min_lr + coef * (max_lr - min_lr)

#optimize
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate= 6e-4, device = device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)


    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f'valiadtion loss: {val_loss_accum.item():.4f}')
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    model.train()
    optimizer.zero_grad()
    loss_accume = 0.0
    for micro_step in range(grad_acum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            #mport code; code.interact(local=locals())
        loss = loss / grad_acum_steps # to add mean reduction
        loss_accume += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_acum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accume, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    delta_t = (t1 - t0) # time diff  in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_acum_steps * ddp_world_size
    tokens_per_sec =  tokens_processed / delta_t
    if master_process:
        print(f'step {step}, loss : {loss_accume.item():.6f} | norm : {norm:.4f} | lr : {lr:.4e} | time_delta: {delta_t:.2f} s | tokens per second: {tokens_per_sec:.2f}')
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accume.item():.6f}\n")

if ddp:
    destroy_process_group()
#print('didnt crash')


