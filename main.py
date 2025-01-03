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