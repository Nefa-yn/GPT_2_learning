from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------

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
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_emb: int = 384


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