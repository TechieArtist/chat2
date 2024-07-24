import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    """
    FeedForward network for the transformer model.

    Args:
        n_embd (int): Embedding dimension.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """
    Single head of self-attention.

    Args:
        head_size (int): Size of each attention head.
        block_size (int): Size of the input block.
        n_embd (int): Embedding dimension.
    """
    def __init__(self, head_size, block_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        num_heads (int): Number of attention heads.
        head_size (int): Size of each attention head.
        block_size (int): Size of the input block.
        n_embd (int): Embedding dimension.
    """
    def __init__(self, num_heads, head_size, block_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.

    Args:
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        block_size (int): Size of the input block.
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
