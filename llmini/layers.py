# layers.py
# Contains reusable building blocks for model architectures

import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    A causal self-attention mechanism for transformer models.

    Attributes:
        n_head (int): Number of attention heads.
        key (nn.Linear): Linear layer for computing keys.
        query (nn.Linear): Linear layer for computing queries.
        value (nn.Linear): Linear layer for computing values.
        proj (nn.Linear): Linear layer for projecting the output.
        mask (torch.Tensor): Causal mask to prevent attention to future tokens.
    """

    def __init__(self, n_embd, n_head, block_size):
        """
        Initialize the CausalSelfAttention module.

        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads.
            block_size (int): Maximum sequence length.
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(
            block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        Perform a forward pass through the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size,
                              T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """
    A transformer block consisting of a causal self-attention layer and a feedforward layer.

    Attributes:
        ln1 (nn.LayerNorm): Layer normalization before the attention layer.
        attn (CausalSelfAttention): Causal self-attention layer.
        ln2 (nn.LayerNorm): Layer normalization before the feedforward layer.
        mlp (nn.Sequential): Feedforward neural network.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initialize the Block module.

        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads.
            block_size (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Perform a forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size,
                              T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
