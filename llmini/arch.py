import math
import torch
import torch.nn as nn
from llmini.layers import CausalSelfAttention, Block


class TinyGPT(nn.Module):
    """
    A lightweight GPT-like model for text generation.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_layer (int): Number of transformer layers.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimension.
        dropout (float): Dropout rate.
    """

    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1):
        """
        Initialize the TinyGPT model.

        Args:
            vocab_size (int): Size of the vocabulary.
            block_size (int): Maximum sequence length.
            n_layer (int): Number of transformer layers.
            n_head (int): Number of attention heads.
            n_embd (int): Embedding dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the model using Xavier initialization.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        """
        Perform a forward pass through the model.

        Args:
            idx (torch.Tensor): Input tensor of token indices.
            targets (torch.Tensor, optional): Target tensor for computing loss.

        Returns:
            tuple: A tuple (logits, loss) where logits are the model predictions and loss is the computed loss (if targets are provided).
        """
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling tokens from the model's predictions.

        Args:
            idx (torch.Tensor): Input tensor of token indices.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): Top-k sampling parameter.

        Returns:
            torch.Tensor: Generated token indices.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                vals, idxs = torch.topk(logits, k)
                mask = torch.full_like(logits, float('-inf'))
                logits = mask.scatter(1, idxs, vals)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


class ComplexGPT(nn.Module):
    """
    A more complex GPT-like model with additional layers and parameters.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_layer (int): Number of transformer layers.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimension.
        dropout (float): Dropout rate.
    """

    def __init__(self, vocab_size, block_size, n_layer=8, n_head=8, n_embd=512, dropout=0.1):
        """
        Initialize the ComplexGPT model.

        Args:
            vocab_size (int): Size of the vocabulary.
            block_size (int): Maximum sequence length.
            n_layer (int): Number of transformer layers.
            n_head (int): Number of attention heads.
            n_embd (int): Embedding dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd, dropout=dropout
            ) for _ in range(n_layer)
        ])

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        Perform a forward pass through the model.

        Args:
            idx (torch.Tensor): Input tensor of token indices.
            targets (torch.Tensor, optional): Target tensor for computing loss.

        Returns:
            tuple: A tuple (logits, loss) where logits are the model predictions and loss is the computed loss (if targets are provided).
        """
        B, T = idx.size()
        assert T <= self.block_size, "Input sequence length exceeds block size."

        # Token and position embeddings
        token_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device))  # (T, n_embd)
        x = token_emb + pos_emb.unsqueeze(0)  # (B, T, n_embd)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm and output head
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
