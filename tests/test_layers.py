# test_layers.py
# Unit tests for layers.py

import pytest
import torch
from llmini.layers import CausalSelfAttention, Block


@pytest.fixture
def dummy_input():
    # Batch size 2, sequence length 10, embedding size 64
    return torch.rand(2, 10, 64)


@pytest.fixture
def dummy_block():
    return Block(n_embd=64, n_head=4, block_size=10, dropout=0.1)


@pytest.fixture
def dummy_attention():
    return CausalSelfAttention(n_embd=64, n_head=4, block_size=10)


def test_causal_self_attention(dummy_input, dummy_attention):
    output = dummy_attention(dummy_input)
    assert output.shape == dummy_input.shape, "Output shape mismatch in CausalSelfAttention"


def test_block(dummy_input, dummy_block):
    output = dummy_block(dummy_input)
    assert output.shape == dummy_input.shape, "Output shape mismatch in Block"
