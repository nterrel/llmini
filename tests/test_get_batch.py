import pytest


def test_get_batch():
    from llmini.data import get_batch
    import torch

    # Test with a small batch size
    batch_size = 4
    device = "cpu"
    x, y = get_batch("train", batch_size, device=device)

    # Assertions to validate the output
    assert x.shape == (batch_size, 128)  # Assuming BLOCK_SIZE = 128
    assert y.shape == (batch_size, 128)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
