import pytest


def test_get_batch():
    from llmini.data import get_batch
    import torch

    # Test with a small batch size
    batch_size = 4
    x, y = get_batch("train", batch_size)

    # Assertions to validate the output
    assert x.shape == (batch_size, 16)
    assert y.shape == (batch_size, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
