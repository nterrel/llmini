import pytest


def test_estimate_loss():
    from llmini.train import estimate_loss

    # Run the actual implementation of estimate_loss
    losses = estimate_loss(iters=5, test_mode=True)

    # Assertions to validate the output
    assert 'train' in losses
    assert 'val' in losses
    assert losses['train'] > 0
    assert losses['val'] > 0
