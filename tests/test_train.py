# Test for loss estimation in `train.py`

import time
from unittest.mock import patch
import cProfile
import pstats


def test_estimate_loss():
    from llmini.train import estimate_loss

    # Mock the estimate_loss function to return predefined values
    with patch("llmini.train.estimate_loss", return_value={"train": 0.1, "val": 0.2}):
        print("Mock applied")  # Debugging output to confirm the mock is used

        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        losses = estimate_loss(iters=1, test_mode=True)
        end_time = time.time()

        # End profiling
        profiler.disable()
        elapsed_time = end_time - start_time

        # Print profiling results
        print("Profiling results:")
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(10)  # Print top 10 cumulative time functions

        print(f"Test execution time: {elapsed_time:.4f} seconds")

        assert 'train' in losses and 'val' in losses
        assert losses['train'] > 0 and losses['val'] > 0
