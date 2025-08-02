"""Global pytest configuration for glide-finetune tests."""

import gc
import os

import pytest
import torch

# Set PyTorch CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def pytest_configure(config):
    """Configure pytest with global settings."""
    # Enable TF32 globally for all tests if CUDA is available
    # This provides better performance/memory tradeoff
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ Enabled TF32 for all GPU tests")


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically clean up GPU memory before and after each test."""
    # Clean before test
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    yield

    # Clean after test
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
