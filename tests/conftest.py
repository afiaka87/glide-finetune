"""Pytest configuration and shared fixtures for GLIDE finetune tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glide_finetune.settings import (
    CheckpointSettings,
    DatasetSettings,
    FP16Settings,
    ModelSettings,
    SamplingSettings,
    SystemSettings,
    TrainingSettings,
)


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture(scope="session")
def test_device() -> torch.device:
    """Get test device (CPU for CI, CUDA if available locally)."""
    if torch.cuda.is_available() and os.environ.get("CI") != "true":
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_dtype() -> torch.dtype:
    """Get test dtype."""
    return torch.float32


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def mock_glide_model() -> Mock:
    """Create a mock GLIDE model."""
    model = Mock(spec=nn.Module)
    model.parameters = Mock(return_value=iter([
        torch.randn(10, 10, requires_grad=True),
        torch.randn(5, 5, requires_grad=True),
    ]))
    model.state_dict = Mock(return_value={
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 5),
        "layer2.bias": torch.randn(5),
    })
    model.eval = Mock(return_value=model)
    model.train = Mock(return_value=model)
    model.to = Mock(return_value=model)
    model.cuda = Mock(return_value=model)
    model.cpu = Mock(return_value=model)
    return model


@pytest.fixture
def mock_diffusion() -> Mock:
    """Create a mock diffusion object."""
    diffusion = Mock()
    diffusion.num_timesteps = 1000
    diffusion.training_losses = Mock(return_value=torch.tensor([0.5]))
    diffusion.p_sample_loop = Mock(return_value=torch.randn(1, 3, 64, 64))
    diffusion.p_sample_loop_progressive = Mock(return_value=iter([
        {"sample": torch.randn(1, 3, 64, 64)}
        for _ in range(50)
    ]))
    return diffusion


@pytest.fixture
def mock_text_encoder() -> Mock:
    """Create a mock text encoder."""
    encoder = Mock(spec=nn.Module)
    encoder.encode = Mock(return_value=torch.randn(1, 77, 512))
    encoder.tokenize = Mock(return_value=torch.randint(0, 1000, (1, 77)))
    encoder.to = Mock(return_value=encoder)
    return encoder


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample PIL image."""
    return Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )


@pytest.fixture
def sample_images(sample_image: Image.Image) -> List[Image.Image]:
    """Create a list of sample images."""
    return [sample_image.copy() for _ in range(4)]


@pytest.fixture
def sample_tensor_batch() -> torch.Tensor:
    """Create a sample batch of image tensors."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def sample_text_batch() -> List[str]:
    """Create a sample batch of text prompts."""
    return [
        "a painting of a sunset",
        "a photo of a cat",
        "an abstract artwork",
        "a landscape with mountains",
    ]


@pytest.fixture
def mock_dataset(sample_images: List[Image.Image], sample_text_batch: List[str]) -> Mock:
    """Create a mock dataset."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(side_effect=lambda idx: (
        sample_images[idx % len(sample_images)],
        sample_text_batch[idx % len(sample_text_batch)],
    ))
    return dataset


@pytest.fixture
def mock_dataloader(sample_tensor_batch: torch.Tensor, sample_text_batch: List[str]) -> Mock:
    """Create a mock dataloader."""
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([
        (sample_tensor_batch, sample_text_batch)
        for _ in range(10)
    ]))
    dataloader.__len__ = Mock(return_value=10)
    dataloader.batch_size = 4
    return dataloader


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def dataset_config(temp_dir: Path) -> DatasetSettings:
    """Create test dataset configuration."""
    return DatasetSettings(
        data_dir=str(temp_dir / "data"),
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        use_webdataset=False,
    )


@pytest.fixture
def model_config(temp_dir: Path) -> ModelSettings:
    """Create test model configuration."""
    return ModelSettings(
        model_path=None,
        resume_ckpt=None,
        train_upsample=False,
        freeze_transformer=False,
        freeze_diffusion=False,
        activation_checkpointing=False,
        use_sdpa=True,
    )


@pytest.fixture
def training_config() -> TrainingSettings:
    """Create test training configuration."""
    return TrainingSettings(
        learning_rate=1e-4,
        adam_weight_decay=0.01,
        adam_eps=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        grad_clip=1.0,
        num_epochs=2,
        warmup_steps=10,
        gradient_accumulation_steps=1,
        batch_size=2,
        microbatch_size=2,
        uncond_p=0.2,
    )


@pytest.fixture
def fp16_config() -> FP16Settings:
    """Create test FP16 configuration."""
    return FP16Settings(
        use_fp16=False,
        fp16_loss_scale=256.0,
        fp16_scale_window=2000,
        fp16_min_loss_scale=1.0,
        fp16_max_loss_scale=2**20,
    )


@pytest.fixture
def sampling_config() -> SamplingSettings:
    """Create test sampling configuration."""
    return SamplingSettings(
        timestep_respacing="50",
        guidance_scale=3.0,
        num_steps=50,
        test_prompt="a test image",
        sample_bs=1,
        sample_gs=4.0,
    )


@pytest.fixture
def checkpoint_config(temp_dir: Path) -> CheckpointSettings:
    """Create test checkpoint configuration."""
    return CheckpointSettings(
        save_directory=temp_dir / "checkpoints",
        checkpoint_frequency=100,
        sample_frequency=100,
        log_frequency=10,
        prefix="test-",
        max_checkpoints=3,
    )


@pytest.fixture
def system_config(test_device: torch.device) -> SystemSettings:
    """Create test system configuration."""
    return SystemSettings(
        seed=42,
        device=str(test_device),
        enable_tf32=False,  # Disable for consistent testing
        debug=True,
        log_level="DEBUG",
    )


# =============================================================================
# Training Component Fixtures
# =============================================================================

@pytest.fixture
def mock_optimizer(mock_glide_model: Mock) -> torch.optim.Optimizer:
    """Create a mock optimizer."""
    # Use a real optimizer with mock model parameters
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    return optimizer


@pytest.fixture
def mock_scheduler(mock_optimizer: torch.optim.Optimizer) -> Mock:
    """Create a mock learning rate scheduler."""
    scheduler = Mock()
    scheduler.step = Mock()
    scheduler.get_last_lr = Mock(return_value=[1e-4])
    return scheduler


@pytest.fixture
def mock_checkpoint_manager(temp_dir: Path) -> Mock:
    """Create a mock checkpoint manager."""
    manager = Mock()
    manager.save_checkpoint = Mock()
    manager.load_checkpoint = Mock(return_value={
        "epoch": 0,
        "step": 0,
        "best_loss": float("inf"),
    })
    manager.checkpoint_dir = temp_dir / "checkpoints"
    manager.cleanup_old_checkpoints = Mock()
    return manager


@pytest.fixture
def mock_metrics_tracker() -> Mock:
    """Create a mock metrics tracker."""
    tracker = Mock()
    tracker.update = Mock()
    tracker.get_average = Mock(return_value=0.5)
    tracker.get_all_averages = Mock(return_value={
        "loss": 0.5,
        "grad_norm": 1.0,
        "learning_rate": 1e-4,
    })
    tracker.reset = Mock()
    tracker.log_metrics = Mock()
    return tracker


# =============================================================================
# Integration Test Fixtures
# =============================================================================

@pytest.fixture
def training_context(
    mock_glide_model: Mock,
    mock_diffusion: Mock,
    mock_text_encoder: Mock,
    mock_dataloader: Mock,
    mock_optimizer: torch.optim.Optimizer,
    mock_scheduler: Mock,
    mock_checkpoint_manager: Mock,
    mock_metrics_tracker: Mock,
    training_config: TrainingSettings,
    test_device: torch.device,
) -> Dict[str, Any]:
    """Create a complete training context for integration tests."""
    return {
        "model": mock_glide_model,
        "diffusion": mock_diffusion,
        "text_encoder": mock_text_encoder,
        "dataloader": mock_dataloader,
        "optimizer": mock_optimizer,
        "scheduler": mock_scheduler,
        "checkpoint_manager": mock_checkpoint_manager,
        "metrics_tracker": mock_metrics_tracker,
        "config": training_config,
        "device": test_device,
        "epoch": 0,
        "global_step": 0,
    }


@pytest.fixture
def mock_accelerator() -> Mock:
    """Create a mock Accelerator for distributed tests."""
    accelerator = Mock()
    accelerator.device = torch.device("cpu")
    accelerator.is_main_process = True
    accelerator.is_local_main_process = True
    accelerator.num_processes = 1
    accelerator.process_index = 0
    accelerator.prepare = Mock(side_effect=lambda *args: args)
    accelerator.backward = Mock()
    accelerator.gather = Mock(side_effect=lambda x: x)
    accelerator.wait_for_everyone = Mock()
    accelerator.save_state = Mock()
    accelerator.load_state = Mock()
    return accelerator


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def capture_logs():
    """Fixture to capture log output."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add handler to root logger
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    # Clean up
    logger.removeHandler(handler)


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb for tests."""
    mock = Mock()
    mock.init = Mock()
    mock.log = Mock()
    mock.finish = Mock()
    mock.config = Mock()
    monkeypatch.setattr("wandb", mock)
    return mock


@pytest.fixture
def assert_tensors_close():
    """Fixture to assert tensors are close."""
    def _assert_close(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """Assert two tensors are close."""
        torch.testing.assert_close(tensor1, tensor2, rtol=rtol, atol=atol)
    return _assert_close


@pytest.fixture
def create_test_checkpoint(temp_dir: Path) -> Path:
    """Create a test checkpoint file."""
    checkpoint_path = temp_dir / "test_checkpoint.pt"
    torch.save({
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "epoch": 5,
        "step": 500,
        "loss": 0.123,
    }, checkpoint_path)
    return checkpoint_path


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def benchmark_timer():
    """Fixture for benchmarking code execution time."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.times.append(time.perf_counter() - self.start)
        
        @property
        def avg_time(self):
            return sum(self.times) / len(self.times) if self.times else 0
        
        @property
        def total_time(self):
            return sum(self.times)
    
    return Timer()


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Clean up CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up environment variables after each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)