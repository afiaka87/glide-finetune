"""Integration test for glide_finetune module."""

import os
import shutil
import tempfile
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from glide_finetune.checkpoint_utils import CheckpointManager
from glide_finetune.glide_finetune import (
    base_train_step,
    get_warmup_lr,
    run_glide_finetune_epoch,
    update_metrics,
    upsample_train_step,
)


class MockGlideModel(nn.Module):
    """Mock GLIDE model for testing."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.device = device
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3])
        self.tokenizer.padded_tokens_and_mask = Mock(
            return_value=([1, 2, 3, 0], [True, True, True, False])
        )

    def forward(self, x, timesteps, tokens=None, mask=None, low_res=None):
        # Return mock output with correct shape for epsilon and variance
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        # Double channels for epsilon and variance split
        # Use linear layer to ensure gradients flow
        x_flat = x.view(batch_size, -1)
        processed = self.linear(x_flat[:, :10])  # Use first 10 values
        # Create output with gradients
        output = torch.randn(
            batch_size, channels * 2, height, width, requires_grad=True
        )
        # Add a small amount of the processed value to ensure gradient flow
        output = output + processed.sum() * 0.001
        return output

    def del_cache(self):
        pass


class MockDiffusion:
    """Mock diffusion for testing."""

    def __init__(self):
        self.betas = torch.linspace(0.0001, 0.02, 1000)
        self.num_timesteps = 1000

    def q_sample(self, x, t, noise):
        return x + noise * 0.1


def create_mock_batch(batch_size=2, device="cpu", upsample=False):
    """Create a mock batch of data."""
    tokens = torch.randint(0, 100, (batch_size, 10))
    masks = torch.ones(batch_size, 10, dtype=torch.bool)

    if upsample:
        low_res = torch.randn(batch_size, 3, 64, 64)
        high_res = torch.randn(batch_size, 3, 256, 256)
        return tokens, masks, low_res, high_res
    else:
        images = torch.randn(batch_size, 3, 64, 64)
        return tokens, masks, images


class TestGlideFinetuneComponents:
    """Test individual components of the glide_finetune module."""

    def test_base_train_step(self):
        """Test base training step."""
        model = MockGlideModel()
        diffusion = MockDiffusion()
        batch = create_mock_batch()
        device = "cpu"

        loss, metrics = base_train_step(model, diffusion, batch, device)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert isinstance(metrics, dict)
        assert all(f"loss_q{i}" in metrics for i in range(4))

    def test_upsample_train_step(self):
        """Test upsampling training step."""
        model = MockGlideModel()
        diffusion = MockDiffusion()
        batch = create_mock_batch(upsample=True)
        device = "cpu"

        loss, metrics = upsample_train_step(model, diffusion, batch, device)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert isinstance(metrics, dict)
        assert all(f"loss_q{i}" in metrics for i in range(4))

    def test_warmup_lr(self):
        """Test learning rate warmup calculation."""
        base_lr = 1e-3
        warmup_steps = 1000

        # Test linear warmup
        assert get_warmup_lr(0, base_lr, warmup_steps, "linear") == 0.0
        assert get_warmup_lr(500, base_lr, warmup_steps, "linear") == base_lr * 0.5
        assert get_warmup_lr(1000, base_lr, warmup_steps, "linear") == base_lr

        # Test cosine warmup
        lr_cosine = get_warmup_lr(500, base_lr, warmup_steps, "cosine")
        assert 0 < lr_cosine < base_lr

        # Test no warmup
        assert get_warmup_lr(0, base_lr, 0, "linear") == base_lr

    def test_update_metrics(self):
        """Test metrics update function."""
        model = MockGlideModel()
        log: dict[str, float] = {}
        loss = torch.tensor(0.5)
        step_metrics = {"loss_q0": 0.1, "loss_q1": 0.2}

        updated_log = update_metrics(
            log,
            loss,
            step_metrics,
            global_step=100,
            train_idx=50,
            current_lr=1e-4,
            batch_size=4,
            gradient_accumualation_steps=1,
            glide_model=model,
        )

        assert updated_log["step"] == 100
        assert updated_log["iter"] == 50
        assert updated_log["loss"] == 0.5
        assert updated_log["lr"] == 1e-4
        assert updated_log["samples_seen"] == 404  # (100 + 1) * 4
        assert "param_norm" in updated_log
        assert updated_log["loss_q0"] == 0.1
        assert updated_log["loss_q1"] == 0.2


class TestGlideFinetuneEpoch:
    """Test the main epoch training function."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for outputs and checkpoints."""
        outputs_dir = tempfile.mkdtemp()
        checkpoints_dir = tempfile.mkdtemp()
        yield outputs_dir, checkpoints_dir
        shutil.rmtree(outputs_dir)
        shutil.rmtree(checkpoints_dir)

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        with patch("glide_finetune.glide_finetune.wandb") as mock:
            mock_run = Mock()
            mock_run.log = Mock()
            mock.Image = Mock(return_value="mock_image")
            yield mock_run, mock

    @pytest.fixture
    def mock_sample(self):
        """Mock the sample function."""
        with patch("glide_finetune.glide_finetune.glide_util.sample") as mock:
            # Return a mock tensor that can be saved as image
            mock.return_value = torch.randn(1, 3, 64, 64)
            yield mock

    @pytest.fixture
    def mock_pred_to_pil(self):
        """Mock the pred_to_pil function."""
        with patch("glide_finetune.glide_finetune.train_util.pred_to_pil") as mock:
            mock_image = Mock()
            mock_image.save = Mock()
            mock.return_value = mock_image
            yield mock

    def test_run_epoch_basic(
        self, temp_dirs, mock_wandb, mock_sample, mock_pred_to_pil
    ):
        """Test basic epoch run without errors."""
        outputs_dir, checkpoints_dir = temp_dirs
        wandb_run, _ = mock_wandb

        # Create mock model and optimizer
        model = MockGlideModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        diffusion = MockDiffusion()

        # Create simple dataloader with a few batches
        dataset = [create_mock_batch() for _ in range(5)]
        dataloader: Any = dataset  # Simple list works for testing, typed as Any to match DataLoader

        # Mock glide_options
        glide_options = {
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "text_ctx": 128,
        }

        # Run epoch
        steps = run_glide_finetune_epoch(
            glide_model=model,
            glide_diffusion=diffusion,
            glide_options=glide_options,
            dataloader=dataloader,
            optimizer=optimizer,
            sample_bs=1,
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
            device="cpu",
            log_frequency=2,
            sample_interval=3,
            wandb_run=wandb_run,
            epoch=1,
            early_stop=5,  # Stop after 5 steps
        )

        assert steps == 5
        # Check that metrics were logged
        assert wandb_run.log.called
        # Check that checkpoint was saved (final checkpoint)
        assert len(os.listdir(checkpoints_dir)) > 0

    def test_run_epoch_with_warmup(
        self, temp_dirs, mock_wandb, mock_sample, mock_pred_to_pil
    ):
        """Test epoch run with learning rate warmup."""
        outputs_dir, checkpoints_dir = temp_dirs
        wandb_run, _ = mock_wandb

        model = MockGlideModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        diffusion = MockDiffusion()

        dataset = [create_mock_batch() for _ in range(10)]
        dataloader: Any = dataset

        glide_options = {
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "text_ctx": 128,
        }

        # Run with warmup
        steps = run_glide_finetune_epoch(
            glide_model=model,
            glide_diffusion=diffusion,
            glide_options=glide_options,
            dataloader=dataloader,
            optimizer=optimizer,
            sample_bs=1,
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
            device="cpu",
            wandb_run=wandb_run,
            warmup_steps=5,
            warmup_type="linear",
            base_lr=1e-3,
            early_stop=10,
        )

        assert steps == 10
        # Verify warmup was applied (through logged metrics)
        logged_lrs = [
            call[0][0]["lr"]
            for call in wandb_run.log.call_args_list
            if "lr" in call[0][0]
        ]
        assert logged_lrs[0] < logged_lrs[-1]  # LR should increase

    def test_run_epoch_with_error_handling(self, temp_dirs, mock_wandb):
        """Test that errors are caught and emergency checkpoint is saved."""
        outputs_dir, checkpoints_dir = temp_dirs
        wandb_run, _ = mock_wandb

        model = MockGlideModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        diffusion = MockDiffusion()

        # Create a dataloader that will cause an error
        def error_batch_generator():
            yield create_mock_batch()
            yield create_mock_batch()
            raise RuntimeError("Simulated training error")

        dataloader = error_batch_generator()

        glide_options = {
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "text_ctx": 128,
        }

        # Run epoch - should raise but save emergency checkpoint
        with pytest.raises(RuntimeError, match="Simulated training error"):
            run_glide_finetune_epoch(
                glide_model=model,
                glide_diffusion=diffusion,
                glide_options=glide_options,
                dataloader=dataloader,
                optimizer=optimizer,
                sample_bs=1,
                outputs_dir=outputs_dir,
                checkpoints_dir=checkpoints_dir,
                device="cpu",
                wandb_run=wandb_run,
            )

        # Check that emergency checkpoint was saved
        checkpoint_files = os.listdir(checkpoints_dir)
        assert any("emergency" in f for f in checkpoint_files)

    def test_checkpoint_integration(
        self, temp_dirs, mock_wandb, mock_sample, mock_pred_to_pil
    ):
        """Test checkpoint saving with all components."""
        outputs_dir, checkpoints_dir = temp_dirs
        wandb_run, _ = mock_wandb

        model = MockGlideModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        diffusion = MockDiffusion()

        # Create dataloader with enough batches to trigger checkpoint
        dataset = [create_mock_batch() for _ in range(10)]
        dataloader: Any = dataset

        glide_options = {
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "text_ctx": 128,
        }

        checkpoint_manager = CheckpointManager(checkpoints_dir)

        run_glide_finetune_epoch(
            glide_model=model,
            glide_diffusion=diffusion,
            glide_options=glide_options,
            dataloader=dataloader,
            optimizer=optimizer,
            sample_bs=1,
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
            device="cpu",
            wandb_run=wandb_run,
            checkpoint_manager=checkpoint_manager,
            sample_interval=1000,  # Don't sample
            log_frequency=1000,  # Don't log
        )

        # Check files were created
        files = os.listdir(checkpoints_dir)
        # Model files end with .pt but NOT .optimizer.pt
        pt_files = [
            f for f in files if f.endswith(".pt") and not f.endswith(".optimizer.pt")
        ]
        json_files = [f for f in files if f.endswith(".json")]
        optimizer_files = [f for f in files if f.endswith(".optimizer.pt")]

        assert len(pt_files) >= 1  # At least final checkpoint
        assert len(json_files) == len(pt_files)  # Matching metadata
        assert len(optimizer_files) == len(pt_files)  # Matching optimizer state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
