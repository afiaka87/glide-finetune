"""Integration test for checkpoint resuming functionality."""

import json
import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch.nn as nn
import torch.optim as optim

from glide_finetune.checkpoint_utils import CheckpointManager
from train_glide import run_glide_finetune


class TestCheckpointResume:
    """Test checkpoint resuming in train_glide.py"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        data_dir = tempfile.mkdtemp()
        checkpoint_dir = tempfile.mkdtemp()

        # Create dummy data files
        for i in range(5):
            img_path = os.path.join(data_dir, f"image_{i}.png")
            txt_path = os.path.join(data_dir, f"image_{i}.txt")
            # Create dummy image (1x1 white pixel)
            import PIL.Image

            img = PIL.Image.new("RGB", (64, 64), color="white")
            img.save(img_path)
            # Create dummy caption
            with open(txt_path, "w") as f:
                f.write(f"test caption {i}")

        yield data_dir, checkpoint_dir

        shutil.rmtree(data_dir)
        shutil.rmtree(checkpoint_dir)

    @patch("train_glide.wandb_setup")
    @patch("train_glide.trange")
    def test_resume_from_checkpoint(self, mock_trange, mock_wandb, temp_dirs):
        """Test resuming training from a checkpoint."""
        data_dir, checkpoint_dir = temp_dirs

        # Mock wandb
        mock_wandb_run = Mock()
        mock_wandb_run.log = Mock()
        mock_wandb.return_value = mock_wandb_run

        # First, run training for 1 epoch to create a checkpoint
        mock_trange.side_effect = lambda start, end: range(start, min(start + 1, end))

        # Run initial training
        run_glide_finetune(
            data_dir=data_dir,
            checkpoints_dir=checkpoint_dir,
            num_epochs=3,
            batch_size=1,
            device="cuda",
            early_stop=5,  # Stop after 5 steps
            use_captions=True,
            side_x=64,
            side_y=64,
            learning_rate=1e-5,
            warmup_steps=10,
            warmup_type="linear",
        )

        # Find the checkpoint that was created
        run_dirs = [
            d
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        assert len(run_dirs) == 1
        run_dir = os.path.join(checkpoint_dir, run_dirs[0])

        # Find checkpoint files
        checkpoint_files = [
            f
            for f in os.listdir(run_dir)
            if f.endswith(".pt") and not f.endswith(".optimizer.pt")
        ]
        assert len(checkpoint_files) >= 1

        # Get the latest checkpoint
        checkpoint_path = os.path.join(run_dir, checkpoint_files[-1])

        # Check that metadata exists
        metadata_path = checkpoint_path.replace(".pt", ".json")
        assert os.path.exists(metadata_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        original_epoch = metadata["epoch"]

        # Reset mocks
        mock_wandb.reset_mock()
        mock_wandb_run.reset_mock()
        mock_trange.reset_mock()

        # Clean up GPU memory before resuming
        import torch
        torch.cuda.empty_cache()

        # Now resume training
        mock_trange.side_effect = lambda start, end: range(start, min(start + 1, end))

        run_glide_finetune(
            data_dir=data_dir,
            checkpoints_dir=checkpoint_dir,
            resume_ckpt=checkpoint_path,
            num_epochs=3,
            batch_size=1,
            device="cuda",
            early_stop=5,
            use_captions=True,
            side_x=64,
            side_y=64,
            learning_rate=1e-5,
            warmup_steps=10,
            warmup_type="linear",
        )

        # Verify that training resumed from the correct epoch
        # Should start from original_epoch + 1
        mock_trange.assert_called_with(original_epoch + 1, 3)

    def test_checkpoint_manager_integration(self):
        """Test that CheckpointManager is properly integrated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple model and optimizer
            model = nn.Linear(10, 10)
            optimizer = optim.Adam(model.parameters())

            # Create checkpoint manager and save
            manager = CheckpointManager(temp_dir)
            paths = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=2,
                step=100,
                global_step=500,
                warmup_steps=1000,
                warmup_type="cosine",
                base_lr=1e-4,
            )

            # Verify files exist
            assert os.path.exists(paths["model"])
            assert os.path.exists(paths["optimizer"])
            assert os.path.exists(paths["metadata"])

            # Create new model/optimizer and load
            new_model = nn.Linear(10, 10)
            new_optimizer = optim.Adam(new_model.parameters())

            state = manager.load_checkpoint(
                checkpoint_path=paths["model"],
                model=new_model,
                optimizer=new_optimizer,
            )

            # Verify state
            assert state["epoch"] == 2
            assert state["step"] == 100
            assert state["global_step"] == 500
            assert state["has_metadata"]
            assert state["has_optimizer_state"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
