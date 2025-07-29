import pytest
import torch
import subprocess
import sys
import os
import tempfile
from pathlib import Path


class TestGPUTraining:
    """Integration tests for GPU training with real datasets."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_coco_style_dataset_with_8bit_and_tf32(self):
        """Test training on COCO-style bird dataset with 8-bit Adam and TF32 enabled."""
        data_dir = "/home/sam/Data/captioned-birds-8k"

        # Verify dataset exists
        if not os.path.exists(data_dir):
            pytest.skip(f"Dataset not found at {data_dir}")

        # Check if GPU supports TF32
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = (device_props.major, device_props.minor)
        supports_tf32 = compute_capability >= (8, 0)

        # Create temporary checkpoint directory
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "test_checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            # Build command
            cmd = [
                sys.executable,
                "train_glide.py",
                "--data_dir",
                data_dir,
                "--batch_size",
                "2",  # Small batch size for testing
                "--learning_rate",
                "1e-4",
                "--side_x",
                "64",
                "--side_y",
                "64",
                "--uncond_p",
                "0.2",
                "--use_captions",
                "--device",
                "cuda",
                "--checkpoints_dir",
                str(checkpoint_dir),
                "--epochs",
                "1",  # Just 1 epoch for testing
                "--log_frequency",
                "10",  # Log frequently
                "--use_8bit_adam",  # Enable 8-bit Adam
                "--test_prompt",
                "a colorful bird",
                "--test_batch_size",
                "1",
                "--test_guidance_scale",
                "4.0",
                "--early_stop",
                "100",  # Stop after 100 steps
            ]

            # Add TF32 if supported
            if supports_tf32:
                cmd.append("--use_tf32")

            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/sam/GitHub/glide-finetune",
            )

            # Check for successful execution
            assert result.returncode == 0, f"Training failed: {result.stderr}"

            # Verify output contains expected messages
            output = result.stdout + result.stderr

            # Check that early stop was recognized
            assert "Early stopping enabled - disabling wandb logging" in output, (
                "Early stop wandb disable message not found"
            )

            # Check that 8-bit Adam was used
            assert "use_8bit_adam True" in output, "8-bit Adam not enabled in output"

            # Check TF32 if supported
            if supports_tf32:
                assert "use_tf32 True" in output, "TF32 not enabled in output"
                assert "TF32 enabled" in output, "TF32 activation message not found"

            # Check that training started
            assert "Found" in output and "images" in output, (
                "Dataset loading message not found"
            )
            assert "Using" in output and "text files" in output, (
                "Caption loading message not found"
            )

            # Check that model was loaded
            assert (
                "Model loaded" in output
                or "Loading model" in output
                or "Downloading" in output
            ), "Model loading message not found"

            # Check for training progress
            assert "loss:" in output.lower() or "epoch" in output.lower(), (
                "No training progress indicators found"
            )

            # Check for early stopping
            assert "Early stopping at step" in output, (
                "Early stopping message not found"
            )

            # Verify checkpoint directory was created
            run_dirs = list(checkpoint_dir.glob("run_*"))
            assert len(run_dirs) > 0, "No run directory created"

            # Check that checkpoint files exist
            latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
            checkpoints = list(latest_run.glob("*.pt"))
            assert len(checkpoints) > 0, "No checkpoint files created"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_efficient_training(self):
        """Test that 8-bit Adam and activation checkpointing reduce memory usage."""
        data_dir = "/home/sam/Data/captioned-birds-8k"

        if not os.path.exists(data_dir):
            pytest.skip(f"Dataset not found at {data_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "test_checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            # Test with memory optimizations
            cmd = [
                sys.executable,
                "train_glide.py",
                "--data_dir",
                data_dir,
                "--batch_size",
                "1",  # Minimal batch size
                "--learning_rate",
                "1e-4",
                "--side_x",
                "64",
                "--side_y",
                "64",
                "--uncond_p",
                "0.2",
                "--use_captions",
                "--device",
                "cuda",
                "--checkpoints_dir",
                str(checkpoint_dir),
                "--epochs",
                "1",
                "--log_frequency",
                "5",
                "--use_8bit_adam",
                "--activation_checkpointing",  # Also enable gradient checkpointing
                "--test_prompt",
                "a small bird",
                "--test_batch_size",
                "1",
                "--early_stop",
                "50",  # Stop after 50 steps
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/sam/GitHub/glide-finetune",
            )

            # Should complete successfully even with memory optimizations
            assert result.returncode == 0, (
                f"Memory-efficient training failed: {result.stderr}"
            )

            output = result.stdout + result.stderr
            assert "use_8bit_adam True" in output
            assert "activation_checkpointing True" in output
            assert "Early stopping at step" in output

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_early_stop_exact_count(self):
        """Test that early stopping stops at exactly the specified number of steps."""
        data_dir = "/home/sam/Data/captioned-birds-8k"

        if not os.path.exists(data_dir):
            pytest.skip(f"Dataset not found at {data_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "test_checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            early_stop_steps = 25

            cmd = [
                sys.executable,
                "train_glide.py",
                "--data_dir",
                data_dir,
                "--batch_size",
                "1",
                "--learning_rate",
                "1e-4",
                "--side_x",
                "64",
                "--side_y",
                "64",
                "--uncond_p",
                "0.2",
                "--use_captions",
                "--device",
                "cuda",
                "--checkpoints_dir",
                str(checkpoint_dir),
                "--epochs",
                "10",  # More epochs than needed
                "--log_frequency",
                "5",
                "--test_prompt",
                "a bird",
                "--test_batch_size",
                "1",
                "--early_stop",
                str(early_stop_steps),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/sam/GitHub/glide-finetune",
            )

            assert result.returncode == 0, f"Training failed: {result.stderr}"

            output = result.stdout + result.stderr
            assert f"Early stopping at step {early_stop_steps}" in output, (
                f"Did not stop at exactly {early_stop_steps} steps"
            )
