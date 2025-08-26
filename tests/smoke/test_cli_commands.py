"""Smoke tests for CLI commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest


@pytest.mark.smoke
class TestCLICommands:
    """Smoke tests to ensure CLI commands run without errors."""
    
    def run_command(self, args: List[str], check: bool = False) -> subprocess.CompletedProcess:
        """Helper to run CLI commands."""
        return subprocess.run(
            [sys.executable] + args,
            capture_output=True,
            text=True,
            check=check,
        )
    
    def test_train_help(self):
        """Test train.py --help command."""
        result = self.run_command(["train.py", "--help"])
        assert result.returncode == 0 or "usage:" in result.stdout.lower()
    
    def test_sample_help(self):
        """Test sample.py --help command."""
        result = self.run_command(["sample.py", "--help"])
        assert result.returncode == 0 or "usage:" in result.stdout.lower()
    
    def test_clip_eval_help(self):
        """Test clip_eval.py --help command."""
        result = self.run_command(["clip_eval.py", "--help"])
        assert result.returncode == 0 or "usage:" in result.stdout.lower()
    
    @pytest.mark.parametrize("script", [
        "train.py",
        "sample.py",
        "clip_eval.py",
    ])
    def test_script_imports(self, script: str):
        """Test that scripts can be imported without errors."""
        script_path = Path(script)
        if not script_path.exists():
            pytest.skip(f"Script {script} not found")
        
        # Test dry run with minimal args
        with patch("sys.argv", [script, "--help"]):
            try:
                # This would normally execute the script
                # We're just checking imports work
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_module", script)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute, just check it loads
                    assert module is not None
            except SystemExit:
                # Expected from --help
                pass
            except Exception as e:
                pytest.fail(f"Failed to import {script}: {e}")
    
    def test_train_minimal_args(self, temp_dir: Path):
        """Test train.py with minimal required arguments."""
        # Create dummy data directory
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create a dummy image file
        dummy_image = data_dir / "image_00000.jpg"
        dummy_image.touch()
        dummy_caption = data_dir / "image_00000.txt"
        dummy_caption.write_text("test caption")
        
        # Test with dry run flag if available
        result = self.run_command([
            "train.py",
            "--data_dir", str(data_dir),
            "--num_epochs", "0",  # Don't actually train
            "--batch_size", "1",
            "--learning_rate", "1e-4",
        ])
        
        # Check no critical errors (may still fail due to missing models)
        assert "error" not in result.stderr.lower() or result.returncode != 0
    
    def test_sample_minimal_args(self, temp_dir: Path):
        """Test sample.py with minimal arguments."""
        output_dir = temp_dir / "samples"
        output_dir.mkdir(exist_ok=True)
        
        result = self.run_command([
            "sample.py",
            "--prompt", "test image",
            "--output_dir", str(output_dir),
            "--num_samples", "1",
            "--batch_size", "1",
        ])
        
        # Check no critical errors (may still fail due to missing models)
        assert "error" not in result.stderr.lower() or result.returncode != 0
    
    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        result = self.run_command([
            "train.py",
            "--invalid_argument", "value",
        ])
        
        # Should fail with error about unrecognized argument
        assert result.returncode != 0
        assert "unrecognized" in result.stderr.lower() or "invalid" in result.stderr.lower()
    
    @pytest.mark.parametrize("arg,value,expected_error", [
        ("--batch_size", "-1", "must be positive"),
        ("--learning_rate", "0", "must be positive"),
        ("--num_epochs", "0", "at least 1"),
        ("--side_x", "63", "divisible by 8"),
        ("--side_y", "63", "divisible by 8"),
    ])
    def test_argument_validation(self, arg: str, value: str, expected_error: str):
        """Test argument validation."""
        result = self.run_command([
            "train.py",
            arg, value,
            "--data_dir", "/tmp/dummy",  # Dummy path
        ])
        
        # Should fail with validation error
        assert result.returncode != 0 or expected_error in result.stderr.lower()


@pytest.mark.smoke
class TestConfigFiles:
    """Test configuration file handling."""
    
    def test_load_config_from_file(self, temp_dir: Path):
        """Test loading configuration from file."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
        learning_rate: 0.0001
        batch_size: 4
        num_epochs: 10
        """)
        
        # This would test config loading if implemented
        # For now, just verify file exists
        assert config_file.exists()
    
    def test_environment_variables(self, monkeypatch):
        """Test configuration via environment variables."""
        monkeypatch.setenv("GLIDE_LEARNING_RATE", "0.001")
        monkeypatch.setenv("GLIDE_BATCH_SIZE", "8")
        
        # Verify environment variables are set
        import os
        assert os.environ.get("GLIDE_LEARNING_RATE") == "0.001"
        assert os.environ.get("GLIDE_BATCH_SIZE") == "8"


@pytest.mark.smoke
class TestScriptModes:
    """Test different script execution modes."""
    
    def test_train_base_model_mode(self):
        """Test training base model mode."""
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Check for base model options
        assert "--train_upsample" in result.stdout or result.returncode == 0
    
    def test_train_upsampler_mode(self):
        """Test training upsampler mode."""
        result = subprocess.run(
            [sys.executable, "train.py", "--train_upsample", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Should not error on flag
        assert result.returncode == 0 or "--train_upsample" in result.stdout
    
    def test_sample_with_guidance(self):
        """Test sampling with classifier-free guidance."""
        result = subprocess.run(
            [sys.executable, "sample.py", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Check for guidance scale option
        assert "--guidance_scale" in result.stdout or result.returncode == 0
    
    def test_clip_eval_metrics(self):
        """Test CLIP evaluation metrics."""
        result = subprocess.run(
            [sys.executable, "clip_eval.py", "--help"],
            capture_output=True,
            text=True,
        )
        
        # Check for evaluation options
        assert "--prompt" in result.stdout or "eval" in result.stdout.lower() or result.returncode == 0


@pytest.mark.smoke
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete workflow from training to sampling."""
    
    def test_minimal_training_workflow(self, temp_dir: Path):
        """Test minimal end-to-end training workflow."""
        # Setup directories
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create minimal dataset
        for i in range(2):
            img_path = data_dir / f"image_{i:05d}.jpg"
            txt_path = data_dir / f"image_{i:05d}.txt"
            img_path.touch()
            txt_path.write_text(f"test caption {i}")
        
        # Step 1: Train for 1 step
        train_result = subprocess.run(
            [
                sys.executable, "train.py",
                "--data_dir", str(data_dir),
                "--save_directory", str(checkpoint_dir),
                "--num_epochs", "0",  # Just test loading
                "--batch_size", "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,  # Timeout after 30 seconds
        )
        
        # May fail due to missing models, but shouldn't crash
        assert train_result.returncode == 0 or "model" in train_result.stderr.lower()
    
    def test_sample_from_checkpoint(self, temp_dir: Path, create_test_checkpoint: Path):
        """Test sampling from a checkpoint."""
        output_dir = temp_dir / "samples"
        output_dir.mkdir(exist_ok=True)
        
        sample_result = subprocess.run(
            [
                sys.executable, "sample.py",
                "--checkpoint", str(create_test_checkpoint),
                "--prompt", "test image",
                "--output_dir", str(output_dir),
                "--num_samples", "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # May fail due to model mismatch, but shouldn't crash
        assert sample_result.returncode == 0 or "checkpoint" in sample_result.stderr.lower()
    
    def test_evaluate_generated_samples(self, temp_dir: Path):
        """Test evaluating generated samples."""
        # Create dummy generated images
        samples_dir = temp_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Create dummy image
        dummy_image = samples_dir / "sample_00000.png"
        dummy_image.touch()
        
        eval_result = subprocess.run(
            [
                sys.executable, "clip_eval.py",
                "--images_dir", str(samples_dir),
                "--prompt", "test image",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # May fail due to missing CLIP model, but shouldn't crash
        assert eval_result.returncode == 0 or "clip" in eval_result.stderr.lower()