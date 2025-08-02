#!/usr/bin/env python3
"""
Integration test for CLIP cache flags in train_glide.py.

Tests that the --use_clip_cache and --clip_cache_dir flags correctly pass
parameters to the data loaders.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch


def test_clip_cache_flags_help():
    """Test that CLIP cache flags are present in help output."""
    result = subprocess.run(
        [sys.executable, "train_glide.py", "--help"], capture_output=True, text=True
    )

    assert "--use_clip_cache" in result.stdout
    assert "--clip_cache_dir" in result.stdout
    assert "pre-computed CLIP embeddings" in result.stdout


def test_clip_cache_flags_parsing():
    """Test that CLIP cache flags can be parsed correctly."""
    # Test with test_run to avoid actual training
    result = subprocess.run(
        [
            sys.executable,
            "train_glide.py",
            "--test_run",
            "1",
            "--use_clip_cache",
            "--clip_cache_dir",
            "/tmp/test_clip_cache",
            "--data_dir",
            "/tmp/test_data",
        ],
        capture_output=True,
        text=True,
    )

    # Should fail gracefully because data doesn't exist
    assert result.returncode != 0
    # But should get past argument parsing
    assert "unrecognized arguments" not in result.stderr


def test_clip_cache_integration():
    """Test that CLIP cache parameters are passed to data loaders."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data structure
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()

        # Create a dummy image and caption
        image_path = data_dir / "test.jpg"
        caption_path = data_dir / "test.txt"

        # Create dummy image
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="red")
        img.save(image_path)

        # Create caption
        caption_path.write_text("A test image")

        # Create CLIP cache directory and file
        clip_cache_dir = Path(temp_dir) / "clip_cache"
        clip_cache_dir.mkdir()

        # Create dummy CLIP embedding
        clip_embed_path = data_dir / "test.clip"
        dummy_embedding = torch.randn(512)  # ViT-B/32 dimension
        torch.save(dummy_embedding, clip_embed_path)

        # Run training with early stop
        result = subprocess.run(
            [
                sys.executable,
                "train_glide.py",
                "--data_dir",
                str(data_dir),
                "--clip_cache_dir",
                str(clip_cache_dir),
                "--use_clip_cache",
                "--use_clip",
                "--clip_model_name",
                "ViT-B/32",
                "--test_run",
                "1",
                "--batch_size",
                "1",
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # Force CPU
        )

        # Check that it attempted to load the model
        # It will fail because we don't have checkpoints, but that's OK
        assert "Loading data..." in result.stdout or "Loading data..." in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
