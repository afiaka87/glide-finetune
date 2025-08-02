#!/usr/bin/env python3
"""
Integration test for CLIP text embedding pre-computation script.

Tests the full pipeline:
1. Create a temporary dataset with images and text files
2. Run the pre-computation script
3. Verify .clip files are created with correct format
4. Load embeddings and verify they work with TextImageDataset
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from glide_finetune.loader import TextImageDataset
from scripts.precompute_clip_text_embeddings import precompute_embeddings


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory with sample images and captions."""
    temp_dir = tempfile.mkdtemp()

    # Create sample images and captions
    samples = [
        ("cat", "A fluffy orange cat sitting on a windowsill"),
        ("dog", "A golden retriever playing in the park"),
        ("landscape", "Beautiful mountain landscape at sunset"),
        ("city", ""),  # Empty caption to test edge case
        (
            "ocean",
            "Deep blue ocean waves crashing on the shore\nWith seagulls flying overhead",
        ),  # Multi-line
    ]

    for name, caption in samples:
        # Create a simple test image
        img = Image.new(
            "RGB",
            (256, 256),
            color=(
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            ),
        )
        img.save(Path(temp_dir) / f"{name}.jpg")

        # Create caption file
        if caption:  # Skip creating text file for empty caption test
            with open(Path(temp_dir) / f"{name}.txt", "w") as f:
                f.write(caption)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


def test_precompute_clip_embeddings_basic(temp_dataset_dir):
    """Test basic pre-computation functionality."""
    # Run pre-computation
    stats = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",  # Use smaller model for tests
        batch_size=2,
        device="cpu",  # Use CPU for tests
        force_recompute=False,
        dry_run=False,
    )

    # Check statistics
    assert stats["processed"] == 4  # Should process 4 text files
    assert stats["skipped"] == 0
    assert stats["errors"] == 0

    # Verify .clip files were created
    clip_files = list(Path(temp_dataset_dir).glob("*.clip"))
    assert len(clip_files) == 4

    # Check file format for each clip file
    for clip_file in clip_files:
        data = torch.load(clip_file, map_location="cpu")

        # Verify required fields
        assert "clip_model" in data
        assert data["clip_model"] == "ViT-B/32"
        assert "embedding" in data
        assert "caption" in data
        assert "created_at" in data
        assert "embedding_dim" in data

        # Verify embedding shape
        embedding = data["embedding"]
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (512,)  # ViT-B/32 outputs 512-d embeddings
        assert data["embedding_dim"] == 512

        # Verify embedding is normalized (CLIP standard)
        norm = torch.norm(embedding)
        assert abs(norm - 1.0) < 1e-5


def test_precompute_with_existing_cache(temp_dataset_dir):
    """Test behavior with existing cache files."""
    # First run
    stats1 = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        batch_size=2,
        device="cpu",
        force_recompute=False,
        dry_run=False,
    )

    assert stats1["processed"] == 4
    assert stats1["skipped"] == 0

    # Second run - should skip existing files
    stats2 = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        batch_size=2,
        device="cpu",
        force_recompute=False,
        dry_run=False,
    )

    assert stats2["processed"] == 0
    assert stats2["skipped"] == 4

    # Force recompute
    stats3 = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        batch_size=2,
        device="cpu",
        force_recompute=True,
        dry_run=False,
    )

    assert stats3["processed"] == 4
    assert stats3["skipped"] == 0


def test_integration_with_text_image_dataset(temp_dataset_dir):
    """Test that pre-computed embeddings work with TextImageDataset."""
    # Pre-compute embeddings
    precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        batch_size=2,
        device="cpu",
    )

    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.n_vocab = 50257
            self.end_token = self.n_vocab - 1

        def encode(self, text):
            return list(range(10))  # Simple mock tokens

        def padded_tokens_and_mask(self, tokens, text_ctx):
            tokens = tokens[:text_ctx]
            padding = text_ctx - len(tokens)
            padded_tokens = tokens + [self.end_token] * padding
            mask = [True] * len(tokens) + [False] * padding
            return padded_tokens, mask

    # Load dataset with CLIP cache enabled
    dataset = TextImageDataset(
        folder=temp_dataset_dir,
        side_x=64,
        side_y=64,
        tokenizer=MockTokenizer(),
        use_captions=True,
        use_clip_cache=True,
        clip_model_name="ViT-B/32",
    )

    # Check cache statistics were initialized
    stats = dataset.get_clip_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["errors"] == 0

    # Load a sample with cached embedding
    tokens, mask, image, clip_embedding = dataset[0]

    # Verify we got a 4-tuple (with CLIP embedding)
    assert clip_embedding is not None
    assert isinstance(clip_embedding, torch.Tensor)
    assert clip_embedding.shape == (512,)

    # Check cache hit
    stats = dataset.get_clip_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0

    # Verify the dataset doesn't include "city" (no text file)
    assert "city" not in dataset.keys

    # All samples should have embeddings since they all have text files
    assert len(dataset.keys) == 4  # Should exclude "city"


def test_different_clip_models(temp_dataset_dir):
    """Test pre-computation with different CLIP models."""
    # Pre-compute with ViT-B/32
    stats1 = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    assert stats1["processed"] == 4

    # Try to load with different model - should not use cache
    class MockTokenizer:
        def __init__(self):
            self.n_vocab = 50257
            self.end_token = self.n_vocab - 1

        def encode(self, text):
            return list(range(10))

        def padded_tokens_and_mask(self, tokens, text_ctx):
            tokens = tokens[:text_ctx]
            padding = text_ctx - len(tokens)
            padded_tokens = tokens + [self.end_token] * padding
            mask = [True] * len(tokens) + [False] * padding
            return padded_tokens, mask

    dataset = TextImageDataset(
        folder=temp_dataset_dir,
        side_x=64,
        side_y=64,
        tokenizer=MockTokenizer(),
        use_captions=True,
        use_clip_cache=True,
        clip_model_name="ViT-L/14",  # Different model
    )

    # Should get cache miss due to model mismatch
    tokens, mask, image, clip_embedding = dataset[0]
    assert clip_embedding is None

    stats = dataset.get_clip_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_dry_run_mode(temp_dataset_dir):
    """Test dry run mode doesn't create files."""
    # Run in dry run mode
    stats = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
        dry_run=True,
    )

    # Should report what would be processed
    assert stats["processed"] == 4
    assert stats["skipped"] == 0

    # But no .clip files should be created
    clip_files = list(Path(temp_dataset_dir).glob("*.clip"))
    assert len(clip_files) == 0


def test_multiline_captions(temp_dataset_dir):
    """Test handling of multi-line captions."""
    # Pre-compute embeddings
    precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
    )

    # Load the ocean.clip file (which has multi-line caption)
    ocean_clip = Path(temp_dataset_dir) / "ocean.clip"
    data = torch.load(ocean_clip, map_location="cpu")

    # Should use first line only for pre-computation
    assert data["caption"] == "Deep blue ocean waves crashing on the shore"
    assert "seagulls" not in data["caption"]


def test_batch_processing(temp_dataset_dir):
    """Test that batch processing works correctly."""
    # Create more files to test batching
    for i in range(10):
        with open(Path(temp_dataset_dir) / f"extra_{i}.txt", "w") as f:
            f.write(f"Test caption number {i}")
        # Create dummy image
        img = Image.new("RGB", (64, 64))
        img.save(Path(temp_dataset_dir) / f"extra_{i}.jpg")

    # Run with small batch size
    stats = precompute_embeddings(
        data_dir=temp_dataset_dir,
        clip_model_name="ViT-B/32",
        batch_size=3,
        device="cpu",
    )

    # Should process all files
    assert stats["processed"] == 14  # 4 original + 10 extra
    assert stats["errors"] == 0

    # Verify all clip files exist
    clip_files = list(Path(temp_dataset_dir).glob("*.clip"))
    assert len(clip_files) == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
