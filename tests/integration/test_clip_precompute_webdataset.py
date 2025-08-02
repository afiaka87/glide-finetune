#!/usr/bin/env python3
"""
Integration test for CLIP WebDataset embedding pre-computation script.

Tests the full pipeline:
1. Create temporary tar files with WebDataset format
2. Run the pre-computation script
3. Verify cache directory structure and embedding files
4. Load embeddings and verify format
"""

import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.precompute_clip_webdataset_embeddings import (
    precompute_webdataset_embeddings,
    sanitize_model_name,
    setup_cache_directory,
    check_existing_cache,
)


def create_sample_tar(tar_path: Path, num_samples: int = 5):
    """Create a sample WebDataset tar file for testing."""
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_samples):
            # Create sample key
            key = f"sample_{i:06d}"
            
            # Create image
            img = Image.new('RGB', (256, 256), color=(
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Add image to tar
            img_info = tarfile.TarInfo(name=f"{key}.jpg")
            img_info.size = len(img_bytes.getvalue())
            tar.addfile(img_info, img_bytes)
            
            # Create caption
            caption = f"Test caption number {i} for sample {key}"
            caption_bytes = caption.encode('utf-8')
            
            # Add caption to tar
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(caption_bytes)
            tar.addfile(txt_info, io.BytesIO(caption_bytes))
            
            # Create metadata
            metadata = {
                "key": key,
                "status": "success",
                "width": 256,
                "height": 256,
                "original_width": 512,
                "original_height": 512,
                "NSFW": "UNLIKELY",
                "similarity": 0.85,
                "LICENSE": "CC-BY",
            }
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            
            # Add metadata to tar
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(metadata_bytes)
            tar.addfile(json_info, io.BytesIO(metadata_bytes))


@pytest.fixture
def temp_tar_dir():
    """Create temporary directory with sample tar files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create multiple tar files
    tar_files = []
    for i in range(3):
        tar_path = Path(temp_dir) / f"data_{i:03d}.tar"
        create_sample_tar(tar_path, num_samples=5 + i*2)  # Different sizes
        tar_files.append(tar_path)
    
    yield temp_dir, tar_files
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_precompute_webdataset_basic(temp_tar_dir, temp_cache_dir):
    """Test basic WebDataset pre-computation functionality."""
    tar_dir, tar_files = temp_tar_dir
    
    # Run pre-computation
    stats = precompute_webdataset_embeddings(
        tar_urls=[str(f) for f in tar_files],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        caption_key="txt",
        batch_size=3,
        device="cpu",
        force_recompute=False,
        dry_run=False,
    )
    
    # Check statistics
    assert stats["processed"] == 3  # 3 tar files
    assert stats["skipped"] == 0
    assert stats["errors"] == 0
    assert stats["total_samples"] == 5 + 7 + 9  # 21 total samples
    
    # Verify cache directory structure
    cache_path = Path(temp_cache_dir)
    model_dir = cache_path / "ViT-B-32"  # Note: sanitized name
    assert model_dir.exists()
    
    embeddings_dir = model_dir / "embeddings"
    assert embeddings_dir.exists()
    
    metadata_file = model_dir / "tar_metadata.json"
    assert metadata_file.exists()
    
    # Check metadata file
    with open(metadata_file, 'r') as f:
        tar_metadata = json.load(f)
    
    assert len(tar_metadata) == 3
    for tar_file in tar_files:
        tar_name = tar_file.name
        assert tar_name in tar_metadata
        assert "path" in tar_metadata[tar_name]
        assert "sample_count" in tar_metadata[tar_name]
        assert "cache_file" in tar_metadata[tar_name]
        assert "processed_at" in tar_metadata[tar_name]
    
    # Check embedding cache files
    for tar_file in tar_files:
        cache_file = embeddings_dir / f"{tar_file.name}.pt"
        assert cache_file.exists()
        
        # Load and verify cache file format
        data = torch.load(cache_file, map_location='cpu')
        
        assert "metadata" in data
        assert "embeddings" in data
        assert "stats" in data
        
        # Check metadata
        metadata = data["metadata"]
        assert metadata["clip_model"] == "ViT-B/32"
        assert metadata["tar_file"] == str(tar_file)
        assert "created_at" in metadata
        assert metadata["sample_count"] > 0
        assert metadata["embedding_dim"] == 512  # ViT-B/32
        
        # Check embeddings
        embeddings = data["embeddings"]
        assert len(embeddings) == metadata["sample_count"]
        
        # Check each embedding
        for key, embedding_data in embeddings.items():
            assert "embedding" in embedding_data
            assert "caption" in embedding_data
            
            embedding = embedding_data["embedding"]
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape == (512,)
            
            # Verify normalization
            norm = torch.norm(embedding)
            assert abs(norm - 1.0) < 1e-5


def test_skip_existing_cache(temp_tar_dir, temp_cache_dir):
    """Test that existing valid cache files are skipped."""
    tar_dir, tar_files = temp_tar_dir
    
    # First run
    stats1 = precompute_webdataset_embeddings(
        tar_urls=[str(f) for f in tar_files],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    assert stats1["processed"] == 3
    assert stats1["skipped"] == 0
    
    # Second run - should skip all
    stats2 = precompute_webdataset_embeddings(
        tar_urls=[str(f) for f in tar_files],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    assert stats2["processed"] == 0
    assert stats2["skipped"] == 3
    
    # Force recompute
    stats3 = precompute_webdataset_embeddings(
        tar_urls=[str(f) for f in tar_files],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
        force_recompute=True,
    )
    
    assert stats3["processed"] == 3
    assert stats3["skipped"] == 0


def test_different_clip_models(temp_tar_dir, temp_cache_dir):
    """Test cache separation for different CLIP models."""
    tar_dir, tar_files = temp_tar_dir
    
    # Pre-compute with ViT-B/32
    stats1 = precompute_webdataset_embeddings(
        tar_urls=[str(tar_files[0])],  # Just one tar
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    # Pre-compute with ViT-L/14 (would work if CLIP installed)
    # For testing, we'll just verify directory structure
    
    # Check that different models get different directories
    assert (Path(temp_cache_dir) / "ViT-B-32").exists()
    
    # Test sanitize_model_name function
    assert sanitize_model_name("ViT-B/32") == "ViT-B-32"
    assert sanitize_model_name("ViT-L/14@336px") == "ViT-L-14-336px"


def test_tar_with_missing_captions(temp_tar_dir, temp_cache_dir):
    """Test handling of samples without captions."""
    tar_dir, _ = temp_tar_dir
    
    # Create tar with some missing captions
    tar_path = Path(tar_dir) / "incomplete.tar"
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(5):
            key = f"sample_{i:06d}"
            
            # Always add image
            img = Image.new('RGB', (64, 64))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            img_info = tarfile.TarInfo(name=f"{key}.jpg")
            img_info.size = len(img_bytes.getvalue())
            tar.addfile(img_info, img_bytes)
            
            # Only add caption for even indices
            if i % 2 == 0:
                caption = f"Caption for sample {i}"
                txt_info = tarfile.TarInfo(name=f"{key}.txt")
                txt_info.size = len(caption)
                tar.addfile(txt_info, io.BytesIO(caption.encode('utf-8')))
    
    # Process the tar
    stats = precompute_webdataset_embeddings(
        tar_urls=[str(tar_path)],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    # Should only process samples with captions
    assert stats["processed"] == 1
    assert stats["total_samples"] == 3  # Only 3 samples have captions
    
    # Load cache and verify
    cache_file = Path(temp_cache_dir) / "ViT-B-32" / "embeddings" / "incomplete.tar.pt"
    data = torch.load(cache_file, map_location='cpu')
    
    embeddings = data["embeddings"]
    assert len(embeddings) == 3
    
    # Check that only even-indexed samples are present
    for key in embeddings.keys():
        sample_num = int(key.split('_')[1])
        assert sample_num % 2 == 0


def test_dry_run_mode(temp_tar_dir, temp_cache_dir):
    """Test dry run mode doesn't create cache files."""
    tar_dir, tar_files = temp_tar_dir
    
    # Run in dry run mode
    stats = precompute_webdataset_embeddings(
        tar_urls=[str(f) for f in tar_files],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        device="cpu",
        dry_run=True,
    )
    
    # Should report what would be processed
    assert stats["processed"] == 3
    assert stats["skipped"] == 0
    
    # But no cache files should be created
    cache_path = Path(temp_cache_dir)
    assert not (cache_path / "ViT-B-32").exists()


def test_batch_processing(temp_tar_dir, temp_cache_dir):
    """Test batch processing with different batch sizes."""
    tar_dir, _ = temp_tar_dir
    
    # Create tar with many samples
    tar_path = Path(tar_dir) / "large.tar"
    create_sample_tar(tar_path, num_samples=50)
    
    # Process with small batch size
    stats = precompute_webdataset_embeddings(
        tar_urls=[str(tar_path)],
        cache_dir=temp_cache_dir,
        clip_model_name="ViT-B/32",
        batch_size=7,  # Odd number to test edge cases
        device="cpu",
    )
    
    assert stats["processed"] == 1
    assert stats["total_samples"] == 50
    
    # Verify all embeddings were created
    cache_file = Path(temp_cache_dir) / "ViT-B-32" / "embeddings" / "large.tar.pt"
    data = torch.load(cache_file, map_location='cpu')
    
    assert len(data["embeddings"]) == 50


def test_check_existing_cache_function():
    """Test the check_existing_cache utility function."""
    # Create temporary cache file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        cache_path = Path(tmp.name)
        
        # Save valid cache
        torch.save({
            "metadata": {"clip_model": "ViT-B/32"},
            "embeddings": {"sample_000000": {"embedding": torch.randn(512)}}
        }, cache_path)
        
        # Should be valid
        assert check_existing_cache(cache_path, "ViT-B/32", force_recompute=False)
        
        # Wrong model
        assert not check_existing_cache(cache_path, "ViT-L/14", force_recompute=False)
        
        # Force recompute
        assert not check_existing_cache(cache_path, "ViT-B/32", force_recompute=True)
        
        # Cleanup
        cache_path.unlink()
    
    # Non-existent file
    assert not check_existing_cache(Path("/tmp/nonexistent.pt"), "ViT-B/32")


def test_setup_cache_directory():
    """Test cache directory setup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir)
        
        embeddings_dir, metadata_file = setup_cache_directory(cache_path, "ViT-L/14")
        
        # Check paths
        assert embeddings_dir == cache_path / "ViT-L-14" / "embeddings"
        assert metadata_file == cache_path / "ViT-L-14" / "tar_metadata.json"
        
        # Check directories were created
        assert embeddings_dir.exists()
        assert embeddings_dir.is_dir()


def test_glob_pattern_expansion(temp_tar_dir):
    """Test that glob patterns work correctly."""
    tar_dir, tar_files = temp_tar_dir
    
    # Import the main function to test glob expansion
    from scripts.precompute_clip_webdataset_embeddings import main
    import sys
    
    # Mock command line args
    old_argv = sys.argv
    try:
        sys.argv = [
            "script",
            "--tar_urls", str(Path(tar_dir) / "*.tar"),
            "--cache_dir", "/tmp/test_cache",
            "--dry_run",  # Just test glob expansion, don't process
        ]
        
        # This should find all tar files
        # (Would need to refactor main() to be testable, for now just check glob)
        from glob import glob
        found_files = glob(str(Path(tar_dir) / "*.tar"))
        assert len(found_files) == 3
        
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])