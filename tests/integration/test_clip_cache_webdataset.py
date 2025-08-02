#!/usr/bin/env python3
"""
Integration test for CLIP cache loading in WebDataset.

Tests the full pipeline:
1. Create temporary tar files
2. Pre-compute CLIP embeddings 
3. Load WebDataset with CLIP cache enabled
4. Verify embeddings are loaded correctly
"""

import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from glide_finetune.wds_loader import glide_wds_loader
from scripts.precompute_clip_webdataset_embeddings import precompute_webdataset_embeddings


def create_test_tar(tar_path: Path, num_samples: int = 5):
    """Create a test WebDataset tar file."""
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_samples):
            key = f"sample_{i:06d}"
            
            # Create image
            img = Image.new('RGB', (256, 256), color=(
                (i * 50) % 255,
                (i * 75) % 255,
                (i * 100) % 255
            ))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Add image
            img_info = tarfile.TarInfo(name=f"{key}.jpg")
            img_info.size = len(img_bytes.getvalue())
            tar.addfile(img_info, img_bytes)
            
            # Add caption
            caption = f"Test image number {i} with unique caption"
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(caption)
            tar.addfile(txt_info, io.BytesIO(caption.encode('utf-8')))
            
            # Add metadata
            metadata = {
                "key": key,
                "width": 256,
                "height": 256,
                "original_width": 512,
                "original_height": 512,
                "NSFW": "UNLIKELY",
                "similarity": 0.85,
                "LICENSE": "CC-BY",
            }
            json_bytes = json.dumps(metadata).encode('utf-8')
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))


@pytest.fixture
def temp_dataset():
    """Create temporary dataset with tar files and CLIP cache."""
    temp_dir = tempfile.mkdtemp()
    
    # Create tar files
    tar_dir = Path(temp_dir) / "tars"
    tar_dir.mkdir(exist_ok=True)
    
    tar_files = []
    for i in range(2):
        tar_path = tar_dir / f"test_{i:03d}.tar"
        create_test_tar(tar_path, num_samples=5)
        tar_files.append(tar_path)
    
    # Create cache directory
    cache_dir = Path(temp_dir) / "clip_cache"
    cache_dir.mkdir(exist_ok=True)
    
    yield {
        "temp_dir": temp_dir,
        "tar_files": tar_files,
        "cache_dir": cache_dir,
    }
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_webdataset_clip_cache_loading(temp_dataset):
    """Test loading CLIP embeddings from cache in WebDataset."""
    tar_files = temp_dataset["tar_files"]
    cache_dir = temp_dataset["cache_dir"]
    
    # Pre-compute CLIP embeddings
    stats = precompute_webdataset_embeddings(
        tar_urls=[str(f) for f in tar_files],
        cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    assert stats["processed"] == 2
    assert stats["total_samples"] == 10
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.n_vocab = 50257
            self.end_token = self.n_vocab - 1
            
        def encode(self, text):
            return list(range(min(10, len(text))))
            
        def padded_tokens_and_mask(self, tokens, text_ctx):
            tokens = tokens[:text_ctx]
            padding = text_ctx - len(tokens)
            padded_tokens = tokens + [self.end_token] * padding
            mask = [True] * len(tokens) + [False] * padding
            return padded_tokens, mask
    
    # Load WebDataset without CLIP cache
    dataset_no_cache, stats_no_cache = glide_wds_loader(
        [str(f) for f in tar_files],
        tokenizer=MockTokenizer(),
        base_x=64,
        base_y=64,
        uncond_p=0.0,  # No unconditional for testing
        use_clip_cache=False,
        laion_no_filter=True,  # Disable filtering for test
    )
    
    # Load WebDataset with CLIP cache
    dataset_with_cache, stats_with_cache = glide_wds_loader(
        [str(f) for f in tar_files],
        tokenizer=MockTokenizer(),
        base_x=64,
        base_y=64,
        uncond_p=0.0,
        use_clip_cache=True,
        clip_cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        laion_no_filter=True,
    )
    
    # Verify outputs without cache (should be 3-tuples)
    samples_no_cache = []
    for i, sample in enumerate(dataset_no_cache):
        if i >= 5:  # Just check first 5 samples
            break
        samples_no_cache.append(sample)
        assert len(sample) == 3  # tokens, mask, image
        tokens, mask, image = sample
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
    
    # Verify outputs with cache (should be 4-tuples)
    samples_with_cache = []
    clip_embeddings_found = 0
    for i, sample in enumerate(dataset_with_cache):
        if i >= 5:  # Just check first 5 samples
            break
        samples_with_cache.append(sample)
        assert len(sample) == 4  # tokens, mask, image, clip_embedding
        tokens, mask, image, clip_embedding = sample
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        
        # Check CLIP embedding
        if clip_embedding is not None:
            assert isinstance(clip_embedding, torch.Tensor)
            assert clip_embedding.shape == (512,)  # ViT-B/32
            # Verify normalization
            norm = torch.norm(clip_embedding)
            assert abs(norm - 1.0) < 1e-5
            clip_embeddings_found += 1
    
    # All samples should have CLIP embeddings
    assert clip_embeddings_found == 5
    
    # Check statistics
    summary = stats_with_cache.get_summary()
    assert "clip_cache_hits" in summary
    assert "clip_cache_misses" in summary
    assert "clip_cache_hit_rate" in summary
    assert summary["clip_cache_hits"] > 0
    assert summary["clip_cache_hit_rate"] > 0.9  # Should be close to 100%


def test_webdataset_clip_cache_missing_samples(temp_dataset):
    """Test handling of missing samples in CLIP cache."""
    tar_files = temp_dataset["tar_files"]
    cache_dir = temp_dataset["cache_dir"]
    
    # Pre-compute only for first tar
    stats = precompute_webdataset_embeddings(
        tar_urls=[str(tar_files[0])],
        cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.n_vocab = 50257
            self.end_token = self.n_vocab - 1
            
        def encode(self, text):
            return list(range(min(10, len(text))))
            
        def padded_tokens_and_mask(self, tokens, text_ctx):
            tokens = tokens[:text_ctx]
            padding = text_ctx - len(tokens)
            padded_tokens = tokens + [self.end_token] * padding
            mask = [True] * len(tokens) + [False] * padding
            return padded_tokens, mask
    
    # Load both tars with cache enabled
    dataset, stats = glide_wds_loader(
        [str(f) for f in tar_files],
        tokenizer=MockTokenizer(),
        base_x=64,
        base_y=64,
        uncond_p=0.0,
        use_clip_cache=True,
        clip_cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        laion_no_filter=True,
    )
    
    # Process all samples
    samples_with_embedding = 0
    samples_without_embedding = 0
    
    for sample in dataset:
        assert len(sample) == 4
        tokens, mask, image, clip_embedding = sample
        
        if clip_embedding is not None:
            samples_with_embedding += 1
        else:
            samples_without_embedding += 1
    
    # First tar should have embeddings, second should not
    assert samples_with_embedding == 5
    assert samples_without_embedding == 5
    
    # Check statistics
    summary = stats.get_summary()
    assert summary["clip_cache_hits"] == 5
    assert summary["clip_cache_misses"] == 5
    assert abs(summary["clip_cache_hit_rate"] - 0.5) < 0.01


def test_webdataset_clip_cache_model_mismatch(temp_dataset):
    """Test handling of model mismatch in CLIP cache."""
    tar_files = temp_dataset["tar_files"]
    cache_dir = temp_dataset["cache_dir"]
    
    # Pre-compute with ViT-B/32
    precompute_webdataset_embeddings(
        tar_urls=[str(tar_files[0])],
        cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.n_vocab = 50257
            self.end_token = self.n_vocab - 1
            
        def encode(self, text):
            return list(range(min(10, len(text))))
            
        def padded_tokens_and_mask(self, tokens, text_ctx):
            tokens = tokens[:text_ctx]
            padding = text_ctx - len(tokens)
            padded_tokens = tokens + [self.end_token] * padding
            mask = [True] * len(tokens) + [False] * padding
            return padded_tokens, mask
    
    # Try to load with different model
    dataset, stats = glide_wds_loader(
        [str(tar_files[0])],
        tokenizer=MockTokenizer(),
        base_x=64,
        base_y=64,
        uncond_p=0.0,
        use_clip_cache=True,
        clip_cache_dir=str(cache_dir),
        clip_model_name="ViT-L/14",  # Different model
        laion_no_filter=True,
    )
    
    # All embeddings should be None due to model mismatch
    for i, sample in enumerate(dataset):
        if i >= 5:
            break
        assert len(sample) == 4
        tokens, mask, image, clip_embedding = sample
        assert clip_embedding is None
    
    # Check statistics - all should be misses
    summary = stats.get_summary()
    assert summary["clip_cache_hits"] == 0
    assert summary["clip_cache_misses"] == 5


def test_webdataset_clip_cache_with_upsampling(temp_dataset):
    """Test CLIP cache loading with upsampling enabled."""
    tar_files = temp_dataset["tar_files"]
    cache_dir = temp_dataset["cache_dir"]
    
    # Pre-compute embeddings
    precompute_webdataset_embeddings(
        tar_urls=[str(tar_files[0])],
        cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        device="cpu",
    )
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.n_vocab = 50257
            self.end_token = self.n_vocab - 1
            
        def encode(self, text):
            return list(range(min(10, len(text))))
            
        def padded_tokens_and_mask(self, tokens, text_ctx):
            tokens = tokens[:text_ctx]
            padding = text_ctx - len(tokens)
            padded_tokens = tokens + [self.end_token] * padding
            mask = [True] * len(tokens) + [False] * padding
            return padded_tokens, mask
    
    # Load with upsampling and cache
    dataset, stats = glide_wds_loader(
        [str(tar_files[0])],
        tokenizer=MockTokenizer(),
        base_x=64,
        base_y=64,
        uncond_p=0.0,
        enable_upsample=True,
        upscale_factor=4,
        use_clip_cache=True,
        clip_cache_dir=str(cache_dir),
        clip_model_name="ViT-B/32",
        laion_no_filter=True,
    )
    
    # Verify 5-tuple output
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        assert len(sample) == 5  # tokens, mask, base_image, up_image, clip_embedding
        tokens, mask, base_image, up_image, clip_embedding = sample
        
        assert base_image.shape == (3, 64, 64)
        assert up_image.shape == (3, 256, 256)
        assert clip_embedding is not None
        assert clip_embedding.shape == (512,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])