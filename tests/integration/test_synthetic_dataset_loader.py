"""Test loading synthetic DALLE-3 dataset with JSON captions."""

import pytest
import torch as th
from pathlib import Path
from glob import glob
from unittest.mock import MagicMock

from glide_finetune.glide_util import get_tokens_and_mask
from glide_finetune.wds_loader import glide_wds_loader


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def encode(self, text):
        # Simple mock encoding - just convert to char codes
        return [ord(c) for c in text[:128]]  # Limit to 128 tokens
    
    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens if t < 256])


@pytest.fixture
def synthetic_dataset_path():
    """Path to synthetic dataset tar file."""
    # Use local test tar file
    local_tar = Path(__file__).parent.parent.parent / "data-000000.tar"
    if local_tar.exists():
        return str(local_tar)
    
    # Fall back to full dataset if local tar not available
    path = "/mnt/t7_2tb/Data/synthetic/synthetic-dataset-1m-dalle3-high-quality-captions/data"
    if not Path(path).exists():
        pytest.skip(f"Neither local tar nor synthetic dataset found")
    return path


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MockTokenizer()


def test_synthetic_dataset_basic_loading(synthetic_dataset_path, mock_tokenizer):
    """Test basic loading of synthetic dataset with JSON captions."""
    # Get list of tar files
    tar_files = glob(str(Path(synthetic_dataset_path) / "*.tar"))
    assert len(tar_files) > 0, f"No tar files found in {synthetic_dataset_path}"
    
    # Create dataset loader
    dataset = glide_wds_loader(
        urls=tar_files[:1],  # Just use first tar file for testing
        enable_text=True,
        enable_image=True,
        enable_metadata=True,
        image_key="jpg",  # Primary image format
        caption_key="json",  # Captions are in JSON
        metadata_key="json",
        tokenizer=mock_tokenizer,
        base_x=64,
        base_y=64,
        uncond_p=0.0,  # Always use captions for testing
        dataset_name="webdataset",  # No filtering
        laion_no_filter=True,
    )
    
    # Get first sample
    sample = None
    for i, s in enumerate(dataset):
        sample = s
        break
    
    assert sample is not None, "Dataset should return at least one sample"
    
    # Check sample structure
    assert isinstance(sample, tuple), "Sample should be a tuple"
    assert len(sample) == 3, f"Sample should have 3 elements (tokens, mask, image), got {len(sample)}"
    
    tokens, mask, image = sample
    
    # Verify tokens
    assert isinstance(tokens, th.Tensor), "Tokens should be a tensor"
    assert tokens.dim() == 1, "Tokens should be 1D"
    assert tokens.dtype == th.long or tokens.dtype == th.int64, f"Tokens should be int64, got {tokens.dtype}"
    
    # Verify mask
    assert isinstance(mask, th.Tensor), "Mask should be a tensor"
    assert mask.dim() == 1, "Mask should be 1D"
    assert mask.dtype == th.bool, f"Mask should be bool, got {mask.dtype}"
    assert mask.shape == tokens.shape, "Mask and tokens should have same shape"
    
    # Verify image
    assert isinstance(image, th.Tensor), "Image should be a tensor"
    assert image.shape == (3, 64, 64), f"Image should be 3x64x64, got {image.shape}"
    assert image.dtype == th.float32, f"Image should be float32, got {image.dtype}"
    assert image.min() >= -1.0, "Image values should be >= -1.0"
    assert image.max() <= 1.0, "Image values should be <= 1.0"


def test_synthetic_dataset_multiple_samples(synthetic_dataset_path, mock_tokenizer):
    """Test loading multiple samples from synthetic dataset."""
    tar_files = glob(str(Path(synthetic_dataset_path) / "*.tar"))
    
    dataset = glide_wds_loader(
        urls=tar_files[:1],
        enable_text=True,
        enable_image=True,
        image_key="jpg",
        caption_key="json",
        tokenizer=mock_tokenizer,
        base_x=64,
        base_y=64,
        uncond_p=0.0,
        dataset_name="webdataset",
        laion_no_filter=True,
    )
    
    # Test first 5 samples
    samples_tested = 0
    for i, (tokens, mask, image) in enumerate(dataset):
        if i >= 5:
            break
            
        # Basic shape checks
        assert tokens.dim() == 1, f"Sample {i}: Tokens should be 1D"
        assert mask.dim() == 1, f"Sample {i}: Mask should be 1D"
        assert image.shape == (3, 64, 64), f"Sample {i}: Image should be 3x64x64"
        
        # Verify mask is properly set (at least some True values)
        assert mask.any(), f"Sample {i}: Mask should have at least some True values"
        
        samples_tested += 1
    
    assert samples_tested == 5, f"Should test 5 samples, got {samples_tested}"


def test_synthetic_dataset_uncond_probability(synthetic_dataset_path, mock_tokenizer):
    """Test unconditional token probability."""
    tar_files = glob(str(Path(synthetic_dataset_path) / "*.tar"))
    
    dataset = glide_wds_loader(
        urls=tar_files[:1],
        enable_text=True,
        enable_image=True,
        image_key="jpg",
        caption_key="json",
        tokenizer=mock_tokenizer,
        base_x=64,
        base_y=64,
        uncond_p=1.0,  # Always unconditional
        dataset_name="webdataset",
        laion_no_filter=True,
    )
    
    # Get first sample
    sample = None
    for s in dataset:
        sample = s
        break
    
    tokens, mask, image = sample
    
    # With uncond_p=1.0, should get empty tokens/mask
    assert mask.sum() == 0, "With uncond_p=1.0, mask should be all False"


def test_synthetic_dataset_image_formats(synthetic_dataset_path, mock_tokenizer):
    """Test that dataset can handle different image formats (jpg, jpeg, png)."""
    # Note: This test assumes the dataset contains various image formats
    # It will still pass if only jpg is present
    tar_files = glob(str(Path(synthetic_dataset_path) / "*.tar"))
    
    dataset = glide_wds_loader(
        urls=tar_files[:1],
        enable_text=True,
        enable_image=True,
        image_key="jpg",  # Primary key, but loader should fall back to jpeg/png
        caption_key="json",
        tokenizer=mock_tokenizer,
        base_x=64,
        base_y=64,
        dataset_name="webdataset",
        laion_no_filter=True,
    )
    
    # Test first few samples - they should all load successfully
    samples_tested = 0
    for i, (tokens, mask, image) in enumerate(dataset):
        if i >= 10:
            break
        
        # If we got here, the image loaded successfully regardless of format
        assert image.shape == (3, 64, 64), f"Sample {i}: Image loaded with correct shape"
        samples_tested += 1
    
    assert samples_tested > 0, "Should load at least one sample"


def test_synthetic_dataset_with_upsampling(synthetic_dataset_path, mock_tokenizer):
    """Test dataset with upsampling enabled."""
    tar_files = glob(str(Path(synthetic_dataset_path) / "*.tar"))
    
    dataset = glide_wds_loader(
        urls=tar_files[:1],
        enable_text=True,
        enable_image=True,
        enable_upsample=True,
        image_key="jpg",
        caption_key="json",
        tokenizer=mock_tokenizer,
        base_x=64,
        base_y=64,
        upscale_factor=4,
        uncond_p=0.0,
        dataset_name="webdataset",
        laion_no_filter=True,
    )
    
    # Get first sample
    sample = None
    for s in dataset:
        sample = s
        break
    
    assert len(sample) == 4, "With upsampling, should get 4 elements"
    tokens, mask, base_image, up_image = sample
    
    # Check base image
    assert base_image.shape == (3, 64, 64), "Base image should be 64x64"
    
    # Check upsampled image
    assert up_image.shape == (3, 256, 256), "Upsampled image should be 256x256"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_synthetic_dataset_batching(synthetic_dataset_path, mock_tokenizer, batch_size):
    """Test dataset batching functionality."""
    tar_files = glob(str(Path(synthetic_dataset_path) / "*.tar"))
    
    dataset = glide_wds_loader(
        urls=tar_files[:1],
        enable_text=True,
        enable_image=True,
        image_key="jpg",
        caption_key="json",
        tokenizer=mock_tokenizer,
        base_x=64,
        base_y=64,
        dataset_name="webdataset",
        laion_no_filter=True,
    )
    
    # Batch the dataset
    from glide_finetune.wds_loader import custom_webdataset_collate
    batched_dataset = dataset.batched(batch_size, collation_fn=custom_webdataset_collate)
    
    # Get first batch
    batch = None
    for b in batched_dataset:
        batch = b
        break
    
    assert batch is not None, "Should get at least one batch"
    assert isinstance(batch, tuple), "Batch should be a tuple"
    assert len(batch) == 3, "Batch should have 3 elements"
    
    tokens, masks, images = batch
    
    # Check batch dimensions
    assert tokens.shape[0] == batch_size, f"Batch size should be {batch_size}"
    assert masks.shape[0] == batch_size, f"Mask batch size should be {batch_size}"
    assert images.shape[0] == batch_size, f"Image batch size should be {batch_size}"
    assert images.shape == (batch_size, 3, 64, 64), f"Images shape should be ({batch_size}, 3, 64, 64)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])