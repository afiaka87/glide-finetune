#!/usr/bin/env python3
"""
Test script for precomputed CLIP feature loading.
Tests both COCO-style and WebDataset formats.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from glide_finetune.clip_features_loader import (
    NPYClipFeatureLoader, 
    ParquetClipFeatureLoader,
    load_clip_features
)
from glide_finetune.loader import TextImageDataset
from glide_finetune.utils.logging_utils import get_logger
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

logger = get_logger("test_clip_features")


def test_precompute_coco_features():
    """Test precomputing features for COCO-style dataset."""
    logger.info("=" * 80)
    logger.info("Testing COCO-style feature precomputation")
    logger.info("=" * 80)
    
    # Precompute features for birds dataset
    import subprocess
    result = subprocess.run([
        "uv", "run", "python", "scripts/precompute_clip_features.py",
        "--dataset_type", "coco",
        "--data_path", "/home/sam/Data/birds_00100",
        "--output_path", "/home/sam/Data/birds_00100_clip_features",
        "--batch_size", "16",
        "--max_samples", "20",  # Just test with first 20 samples
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to precompute features: {result.stderr}")
        return False
    
    logger.info("✓ Successfully precomputed COCO features")
    return True


def test_load_coco_features():
    """Test loading precomputed COCO features."""
    logger.info("=" * 80)
    logger.info("Testing COCO feature loading")
    logger.info("=" * 80)
    
    features_dir = Path("/home/sam/Data/birds_00100_clip_features")
    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        return False
    
    # Test NPY loader directly
    try:
        loader = NPYClipFeatureLoader(features_dir)
        logger.info(f"✓ Loaded NPY features: {len(loader)} samples, dim={loader.clip_dim}")
        
        # Test getting a feature
        if loader.stem_to_idx:
            first_stem = next(iter(loader.stem_to_idx.keys()))
            feature = loader.get_feature(first_stem)
            if feature is not None:
                logger.info(f"✓ Retrieved feature for '{first_stem}': shape={feature.shape}")
            else:
                logger.error(f"Failed to retrieve feature for '{first_stem}'")
                return False
    except Exception as e:
        logger.error(f"Failed to load NPY features: {e}")
        return False
    
    return True


def test_dataset_with_features():
    """Test TextImageDataset with precomputed features."""
    logger.info("=" * 80) 
    logger.info("Testing TextImageDataset integration")
    logger.info("=" * 80)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create dataset without features
    dataset_no_clip = TextImageDataset(
        folder="/home/sam/Data/birds_00100",
        side_x=64,
        side_y=64,
        tokenizer=tokenizer,
        use_captions=True,
        clip_features_path=None,
    )
    
    # Get a sample without CLIP
    sample = dataset_no_clip[0]
    logger.info(f"Sample without CLIP: {len(sample)} elements")
    for i, elem in enumerate(sample):
        if isinstance(elem, torch.Tensor):
            logger.info(f"  Element {i}: shape={elem.shape}, dtype={elem.dtype}")
    
    # Create dataset with features
    dataset_with_clip = TextImageDataset(
        folder="/home/sam/Data/birds_00100",
        side_x=64,
        side_y=64,
        tokenizer=tokenizer,
        use_captions=True,
        clip_features_path="/home/sam/Data/birds_00100_clip_features",
    )
    
    # Get a sample with CLIP
    sample = dataset_with_clip[0]
    logger.info(f"Sample with CLIP: {len(sample)} elements")
    for i, elem in enumerate(sample):
        if isinstance(elem, torch.Tensor):
            logger.info(f"  Element {i}: shape={elem.shape}, dtype={elem.dtype}")
    
    # Verify we got CLIP features
    if len(sample) == 4:  # tokens, mask, image, clip_features
        clip_features = sample[3]
        if clip_features.shape[-1] == 512:  # ViT-B/32 dimension
            logger.info("✓ CLIP features successfully integrated into dataset")
            return True
        else:
            logger.error(f"Unexpected CLIP dimension: {clip_features.shape}")
            return False
    else:
        logger.error(f"Expected 4 elements with CLIP, got {len(sample)}")
        return False


def test_precompute_webdataset_features():
    """Test precomputing features for WebDataset."""
    logger.info("=" * 80)
    logger.info("Testing WebDataset feature precomputation")
    logger.info("=" * 80)
    
    # Find tar files
    import glob
    tar_pattern = "/home/sam/Data/captioned-birds-subset-wds/*.tar"
    tar_files = glob.glob(tar_pattern)
    if not tar_files:
        logger.error(f"No tar files found matching: {tar_pattern}")
        return False
    
    logger.info(f"Found {len(tar_files)} tar files")
    
    # Use just first tar file for testing
    first_tar = tar_files[0]
    logger.info(f"Using first tar file for test: {first_tar}")
    
    # Precompute features for WebDataset
    import subprocess
    result = subprocess.run([
        "uv", "run", "python", "scripts/precompute_clip_features.py",
        "--dataset_type", "webdataset",
        "--data_path", first_tar,  # Use single tar file
        "--output_path", "/home/sam/Data/captioned-birds-subset-wds_clip.parquet",
        "--batch_size", "16",
        "--caption_key", "txt",
        "--max_samples", "20",  # Just test with first 20 samples
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to precompute features: {result.stderr}")
        return False
    
    logger.info("✓ Successfully precomputed WebDataset features")
    return True


def test_load_parquet_features():
    """Test loading precomputed Parquet features."""
    logger.info("=" * 80)
    logger.info("Testing Parquet feature loading")
    logger.info("=" * 80)
    
    parquet_path = Path("/home/sam/Data/captioned-birds-subset-wds_clip.parquet")
    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return False
    
    # Test Parquet loader directly
    try:
        loader = ParquetClipFeatureLoader(parquet_path)
        logger.info(f"✓ Loaded Parquet features: {len(loader)} samples, dim={loader.clip_dim}")
        
        # Test getting a feature
        if loader.member_to_features:
            first_key = next(iter(loader.member_to_features.keys()))
            feature = loader.get_feature(first_key)
            if feature is not None:
                logger.info(f"✓ Retrieved feature for '{first_key}': shape={feature.shape}")
            else:
                logger.error(f"Failed to retrieve feature for '{first_key}'")
                return False
    except Exception as e:
        logger.error(f"Failed to load Parquet features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    logger.info("Testing precomputed CLIP feature loading")
    logger.info("=" * 80)
    
    tests = [
        ("Precompute COCO features", test_precompute_coco_features),
        ("Load COCO features", test_load_coco_features),
        ("Dataset with features", test_dataset_with_features),
        ("Precompute WebDataset features", test_precompute_webdataset_features),
        ("Load Parquet features", test_load_parquet_features),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.info("\n✗ Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())