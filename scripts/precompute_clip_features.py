#!/usr/bin/env python3
"""
Precompute CLIP features for a dataset and save them for efficient training.

Supports:
- COCO-style datasets: Outputs NPY format indexed by filename stem
- WebDataset: Outputs Parquet format indexed by tar member key
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from glide_finetune.clip_adapter import load_openai_clip, get_clip_text_features
from glide_finetune.loader import get_image_files_dict, get_text_files_dict, get_shared_stems
from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("precompute_clip_features")

# Check for optional dependencies
try:
    import pandas as pd
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("Parquet support not available. Install with: uv add pandas pyarrow")

try:
    import webdataset as wds
    WDS_AVAILABLE = True
except ImportError:
    WDS_AVAILABLE = False
    logger.warning("WebDataset support not available. Install with: uv add webdataset")


def process_coco_dataset(
    data_dir: Path,
    output_dir: Path, 
    clip_model: Any,
    device: str,
    batch_size: int = 32,
    resume_from: int = 0,
) -> None:
    """Process COCO-style dataset and save features in NPY format.
    
    Args:
        data_dir: Directory containing images and text files
        output_dir: Directory to save features.npy, index.json, metadata.json
        clip_model: Loaded CLIP model
        device: Device to run on
        batch_size: Batch size for processing
        resume_from: Resume from this index (for interrupted runs)
    """
    # Get all text files
    text_files = get_text_files_dict(data_dir)
    image_files = get_image_files_dict(data_dir)
    stems = get_shared_stems(image_files, text_files)
    
    if not stems:
        logger.error("No matching image/text pairs found")
        return
    
    logger.info(f"Found {len(stems)} image/text pairs")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize storage
    clip_dim = 512  # ViT-B/32 dimension
    features_list = []
    stem_to_idx = {}
    
    # Process in batches
    for batch_start in tqdm(range(resume_from, len(stems), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(stems))
        batch_stems = stems[batch_start:batch_end]
        
        # Load captions for batch
        batch_texts = []
        for stem in batch_stems:
            text_file = text_files[stem]
            with open(text_file) as f:
                lines = f.readlines()
                # Use first non-empty line
                caption = next((line.strip() for line in lines if line.strip()), "")
                batch_texts.append(caption)
        
        # Compute CLIP features
        if batch_texts:
            with torch.no_grad():
                clip_features = get_clip_text_features(
                    clip_model, 
                    batch_texts, 
                    device=device
                )
                features_list.append(clip_features.cpu().numpy())
            
            # Update index
            for i, stem in enumerate(batch_stems):
                stem_to_idx[stem] = batch_start + i
    
    # Save features
    if features_list:
        all_features = np.vstack(features_list)
        features_path = output_dir / "features.npy"
        np.save(features_path, all_features)
        logger.info(f"Saved features to {features_path}, shape: {all_features.shape}")
        
        # Save index
        index_path = output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(stem_to_idx, f, indent=2)
        logger.info(f"Saved index to {index_path}")
        
        # Save metadata
        metadata = {
            "clip_model": "openai/clip-vit-base-patch32",
            "clip_dim": clip_dim,
            "num_samples": len(stem_to_idx),
            "feature_shape": list(all_features.shape),
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def process_webdataset(
    tar_pattern: str,
    output_path: Path,
    clip_model: Any,
    device: str, 
    batch_size: int = 32,
    caption_key: str = "txt",
    max_samples: int | None = None,
) -> None:
    """Process WebDataset and save features in Parquet format.
    
    Args:
        tar_pattern: Pattern for tar files (e.g., "data-*.tar")
        output_path: Path to output Parquet file
        clip_model: Loaded CLIP model
        device: Device to run on
        batch_size: Batch size for processing
        caption_key: Key for captions in tar files
        max_samples: Maximum number of samples to process (for testing)
    """
    if not PARQUET_AVAILABLE:
        logger.error("Parquet support not available. Install pandas and pyarrow.")
        return
    
    if not WDS_AVAILABLE:
        logger.error("WebDataset support not available. Install webdataset.")
        return
    # Expand glob patterns in tar files
    if "*" in tar_pattern or "?" in tar_pattern:
        tar_files = sorted(glob.glob(tar_pattern))
        if not tar_files:
            logger.error(f"No tar files found matching pattern: {tar_pattern}")
            return
        logger.info(f"Found {len(tar_files)} tar files from pattern")
    else:
        # Single file or list of files
        tar_files = [tar_pattern]
        logger.info(f"Using single tar file: {tar_pattern}")
    
    # Create dataset with proper shardshuffle setting
    dataset = wds.WebDataset(tar_files, shardshuffle=False).decode()

    # Storage for results
    records = []
    batch_texts = []
    batch_keys = []
    
    sample_count = 0
    for sample in tqdm(dataset, desc="Processing WebDataset"):
        if max_samples and sample_count >= max_samples:
            break
        
        # Get caption
        caption = sample.get(caption_key, "")
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8")
        
        # Get tar member key (usually __key__)
        tar_key = sample.get("__key__", f"sample_{sample_count}")
        
        batch_texts.append(caption)
        batch_keys.append(tar_key)
        sample_count += 1
        
        # Process batch when full
        if len(batch_texts) >= batch_size:
            with torch.no_grad():
                clip_features = get_clip_text_features(
                    clip_model,
                    batch_texts,
                    device=device
                )
                
                for key, features in zip(batch_keys, clip_features):
                    records.append({
                        "tar_member": key,
                        "clip_features": features.cpu().numpy(),
                    })
            
            batch_texts = []
            batch_keys = []
    
    # Process remaining batch
    if batch_texts:
        with torch.no_grad():
            clip_features = get_clip_text_features(
                clip_model,
                batch_texts, 
                device=device
            )
            
            for key, features in zip(batch_keys, clip_features):
                records.append({
                    "tar_member": key,
                    "clip_features": features.cpu().numpy(),
                })
    
    # Save to Parquet
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(records)} features to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP features for datasets")
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["coco", "webdataset"],
        help="Type of dataset to process",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset (directory for COCO, tar pattern for WebDataset)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path (directory for COCO, Parquet file for WebDataset)",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use (default: ViT-B/32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=0,
        help="Resume from this index for COCO datasets (default: 0)",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="txt",
        help="Caption key for WebDataset (default: txt)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Load CLIP model
    logger.info(f"Loading CLIP model: {args.clip_model}")
    clip_model, preprocess = load_openai_clip(args.clip_model, device=args.device)
    clip_model.eval()

    # Expand User
    data_dir = args.data_path
    output_dir = args.output_path

    # Process dataset
    if args.dataset_type == "coco":
        data_dir = Path(args.data_path).expanduser()
        output_dir = Path(args.output_path)
        process_coco_dataset(
            data_dir,
            output_dir,
            clip_model,
            args.device,
            args.batch_size,
            args.resume_from,
        )
    else:  # webdataset
        output_path = Path(args.output_path)
        process_webdataset(
            args.data_path,
            output_path,
            clip_model,
            args.device,
            args.batch_size,
            args.caption_key,
            args.max_samples,
        )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()