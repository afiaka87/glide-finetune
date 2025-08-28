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
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from glide_finetune.clip_adapter import load_openai_clip, get_clip_text_features
from glide_finetune.loaders.loader import get_image_files_dict, get_text_files_dict, get_shared_stems
from glide_finetune.utils.logging_utils import get_logger
from torch.utils.data import DataLoader

logger = get_logger("precompute_clip_features")

# Enable TF32 for massive speedup on Ampere/Hopper GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("TF32 enabled for accelerated computation")

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
    use_amp: bool = True,
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
            with torch.amp.autocast('cuda', enabled=use_amp and device == "cuda"):
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
    use_amp: bool = True,
    warmup_steps: int = 10,
    num_workers: int = 4,
    prefetch_factor: int = 2,
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
    if "*" in tar_pattern or "?" in tar_pattern or "{" in tar_pattern:
        # Use braceexpand for proper brace expansion
        from braceexpand import braceexpand
        
        # First expand braces
        expanded_patterns = list(braceexpand(tar_pattern))
        
        tar_files = []
        for pattern in expanded_patterns:
            # Then apply glob to each expanded pattern
            if "*" in pattern or "?" in pattern:
                tar_files.extend(sorted(glob.glob(pattern)))
            else:
                # Check if file exists
                if Path(pattern).exists():
                    tar_files.append(pattern)
        
        if not tar_files:
            logger.error(f"No tar files found matching pattern: {tar_pattern}")
            logger.error(f"Expanded to: {expanded_patterns[:5]}...")
            return
            
        logger.info(f"Found {len(tar_files)} tar files from pattern")
    else:
        # Single file or list of files
        tar_files = [tar_pattern]
        logger.info(f"Using single tar file: {tar_pattern}")
    
    # Create dataset with optimized settings
    dataset = (
        wds.WebDataset(
            tar_files, 
            shardshuffle=False,
            handler=wds.warn_and_continue,  # Continue on errors
        )
        .decode("rgb", handler=wds.warn_and_continue)  # Handle decode errors gracefully
    )
    
    # Wrap in DataLoader for parallel I/O and prefetching (helps with pausing)
    if num_workers > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=None,  # WebDataset handles its own batching
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            prefetch_factor=prefetch_factor,
            persistent_workers=True,  # Keep workers alive
        )
        logger.info(f"Using DataLoader with {num_workers} workers for parallel I/O")
    else:
        dataloader = dataset
        logger.info("Using direct WebDataset iteration")

    # Storage for results
    records = []
    batch_texts = []
    batch_keys = []
    
    # Performance monitoring
    sample_count = 0
    batch_count = 0
    start_time = time.time()
    processing_times = []
    
    # Progress bar with dynamic speed display
    pbar = tqdm(dataloader, desc="Processing WebDataset")
    
    for sample in pbar:
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
            batch_start = time.time()
            
            # Use AMP for inference
            with torch.amp.autocast('cuda', enabled=use_amp and device == "cuda"):
                with torch.no_grad():
                    clip_features = get_clip_text_features(
                        clip_model,
                        batch_texts,
                        device=device
                    )
            
            # Move to CPU and convert to numpy
            clip_features = clip_features.cpu().numpy()
            
            for key, features in zip(batch_keys, clip_features):
                records.append({
                    "tar_member": key,
                    "clip_features": features,
                })
            
            batch_time = time.time() - batch_start
            batch_count += 1
            
            # Track performance after warmup
            if batch_count > warmup_steps:
                processing_times.append(batch_time)
                avg_time = np.mean(processing_times)
                samples_per_sec = batch_size / avg_time
                pbar.set_postfix({
                    "samples/s": f"{samples_per_sec:.1f}",
                    "batch_ms": f"{avg_time*1000:.1f}",
                    "total": sample_count,
                })
            
            batch_texts = []
            batch_keys = []
            
            # Periodic cache clearing to prevent fragmentation
            if batch_count % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Process remaining batch
    if batch_texts:
        with torch.amp.autocast('cuda', enabled=use_amp and device == "cuda"):
            with torch.no_grad():
                clip_features = get_clip_text_features(
                    clip_model,
                    batch_texts,
                    device=device
                )
        
        clip_features = clip_features.cpu().numpy()
        
        for key, features in zip(batch_keys, clip_features):
            records.append({
                "tar_member": key,
                "clip_features": features,
            })
    
    pbar.close()
    
    # Report performance statistics
    total_time = time.time() - start_time
    if processing_times:
        avg_batch_time = np.mean(processing_times)
        std_batch_time = np.std(processing_times)
        logger.info(f"\nPerformance Statistics:")
        logger.info(f"  Total samples: {sample_count:,}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average speed: {sample_count/total_time:.1f} samples/s")
        logger.info(f"  Batch time: {avg_batch_time*1000:.1f} ± {std_batch_time*1000:.1f} ms")
        logger.info(f"  Throughput: {batch_size/avg_batch_time:.1f} samples/s")
    
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
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=True,
        help="Use torch.compile for optimization (default: True)",
    )
    parser.add_argument(
        "--no_compile",
        dest="use_compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true", 
        default=True,
        help="Use automatic mixed precision (default: True)",
    )
    parser.add_argument(
        "--no_amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch (default: 2)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Warmup steps before measuring speed (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Load CLIP model
    logger.info(f"Loading CLIP model: {args.clip_model}")
    clip_model, preprocess = load_openai_clip(args.clip_model, device=args.device)
    clip_model.eval()
    
    # Apply torch.compile if requested and available
    if args.use_compile and torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            logger.info(f"Compiling CLIP model with mode: {args.compile_mode}")
            clip_model = torch.compile(clip_model, mode=args.compile_mode)
            logger.info("✓ Model compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            logger.warning("Continuing without compilation")

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
            use_amp=args.use_amp,
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
            use_amp=args.use_amp,
            warmup_steps=args.warmup_steps,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
