#!/usr/bin/env python3
"""
Pre-compute CLIP text embeddings for WebDataset tar files.

This script processes WebDataset tar files, extracts text captions, encodes them with CLIP,
and saves the embeddings in a cache directory organized by CLIP model and tar file.

Usage:
    uv run python scripts/precompute_clip_webdataset_embeddings.py \
        --tar_urls "/path/to/tars/*.tar" \
        --cache_dir ./clip_cache \
        --clip_model_name "ViT-L/14" \
        --batch_size 32 \
        --device cuda

Cache structure:
    clip_cache/
    ├── ViT-L-14/                    # CLIP model variant (sanitized name)
    │   ├── tar_metadata.json       # Maps tar files to sample counts
    │   └── embeddings/
    │       ├── 000000.tar.pt       # One cache file per tar
    │       └── 000001.tar.pt
    └── ViT-B-32/
        ├── tar_metadata.json
        └── embeddings/
            └── data.tar.pt
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from glob import glob
import io

import clip
import torch
import webdataset as wds
from tqdm import tqdm

from glide_finetune.adapters.clip_adapter import CLIP_DIMENSIONS


def sanitize_model_name(model_name: str) -> str:
    """Convert CLIP model name to filesystem-safe string."""
    return model_name.replace("/", "-").replace("@", "-")


def load_clip_model(clip_model_name: str, device: str):
    """Load CLIP model and preprocessing."""
    print(f"Loading CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()
    
    # Enable TF32 for Ampere GPUs
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("TF32 enabled for better performance")
    
    # Verify expected dimensions
    expected_dim = CLIP_DIMENSIONS.get(clip_model_name)
    if expected_dim is None:
        print(f"Warning: Unknown CLIP model {clip_model_name}, dimension not verified")
        
    return model, preprocess


def setup_cache_directory(cache_dir: Path, clip_model_name: str) -> tuple[Path, Path]:
    """Setup cache directory structure."""
    model_dir = cache_dir / sanitize_model_name(clip_model_name)
    embeddings_dir = model_dir / "embeddings"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(exist_ok=True)
    
    metadata_file = model_dir / "tar_metadata.json"
    
    return embeddings_dir, metadata_file


def load_tar_metadata(metadata_file: Path) -> Dict[str, Dict]:
    """Load existing tar metadata."""
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
    
    return {}


def save_tar_metadata(metadata_file: Path, metadata: Dict[str, Dict]):
    """Save tar metadata."""
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def extract_text_from_sample(sample: Dict, caption_key: str = "txt") -> Optional[str]:
    """Extract text caption from WebDataset sample."""
    try:
        if caption_key not in sample:
            return None
            
        caption_bytes = sample[caption_key]
        if isinstance(caption_bytes, bytes):
            caption = caption_bytes.decode('utf-8').strip()
        else:
            caption = str(caption_bytes).strip()
            
        return caption if caption else None
        
    except Exception as e:
        return None


def process_tar_file(
    tar_path: str,
    model,
    device: str,
    clip_model_name: str,
    caption_key: str = "txt",
    batch_size: int = 32,
    dry_run: bool = False
) -> Dict[str, torch.Tensor]:
    """Process a single tar file and return CLIP embeddings."""
    
    print(f"Processing {tar_path}")
    
    # Create WebDataset from single tar
    dataset = wds.WebDataset(tar_path)
    
    embeddings = {}
    batch_samples = []
    batch_keys = []
    batch_texts = []
    
    total_samples = 0
    valid_samples = 0
    
    try:
        for sample in dataset:
            total_samples += 1
            
            # Extract sample key (__key__ is WebDataset standard)
            sample_key = sample.get("__key__", f"sample_{total_samples:06d}")
            
            # Extract text caption
            caption = extract_text_from_sample(sample, caption_key)
            if caption is None:
                continue
                
            valid_samples += 1
            batch_samples.append(sample)
            batch_keys.append(sample_key)
            batch_texts.append(caption)
            
            # Process batch when full
            if len(batch_texts) >= batch_size:
                if not dry_run:
                    batch_embeddings = encode_texts_batch(model, batch_texts, device)
                    for i, (key, embedding) in enumerate(zip(batch_keys, batch_embeddings)):
                        embeddings[key] = {
                            "embedding": embedding,
                            "caption": batch_texts[i]
                        }
                else:
                    print(f"[DRY RUN] Would encode batch of {len(batch_texts)} texts")
                
                # Reset batch
                batch_samples = []
                batch_keys = []  
                batch_texts = []
        
        # Process remaining samples
        if batch_texts:
            if not dry_run:
                batch_embeddings = encode_texts_batch(model, batch_texts, device)
                for i, (key, embedding) in enumerate(zip(batch_keys, batch_embeddings)):
                    embeddings[key] = {
                        "embedding": embedding,
                        "caption": batch_texts[i]
                    }
            else:
                print(f"[DRY RUN] Would encode final batch of {len(batch_texts)} texts")
                
    except Exception as e:
        print(f"Error processing tar {tar_path}: {e}")
        return {}
    
    print(f"  Processed {valid_samples}/{total_samples} samples from {tar_path}")
    return embeddings


def encode_texts_batch(
    model, 
    texts: List[str], 
    device: str, 
    max_length: int = 77
) -> torch.Tensor:
    """Encode a batch of texts with CLIP."""
    # Tokenize texts
    tokens = clip.tokenize(texts, context_length=max_length, truncate=True)
    tokens = tokens.to(device)
    
    # Encode with CLIP
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        # Normalize features (CLIP standard practice)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu()


def check_existing_cache(
    cache_file: Path, 
    clip_model_name: str,
    force_recompute: bool = False
) -> bool:
    """Check if valid cache file already exists."""
    if force_recompute:
        return False
        
    if not cache_file.exists():
        return False
        
    try:
        data = torch.load(cache_file, map_location='cpu')
        
        # Check metadata
        metadata = data.get("metadata", {})
        if metadata.get("clip_model") != clip_model_name:
            return False
            
        # Check if embeddings exist
        embeddings = data.get("embeddings", {})
        if not embeddings:
            return False
            
        return True
        
    except Exception as e:
        print(f"Invalid existing cache {cache_file}: {e}")
        return False


def save_embeddings_cache(
    embeddings: Dict[str, torch.Tensor],
    cache_file: Path,
    clip_model_name: str,
    tar_path: str,
    stats: Dict[str, int]
):
    """Save embeddings cache with metadata."""
    data = {
        "metadata": {
            "clip_model": clip_model_name,
            "tar_file": str(tar_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": len(embeddings),
            "embedding_dim": embeddings[next(iter(embeddings))]["embedding"].shape[-1] if embeddings else 0,
        },
        "embeddings": embeddings,
        "stats": stats
    }
    
    torch.save(data, cache_file)


def precompute_webdataset_embeddings(
    tar_urls: List[str],
    cache_dir: str,
    clip_model_name: str = "ViT-L/14",
    caption_key: str = "txt",
    batch_size: int = 32,
    device: str = "cuda",
    force_recompute: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Pre-compute CLIP embeddings for WebDataset tar files."""
    
    cache_path = Path(cache_dir).resolve()
    
    # Setup cache directory (only if not dry run)
    if not dry_run:
        embeddings_dir, metadata_file = setup_cache_directory(cache_path, clip_model_name)
        # Load existing metadata
        tar_metadata = load_tar_metadata(metadata_file)
    else:
        # For dry run, just construct paths without creating directories
        model_dir = cache_path / sanitize_model_name(clip_model_name)
        embeddings_dir = model_dir / "embeddings"
        metadata_file = model_dir / "tar_metadata.json"
        tar_metadata = {}
    
    # Load CLIP model if not dry run
    if not dry_run:
        model, _ = load_clip_model(clip_model_name, device)
    else:
        model = None
    
    stats = {"processed": 0, "skipped": 0, "errors": 0, "total_samples": 0}
    
    # Process each tar file
    for tar_path in tqdm(tar_urls, desc="Processing tar files"):
        tar_name = Path(tar_path).name
        cache_file = embeddings_dir / f"{tar_name}.pt"
        
        # Skip if valid cache already exists
        if check_existing_cache(cache_file, clip_model_name, force_recompute):
            stats["skipped"] += 1
            continue
        
        try:
            # Process tar file
            embeddings = process_tar_file(
                tar_path, model, device, clip_model_name, 
                caption_key, batch_size, dry_run
            )
            
            if not embeddings and not dry_run:
                print(f"No valid embeddings extracted from {tar_path}")
                stats["errors"] += 1
                continue
            
            if not dry_run:
                # Save embeddings cache
                save_embeddings_cache(
                    embeddings, cache_file, clip_model_name, tar_path, 
                    {"valid_samples": len(embeddings)}
                )
            
            # Update tar metadata
            tar_metadata[tar_name] = {
                "path": str(tar_path),
                "sample_count": len(embeddings) if embeddings else 0,
                "cache_file": str(cache_file.relative_to(cache_path)),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            stats["processed"] += 1
            stats["total_samples"] += len(embeddings) if embeddings else 0
            
        except Exception as e:
            print(f"Error processing {tar_path}: {e}")
            stats["errors"] += 1
            continue
    
    # Save updated metadata
    if not dry_run:
        save_tar_metadata(metadata_file, tar_metadata)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for WebDataset tar files"
    )
    parser.add_argument(
        "--tar_urls",
        type=str,
        required=True,
        help="Glob pattern for tar files (e.g., '/path/to/data/*.tar')"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./clip_cache",
        help="Directory to store CLIP embedding cache"
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-L/14",
        choices=list(CLIP_DIMENSIONS.keys()),
        help="CLIP model to use for encoding"
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="txt",
        help="WebDataset key for text captions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for CLIP encoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for CLIP model"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute embeddings even if cache already exists"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually computing embeddings"
    )
    
    args = parser.parse_args()
    
    # Expand glob pattern to get tar file list
    tar_files = glob(args.tar_urls)
    if not tar_files:
        print(f"No tar files found matching pattern: {args.tar_urls}")
        return 1
    
    print("CLIP WebDataset Embedding Precomputation")
    print("=" * 45)
    print(f"Tar pattern: {args.tar_urls}")
    print(f"Found {len(tar_files)} tar files")
    print(f"Cache directory: {args.cache_dir}")
    print(f"CLIP model: {args.clip_model_name}")
    print(f"Caption key: {args.caption_key}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Force recompute: {args.force_recompute}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    start_time = time.time()
    
    try:
        stats = precompute_webdataset_embeddings(
            tar_urls=tar_files,
            cache_dir=args.cache_dir,
            clip_model_name=args.clip_model_name,
            caption_key=args.caption_key,
            batch_size=args.batch_size,
            device=args.device,
            force_recompute=args.force_recompute,
            dry_run=args.dry_run,
        )
        
        elapsed = time.time() - start_time
        
        print("\nResults:")
        print(f"  Tar files processed: {stats['processed']}")
        print(f"  Tar files skipped: {stats['skipped']}")
        print(f"  Tar files with errors: {stats['errors']}")
        print(f"  Total samples processed: {stats['total_samples']}")
        print(f"  Total time: {elapsed:.1f}s")
        
        if stats['total_samples'] > 0:
            print(f"  Average time per sample: {elapsed/stats['total_samples']:.3f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())