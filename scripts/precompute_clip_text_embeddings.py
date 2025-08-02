#!/usr/bin/env python3
"""
Pre-compute CLIP text embeddings for TextImageDataset format.

This script scans a directory for .txt files (captions), encodes them with CLIP,
and saves the embeddings as .clip files alongside the original text files.

Usage:
    uv run python scripts/precompute_clip_text_embeddings.py \
        --data_dir /path/to/data \
        --clip_model_name "ViT-L/14" \
        --batch_size 32 \
        --device cuda

Directory structure:
    data/
    ├── image1.jpg
    ├── image1.txt      # Caption file
    ├── image1.clip     # Generated embedding file
    ├── image2.png
    ├── image2.txt
    └── image2.clip
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import clip
import torch
from tqdm import tqdm

from glide_finetune.adapters.clip_adapter import CLIP_DIMENSIONS


def find_text_files(data_dir: Path) -> List[Path]:
    """Find all .txt files in the data directory."""
    text_files = list(data_dir.glob("**/*.txt"))
    print(f"Found {len(text_files)} text files in {data_dir}")
    return sorted(text_files)


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


def read_caption_file(text_file: Path) -> Optional[str]:
    """Read caption from text file, handling various edge cases."""
    try:
        if not text_file.exists():
            return None
            
        # Check file size
        if text_file.stat().st_size == 0:
            return None
            
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Filter out empty lines and strip whitespace
        valid_lines = [line.strip() for line in lines if line.strip()]
        
        if not valid_lines:
            return None
            
        # Use first non-empty line as caption
        # (TextImageDataset uses random.choice, but for pre-computation we'll use first)
        return valid_lines[0]
        
    except Exception as e:
        print(f"Error reading {text_file}: {e}")
        return None


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


def save_clip_embedding(
    embedding: torch.Tensor,
    clip_model_name: str,
    output_path: Path,
    original_caption: str
) -> None:
    """Save CLIP embedding with metadata."""
    data = {
        "clip_model": clip_model_name,
        "embedding": embedding,
        "caption": original_caption,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": embedding.shape[-1],
    }
    
    torch.save(data, output_path)


def check_existing_embedding(
    clip_file: Path, 
    clip_model_name: str,
    force_recompute: bool = False
) -> bool:
    """Check if valid embedding already exists."""
    if force_recompute:
        return False
        
    if not clip_file.exists():
        return False
        
    try:
        data = torch.load(clip_file, map_location='cpu')
        
        # Check if it's for the same CLIP model
        if data.get("clip_model") != clip_model_name:
            return False
            
        # Check if embedding has correct dimensions
        expected_dim = CLIP_DIMENSIONS.get(clip_model_name)
        if expected_dim and data["embedding"].shape[-1] != expected_dim:
            return False
            
        return True
        
    except Exception as e:
        print(f"Invalid existing embedding {clip_file}: {e}")
        return False


def precompute_embeddings(
    data_dir: str,
    clip_model_name: str = "ViT-L/14",
    batch_size: int = 32,
    device: str = "cuda",
    force_recompute: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Pre-compute CLIP embeddings for all text files in directory."""
    
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")
    
    # Find all text files
    text_files = find_text_files(data_path)
    if not text_files:
        print("No text files found!")
        return {"processed": 0, "skipped": 0, "errors": 0}
    
    print(f"\n📁 Dataset Information:")
    print(f"   Data directory: {data_dir}")
    print(f"   Text files found: {len(text_files)}")
    
    print(f"\n🤖 Model Configuration:")
    print(f"   CLIP model: {clip_model_name}")
    print(f"   Device: {device}")
    
    print(f"\n⚡ Performance Settings:")
    print(f"   Batch size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print(f"   Force recompute: {force_recompute}")
    print(f"   Dry run: {dry_run}")
    
    # Load CLIP model if not dry run
    if not dry_run:
        model, _ = load_clip_model(clip_model_name, device)
    
    stats = {"processed": 0, "skipped": 0, "errors": 0}
    
    # Process files in batches
    batch_texts = []
    batch_files = []
    batch_captions = []
    
    pbar = tqdm(text_files, desc="Processing text files")
    
    for text_file in pbar:
        # Generate corresponding .clip file path
        clip_file = text_file.with_suffix('.clip')
        
        # Skip if valid embedding already exists
        if check_existing_embedding(clip_file, clip_model_name, force_recompute):
            stats["skipped"] += 1
            pbar.set_postfix(stats)
            continue
        
        # Read caption
        caption = read_caption_file(text_file)
        if caption is None:
            print(f"Skipping {text_file}: no valid caption")
            stats["errors"] += 1
            pbar.set_postfix(stats)
            continue
        
        # Add to batch
        batch_texts.append(caption)
        batch_files.append(clip_file)
        batch_captions.append(caption)
        
        # Process batch when full or at end
        if len(batch_texts) >= batch_size or text_file == text_files[-1]:
            if dry_run:
                print(f"[DRY RUN] Would process batch of {len(batch_texts)} texts")
                stats["processed"] += len(batch_texts)
            else:
                try:
                    # Encode batch
                    embeddings = encode_texts_batch(model, batch_texts, device)
                    
                    # Save individual embeddings
                    for i, (embedding, clip_path, caption) in enumerate(
                        zip(embeddings, batch_files, batch_captions)
                    ):
                        save_clip_embedding(
                            embedding, clip_model_name, clip_path, caption
                        )
                    
                    stats["processed"] += len(batch_texts)
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    stats["errors"] += len(batch_texts)
            
            # Reset batch
            batch_texts = []
            batch_files = []
            batch_captions = []
            
            pbar.set_postfix(stats)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for TextImageDataset"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Directory containing image and text files"
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-L/14",
        choices=list(CLIP_DIMENSIONS.keys()),
        help="CLIP model to use for encoding"
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
        help="Recompute embeddings even if they already exist"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually computing embeddings"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CLIP TEXT EMBEDDING PRECOMPUTATION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        stats = precompute_embeddings(
            data_dir=args.data_dir,
            clip_model_name=args.clip_model_name,
            batch_size=args.batch_size,
            device=args.device,
            force_recompute=args.force_recompute,
            dry_run=args.dry_run,
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        
        print(f"\n📊 Summary:")
        print(f"   Files processed: {stats['processed']:,}")
        print(f"   Files skipped: {stats['skipped']:,}")
        print(f"   Files with errors: {stats['errors']:,}")
        
        print(f"\n⏱️  Performance:")
        print(f"   Total time: {elapsed:.1f}s")
        
        if stats['processed'] > 0:
            avg_time = elapsed/stats['processed']
            avg_throughput = stats['processed']/elapsed
            print(f"   Average time per file: {avg_time:.3f}s")
            print(f"   Average throughput: {avg_throughput:,.1f} files/s")
            
            # Estimate for common dataset sizes
            print(f"\n📈 Estimated times at this rate:")
            for size, count in [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)]:
                eta_seconds = count / avg_throughput
                eta_hours = eta_seconds / 3600
                if eta_hours < 1:
                    print(f"   {size} files: ~{eta_seconds/60:.1f} minutes")
                else:
                    print(f"   {size} files: ~{eta_hours:.1f} hours")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())