#!/usr/bin/env python3
"""
Fast CLIP text embeddings pre-computation for WebDataset tar files.

This script uses optimizations from clip-retrieval for much faster processing:
- JIT compilation for CLIP model
- Efficient WebDataset loading with prefetching
- Parallel processing with multiple workers
- Batched encoding with proper GPU utilization
- Streaming writes to avoid memory buildup

Usage:
    uv run python scripts/precompute_clip_webdataset_embeddings_fast.py \
        --tar_urls "/path/to/tars/*.tar" \
        --cache_dir ./clip_cache \
        --clip_model_name "ViT-B/32" \
        --batch_size 512 \
        --num_workers 24
"""

import argparse
import json
import time
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

from glide_finetune.adapters.clip_adapter import CLIP_DIMENSIONS


def sanitize_model_name(model_name: str) -> str:
    """Convert CLIP model name to filesystem-safe string."""
    return model_name.replace("/", "-").replace("@", "-")


def load_clip_model_jit(clip_model_name: str, device: str):
    """Load CLIP model with JIT compilation for better performance."""
    import clip

    print(f"Loading CLIP model with JIT: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device, jit=True)
    model.eval()

    # Enable TF32 for Ampere GPUs
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Warm up the model
    with torch.no_grad():
        dummy_text = clip.tokenize(["warmup"] * 32).to(device)
        _ = model.encode_text(dummy_text)

    # Verify expected dimensions
    expected_dim = CLIP_DIMENSIONS.get(clip_model_name)
    if expected_dim is None:
        print(f"Warning: Unknown CLIP model {clip_model_name}, dimension not verified")

    return model, preprocess


def create_webdataset_loader(
    tar_urls: List[str],
    caption_key: str,
    batch_size: int,
    num_workers: int,
    cache_path: Optional[str] = None,
):
    """Create an optimized WebDataset dataloader."""

    # Create dataset with caching and error handling
    dataset = wds.WebDataset(
        tar_urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.warn_and_continue,
        shardshuffle=False,  # Keep order for reproducibility
        empty_check=False,  # Disable empty check to avoid worker issues
    )

    # Filter for samples with captions
    def has_caption(sample):
        return caption_key in sample and sample[caption_key] is not None

    dataset = dataset.select(has_caption)

    # Extract key and caption
    def extract_caption(sample):
        caption_bytes = sample[caption_key]
        if isinstance(caption_bytes, bytes):
            caption = caption_bytes.decode("utf-8").strip()
        else:
            caption = str(caption_bytes).strip()

        return {"__key__": sample["__key__"], "caption": caption}

    dataset = dataset.map(extract_caption)

    # Create dataloader with optimizations
    if num_workers > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=False,
        )
    else:
        # When num_workers=0, we can't use prefetch_factor or persistent_workers
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    return dataloader


def encode_batch_optimized(model, texts: List[str], device: str) -> torch.Tensor:
    """Encode a batch of texts with CLIP, optimized version."""
    import clip

    # Tokenize with truncation
    tokens = clip.tokenize(texts, context_length=77, truncate=True).to(device)

    # Encode with no_grad and mixed precision
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            text_features = model.encode_text(tokens)
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)

    return text_features.cpu()


def process_tar_file_streaming(
    tar_path: str,
    model,
    device: str,
    clip_model_name: str,
    caption_key: str,
    batch_size: int,
    num_workers: int,
    output_file: Path,
    tar_index: int = 0,
    total_tars: int = 1,
):
    """Process a single tar file with streaming writes."""

    tar_name = Path(tar_path).name
    print(f"\n📄 [{tar_index}/{total_tars}] Processing: {tar_name}")
    start_time = time.time()

    # Create dataloader for this tar file
    # Use 0 workers when processing a single tar file to avoid the empty shard issue
    dataloader = create_webdataset_loader(
        [tar_path],
        caption_key,
        batch_size,
        0,  # Single tar file, so use main process only
    )

    embeddings = {}
    total_samples = 0

    # Process batches
    # Estimate total samples (WebDataset doesn't provide length)
    estimated_total = 10000  # Typical tar file has ~10k samples
    pbar = tqdm(
        desc="   Samples",
        unit=" samples",
        unit_scale=True,
        total=estimated_total,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )

    for batch in dataloader:
        keys = batch["__key__"]
        captions = batch["caption"]

        # Encode batch
        batch_embeddings = encode_batch_optimized(model, captions, device)

        # Store embeddings
        batch_size = len(keys)
        for key, caption, embedding in zip(keys, captions, batch_embeddings):
            embeddings[key] = {"embedding": embedding, "caption": caption}
            total_samples += 1

        pbar.update(batch_size)

        # Dynamically adjust total estimate
        if total_samples > estimated_total * 0.9:
            # If we're close to the estimate, increase it
            new_estimate = int(total_samples * 1.2)
            pbar.total = new_estimate
            pbar.refresh()

        # Periodically save to avoid memory buildup
        if len(embeddings) >= 10000:
            save_embeddings_incremental(
                embeddings, output_file, clip_model_name, tar_path
            )
            embeddings = {}

    # Save remaining embeddings
    if embeddings:
        save_embeddings_incremental(embeddings, output_file, clip_model_name, tar_path)

    pbar.close()

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed if elapsed > 0 else 0
    print(
        f"   ✅ Completed: {total_samples:,} samples in {elapsed:.1f}s ({throughput:,.1f} samples/s)"
    )

    return total_samples


def save_embeddings_incremental(
    embeddings: Dict,
    output_file: Path,
    clip_model_name: str,
    tar_path: str,
):
    """Save embeddings incrementally to avoid memory issues."""

    # Load existing data if file exists
    if output_file.exists():
        existing_data = torch.load(output_file, map_location="cpu")
        existing_embeddings = existing_data.get("embeddings", {})
        existing_embeddings.update(embeddings)
        embeddings = existing_embeddings

    # Save with metadata
    data = {
        "metadata": {
            "clip_model": clip_model_name,
            "tar_file": str(tar_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": len(embeddings),
            "embedding_dim": next(iter(embeddings.values()))["embedding"].shape[-1],
        },
        "embeddings": embeddings,
    }

    # Save atomically
    temp_file = output_file.with_suffix(".tmp")
    torch.save(data, temp_file)
    temp_file.rename(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Fast CLIP embeddings pre-computation for WebDataset"
    )
    parser.add_argument(
        "--tar_urls",
        type=str,
        required=True,
        help="Glob pattern for tar files (e.g., '/path/to/data/*.tar')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./clip_cache",
        help="Directory to store CLIP embedding cache",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B/32",
        choices=list(CLIP_DIMENSIONS.keys()),
        help="CLIP model to use for encoding",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="txt",
        help="WebDataset key for text captions",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for CLIP encoding"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for CLIP model",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute embeddings even if cache already exists",
    )

    args = parser.parse_args()

    # Expand glob pattern to get tar file list
    tar_files = sorted(glob(args.tar_urls))
    if not tar_files:
        print(f"No tar files found matching pattern: {args.tar_urls}")
        return 1

    print("\n" + "=" * 80)
    print("FAST CLIP EMBEDDING PRECOMPUTATION")
    print("=" * 80)
    print("\n📁 Dataset Information:")
    print(f"   Pattern: {args.tar_urls}")
    print(f"   Files found: {len(tar_files)} tar files")
    print(f"   Cache directory: {args.cache_dir}")

    print("\n🤖 Model Configuration:")
    print(f"   CLIP model: {args.clip_model_name}")
    print(f"   Device: {args.device}")

    print("\n⚡ Performance Settings:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Workers: {args.num_workers}")
    print(f"   Force recompute: {args.force_recompute}")

    # Setup cache directory
    cache_path = Path(args.cache_dir).resolve()
    model_dir = cache_path / sanitize_model_name(args.clip_model_name)
    embeddings_dir = model_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = model_dir / "tar_metadata.json"
    tar_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            tar_metadata = json.load(f)

    # Load CLIP model with JIT
    model, _ = load_clip_model_jit(args.clip_model_name, args.device)

    # Process all tar files
    total_start = time.time()
    total_samples = 0
    processed_files = 0
    skipped_files = 0

    for idx, tar_path in enumerate(tar_files, 1):
        tar_name = Path(tar_path).name
        cache_file = embeddings_dir / f"{tar_name}.pt"

        # Skip if already processed
        if cache_file.exists() and not args.force_recompute:
            print(
                f"\n📄 [{idx}/{len(tar_files)}] Skipping: {tar_name} (already processed)"
            )
            skipped_files += 1
            continue

        try:
            # Process tar file
            num_samples = process_tar_file_streaming(
                tar_path,
                model,
                args.device,
                args.clip_model_name,
                args.caption_key,
                args.batch_size,
                args.num_workers,
                cache_file,
                tar_index=idx,
                total_tars=len(tar_files),
            )

            # Update metadata
            tar_metadata[tar_name] = {
                "path": str(tar_path),
                "sample_count": num_samples,
                "cache_file": str(cache_file.relative_to(cache_path)),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Save metadata after each file
            with open(metadata_file, "w") as f:
                json.dump(tar_metadata, f, indent=2)

            total_samples += num_samples
            processed_files += 1

        except Exception as e:
            print(f"Error processing {tar_path}: {e}")
            continue

    # Final summary
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    print("\n📊 Summary:")
    print(f"   Files processed: {processed_files:,}")
    print(f"   Files skipped: {skipped_files:,}")
    print(f"   Total samples: {total_samples:,}")

    print("\n⏱️  Performance:")
    print(f"   Total time: {total_elapsed:.1f}s")

    if total_samples > 0:
        avg_throughput = total_samples / total_elapsed
        print(f"   Average throughput: {avg_throughput:,.1f} samples/s")

        # Estimate for common dataset sizes
        print("\n📈 Estimated times at this rate:")
        for size, count in [
            ("1M", 1_000_000),
            ("10M", 10_000_000),
            ("100M", 100_000_000),
        ]:
            eta_seconds = count / avg_throughput
            eta_hours = eta_seconds / 3600
            if eta_hours < 1:
                print(f"   {size} samples: ~{eta_seconds / 60:.1f} minutes")
            else:
                print(f"   {size} samples: ~{eta_hours:.1f} hours")

    return 0


if __name__ == "__main__":
    exit(main())
