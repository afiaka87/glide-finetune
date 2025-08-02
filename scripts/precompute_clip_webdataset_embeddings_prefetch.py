#!/usr/bin/env python3
"""
Precompute CLIP text embeddings for WebDataset tar files with prefetching.
Uses producer-consumer pattern to prefetch tar files while GPU processes embeddings.
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Iterator
import webdataset as wds
from queue import Queue
from threading import Thread, Event
from dataclasses import dataclass
import gc

# Configure PyTorch for better performance
torch.set_float32_matmul_precision('high')

@dataclass
class TarBatch:
    """Container for a batch of samples from a tar file."""
    tar_name: str
    tar_index: int
    texts: List[str]
    indices: List[str]
    batch_start: int
    
class PrefetchingTarLoader:
    """Loads tar files in background thread with prefetching."""
    
    def __init__(self, tar_urls: List[str], caption_key: str, prefetch_size: int = 2):
        self.tar_urls = tar_urls
        self.caption_key = caption_key
        self.prefetch_size = prefetch_size
        self.queue = Queue(maxsize=prefetch_size)
        self.stop_event = Event()
        self.exception = None
        
    def _load_tar_samples(self, tar_url: str, tar_index: int):
        """Load all samples from a tar file."""
        texts = []
        indices = []
        
        try:
            dataset = wds.WebDataset(tar_url, shardshuffle=False)
            
            for sample in dataset:
                if self.caption_key in sample:
                    caption = sample[self.caption_key]
                    if isinstance(caption, bytes):
                        caption = caption.decode('utf-8', errors='replace')
                    
                    texts.append(caption.strip())
                    indices.append(sample.get('__key__', str(len(texts))))
                    
                if self.stop_event.is_set():
                    break
                    
        except Exception as e:
            print(f"Error loading {tar_url}: {e}")
            
        return tar_url, tar_index, texts, indices
    
    def _producer_thread(self):
        """Background thread that loads tar files."""
        try:
            for i, tar_url in enumerate(self.tar_urls):
                if self.stop_event.is_set():
                    break
                    
                # Load tar file
                tar_data = self._load_tar_samples(tar_url, i)
                
                # Put in queue (blocks if queue is full)
                self.queue.put(tar_data)
                
        except Exception as e:
            self.exception = e
        finally:
            # Signal end of data
            self.queue.put(None)
    
    def start(self):
        """Start the background loading thread."""
        self.thread = Thread(target=self._producer_thread, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the background thread."""
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)
            
    def get_batches(self, batch_size: int) -> Iterator[TarBatch]:
        """Yield batches of samples from prefetched tar files."""
        while True:
            # Get next tar file from queue
            tar_data = self.queue.get()
            
            # Check for end of data or exception
            if tar_data is None:
                if self.exception:
                    raise self.exception
                break
                
            tar_url, tar_index, texts, indices = tar_data
            tar_name = os.path.basename(tar_url)
            
            # Yield batches from this tar file
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                
                yield TarBatch(
                    tar_name=tar_name,
                    tar_index=tar_index,
                    texts=texts[batch_start:batch_end],
                    indices=indices[batch_start:batch_end],
                    batch_start=batch_start
                )
                
            # Force garbage collection after each tar
            gc.collect()

def load_clip_model_optimized(clip_model_name: str, device: str):
    """Load CLIP model with maximum optimizations."""
    import clip
    
    print(f"\n🚀 Loading CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model.eval()
    
    # Enable TF32 for Ampere GPUs
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ TF32 and cuDNN optimizations enabled")
    
    # Compile the encode_text function for better performance
    if torch.__version__ >= "2.0.0" and device == "cuda":
        print("🔥 Compiling CLIP text encoder with torch.compile...")
        
        # Move attention masks to GPU before compilation
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
            for block in model.transformer.resblocks:
                if hasattr(block, 'attn_mask') and block.attn_mask is not None:
                    block.attn_mask = block.attn_mask.to(device)
        
        # Also handle the positional embedding if it exists
        if hasattr(model, 'positional_embedding'):
            model.positional_embedding = model.positional_embedding.to(device)
        
        model.encode_text = torch.compile(
            model.encode_text,
            mode="default",
            backend="inductor",
            fullgraph=False,
            disable=False,
        )
        print("✅ Model compilation complete")
    
    return model, clip.tokenize

def process_batch(
    batch: TarBatch,
    model,
    tokenize,
    device: str,
    cache_dir: str,
    normalize: bool
) -> int:
    """Process a batch of texts and save embeddings."""
    
    # Tokenize texts
    try:
        tokens = tokenize(batch.texts, truncate=True).to(device)
    except Exception as e:
        print(f"Error tokenizing batch: {e}")
        return 0
    
    # Compute embeddings
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            embeddings = model.encode_text(tokens)
            
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            embeddings = embeddings.cpu().numpy()
    
    # Save embeddings
    output_file = os.path.join(cache_dir, f"{batch.tar_name[:-4]}_batch_{batch.batch_start:06d}.npz")
    
    try:
        np.savez_compressed(
            output_file,
            embeddings=embeddings,
            indices=batch.indices
        )
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return 0
        
    return len(batch.texts)

def main():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP embeddings with tar prefetching for maximum GPU utilization"
    )
    parser.add_argument(
        "--tar_urls",
        type=str,
        required=True,
        help="Glob pattern or comma-separated list of tar URLs"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Output directory for embeddings cache"
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use"
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="txt",
        help="Key for text captions in WebDataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for CLIP encoding"
    )
    parser.add_argument(
        "--prefetch_size",
        type=int,
        default=3,
        help="Number of tar files to prefetch"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings to unit vectors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for CLIP model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("🚀 CLIP WebDataset Precomputation with Prefetching")
    print("="*60)
    print(f"📁 Cache directory: {args.cache_dir}")
    print(f"🤖 CLIP model: {args.clip_model_name}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔄 Prefetch size: {args.prefetch_size}")
    print(f"🖥️  Device: {args.device}")
    print("="*60 + "\n")
    
    # Parse tar URLs
    if ',' in args.tar_urls:
        tar_urls = [url.strip() for url in args.tar_urls.split(',')]
    else:
        tar_urls = sorted(glob(args.tar_urls))
    
    if not tar_urls:
        print(f"❌ No tar files found matching: {args.tar_urls}")
        return
    
    print(f"📊 Found {len(tar_urls)} tar files to process")
    
    # Create cache directory structure
    model_name_safe = args.clip_model_name.replace('/', '-').replace('@', '-')
    cache_dir = os.path.join(args.cache_dir, model_name_safe, "embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load CLIP model
    model, tokenize = load_clip_model_optimized(args.clip_model_name, args.device)
    
    # Create prefetching loader
    loader = PrefetchingTarLoader(tar_urls, args.caption_key, args.prefetch_size)
    loader.start()
    
    # Process all batches with progress tracking
    total_samples = 0
    start_time = time.time()
    current_tar_index = -1
    tar_progress = None
    estimated_total = len(tar_urls) * 10000  # Estimate 10k samples per tar
    
    try:
        # Main progress bar for overall progress
        main_progress = tqdm(
            total=estimated_total,
            desc="🔥 Total Progress",
            unit="samples",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for batch in loader.get_batches(args.batch_size):
            # Update tar-specific progress bar
            if batch.tar_index != current_tar_index:
                if tar_progress is not None:
                    tar_progress.close()
                
                current_tar_index = batch.tar_index
                tar_name = batch.tar_name
                
                # Create new progress bar for this tar
                tar_progress = tqdm(
                    total=10000,  # Estimate
                    desc=f"📦 [{current_tar_index+1}/{len(tar_urls)}] {tar_name}",
                    unit="samples",
                    leave=False,
                    bar_format="{desc}: {n_fmt}/{total_fmt} [{rate_fmt}]"
                )
            
            # Process batch
            n_processed = process_batch(
                batch, model, tokenize, args.device, cache_dir, args.normalize
            )
            
            # Update progress
            total_samples += n_processed
            tar_progress.update(n_processed)
            main_progress.update(n_processed)
            
            # Adjust total if we're getting close to estimate
            if total_samples > estimated_total * 0.9:
                new_estimate = int(total_samples * 1.2)
                main_progress.total = new_estimate
                main_progress.refresh()
            
            # Print GPU utilization periodically
            if args.verbose and total_samples % 10000 == 0:
                elapsed = time.time() - start_time
                rate = total_samples / elapsed
                print(f"\n⚡ Processing rate: {rate:.1f} samples/sec")
                
                if torch.cuda.is_available():
                    print(f"🎮 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                    print(f"🔥 GPU Utilization: Check nvidia-smi")
        
        # Close final tar progress
        if tar_progress is not None:
            tar_progress.close()
            
        main_progress.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
    finally:
        # Clean up
        loader.stop()
        
    # Save metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "clip_model_name": args.clip_model_name,
        "caption_key": args.caption_key,
        "normalized": args.normalize,
        "total_samples": total_samples,
        "num_tars": len(tar_urls),
        "processing_time": elapsed_time,
        "samples_per_second": total_samples / elapsed_time if elapsed_time > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "tar_files": [os.path.basename(url) for url in tar_urls]
    }
    
    metadata_path = os.path.join(args.cache_dir, model_name_safe, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ Precomputation Complete!")
    print("="*60)
    print(f"📊 Total samples processed: {total_samples:,}")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"⚡ Average speed: {total_samples/elapsed_time:.1f} samples/sec")
    print(f"💾 Embeddings saved to: {cache_dir}")
    print(f"📋 Metadata saved to: {metadata_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()