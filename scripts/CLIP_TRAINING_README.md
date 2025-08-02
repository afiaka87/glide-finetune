# CLIP Adapter Training Scripts for LAION

This directory contains scripts for training GLIDE with CLIP adapters on the LAION dataset.

## Scripts Overview

### 1. `precompute-clip-laion.sh`
Basic CLIP text embedding precomputation. This is a one-time operation that speeds up training by 5-10x.

**Usage:**
```bash
./scripts/precompute-clip-laion.sh
```

### 2. `precompute-clip-laion-prefetch.sh` (Recommended)
High-performance version with tar prefetching for maximum GPU utilization. Uses producer-consumer pattern to load tar files in background while GPU processes embeddings.

**Usage:**
```bash
./scripts/precompute-clip-laion-prefetch.sh
```

**Benefits over basic version:**
- 🚀 Keeps GPU at 100% utilization by prefetching tar files
- 🔄 Producer-consumer pattern eliminates I/O bottlenecks
- 📊 Better progress tracking with per-tar statistics
- 🎯 Typically 2-3x faster than basic version

**What they do:**
- Process all WebDataset tar files in the LAION directory
- Compute CLIP embeddings for each text caption
- Save embeddings in an organized cache directory
- Show progress with detailed statistics

**Configuration:**
- Edit the scripts to change `LAION_DATA_DIR`, `OUTPUT_DIR`, or `CLIP_MODEL`
- Default CLIP model is ViT-B/32 (you can use ViT-L/14 for better quality)
- Prefetch version allows adjusting `PREFETCH_SIZE` (default: 3 tar files)

### 3. `run-finetune-laion-clip.sh`
Basic training script for GLIDE with CLIP adapters on LAION dataset.

**Usage:**
```bash
./scripts/run-finetune-laion-clip.sh
```

**Features:**
- Uses pre-computed CLIP embeddings for fast training
- Includes all stability features (gradient clipping, KL regularization, early stopping)
- Logs to Weights & Biases
- Saves checkpoints every 2500 steps
- Generates sample images every 1000 steps

### 4. `run-finetune-laion-clip-3phase.sh`
Advanced three-phase training script for optimal results.

**Usage:**
```bash
# Phase 1: Train adapter only (10k steps)
./scripts/run-finetune-laion-clip-3phase.sh 1

# Phase 2: Train adapter + gates (5k steps)
./scripts/run-finetune-laion-clip-3phase.sh 2

# Phase 3: Full fine-tuning (10k steps)
./scripts/run-finetune-laion-clip-3phase.sh 3

# Resume from specific checkpoint
./scripts/run-finetune-laion-clip-3phase.sh 2 /path/to/checkpoint.pt
```

**Phase Details:**
- **Phase 1**: Trains only CLIP adapter components with higher LR (1e-5)
- **Phase 2**: Adds attention gates to training with reduced LR (5e-6)
- **Phase 3**: Full model fine-tuning with very low LR (1e-7 for main, 1e-6 for adapter)

## Workflow Example

1. **First, precompute CLIP embeddings** (one-time, ~1-2 hours for 5M images):
   ```bash
   # Basic version
   ./scripts/precompute-clip-laion.sh
   
   # OR use the faster prefetch version (recommended)
   ./scripts/precompute-clip-laion-prefetch.sh
   ```

2. **Option A: Simple training**
   ```bash
   ./scripts/run-finetune-laion-clip.sh
   ```

3. **Option B: Three-phase training** (recommended):
   ```bash
   # Run each phase sequentially
   ./scripts/run-finetune-laion-clip-3phase.sh 1
   ./scripts/run-finetune-laion-clip-3phase.sh 2
   ./scripts/run-finetune-laion-clip-3phase.sh 3
   ```

## Customization

### Changing CLIP Model
Edit the `CLIP_MODEL` variable in the scripts:
- `ViT-B/32`: Faster, uses less memory (default)
- `ViT-L/14`: Better quality, slower
- `RN50`: ResNet-based, different characteristics

### Adjusting Batch Sizes
The scripts use different batch sizes for each phase:
- Phase 1: 12 (can go higher)
- Phase 2: 10
- Phase 3: 6 (memory intensive)

Adjust based on your GPU memory.

### Monitoring Training
- Check Weights & Biases for real-time metrics
- Sample images saved to checkpoint directory
- Look for:
  - Decreasing adapter loss
  - Stable KL divergence (around 0.01-0.1)
  - Gate values increasing from 0 to 0.5 during warmup
  - No gradient explosions

## Troubleshooting

1. **Out of Memory**: Reduce batch size or enable more aggressive checkpointing
2. **Slow Training**: Ensure you're using pre-computed embeddings (`--use_clip_cache`)
3. **Poor Quality**: Check if early stopping triggered - may need to adjust thresholds
4. **Can't Find Embeddings**: Make sure CLIP model name matches between precompute and training

## Expected Results

- Phase 1: Should see loss improvement within 2-3k steps
- Phase 2: Subtle improvements in following complex prompts
- Phase 3: Overall quality refinement, better detail generation

Total training time: ~24-48 hours on a single A100/A6000 GPU