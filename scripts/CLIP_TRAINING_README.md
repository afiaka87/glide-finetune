# GLIDE Training Scripts

This directory contains unified scripts for training GLIDE models with optional CLIP adapter support.

## Overview

The scripts have been consolidated into three main tools:
- `precompute-clip-embeddings.sh` - Unified CLIP embedding precomputation
- `finetune-glide.sh` - Regular GLIDE fine-tuning (no CLIP)
- `finetune-glide-clip.sh` - GLIDE fine-tuning with CLIP adapters

## Script Details

### 1. `precompute-clip-embeddings.sh`
High-performance script for precomputing CLIP embeddings using the ultra-fast implementation.

**Usage:**
```bash
# Basic usage with dataset preset
./scripts/precompute-clip-embeddings.sh --dataset laion

# Custom dataset with options
./scripts/precompute-clip-embeddings.sh \
    --data-dir /path/to/webdataset \
    --output-dir /path/to/clip_cache \
    --clip-model ViT-L/14 \
    --batch-size 4096
```

**Options:**
- `--dataset` - Preset datasets: laion, cc12m, birds, custom (default: custom)
- `--data-dir` - Path to WebDataset tar files (required for custom)
- `--output-dir` - Output directory for CLIP cache (default: ./clip_cache)
- `--clip-model` - CLIP model: ViT-B/32, ViT-L/14, etc. (default: ViT-B/32)
- `--batch-size` - Batch size (default: 2048)
- `--num-workers` - Number of workers (default: 12)

**Features:**
- Uses torch.compile and BF16 for maximum speed
- Concurrent tar processing for better throughput
- Optimized for large-scale datasets

### 2. `finetune-glide.sh`
Regular GLIDE fine-tuning without CLIP adapters.

**Usage:**
```bash
# Basic usage with dataset preset
./scripts/finetune-glide.sh --dataset laion --config-preset laion

# Custom dataset
./scripts/finetune-glide.sh \
    --data-dir /path/to/dataset \
    --checkpoint-dir ./checkpoints \
    --batch-size 8 \
    --epochs 20 \
    --learning-rate 1e-4
```

**Options:**
- `--dataset` - Preset datasets: laion, cc12m, custom (default: custom)
- `--data-dir` - Path to dataset (required for custom)
- `--checkpoint-dir` - Directory for checkpoints (default: ./checkpoints)
- `--config-preset` - Configuration preset: default, laion, cc12m
- `--batch-size` - Batch size (default: 8)
- `--epochs` - Number of epochs (default: 20)
- `--learning-rate` - Learning rate (default: 1e-4)
- `--resume` - Resume from checkpoint path
- `--eval-prompts` - Evaluation prompts file
- `--project-name` - W&B project name

### 3. `finetune-glide-clip.sh`
GLIDE fine-tuning with CLIP adapters, supporting three-phase training.

**Usage:**
```bash
# Phase 1: Train adapter only
./scripts/finetune-glide-clip.sh --dataset laion --phase 1

# Phase 2: Train adapter + gates (auto-resumes from phase 1)
./scripts/finetune-glide-clip.sh --dataset laion --phase 2

# Phase 3: Full fine-tuning (auto-resumes from phase 2)
./scripts/finetune-glide-clip.sh --dataset laion --phase 3

# Custom dataset with specific options
./scripts/finetune-glide-clip.sh \
    --data-dir /path/to/webdataset \
    --clip-cache-dir /path/to/clip_cache \
    --phase 1 \
    --batch-size 4 \
    --test-mode 100  # Run 100 steps only for testing
```

**Options:**
- `--dataset` - Presets: laion, laion-synthetic, birds, custom (default: custom)
- `--phase` - Training phase: 1, 2, or 3 (default: 1)
- `--data-dir` - Path to WebDataset tar files (required for custom)
- `--clip-cache-dir` - Path to CLIP cache (default: auto-detect)
- `--checkpoint-dir` - Base checkpoint directory (default: ./checkpoints-clip)
- `--clip-model` - CLIP model name (default: ViT-B/32)
- `--batch-size` - Batch size (default: auto based on phase)
- `--epochs` - Number of epochs (default: auto based on phase)
- `--resume` - Resume from specific checkpoint
- `--test-mode` - Run N steps in test mode (disables W&B)

**Phase Details:**
- **Phase 1**: Adapter only training (higher LR: 1e-5)
- **Phase 2**: Adapter + gates training (reduced LR: 5e-6)
- **Phase 3**: Full model fine-tuning (very low LR: 1e-7 main, 1e-6 adapter)

## Complete Workflow Example

### Standard Fine-tuning (No CLIP)
```bash
# Simple LAION fine-tuning
./scripts/finetune-glide.sh --dataset laion --config-preset laion
```

### CLIP-Enhanced Fine-tuning
```bash
# Step 1: Precompute CLIP embeddings (one-time, ~1-2 hours for 5M images)
./scripts/precompute-clip-embeddings.sh --dataset laion

# Step 2: Run three-phase training
./scripts/finetune-glide-clip.sh --dataset laion --phase 1
./scripts/finetune-glide-clip.sh --dataset laion --phase 2
./scripts/finetune-glide-clip.sh --dataset laion --phase 3
```

### Testing and Development
```bash
# Test precomputation with smaller batch size
./scripts/precompute-clip-embeddings.sh \
    --dataset laion \
    --batch-size 512 \
    --num-workers 4

# Test training with limited steps
./scripts/finetune-glide-clip.sh \
    --dataset laion \
    --phase 1 \
    --test-mode 100  # Stop after 100 steps
```

## Migration from Old Scripts

The old scripts have been moved to `.scratch/scripts/` for reference. To migrate:

1. **Precomputation**: Replace dataset-specific scripts with unified script
   - Old: `precompute-clip-laion.sh`
   - New: `precompute-clip-embeddings.sh --dataset laion`

2. **Regular training**: Use `finetune-glide.sh` with presets
   - Old: `run-finetune-laion.sh`
   - New: `finetune-glide.sh --dataset laion --config-preset laion`

3. **CLIP training**: Use `finetune-glide-clip.sh` with phases
   - Old: `run-finetune-laion-clip-3phase.sh 1`
   - New: `finetune-glide-clip.sh --dataset laion --phase 1`

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