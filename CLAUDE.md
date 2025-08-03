# CLAUDE.md (Condensed)

This file provides guidance to Claude Code when working with the GLIDE fine-tuning repository.

## Project Overview

Repository for fine-tuning OpenAI's GLIDE text-to-image model on custom datasets using diffusion-based generation.

## Critical Guidelines

- **Don't break existing functionality** - Ask before running GPU-intensive operations
- **Use TF32 instead of FP16** for better stability with minimal code changes
- **Git commits**: Use granular commits, no attribution in messages
- **Dependencies**: Always use `uv` for package management

## CLIP Adapter Status (2025-08-02)

### Recently Completed
- Comprehensive CLIP adapter integration with KL divergence loss
- Fixed critical issues: gate initialization, WebDataset batching, parameter freezing
- Created high-performance precomputation scripts
- Implemented three-phase training system

### Key Technical Insights
1. **Gates use logit space**: Initialize to -10 for gate=0
2. **Load pretrained weights**: Prevents NaN from random initialization  
3. **WebDataset batching**: Handle multiple formats with smart collation
4. **CLIP dimensions**: ViT-B/32=512d, ViT-L/14=768d (not 768/1024)

### Active Training Scripts
```bash
# Three-phase CLIP training (recommended)
./scripts/run-finetune-laion-synthetic-clip-3phase.sh 1  # Phase 1
./scripts/run-finetune-laion-synthetic-clip-3phase.sh 2  # Phase 2  
./scripts/run-finetune-laion-synthetic-clip-3phase.sh 3  # Phase 3

# Python version with better logging
uv run python scripts/train_glide_clip_adapter.py
```

## Common Commands

### Training Base Model
```bash
uv run python train_glide.py \
  --data_dir /path/to/data \
  --use_webdataset \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --use_8bit_adam \
  --use_tf32 \
  --laion_no_filter
```

### Development
```bash
uv run pytest                    # Run tests
uv run ruff format .            # Format code
uv run ruff check --fix .       # Fix linting
uv run mypy .                   # Type check
```

## Known Issues & Solutions

### Training Appears Frozen
- Set `--log_frequency 1` for immediate feedback (default 100 can take minutes)

### Checkpoint Compatibility
- Use `strict=False` when loading non-CLIP checkpoints into CLIP models
- Missing CLIP components will use initialized values

### Memory Management
- Use batch_size=2-4, TF32, 8-bit Adam, activation checkpointing
- Clean up with `torch.cuda.empty_cache()` between phases

## Architecture Notes

### CLIP Adapter Components
- **DualAttentionBlock**: Separate K/V projections for text and CLIP
- **ClipAdapter**: 2-layer MLP with residual connections and learnable gates
- **Three-phase training**: adapter_only → adapter_gates → full

### Critical Implementation Details
- CLIP tokenizer outputs CPU tensors - must move to device
- Test with real CLIP embeddings, not random (causes NaN)
- Image preprocessing: Remove padding BEFORE resizing to 64x64

## Files to Check
- **TODOS.md**: Ongoing tasks and implementation plans
- **THE_PLAN.md**: Detailed CLIP adapter architectural guidance
- **scripts/CLIP_TRAINING_README.md**: CLIP training documentation

## Best Practices
- Type annotations prevent runtime errors (use MyPy)
- Granular commits with logical groupings
- Update CLAUDE.md after major milestones
- Use TodoWrite tool for complex multi-step tasks