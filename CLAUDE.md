# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains code for fine-tuning OpenAI's GLIDE text-to-image model on custom datasets. GLIDE is a diffusion-based model that generates images from text descriptions.

## Common Commands

### Installation
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install dev dependencies
uv sync --all-extras
```

### Training Base Model
```bash
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.2 \
  --checkpoints_dir './finetune_checkpoints'
```

### Training Upsampler Model
```bash
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --train_upsample \
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0 \
  --checkpoints_dir './finetune_checkpoints'
```

### Training with WebDataset (LAION/Alamy)
```bash
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion'
```

### Development Commands
```bash
# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

## Architecture Overview

### Core Components

1. **train_glide.py**: Main entry point for training. Handles argument parsing, model initialization, data loading, and training loop orchestration.

2. **glide_finetune/**: Package containing training implementation
   - `glide_finetune.py`: Core training loop implementation with `run_glide_finetune_epoch()` and separate training steps for base and upsampler models
   - `glide_util.py`: Model loading utilities, tokenization helpers, and sampling functions
   - `loader.py`: Standard dataset implementation for loading local image-caption pairs
   - `wds_loader.py`: WebDataset loader for large-scale datasets like LAION
   - `train_util.py`: Training utilities including wandb setup, checkpoint saving, and image generation
   - `fp16_util.py`: Mixed precision training utilities
   - `noisy_clip_finetune.py`: Alternative training approach using noisy CLIP

### Key Concepts

- **Two-stage generation**: Base model (64x64) + Upsampler (64x64 → 256x256)
- **Classifier-free guidance**: Base model uses `uncond_p=0.2` to randomly replace captions with empty tokens during training
- **Checkpoint management**: Automatically creates numbered run directories for each training session
- **Wandb integration**: Built-in experiment tracking and visualization

### Training Flow

1. Model loads from OpenAI checkpoints or resume from custom checkpoint
2. Data is loaded either from local directory (expects images + captions) or WebDataset format
3. Training runs for specified epochs with periodic:
   - Checkpoint saving
   - Sample generation using test prompt
   - Wandb logging of metrics and generated images
4. Checkpoints are saved in numbered subdirectories under `checkpoints_dir`

### Important Parameters

- `--uncond_p`: Set to 0.2 for base model (classifier-free guidance), 0.0 for upsampler
- `--side_x/--side_y`: Always 64 for both base and upsampler (upsampler scales internally)
- `--use_fp16`: Mixed precision training (noted as potentially unstable)
- `--freeze_transformer/--freeze_diffusion`: Freeze parts of the model during fine-tuning
- `--activation_checkpointing`: Gradient checkpointing to reduce memory usage