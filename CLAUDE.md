# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Environment
```bash
# Clone required dependency
git clone https://github.com/crowsonkb/glide-text2im
cd glide-text2im
pip install -e .
cd ..

# Install dependencies with uv
uv sync

# Run commands with uv
uv run python train_glide.py [args]
```

### Training Commands

#### Train base model (64x64 text-to-image)
```bash
uv run python train_glide.py \
  --data_dir '/path/to/dataset' \
  --train_upsample False \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.2 \
  --checkpoints_dir 'checkpoint_directory'
```

#### Train upsampler model (64x64 → 256x256)
```bash
uv run python train_glide.py \
  --data_dir '/path/to/dataset' \
  --train_upsample True \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0 \
  --upscale_factor 4 \
  --checkpoints_dir 'checkpoint_directory'
```

#### Train on webdataset (LAION/Alamy)
```bash
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion'
```

## Architecture

This repository implements finetuning for GLIDE (Guided Language to Image Diffusion for Generation and Editing), a text-to-image generation model based on diffusion. The codebase consists of two main training modes:

### Core Components

- **Base Model Training** (`train_upsample=False`): Trains the 64x64 text-to-image generation model with classifier-free guidance (randomly replacing captions with empty tokens ~20% of the time via `uncond_p`)

- **Upsampler Training** (`train_upsample=True`): Trains the prompt-aware super-resolution model that upscales 64x64 images to 256x256

### Key Modules

- `glide_finetune/`: Main training logic and model utilities
  - `glide_finetune.py`: Core training loop implementation
  - `glide_util.py`: Model loading and tokenization utilities, interfaces with `glide_text2im` package
  - `loader.py`: Standard dataset loader for image-caption pairs
  - `wds_loader.py`: WebDataset loader for large-scale datasets (LAION2B, Alamy)
  - `train_util.py`: Training utilities including wandb integration
  - `fp16_util.py`: Mixed precision training utilities

### Data Loading

The system supports two data loading modes:
1. **Standard datasets**: Image-caption pairs from local directories
2. **WebDatasets**: TAR-based datasets for efficient large-scale training (LAION, Alamy)

### Important Training Notes

- Base model uses `uncond_p=0.2` for classifier-free guidance
- Upsampler uses `uncond_p=0.0` (no unconditional training)
- Both models train on 64x64 inputs (upsampler scales internally to 256x256)
- Checkpoint saving creates numbered folders per run and per epoch
- The `glide_text2im` package must be installed from the crowsonkb fork for compatibility

## Sampling Methods

The repository now supports multiple advanced sampling algorithms beyond the default PLMS and DDIM:

### Available Samplers

1. **PLMS** (default): Pseudo Linear Multi-Step method - good balance of quality and speed
2. **DDIM**: Denoising Diffusion Implicit Models - deterministic or stochastic via eta parameter
3. **Euler**: Deterministic ODE solver - faster convergence than DDPM
4. **Euler Ancestral**: Stochastic variant of Euler with controllable noise (eta parameter)
5. **DPM++**: DPM-Solver++ - advanced multi-step solver with order 1 or 2

### Using Different Samplers

In Python code:
```python
from glide_finetune.glide_util import load_model, sample

# Load model
model, diffusion, options = load_model(model_type="base")

# Generate with different samplers
sample_euler = sample(
    model, options, 64, 64,
    prompt="a painting of a cat",
    sampler="euler",  # Choose: "plms", "ddim", "euler", "euler_a", "dpm++"
    sampler_eta=0.0,  # For stochastic samplers (euler_a, ddim)
    dpm_order=2,      # For DPM++ (1 or 2)
)
```

### Testing Samplers

Run the comparison script to test all samplers:
```bash
uv run python test_samplers.py \
    --prompt "a beautiful landscape" \
    --steps 50 \
    --guidance-scale 3.0 \
    --output-dir sampler_outputs
```

### Performance Characteristics

- **Euler**: ~15-20% faster than PLMS, deterministic
- **Euler Ancestral**: Similar speed to Euler, adds controlled stochasticity
- **DPM++**: Can achieve good quality with fewer steps (20-30 vs 50-100)
- **DDIM**: Flexible deterministic/stochastic balance via eta parameter