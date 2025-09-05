# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Environment
```bash
# Clone required dependency (crowsonkb fork for compatibility)
git clone https://github.com/crowsonkb/glide-text2im
cd glide-text2im
pip install -e .
cd ..

# Install dependencies with uv (Python 3.12+)
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
  --checkpoints_dir 'checkpoint_directory' \
  --wandb_project_name 'my_project' \
  --sample_interval 500 \
  --log_frequency 100
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
  --image_to_upsample 'low_res_image.png' \
  --checkpoints_dir 'checkpoint_directory'
```

#### Train on webdataset (LAION/Alamy/Custom)
```bash
# Using a directory with tar files
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion' \
  --use_captions

# Using glob patterns
uv run python train_glide.py \
  --data_dir '/path/to/dataset/*.tar' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'png' \
  --wds_dataset_name 'simple' \
  --use_captions

# Using brace expansion for numbered shards
uv run python train_glide.py \
  --data_dir '/path/to/dataset/shard_{00000..00115}.tar' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'png' \
  --wds_dataset_name 'simple' \
  --use_captions
```

### Testing and Evaluation

#### Test all samplers
```bash
uv run python test_samplers.py \
  --prompt "a beautiful landscape" \
  --steps 50 \
  --guidance-scale 3.0 \
  --output-dir sampler_outputs
```

#### Quick sampler test
```bash
uv run python quick_test_samplers.py
```

#### Test WebDataset loading
```bash
uv run python test_webdataset.py
uv run python test_wds_detailed.py
```

## Architecture

This repository implements finetuning for GLIDE (Guided Language to Image Diffusion for Generation and Editing), a text-to-image generation model based on diffusion. The codebase consists of two main training modes:

### Core Components

- **Base Model Training** (`train_upsample=False`): Trains the 64x64 text-to-image generation model with classifier-free guidance (randomly replacing captions with empty tokens ~20% of the time via `uncond_p`)

- **Upsampler Training** (`train_upsample=True`): Trains the prompt-aware super-resolution model that upscales 64x64 images to 256x256

### Key Modules

- `glide_finetune/`: Main training logic and model utilities
  - `glide_finetune.py`: Core training loop implementation with wandb integration
  - `glide_util.py`: Model loading, tokenization, and sampling utilities
  - `loader.py`: Standard dataset loader for image-caption pairs
  - `wds_loader.py`: WebDataset loader for large-scale datasets (LAION2B, Alamy, custom)
  - `train_util.py`: Training utilities including wandb integration
  - `fp16_util.py`: Mixed precision training utilities
  - `enhanced_samplers.py`: Advanced sampling methods (Euler, Euler-A, DPM++)

### Data Loading

The system supports two data loading modes:
1. **Standard datasets**: Image-caption pairs from local directories
2. **WebDatasets**: TAR-based datasets for efficient large-scale training
   - Supports glob patterns, brace expansion, and directory scanning
   - Configurable image/caption keys for different dataset formats
   - Dataset presets: 'laion', 'alamy', 'simple'

### Training Configuration

#### Key Parameters
- `--uncond_p`: Probability of unconditional training (0.2 for base, 0.0 for upsampler)
- `--sample_interval`: How often to generate sample images (default: 500 steps)
- `--log_frequency`: Console logging frequency (default: 100 steps)
- `--wandb_project_name`: W&B project name for experiment tracking
- `--use_captions`: Enable text conditioning (required for text-to-image)
- `--activation_checkpointing`: Reduce memory usage via gradient checkpointing
- `--use_fp16`: Enable mixed precision training

#### Training Notes
- Metrics are logged to wandb every step for smooth monitoring
- Sample images use random captions from `eval_captions.txt` if available
- Checkpoints saved per epoch in numbered directories
- Default sampler during training: Euler with 30 steps
- The `glide_text2im` package must be installed from the crowsonkb fork

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
- If I make an obvious oversight or mistake - feel free to point it out rather than making code changes. It won't hurt my feelings.