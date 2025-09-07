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

# Install glide-text2im in uv environment if not already installed
uv pip install -e glide-text2im

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
  --log_frequency 100 \
  --save_checkpoint_interval 5000
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

#### Train with LoRA (efficient finetuning)
```bash
uv run python train_glide.py \
  --data_dir '/path/to/dataset' \
  --use_lora \
  --lora_rank 4 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --lora_target_mode 'attention' \
  --freeze_transformer \
  --freeze_diffusion
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

### Gradio Web Interface

#### Running the Interactive Web App
```bash
# Start the Gradio interface
./run_gradio.sh

# Or with custom port
PORT=8080 ./run_gradio.sh

# Enable public sharing (generates a public URL)
SHARE=true ./run_gradio.sh

# Direct Python execution
uv run python gradio_app.py
```

The web interface provides:
- Single prompt text input with batch generation
- Adjustable parameters (batch size, sampler, steps, CFG scale)
- Model selection and optimization options
- Real-time progress updates
- Image gallery with download options
- GPU memory monitoring
- Example prompts for quick testing

Access at: http://localhost:7860

### Testing and Evaluation

#### Batch Evaluation with Rich UI
```bash
# Basic evaluation
uv run python evaluate_glide.py \
  --prompt_file prompts.txt \
  --base_model checkpoints/base.pt \
  --sr_model checkpoints/sr.pt

# Advanced options
uv run python evaluate_glide.py \
  --prompt_file prompts.txt \
  --base_model checkpoints/base.pt \
  --sr_model checkpoints/sr.pt \
  --sampler euler \
  --base_steps 30 \
  --sr_steps 30 \
  --cfg 4.0 \
  --batch_size 4 \
  --save_grid \
  --seed 42 \
  --wandb_project my-evaluation

# With torch.compile for optimized inference
uv run python evaluate_glide.py \
  --prompt_file prompts.txt \
  --base_model checkpoints/base.pt \
  --sr_model checkpoints/sr.pt \
  --use_torch_compile \
  --batch_size 4

# Dry run to test configuration
uv run python evaluate_glide.py \
  --prompt_file prompts.txt \
  --base_model checkpoints/base.pt \
  --sr_model checkpoints/sr.pt \
  --dry_run
```

**Note**: Prompt file must contain a power-of-2 number of prompts (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, or 1024).

#### Quick smoke test
```bash
./smoke_test.sh
```

#### Test with compiled models
```bash
./smoke_test_compiled.sh
```

#### Dry run test
```bash
./smoke_test_dry.sh
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
  - `lora_wrapper.py`: LoRA (Low-Rank Adaptation) implementation for efficient finetuning
  - `cli_utils.py`: Command-line interface utilities

- `train_glide.py`: Main training script with CLI argument parsing
- `evaluate_glide.py`: Batch evaluation script with rich progress UI
- `gradio_app.py`: Web interface for interactive generation

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
- `--save_checkpoint_interval`: How often to save model checkpoints (default: 5000 steps)
- `--log_frequency`: Console logging frequency (default: 100 steps)
- `--wandb_project_name`: W&B project name for experiment tracking
- `--use_captions`: Enable text conditioning (required for text-to-image)
- `--activation_checkpointing`: Reduce memory usage via gradient checkpointing
- `--use_fp16`: Enable mixed precision training
- `--freeze_transformer`: Freeze transformer weights during training
- `--freeze_diffusion`: Freeze diffusion model weights during training
- `--use_lora`: Enable LoRA for parameter-efficient finetuning

#### Training Notes
- Metrics are logged to wandb every step for smooth monitoring
- Sample images use random captions from `eval_captions.txt` if available
- Checkpoints saved at specified intervals in numbered directories
- Default sampler during training: Euler with 30 steps
- The `glide_text2im` package must be installed from the crowsonkb fork

## Sampling Methods

The repository supports multiple advanced sampling algorithms beyond the default PLMS and DDIM:

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

### Performance Characteristics

- **Euler**: ~15-20% faster than PLMS, deterministic
- **Euler Ancestral**: Similar speed to Euler, adds controlled stochasticity
- **DPM++**: Can achieve good quality with fewer steps (20-30 vs 50-100)
- **DDIM**: Flexible deterministic/stochastic balance via eta parameter

### Enhanced Samplers with Upsampling

The enhanced samplers (Euler, Euler Ancestral, DPM++) work with the upsampling pipeline in `sample_with_superres`:

```python
from glide_finetune.glide_util import load_model, sample_with_superres

# Load base and upsampler models
base_model, _, base_options = load_model(model_type="base")
upsampler_model, _, upsampler_options = load_model(model_type="upsample")

# Generate with full pipeline using enhanced samplers
samples = sample_with_superres(
    base_model, base_options,
    upsampler_model, upsampler_options,
    prompt="a beautiful landscape",
    sampler="euler",  # Works for both base and upsampler
    base_respacing="27",
    upsampler_respacing="17",
)
```

**Upsampler Performance**: Testing shows ~1.5x speedup when using Euler/DPM++ vs PLMS for the upsampling stage.