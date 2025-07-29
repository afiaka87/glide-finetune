# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Finetune GLIDE-text2im on your own image-text dataset.

--- 

## Features

- Finetune both base model (64x64) and upsampler (64x64 → 256x256)
- Memory-efficient 8-bit AdamW optimizer support
- TensorFloat-32 (TF32) support for faster training on Ampere GPUs
- WebDataset support with custom filtering modes
- Gradient checkpointing for reduced memory usage
- Learning rate warmup (linear or cosine) for stable training
- Early stopping for testing and integration
- Built-in Weights & Biases (wandb) logging
- Drop-in support for LAION and Alamy datasets
- Modern diffusion samplers with memory-optimized implementations
- Comprehensive checkpoint system with automatic saves and full training state
- Graceful interruption handling (Ctrl+C saves checkpoint before exit)
- Automatic emergency checkpoints on crashes
- Multi-prompt evaluation with grid visualization


## Installation

```sh
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# To run commands, use uv run:
# uv run python train_glide.py --help
```

## Example usage

### Finetune the base model

The base model should be tuned for "classifier free guidance". This means you want to randomly replace captions with an unconditional (empty) token about 20% of the time. This is controlled by the argument `--uncond_p`, which is set to 0.2 by default and should only be disabled for the upsampler.

```sh
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --resize_ratio 1.0 \
  --uncond_p 0.2 \
  --checkpoints_dir 'my_local_checkpoint_directory'

# With memory optimizations
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --use_8bit_adam \
  --activation_checkpointing \
  --checkpoints_dir './finetune_checkpoints'
```

### Finetune the prompt-aware super-res model (stage 2 of generating)

Note that the `--side_x` and `--side_y` args here should still be 64. They are scaled to 256 after mutliplying by the upscaling factor (4, by default.)

```sh
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample True \
  --image_to_upsample 'low_res_face.png' \
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0 \
  --checkpoints_dir 'my_local_checkpoint_directory'

# Resume from a previous upsampler training run
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample True \
  --resume_ckpt './checkpoints/0000/glide-ft-3x1000.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory'
```

### Finetune on WebDataset (LAION, Alamy, or custom)

The project supports WebDataset format for efficient large-scale training.

```sh
# For LAION dataset (applies metadata filtering)
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion'

# For custom WebDataset (no filtering, faster loading)
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'png' \
  --wds_dataset_name 'webdataset'
```

### Training Tips

1. **Memory optimization**: If you run out of GPU memory, try:
   - `--use_8bit_adam` to reduce optimizer memory by ~50%
   - `--activation_checkpointing` for gradient checkpointing
   - Reduce `--batch_size` (can go as low as 1)
   - `--freeze_transformer` or `--freeze_diffusion` to train fewer parameters
   - Disable `--use_esrgan` if you don't need 256x256 upsampling during training

2. **Speed optimization**: For faster training on Ampere GPUs (RTX 30xx, A100):
   - `--use_tf32` for up to 3x speedup with minimal precision loss

3. **Wandb logging**: The training automatically logs to Weights & Biases unless `--early_stop` is set
   - Training metrics: loss, learning rate, quartile losses, parameter norms
   - VRAM usage: allocated, reserved, and percentage used
   - Sample images: both 64x64 base and 256x256 ESRGAN versions if enabled
   - WebDataset statistics (when using `--use_webdataset`):
     - Sample processing rates and counts
     - Filtering statistics by type (NSFW, similarity, size, aspect ratio)
     - Metadata distributions (e.g., NSFW ratings, similarity scores)
     - Average preprocessing time per sample

4. **Image Compression**: All generated images are automatically saved as high-quality JPEG files
   - Uses quality=95 for excellent visual fidelity
   - Reduces file sizes by 50-70% compared to PNG
   - Speeds up uploads to Weights & Biases
   - Handles transparency by converting to RGB with white background
   - File extensions automatically changed from .png to .jpg

### VRAM Requirements

The following are typical VRAM usage patterns:

| Configuration | VRAM Usage | Notes |
|--------------|------------|-------|
| Base GLIDE (inference) | ~1.45 GB | 64x64 generation only |
| Base GLIDE + ESRGAN (inference) | ~1.85 GB | 64x64 → 256x256 upsampling |
| Training (batch_size=1) | ~4-5 GB | Without ESRGAN |
| Training + ESRGAN (batch_size=1) | ~7.4 GB | Includes optimizer states |
| Training with 8-bit Adam | ~3-4 GB | ~50% optimizer memory reduction |

**ESRGAN Integration**: The `--use_esrgan` flag enables automatic upsampling from 64x64 to 256x256:
- ESRGAN model itself only uses ~70MB
- Upsampling overhead during training: ~2-3GB (includes temporary buffers)
- Both 64x64 and 256x256 versions are saved and logged to wandb
- Can be used with both training and inference scripts
- Model downloaded automatically on first use to `esrgan_models/` directory
- Supports RealESRGAN_x4plus.pth (default) and RealESRGAN_x4plus_anime_6B.pth models

## Checkpointing and Resuming

The project automatically saves comprehensive checkpoints that include everything needed to resume training exactly where you left off.

### What Gets Saved

Every checkpoint consists of three files with the same basename:
- `basename.pt` - Model weights
- `basename.optimizer.pt` - Complete optimizer state (including momentum buffers)
- `basename.json` - Training metadata (epoch, step, learning rate, etc.)

### When Checkpoints Are Saved

1. **Regular checkpoints**: Every 5000 training steps
2. **End of epoch**: After each complete epoch
3. **On interruption**: Press Ctrl+C to gracefully save and exit
4. **On error**: Emergency checkpoint saved if training crashes

### Resuming Training

To resume from a checkpoint, use the `--resume_ckpt` flag with any of the checkpoint files:

```sh
# Resume using model weights file (recommended)
uv run python train_glide.py \
  --resume_ckpt './finetune_checkpoints/0000/glide-ft-2x1500.pt' \
  --data_dir '/path/to/data' \
  # ... other arguments

# Or use the JSON file
uv run python train_glide.py \
  --resume_ckpt './finetune_checkpoints/0000/glide-ft-2x1500.json' \
  --data_dir '/path/to/data' \
  # ... other arguments
```

The system will automatically find all associated files and restore:
- Model weights
- Optimizer state (learning rate, momentum, etc.)
- Training position (epoch and step)
- Learning rate schedule progress
- Random number generator states for reproducibility

**Note**: Training resumes from the beginning of the next epoch after the checkpoint to keep things simple.

### Checkpoint Organization

Checkpoints are organized in numbered run directories:
```
checkpoints_dir/
├── 0000/  # First run
│   ├── glide-ft-0x5000.pt
│   ├── glide-ft-0x5000.optimizer.pt
│   ├── glide-ft-0x5000.json
│   └── ...
├── 0001/  # Second run
│   └── ...
```

### Emergency Recovery

If training crashes unexpectedly, you'll be prompted to save a checkpoint (20s timeout, defaults to yes).
Look for emergency checkpoint files:
- `emergency_checkpoint_epoch{N}_step{M}_{timestamp}.pt` (and associated files)
- `interrupted_checkpoint_epoch{N}_step{M}.pt` (for Ctrl+C interruptions)

These can be used with `--resume_ckpt` just like regular checkpoints.

## Sampling and Inference

The project includes a comprehensive sampling script (`scripts/sample.py`) for generating images with trained models:

```sh
# Single prompt with default PLMS sampler
uv run python scripts/sample.py --prompt "a beautiful sunset"

# Multi-prompt file with specific sampler
uv run python scripts/sample.py --prompt_file examples/eval_prompts_4.txt --sampler euler

# Benchmark all samplers
uv run python scripts/sample.py --prompt "test image" --benchmark

# Use custom checkpoint
uv run python scripts/sample.py --model_path checkpoints/model.pt --prompt "landscape"
```

### Available Samplers

- **`plms`** (default): Pseudo Linear Multi-Step method
  - ✓ Original GLIDE sampler, stable and well-tested
  - ✓ Good quality at 100 steps
  - ✗ Not the fastest option available

- **`ddim`**: Denoising Diffusion Implicit Models
  - ✓ Deterministic when eta=0 (reproducible results)
  - ✓ Good for testing and comparisons
  - ✓ 50-100 steps recommended

- **`euler`**: Euler method adapted for GLIDE
  - ✓ Fast generation (~2 seconds for 50 steps)
  - ✓ Good quality with 50 steps
  - ✓ Uses DDPM parameterization (not k-diffusion style)

- **`euler_a`**: Euler Ancestral (adds noise at each step)
  - ✓ More variation in outputs
  - ✓ Good for exploration and diverse results
  - ✗ Non-deterministic even with same seed

- **`dpm++_2m`**: DPM++ 2nd order multistep
  - ✓ Excellent quality/speed balance
  - ✓ 20-25 steps recommended
  - ✓ Second-order solver with log-SNR formulation

- **`dpm++_2m_karras`**: DPM++ with Karras noise schedule
  - ✓ Best quality at low step counts
  - ✓ 20-25 steps recommended
  - ✓ Optimized noise schedule for better convergence

### Sampler Implementation Details

All samplers use GLIDE's native DDPM parameterization:

```python
# Example: How samplers handle GLIDE's parameterization
# (Simplified from actual implementation)

# Get alpha values from diffusion model
alphas_cumprod = th.cumprod(1 - diffusion.betas, dim=0)

# DDPM forward process
noisy_image = np.sqrt(alpha_bar) * original + np.sqrt(1 - alpha_bar) * noise

# Predict clean image from noise prediction
predicted_clean = (noisy_image - np.sqrt(1 - alpha_bar) * predicted_noise) / np.sqrt(alpha_bar)

# DDIM update step
next_image = np.sqrt(next_alpha_bar) * predicted_clean + np.sqrt(1 - next_alpha_bar) * direction
```

### Troubleshooting Samplers

**Issue: Gray or noisy output**
- Ensure you're using the correct sampler names (not `*_fixed` versions)
- Check that your model checkpoint is valid
- Verify guidance scale is reasonable (2.0-5.0 typical)

**Issue: Poor quality at low steps**
- Use DPM++ samplers for best low-step performance
- PLMS needs ~100 steps, Euler needs ~50, DPM++ needs ~25

**Issue: Non-deterministic results**
- Use `ddim` with eta=0 for deterministic sampling
- `euler_a` is inherently stochastic (adds noise at each step)

### Comparing Samplers

To compare all samplers side-by-side:

```sh
# Benchmark mode generates images with all samplers
uv run python scripts/sample.py --prompt "a red apple" --benchmark

# Or use pytest for comprehensive comparison
uv run pytest tests/integration/test_samplers_comprehensive.py::TestSamplerComparison -v
```

This will create a grid showing the same prompt generated by each sampler, making it easy to compare quality and characteristics.

### Usage Example

```sh
# Train with Euler sampler for faster iteration during development
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --sampler euler \
  --test_guidance_scale 3.0 \
  --test_steps 50

# Train with DPM++ 2M Karras for best quality evaluation
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --sampler dpm++_2m_karras \
  --test_guidance_scale 4.0 \
  --test_steps 20
```

## Sampling and Inference

Generate images using trained models with `scripts/sample.py`:

```sh
# Basic usage
uv run python scripts/sample.py --prompt "a beautiful sunset"

# With ESRGAN upsampling (64x64 → 256x256)
uv run python scripts/sample.py --prompt "a cat" --use_esrgan

# Multiple prompts from file
uv run python scripts/sample.py --prompt_file examples/eval_prompts_16.txt

# Specific sampler and steps
uv run python scripts/sample.py --prompt "landscape" --sampler euler --steps 50

# Using custom checkpoint
uv run python scripts/sample.py --model_path checkpoints/0000/glide-ft-1x5000.pt --prompt "test"

# Benchmark all samplers
uv run python scripts/sample.py --prompt "benchmark test" --benchmark
```

### Sampler Options

- `plms` (default): Original GLIDE sampler, 100 steps recommended
- `ddim`: Deterministic, good for reproducibility
- `euler`: Fast generation, 50 steps recommended  
- `euler_a`: More variation, non-deterministic

## Multi-Prompt Evaluation

Evaluate your model on multiple prompts simultaneously with grid visualization:

```sh
# Create evaluation prompts file (must have 2, 4, 8, 16, or 32 lines)
cat > eval_prompts.txt << EOF
a red apple on a wooden table
a blue car on a highway at sunset
a golden retriever playing in snow
a modern house with large windows
EOF

# Use during training
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --eval_prompts_file eval_prompts.txt \
  --sample_interval 1000  # Generate grid every 1000 steps

# Note: Cannot use both --test_prompt and --eval_prompts_file
```

The evaluation grid:
- Generates images for all prompts at each sampling interval
- Creates a square grid (2x2 for 4 prompts, 4x4 for 16 prompts, etc.)
- Saves as `eval_grid_{step}.png` in outputs directory
- Logs to wandb as both a grid view and individual gallery with captions
- Generates initial samples at step 0 for baseline comparison

Example prompt files are provided in `examples/`:
- `eval_prompts_4.txt`, `eval_prompts_8.txt`, `eval_prompts_16.txt`, `eval_prompts_32.txt`


## Full Usage
```
usage: train_glide.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                      [--learning_rate LEARNING_RATE]
                      [--adam_weight_decay ADAM_WEIGHT_DECAY]
                      [--side_x SIDE_X] [--side_y SIDE_Y]
                      [--resize_ratio RESIZE_RATIO] [--uncond_p UNCOND_P]
                      [--train_upsample] [--resume_ckpt RESUME_CKPT]
                      [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                      [--device DEVICE] [--log_frequency LOG_FREQUENCY]
                      [--freeze_transformer] [--freeze_diffusion]
                      [--project_name PROJECT_NAME]
                      [--activation_checkpointing] [--use_captions]
                      [--epochs EPOCHS] [--test_prompt TEST_PROMPT]
                      [--eval_prompts_file EVAL_PROMPTS_FILE]
                      [--test_batch_size TEST_BATCH_SIZE]
                      [--test_guidance_scale TEST_GUIDANCE_SCALE]
                      [--use_webdataset] [--wds_image_key WDS_IMAGE_KEY]
                      [--wds_caption_key WDS_CAPTION_KEY]
                      [--wds_dataset_name WDS_DATASET_NAME] [--seed SEED]
                      [--cudnn_benchmark] [--upscale_factor UPSCALE_FACTOR]
                      [--image_to_upsample IMAGE_TO_UPSAMPLE]
                      [--use_8bit_adam] [--use_tf32] [--early_stop EARLY_STOP]
                      [--sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}]
                      [--test_steps TEST_STEPS] [--laion_no_filter]
                      [--warmup_steps WARMUP_STEPS] [--warmup_type {linear,cosine}]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
                        Path to dataset directory
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size for training (default: 1)
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate (default: 2e-5)
  --adam_weight_decay ADAM_WEIGHT_DECAY, -adam_wd ADAM_WEIGHT_DECAY
                        Adam weight decay (default: 0.0)
  --side_x SIDE_X, -x SIDE_X
                        Width of training images (default: 64)
  --side_y SIDE_Y, -y SIDE_Y
                        Height of training images (default: 64)
  --resize_ratio RESIZE_RATIO, -crop RESIZE_RATIO
                        Crop ratio for training (default: 0.8)
  --uncond_p UNCOND_P, -p UNCOND_P
                        Probability of using empty/unconditional token instead
                        of caption. OpenAI used 0.2 for their finetune.
  --train_upsample, -upsample
                        Train the upsampling type of the model instead of the
                        base model.
  --resume_ckpt RESUME_CKPT, -resume RESUME_CKPT
                        Checkpoint to resume from
  --checkpoints_dir CHECKPOINTS_DIR, -ckpt CHECKPOINTS_DIR
                        Directory to save checkpoints
  --use_fp16, -fp16     Use mixed precision training
  --device DEVICE, -dev DEVICE
                        Device to use (e.g., cuda, cpu)
  --log_frequency LOG_FREQUENCY, -freq LOG_FREQUENCY
                        How often to log training progress
  --freeze_transformer, -fz_xt
                        Freeze transformer weights during training (text processing components)
  --freeze_diffusion, -fz_unet
                        Freeze diffusion/UNet weights during training (image generation backbone)
  --project_name PROJECT_NAME, -name PROJECT_NAME
                        Weights & Biases project name
  --activation_checkpointing, -grad_ckpt
                        Enable gradient checkpointing to save memory
  --use_captions, -txt  Use captions during training
  --epochs EPOCHS, -epochs EPOCHS
                        Number of epochs to train (default: 20)
  --test_prompt TEST_PROMPT, -prompt TEST_PROMPT
                        Prompt to use for generating test images
  --eval_prompts_file EVAL_PROMPTS_FILE
                        File containing line-separated prompts for evaluation
                        (must have 2,4,8,16, or 32 lines)
  --test_batch_size TEST_BATCH_SIZE, -tbs TEST_BATCH_SIZE
                        Batch size used for model eval, not training.
  --test_guidance_scale TEST_GUIDANCE_SCALE, -tgs TEST_GUIDANCE_SCALE
                        Guidance scale used during model eval, not training.
  --use_webdataset, -wds
                        Enables webdataset (tar) loading
  --wds_image_key WDS_IMAGE_KEY, -wds_img WDS_IMAGE_KEY
                        A 'key' e.g. 'jpg' used to access the image in the
                        webdataset
  --wds_caption_key WDS_CAPTION_KEY, -wds_cap WDS_CAPTION_KEY
                        A 'key' e.g. 'txt' used to access the caption in the
                        webdataset
  --wds_dataset_name WDS_DATASET_NAME, -wds_name WDS_DATASET_NAME
                        Name of the webdataset to use (laion, alamy, or
                        webdataset for no filtering)
  --seed SEED, -seed SEED
                        Random seed for reproducibility
  --cudnn_benchmark, -cudnn
                        Enable cudnn benchmarking. May improve performance.
  --upscale_factor UPSCALE_FACTOR, -upscale UPSCALE_FACTOR
                        Upscale factor for training the upsampling model only
  --image_to_upsample IMAGE_TO_UPSAMPLE, -lowres IMAGE_TO_UPSAMPLE
                        Path to low-res image for upsampling visualization
  --use_8bit_adam, -8bit
                        Use 8-bit AdamW optimizer to save memory (requires
                        bitsandbytes)
  --use_tf32, -tf32     Enable TF32 on Ampere GPUs for faster training with
                        slightly reduced precision
  --early_stop EARLY_STOP
                        Stop training after this many steps (0 = disabled).
                        Useful for testing.
  --sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}
                        Sampler to use for generating test images during
                        training. Options: plms (default) - stable, original
                        GLIDE sampler; ddim - deterministic when eta=0, good
                        for reproducibility; euler - fast first-order solver,
                        good quality; euler_a - euler with added noise, more
                        variation but non-convergent; dpm++_2m - second-order
                        solver, good quality/speed balance; dpm++_2m_karras -
                        dpm++_2m with improved schedule for low step counts
  --test_steps TEST_STEPS
                        Number of sampling steps for test image generation
                        (default: 100)
  --laion_no_filter     Skip LAION metadata filtering (faster loading, no
                        metadata requirements)
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps for learning rate scheduler
                        (0 = no warmup)
  --warmup_type {linear,cosine}
                        Type of warmup schedule
  --sample_interval SAMPLE_INTERVAL
                        Steps between generating sample images (default: 1000)
```
