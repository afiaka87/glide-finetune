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
- **NEW: CLIP adapter integration** for enhanced text-image alignment
- **NEW: KL divergence regularization** to preserve pretrained capabilities
- **NEW: Early stopping protection** to prevent model degradation
- **NEW: Dry-run testing mode** for safe adapter development
- **NEW: Separate gradient clipping** for adapter and main model stability
- **NEW: Production-ready LAION training scripts** with CLIP adapters


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

## Quick Start

Here's the simplest way to start training:

```sh
# Basic training with recommended settings
uv run python train_glide.py \
  --data_dir './path/to/your/images' \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --use_8bit_adam \
  --use_tf32 \
  --checkpoints_dir './checkpoints'
```

This uses:
- 8-bit AdamW optimizer for 50% memory savings
- TensorFloat-32 for ~3x faster training on modern GPUs
- Default settings that work well for most datasets

## Dataset Formats

### Standard Dataset Format

For regular image-caption pairs, organize your dataset as:

```
data_dir/
├── image1.jpg
├── image1.txt  # Caption for image1
├── image2.png
├── image2.txt  # Caption for image2
└── ...
```

- Each image should have a corresponding `.txt` file with the same name
- Caption files should contain a single line of text describing the image
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Images will be automatically resized to 64x64 during training

### WebDataset Format

For large-scale training, use WebDataset format (`.tar` files):

```
data_dir/
├── shard-000000.tar
├── shard-000001.tar
└── ...
```

Each tar file should contain:
- Image files (e.g., `00001.jpg`, `00001.png`)
- Caption files with matching names (e.g., `00001.txt`)
- Optional metadata files (e.g., `00001.json`)

Example WebDataset usage:
```sh
uv run python train_glide.py \
  --data_dir './path/to/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'webdataset'  # or 'laion' for LAION-specific filtering
```

### CLIP Embedding Pre-computation (Optional)

For faster training with CLIP adapters, you can pre-compute CLIP text embeddings for your dataset. This avoids calculating embeddings on-the-fly during training.

#### Pre-computing for Standard Dataset

```sh
# Pre-compute with ViT-B/32 (faster, smaller embeddings)
uv run python scripts/precompute_clip_text_embeddings.py \
  --data_dir ./path/to/images \
  --clip_model_name ViT-B/32 \
  --batch_size 32 \
  --device cuda

# Pre-compute with ViT-L/14 (better quality, larger embeddings)  
uv run python scripts/precompute_clip_text_embeddings.py \
  --data_dir ./path/to/images \
  --clip_model_name ViT-L/14 \
  --batch_size 16 \
  --device cuda
```

This creates `.clip` files alongside your `.txt` files:
```
data_dir/
├── image1.jpg
├── image1.txt
├── image1.clip  # Pre-computed CLIP embedding
└── ...
```

#### Pre-computing for WebDataset

```sh
# Pre-compute embeddings for all tar files
uv run python scripts/precompute_clip_webdataset_embeddings.py \
  --tar_urls "/path/to/tars/*.tar" \
  --cache_dir ./clip_cache \
  --clip_model_name ViT-B/32 \
  --batch_size 32 \
  --device cuda

# Pre-compute with different CLIP models (non-conflicting)
uv run python scripts/precompute_clip_webdataset_embeddings.py \
  --tar_urls "/path/to/tars/*.tar" \
  --cache_dir ./clip_cache \
  --clip_model_name ViT-L/14 \
  --device cuda

# High-resolution variant
uv run python scripts/precompute_clip_webdataset_embeddings.py \
  --tar_urls "/path/to/tars/*.tar" \
  --cache_dir ./clip_cache \
  --clip_model_name ViT-L/14@336px \
  --device cuda
```

This creates an organized cache structure:
```
clip_cache/
├── ViT-B-32/              # Each model gets its own directory
│   ├── tar_metadata.json  # Tracks processed tar files
│   └── embeddings/
│       ├── shard-000.tar.pt
│       └── shard-001.tar.pt
├── ViT-L-14/              # Different model, separate cache
│   ├── tar_metadata.json
│   └── embeddings/
│       └── shard-000.tar.pt
└── ViT-L-14-336px/        # High-res variant
    └── ...
```

#### Using Pre-computed Embeddings

Once embeddings are pre-computed, enable cache loading during training:

```sh
# For standard dataset
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip_cache \
  --clip_model_name ViT-B/32 \
  # ... other args

# For WebDataset  
uv run python train_glide.py \
  --data_dir ./path/to/tars \
  --use_webdataset \
  --use_clip_cache \
  --clip_cache_dir ./clip_cache \
  --clip_model_name ViT-B/32 \
  # ... other args
```

**Benefits:**
- 5-10x faster data loading (no CLIP encoding during training)
- Supports multiple CLIP models without conflicts
- Automatic validation of model compatibility
- Statistics tracking for cache hits/misses
- Can pre-compute once and reuse across experiments

**Notes:**
- Cache files are automatically validated against the requested CLIP model
- Mismatched models will result in cache misses (safe fallback)
- Pre-computation shows progress bars and processing statistics
- Use `--dry_run` flag to preview what would be processed
- Use `--force_recompute` to regenerate existing cache files

## CLIP Adapter Training (Experimental)

This repository includes experimental support for augmenting GLIDE with CLIP visual-language features, following approaches from CLIP-Adapter and IP-Adapter papers. This allows the model to leverage both GLIDE's pretrained capabilities and CLIP's strong image-text alignment.

### Basic CLIP Adapter Usage

```sh
# Train with CLIP adapter (ViT-B/32)
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --clip_model_name ViT-B/32 \
  --adapter_warmup_steps 10000 \
  --adapter_lr 1e-5 \
  --batch_size 4

# Train with larger CLIP model (ViT-L/14)
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --clip_model_name ViT-L/14 \
  --adapter_warmup_steps 10000 \
  --adapter_lr 1e-5 \
  --batch_size 4
```

### CLIP Training Phases

The CLIP adapter supports three training phases to ensure stable integration with pretrained GLIDE:

```sh
# Phase 1: Train adapter only (default)
uv run python train_glide.py \
  --use_clip \
  --adapter_training_phase adapter_only \
  --adapter_lr 1e-5

# Phase 2: Train adapter + attention gates
uv run python train_glide.py \
  --use_clip \
  --adapter_training_phase adapter_gates \
  --adapter_lr 1e-5

# Phase 3: Full fine-tuning
uv run python train_glide.py \
  --use_clip \
  --adapter_training_phase full \
  --adapter_lr 1e-5 \
  --learning_rate 1e-6  # Lower LR for main model
```

### Gradient Clipping for Stability

To ensure stable training, especially when integrating with pretrained models, use separate gradient clipping for adapter and main model parameters:

```sh
uv run python train_glide.py \
  --use_clip \
  --adapter_grad_clip 0.5  # Aggressive clipping for adapter
  --main_grad_clip 2.0     # Looser clipping for main model
```

### Advanced CLIP Options

```sh
# Full example with all stability features
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --clip_model_name ViT-L/14 \
  --adapter_warmup_steps 10000 \
  --adapter_lr 1e-5 \
  --adapter_wd 1e-2 \
  --adapter_beta2 0.98 \
  --adapter_dropout 0.1 \
  --adapter_grad_clip 0.5 \
  --main_grad_clip 1.0 \
  --stability_threshold 10.0 \
  --use_clip_cache \
  --clip_cache_dir ./clip_cache \
  --use_lora \
  --lora_rank 32 \
  --dry_run_interval 1000 \
  --dry_run_samples 5 \
  --kl_loss_interval 100 \
  --kl_loss_weight 0.01
```

**Key CLIP Parameters:**
- `--use_clip`: Enable CLIP adapter
- `--clip_model_name`: CLIP architecture (ViT-B/32, ViT-L/14, etc.)
- `--adapter_warmup_steps`: Steps to gradually increase CLIP influence (0→0.5)
- `--adapter_lr`: Learning rate for adapter (typically 10x smaller than main LR)
- `--adapter_training_phase`: Training strategy (adapter_only, adapter_gates, full)
- `--adapter_grad_clip`: Max gradient norm for adapter parameters
- `--main_grad_clip`: Max gradient norm for main model parameters
- `--use_lora`: Use LoRA for memory-efficient adapter
- `--lora_rank`: LoRA decomposition rank (lower = less parameters)
- `--kl_loss_interval`: Compute KL divergence loss every N steps (regularization)
- `--kl_loss_weight`: Weight for KL divergence between CLIP/non-CLIP outputs
- `--early_stop_threshold`: Max allowed performance degradation (default: 0.1 = 10%)
- `--early_stop_patience`: Steps to wait before stopping after degradation detected
- `--baseline_eval_interval`: How often to check pretrained performance

**Stability Features:**
- Gate initialization at 0.0 ensures zero initial CLIP influence
- Gradual warmup prevents sudden model changes
- Separate optimizers with different learning rates
- Automatic checkpoint rollback on loss spikes
- Frozen GLIDE encoder option for maximum stability
- Early stopping if pretrained performance degrades

### Test Runs vs Early Stopping: Understanding the Difference

This project has two distinct concepts that are important to understand:

1. **Test Runs (`--test_run`)**: Stops training after a fixed number of steps for testing/development
   - Used to verify your setup works before committing to full training
   - Stops after exactly N steps regardless of performance
   - Example: `--test_run 100` stops after 100 steps for quick validation
   - Previously named `--early_stop` but renamed to avoid confusion

2. **ML Early Stopping (`--early_stop_threshold`, `--early_stop_patience`)**: Monitors model performance and stops if it degrades
   - Protects pretrained model from being damaged by new training
   - Monitors baseline performance without CLIP adapter
   - Stops only if performance degrades beyond threshold
   - Example: `--early_stop_threshold 0.1` stops if performance drops >10%

### Dry-Run Mode for Safety Testing

The dry-run mode allows you to test CLIP adapter components without affecting model outputs, ensuring backward compatibility with pretrained GLIDE:

```sh
# Test adapter every 500 steps without affecting outputs
uv run python train_glide.py \
  --use_clip \
  --dry_run_interval 500 \  # Run test every 500 steps
  --dry_run_samples 10      # Test on 10 samples each time
```

**What Dry-Run Mode Does:**
- Computes CLIP features but doesn't use them in the model
- Verifies outputs remain identical with gate=0
- Logs metrics to track any unexpected changes
- Helps debug adapter issues separately from training problems

**Example Output:**
```
[Dry Run Test] Step 500:
  Max output difference: 0.000000
  Mean output difference: 0.000000
  Outputs identical: True
  Adapter gate value: 0.0250
```

### Understanding KL Divergence (For Non-Technical Users)

**What is KL Divergence?**

KL (Kullback-Leibler) divergence is like a "difference meter" between two ways of generating images. When we add CLIP to GLIDE, we want to make sure we're not drastically changing how the model works - just enhancing it.

Think of it like this:
- Your original GLIDE model is like a skilled artist with their own style
- The CLIP adapter is like giving that artist new tools and references
- KL divergence measures how much the artist's style changes when using the new tools

**Why Do We Use It?**

Without KL divergence:
- The CLIP adapter might completely override GLIDE's learned abilities
- The model could "forget" how to generate certain types of images
- Results might become unpredictable or lower quality

With KL divergence:
- The model learns to use CLIP features while staying true to its original training
- Changes are gradual and controlled
- The best of both worlds: CLIP's understanding + GLIDE's generation quality

**What Do the Numbers Mean?**

- KL divergence close to 0: The outputs are very similar (good for stability)
- KL divergence around 0.01-0.1: Moderate differences (typical during training)  
- KL divergence > 1.0: Major differences (might indicate a problem)

The `--kl_loss_weight` parameter controls how much we penalize these differences. A typical value of 0.01 means we gently encourage the model to stay close to its original behavior while still allowing it to learn from CLIP.

### Early Stopping Protection

The early stopping feature acts as a safety net for your pretrained model:

```sh
# Enable early stopping protection
uv run python train_glide.py \
  --use_clip \
  --early_stop_threshold 0.1 \    # Stop if performance drops by 10%
  --early_stop_patience 1000 \     # Wait 1000 steps before stopping
  --baseline_eval_interval 500     # Check every 500 steps
```

**How It Works:**
1. Every 500 steps, the system tests how well the model performs WITHOUT the CLIP adapter
2. If performance drops by more than 10%, it starts a countdown
3. If performance doesn't recover within 1000 steps, training stops automatically
4. This prevents the CLIP adapter from damaging the pretrained model's core abilities

**Example Scenario:**
- Step 5000: Baseline performance = 0.5 (loss value)
- Step 5500: Baseline performance = 0.6 (20% worse) → Warning triggered
- Step 6500: Still degraded → Training stops to protect the model

This is especially useful for:
- Verifying initial setup (gate=0 should produce identical outputs)
- Monitoring stability during training
- Testing after loading checkpoints
- Debugging unexpected behavior

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

## End-to-End CLIP Adapter Fine-tuning Example

This section provides a complete workflow for fine-tuning GLIDE with CLIP adapters, from data preparation to model training.

### Step 1: Prepare Your Dataset

Organize your dataset in one of these formats:

```
# Standard format
data_dir/
├── image1.jpg
├── image1.txt  # Caption for image1
├── image2.png
├── image2.txt  # Caption for image2
└── ...

# WebDataset format (for large datasets)
data_dir/
├── shard-000.tar
├── shard-001.tar
└── ...
```

### Step 2: Pre-compute CLIP Embeddings (Recommended)

Pre-computing CLIP embeddings speeds up training by 5-10x:

```sh
# For standard dataset
uv run python scripts/precompute_clip_text_embeddings.py \
  --data_dir ./path/to/images \
  --clip_model_name ViT-B/32 \
  --batch_size 256 \
  --num_workers 8

# For WebDataset
uv run python scripts/precompute_clip_webdataset_embeddings.py \
  --data_dir ./path/to/tars \
  --output_dir ./clip_cache \
  --clip_model_name ViT-B/32 \
  --batch_size 256
```

This creates `.clip` files alongside your images (or in a cache directory for WebDataset).

### Step 3: Start Fine-tuning with CLIP Adapters

#### Option A: Simple Training (Recommended for beginners)

```sh
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --use_clip_cache \
  --clip_model_name ViT-B/32 \
  --batch_size 4 \
  --learning_rate 1e-6 \
  --adapter_lr 1e-5 \
  --use_8bit_adam \
  --use_tf32 \
  --checkpoints_dir ./clip_checkpoints \
  --sample_interval 500 \
  --test_prompt "a beautiful landscape painting"
```

#### Option B: Advanced Three-Phase Training

For best results, use a three-phase approach:

```sh
# Phase 1: Train CLIP adapter only (5000-10000 steps)
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --use_clip_cache \
  --clip_model_name ViT-B/32 \
  --adapter_training_phase adapter_only \
  --adapter_lr 1e-5 \
  --adapter_warmup_steps 5000 \
  --batch_size 8 \
  --use_8bit_adam \
  --use_tf32 \
  --activation_checkpointing \
  --checkpoints_dir ./clip_phase1 \
  --num_iterations 10000 \
  --sample_interval 1000

# Phase 2: Fine-tune adapter + attention gates (optional, 5000 steps)
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --use_clip_cache \
  --clip_model_name ViT-B/32 \
  --adapter_training_phase adapter_gates \
  --adapter_lr 5e-6 \
  --resume_ckpt ./clip_phase1/latest.pt \
  --batch_size 6 \
  --use_8bit_adam \
  --use_tf32 \
  --checkpoints_dir ./clip_phase2 \
  --num_iterations 5000

# Phase 3: Full fine-tuning with safety features (as needed)
uv run python train_glide.py \
  --data_dir ./path/to/images \
  --use_clip \
  --use_clip_cache \
  --clip_model_name ViT-B/32 \
  --adapter_training_phase full \
  --adapter_lr 1e-6 \
  --learning_rate 1e-7 \
  --resume_ckpt ./clip_phase2/latest.pt \
  --batch_size 4 \
  --use_8bit_adam \
  --use_tf32 \
  --adapter_grad_clip 0.5 \
  --main_grad_clip 2.0 \
  --kl_loss_interval 100 \
  --kl_loss_weight 0.01 \
  --early_stop_threshold 0.1 \
  --early_stop_patience 1000 \
  --baseline_eval_interval 500 \
  --checkpoints_dir ./clip_phase3 \
  --sample_interval 500
```

#### Option C: WebDataset with CLIP (for large-scale training)

```sh
# Pre-compute embeddings first
uv run python scripts/precompute_clip_webdataset_embeddings.py \
  --data_dir ./laion_shards \
  --output_dir ./clip_cache \
  --clip_model_name ViT-L/14 \
  --batch_size 512

# Then train
uv run python train_glide.py \
  --data_dir ./laion_shards \
  --use_webdataset \
  --wds_dataset_name laion \
  --use_clip \
  --use_clip_cache \
  --clip_cache_dir ./clip_cache \
  --clip_model_name ViT-L/14 \
  --adapter_lr 1e-5 \
  --batch_size 16 \
  --use_8bit_adam \
  --use_tf32 \
  --activation_checkpointing \
  --adapter_warmup_steps 10000 \
  --adapter_grad_clip 0.5 \
  --stability_threshold 10.0 \
  --checkpoints_dir ./laion_clip_ft
```

### Step 4: Monitor Training

The training script will:
- Save checkpoints every `--checkpoint_interval` steps
- Generate sample images every `--sample_interval` steps
- Log metrics to wandb (if `--use_wandb` is set)
- Show CLIP adapter metrics: gate values, gradient norms, KL divergence
- Monitor for training instability and performance degradation

### Step 5: Resume from Checkpoint

If training is interrupted:

```sh
uv run python train_glide.py \
  --resume_ckpt ./clip_checkpoints/latest.pt \
  --data_dir ./path/to/images \
  --use_clip \
  --use_clip_cache \
  # ... same other arguments as before
```

### Best Practices for CLIP Adapter Training

1. **Start Conservative**: Use small learning rates (adapter_lr=1e-5, main lr=1e-7)
2. **Monitor Stability**: Watch for loss spikes and gradient explosions
3. **Use Pre-computed Embeddings**: Significantly faster training
4. **Enable Safety Features**: KL regularization, early stopping, gradient clipping
5. **Test Regularly**: Use `--dry_run_interval` to verify adapter doesn't break base model
6. **Save Often**: Set reasonable checkpoint intervals (every 500-1000 steps)

### Expected Results

With proper tuning, CLIP adapters can:
- Improve text-image alignment and semantic understanding
- Enhance detail generation for specific concepts
- Better follow complex prompts
- Maintain the quality of the pretrained GLIDE model

Training typically shows improvement within 5,000-10,000 steps for adapter-only training.

### Production Training Scripts

Ready-to-use training solutions are available for CLIP adapter workflows:

#### Python Training Script (Recommended)

**`train_glide_clip_adapter.py`** - Production-ready Python script with comprehensive logging and error handling:

```sh
# Basic three-phase training
python train_glide_clip_adapter.py --phase 1 --all_phases

# Single phase with custom paths
python train_glide_clip_adapter.py \
  --phase 2 \
  --data_dir /path/to/data \
  --clip_cache_dir /path/to/cache \
  --checkpoint_base_dir /path/to/checkpoints

# Test mode for validation
python train_glide_clip_adapter.py --phase 1 --test_mode 100

# Save/load configurations
python train_glide_clip_adapter.py --save_config my_config.json
python train_glide_clip_adapter.py --config my_config.json --phase 1
```

**Key Features:**
- **Automatic Logging**: All training logs saved to `./logs/<run_id>/` with timestamps
- **Configuration Management**: Save/load training configs as JSON files  
- **Robust Error Handling**: Graceful shutdown, comprehensive error logs
- **Phase Management**: Automatic checkpoint discovery and phase transitions
- **Hardware Optimization**: TF32, cuDNN benchmarking, memory optimizations
- **Signal Handling**: Ctrl+C saves checkpoint before exit
- **Log Organization**: Separate files for training, errors, and phase-specific logs

**Log Directory Structure:**
```
./logs/clip_adapter_20250802_143022/
├── config.json              # Complete training configuration
├── training.log              # Main training log (info/warnings/errors)
├── errors.log                # Error-only log for debugging
├── phase_1.log               # Phase-specific detailed logs
├── phase_2.log               
└── phase_3.log               
```

#### Legacy Bash Scripts

For compatibility, bash scripts are still available in the `scripts/` directory:

- **`precompute-clip-laion.sh`** - Precompute CLIP embeddings for your LAION dataset
- **`run-finetune-laion-clip.sh`** - Single-phase CLIP adapter training  
- **`run-finetune-laion-clip-3phase.sh`** - Advanced three-phase training

See `scripts/CLIP_TRAINING_README.md` for detailed usage instructions.

### Training Tips

1. **Memory optimization**: If you run out of GPU memory, try:
   - `--use_8bit_adam` to reduce optimizer memory by ~50%
   - `--activation_checkpointing` for gradient checkpointing
   - Reduce `--batch_size` (can go as low as 1)
   - `--freeze_transformer` or `--freeze_diffusion` to train fewer parameters
   - Disable `--use_esrgan` if you don't need 256x256 upsampling during training

2. **Speed optimization**: For faster training on Ampere GPUs (RTX 30xx, A100):
   - `--use_tf32` for up to 3x speedup with minimal precision loss

3. **Image preprocessing**: The code automatically handles images with white padding:
   - White borders are detected and removed before resizing
   - Works best with product images, logos, or centered objects
   - Applied consistently across both standard and WebDataset loaders

4. **Wandb logging**: The training automatically logs to Weights & Biases unless `--test_run` is set
   - Training metrics: loss, learning rate, quartile losses, parameter norms
   - VRAM usage: allocated, reserved, and percentage used
   - Sample images: both 64x64 base and 256x256 ESRGAN versions if enabled
   - WebDataset statistics (when using `--use_webdataset`):
     - Sample processing rates and counts
     - Filtering statistics by type (NSFW, similarity, size, aspect ratio)
     - Metadata distributions (e.g., NSFW ratings, similarity scores)
     - Average preprocessing time per sample

5. **Image Compression**: All generated images are automatically saved as high-quality JPEG files
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
                      [--checkpoints_dir CHECKPOINTS_DIR] [--use_tf32]
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
  --use_tf32            Enable TensorFloat-32 for faster training on Ampere+ GPUs (recommended)
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
  --test_run TEST_RUN   Stop training after this many steps (0 = disabled).
                        Useful for testing and development to ensure your
                        setup works without running full training. Previously
                        named --early_stop but renamed to avoid confusion with
                        ML early stopping (which monitors model performance)
  --laion_no_filter     Skip LAION metadata filtering (faster loading, no
                        metadata requirements)
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps for learning rate scheduler
                        (0 = no warmup)
  --warmup_type {linear,cosine}
                        Type of warmup schedule
  --sample_interval SAMPLE_INTERVAL
                        Steps between generating sample images (default: 1000)
  --clip_model_name CLIP_MODEL_NAME
                        CLIP model to use for adapter (default: ViT-L/14).
                        Options: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
  --adapter_warmup_steps ADAPTER_WARMUP_STEPS
                        Warmup steps for CLIP adapter gate (0-0.5 gradual increase)
  --use_clip_cache      Enable loading pre-computed CLIP embeddings
  --clip_cache_dir CLIP_CACHE_DIR
                        Directory containing pre-computed CLIP embeddings (WebDataset only)
```
