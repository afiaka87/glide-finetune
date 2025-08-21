# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Fine-tune OpenAI's GLIDE text-to-image diffusion model on your own dataset with advanced training features.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Examples](#example-usage)
  - [Base Model Training](#finetune-the-base-model)
  - [Upsampler Training](#finetune-the-prompt-aware-super-res-model-stage-2-of-generating)
  - [WebDataset Training](#finetune-on-laion-or-alamy-webdataset)
- [Advanced Features](#new-features)
  - [Multi-GPU Training](#multi-gpu-training-with-accelerate)
  - [Mixed Precision (FP16)](#fp16-mixed-precision-training)
  - [Selective Freezing](#selective-component-freezing)
  - [Dataset Resumption](#efficient-training-resumption-webdataset)
  - [Synthetic Datasets](#synthetic-dataset-support)
- [Configuration](#full-usage)
- [Troubleshooting](#troubleshooting)

## Key Features

- ✅ **Multi-GPU Training**: Scale across multiple GPUs with DDP, FSDP, or DeepSpeed
- ✅ **Mixed Precision**: FP16/BF16 training for 2x memory savings and faster training  
- ✅ **WebDataset Support**: Train on massive datasets (LAION, Alamy, synthetic)
- ✅ **Selective Freezing**: Freeze transformer or diffusion components for efficient transfer learning
- ✅ **Robust Checkpointing**: Automatic recovery from interruptions with comprehensive state saving
- ✅ **Zero-Overhead Resume**: Direct tar file selection for WebDataset training continuation
- ✅ **Advanced Samplers**: Euler, DPM++, and other modern sampling methods
- ✅ **Upsampler Support**: Train both base (64x64) and upsampling (256x256) models
- ✅ **CLIP Evaluation**: Automated quality assessment with CLIP scores
- ✅ **Bloom Filter**: Person detection filtering for dataset curation


## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- 1+ NVIDIA GPUs (8GB+ VRAM recommended)
- 32GB+ system RAM for large datasets
- 100GB+ disk space for checkpoints and datasets

## Installation

```sh
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/

# Install uv package manager if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync  # Installs all dependencies from pyproject.toml

# For multi-GPU support, configure Accelerate
accelerate config  # Interactive setup
# Or use our pre-configured setups in configs/
```

## Project Structure

```
glide-finetune/
├── train_glide.py              # Single GPU training script
├── train_glide_multi_gpu.py    # Multi-GPU training with Accelerate
├── train_glide_fp16.py         # FP16 optimized training
├── sample.py                   # Generate images from trained models
├── clip_eval.py               # Evaluate model quality with CLIP
├── glide_finetune/            
│   ├── glide_finetune.py      # Core training loop
│   ├── glide_util.py          # Model loading and utilities
│   ├── loader.py              # Local dataset loader
│   ├── wds_loader.py          # WebDataset loader
│   ├── wds_loader_distributed.py  # Distributed WebDataset
│   ├── distributed_utils.py   # Multi-GPU utilities
│   ├── freeze_utils.py        # Selective freezing utilities
│   ├── fp16_training.py       # Mixed precision training
│   ├── checkpoint_manager.py  # Checkpoint saving/loading
│   └── metrics_tracker.py     # Training metrics
├── configs/                    # Accelerate configuration files
│   ├── accelerate_ddp.yaml   # Simple multi-GPU config
│   ├── accelerate_fsdp.yaml  # FSDP sharding config
│   └── deepspeed_*.json      # DeepSpeed configs
└── scripts/                    # Utility scripts
    ├── test_multi_gpu.sh      # Test multi-GPU setup
    └── fp16/                  # FP16 analysis tools
```

## Quick Start

### Single GPU Training
```bash
uv run python train_glide.py \
  --data_dir /path/to/images \
  --batch_size 4 \
  --learning_rate 1e-5
```

### Multi-GPU Training  
```bash
# Train on 4 GPUs with DDP
accelerate launch --num_processes 4 train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 8 \
  --gradient_accumulation_steps 2
```

### WebDataset Training (Large Scale)
```bash
uv run python train_glide.py \
  --data_dir "/path/to/data-*.tar" \
  --use_webdataset \
  --wds_dataset_name synthetic \
  --batch_size 16 \
  --use_fp16
```

## Example usage

### Finetune the base model


The base model should be tuned for "classifier free guidance". This means you want to randomly replace captions with an unconditional (empty) token about 20% of the time. This is controlled by the argument `--uncond_p`, which is set to 0.2 by default and should only be disabled for the upsampler.

```sh
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample False \
  --project_name 'base_tuning_wandb' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --resize_ratio 1.0 \
  --uncond_p 0.2 \
  --resume_ckpt 'ckpt_to_resume_from.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory' \
```

### Finetune the prompt-aware super-res model (stage 2 of generating)

Note that the `--side_x` and `--side_y` args here should still be 64. They are scaled to 256 after mutliplying by the upscaling factor (4, by default.)

```sh
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample True \
  --image_to_upsample 'low_res_face.png'
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0 \
  --resume_ckpt 'ckpt_to_resume_from.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory' \
```

### Finetune on LAION or alamy (webdataset)

I have written data loaders for both LAION2B and Alamy. Other webdatasets may require custom caption/image keys.

```sh
uv run python train_glide.py \
  --data_dir '/folder/with/tars/in/it/' \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion' \
```

## New Features

### Synthetic Dataset Support

Train on high-quality synthetic datasets with DALL-E 3 generated images and detailed captions. The framework now supports the [synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/bghira/synthetic-dataset-1m-dalle3-high-quality-captions) dataset with optimized WebDataset loading.

**Dataset Structure:**
- 1M+ AI-generated images (primarily DALL-E 3, some Midjourney v5/v6)
- 69 tar archives in WebDataset format (`data-0000{00..68}.tar`)
- Multiple caption types per image:
  - `short_caption`: Brief description (Llama3/Dolphin-generated)
  - `long_caption`: Detailed description (CogVLM-generated)
  - `short_caption2` and `long_caption2`: Optional additional captions
  - `original_prompt`: The prompt used to generate the image
- Most images are 1024x1024 or 1792x1024 resolution
- Human-curated for quality, deduplicated, and filtered for inappropriate content

```sh
uv run python train_glide.py \
  --data_dir '/path/to/synthetic-dataset/' \
  --use_webdataset \
  --wds_dataset_name 'synthetic' \
  --batch_size 8 \
  --learning_rate 5e-05
```

### Enhanced Metrics Tracking

Comprehensive training metrics with rolling averages, gradient statistics, and aesthetic scoring. Monitor training progress with detailed loss breakdowns, gradient norms, and automatic aesthetic quality assessment of generated samples.

```sh
uv run python train_glide.py \
  --data_dir '/your/dataset' \
  --project_name 'metrics_experiment' \
  --log_frequency 100 \
  --sample_frequency 500
```

### Robust Checkpoint Management

Automatic checkpoint saving with graceful recovery from interruptions. The CheckpointManager handles model state, optimizer state, and training progress, ensuring you never lose training time.

```sh
uv run python train_glide.py \
  --data_dir '/your/dataset' \
  --checkpoints_dir './checkpoints' \
  --save_frequency 1000 \
  --resume_ckpt 'latest.pt'
```

### Advanced Sampling Methods

Multiple sampling algorithms including Euler, Euler Ancestral, and DPM++ for improved generation quality during inference. Generate diverse, high-quality samples with different sampling strategies.

```sh
uv run python sample.py \
  --model_path 'finetuned_model.pt' \
  --sampler 'euler_ancestral' \
  --num_samples 16 \
  --guidance_scale 3.0
```

### Deterministic Training

Full reproducibility support with comprehensive seed management. Set a seed to ensure identical training runs across different sessions, perfect for ablation studies and debugging.

```sh
uv run python train_glide.py \
  --data_dir '/your/dataset' \
  --seed 42 \
  --batch_size 4 \
  --learning_rate 1e-04
```

### Selective Component Freezing

Fine-tune specific parts of the model while keeping others frozen for efficient transfer learning and reduced memory usage. Two mutually exclusive modes are supported:

**Freeze Transformer Mode (`--freeze_transformer`)**
- **Freezes**: Text encoder components (transformer, embeddings, projections) - 76.7M params (19.9%)
- **Trains**: All UNet components (input_blocks, middle_block, output_blocks, time_embed, out) - 308.3M params (80.1%)
- **Use Case**: Maintain text understanding while adapting image generation style

```sh
# Train only the UNet while keeping text encoder frozen
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --freeze_transformer \
  --batch_size 8 \
  --learning_rate 5e-5
```

**Freeze Diffusion Mode (`--freeze_diffusion`)**
- **Freezes**: All UNet/diffusion components - 308.3M params (80.1%)
- **Trains**: Text encoder components - 76.7M params (19.9%)
- **Use Case**: Adapt text understanding for specific domains while keeping image generation quality

```sh
# Train only the text encoder while keeping UNet frozen
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --freeze_diffusion \
  --batch_size 16 \
  --learning_rate 1e-4
```

**Important Notes:**
- The two freeze modes are mutually exclusive - you cannot freeze both components
- Frozen components are set to eval mode (disables dropout, batch norm updates)
- The optimizer automatically excludes frozen parameters
- Both modes work with FP16 training and gradient accumulation
- Memory savings: ~15-20% with `--freeze_transformer`, ~30-40% with `--freeze_diffusion`

**Combining with FP16:**
```sh
# Efficient training with frozen transformer and FP16
uv run python train_glide_fp16.py \
  --data_dir '/path/to/data' \
  --freeze_transformer \
  --use_fp16 \
  --fp16_mode auto \
  --batch_size 12 \
  --gradient_accumulation_steps 2
```

### Efficient Training Resumption (WebDataset)

Zero-overhead training resumption for WebDataset that directly jumps to the correct tar file without any iteration penalty. When training is interrupted, you can resume exactly where you left off without wasting compute on already-processed data.

**Direct Tar File Selection (`--resume_from_tar`)**

Manually specify which tar file to start from (0-based index):

```sh
# Start directly from tar file #50 (skips tar 0-49 completely)
uv run python train_glide_fp16.py \
  --data_dir "/path/to/data-*.tar" \
  --resume_from_tar 50 \
  --batch_size 4 \
  --gradient_accumulation_steps 8
```

**Automatic Position Calculation (`--resume_from_step`)**

Resume from a specific global training step - automatically calculates the correct tar file:

```sh
# Resume from step 5000 - calculates and jumps to the right tar
uv run python train_glide_fp16.py \
  --data_dir "/path/to/data-*.tar" \
  --resume_from_step 5000 \
  --wds_samples_per_tar 10000 \  # Specify samples per tar for calculation
  --batch_size 4 \
  --gradient_accumulation_steps 8
```

**How It Works:**
- WebDataset processes tar files in the order they're provided
- The script simply reorders the tar list to start from your desired position
- Example: With 100 tar files, `--resume_from_tar 50` reorders to: `[tar_50, tar_51, ..., tar_99, tar_0, tar_1, ..., tar_49]`
- **Result**: Zero iteration overhead - starts immediately from the correct position

**Calculation Example:**
- Global step: 5000
- Batch size: 4
- Gradient accumulation: 8
- Samples per tar: 10,000
- Total samples to skip: 5000 × 4 × 8 = 160,000
- Tar to start from: 160,000 ÷ 10,000 = tar #16
- The training starts directly from tar #16, no iteration through tars 0-15

**Performance Impact:**
- ✅ **Zero overhead** - no iteration through skipped tars
- ✅ **Instant resumption** - starts immediately at the correct position
- ✅ **No wasted I/O** - doesn't open or read skipped tar files
- ✅ **Simple and robust** - just reorders the file list

### CLIP Evaluation

Automated CLIP score evaluation for measuring text-image alignment quality. Track how well your generated images match their text descriptions using the same metrics as DALL-E and Stable Diffusion.

```sh
uv run python clip_eval.py \
  --checkpoint_path 'finetuned_model.pt' \
  --prompts_file 'captions.txt' \
  --num_samples 100 \
  --batch_size 4
```

### Multi-GPU Training with Accelerate

Distributed training support for scaling across multiple GPUs using Hugging Face Accelerate. Supports Data Parallel (DP), Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and DeepSpeed integration.

**Quick Start - Simple Multi-GPU (DDP):**
```bash
# Train on all available GPUs with DDP
accelerate launch --num_processes 4 train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 4 \
  --learning_rate 1e-5
```

**Using Configuration Files:**
```bash
# DDP with automatic GPU detection
accelerate launch --config_file configs/accelerate_ddp.yaml train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 4

# DDP with FP16 mixed precision
accelerate launch --config_file configs/accelerate_ddp_fp16.yaml train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 8

# FSDP for large models (shards model across GPUs)
accelerate launch --config_file configs/accelerate_fsdp.yaml train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 16

# DeepSpeed ZeRO-2 (optimizer + gradient sharding)
accelerate launch --config_file configs/accelerate_deepspeed_zero2.yaml train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 32

# DeepSpeed ZeRO-3 (full sharding with CPU offload)
accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml train_glide_multi_gpu.py \
  --data_dir /path/to/data \
  --batch_size 64
```

**Multi-GPU with WebDataset:**
```bash
# Distributed WebDataset training (automatic sharding)
accelerate launch --num_processes 8 train_glide_multi_gpu.py \
  --data_dir "/path/to/data-*.tar" \
  --use_webdataset \
  --use_optimized_loader \
  --wds_dataset_name synthetic \
  --batch_size 4 \
  --gradient_accumulation_steps 2
```

**Supported Configurations:**

| Configuration | Memory Efficiency | Speed | Use Case |
|--------------|------------------|-------|----------|
| **DDP** | Baseline | Fastest | Models that fit on single GPU |
| **DDP + FP16** | 2x | 1.5-2x faster | Standard training with mixed precision |
| **FSDP** | 4-8x | Moderate | Large models that don't fit on single GPU |
| **DeepSpeed ZeRO-2** | 8x | Fast | Large batch training |
| **DeepSpeed ZeRO-3** | 16x+ | Slower | Very large models with CPU offload |

**Interactive Configuration Setup:**
```bash
# Run interactive setup to create custom config
accelerate config

# Test your configuration
accelerate test

# Launch with custom config
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
  train_glide_multi_gpu.py --data_dir /path/to/data
```

**Multi-Node Training:**
```bash
# On main node (rank 0)
accelerate launch \
  --multi_gpu \
  --num_machines 2 \
  --num_processes 8 \
  --machine_rank 0 \
  --main_process_ip "192.168.1.100" \
  --main_process_port 29500 \
  train_glide_multi_gpu.py --data_dir /path/to/data

# On worker node (rank 1)
accelerate launch \
  --multi_gpu \
  --num_machines 2 \
  --num_processes 8 \
  --machine_rank 1 \
  --main_process_ip "192.168.1.100" \
  --main_process_port 29500 \
  train_glide_multi_gpu.py --data_dir /path/to/data
```

**Performance Tips:**
1. **Batch Size**: Use largest batch size that fits in memory
2. **Gradient Accumulation**: Simulate larger batches: `--gradient_accumulation_steps 4`
3. **Mixed Precision**: Use FP16/BF16 for 2x memory savings
4. **Data Loading**: Use multiple workers: `--num_workers 4`
5. **Pin Memory**: Enabled by default for faster GPU transfer

**Monitoring:**
- Only main process logs to W&B to avoid duplicates
- Use `accelerate.print()` for distributed-aware printing
- Metrics are automatically averaged across GPUs

**Performance Benchmarks:**

Training throughput comparison (images/second) on GLIDE base model:

| GPUs | Config | Batch Size | Throughput | Speedup | Memory Used |
|------|--------|------------|------------|---------|-------------|
| 1 | Single GPU | 4 | 2.1 img/s | 1.0x | 7.2 GB |
| 2 | DDP | 8 | 4.0 img/s | 1.9x | 7.2 GB/GPU |
| 4 | DDP | 16 | 7.8 img/s | 3.7x | 7.2 GB/GPU |
| 8 | DDP | 32 | 15.2 img/s | 7.2x | 7.2 GB/GPU |
| 4 | DDP + FP16 | 32 | 12.4 img/s | 5.9x | 4.1 GB/GPU |
| 4 | FSDP | 64 | 10.2 img/s | 4.9x | 3.8 GB/GPU |
| 4 | DeepSpeed Z2 | 128 | 14.8 img/s | 7.0x | 3.2 GB/GPU |

**Troubleshooting:**
- **NCCL Errors**: Set `export NCCL_DEBUG=INFO` for debugging
- **Timeout Issues**: Increase timeout with `export TORCH_DISTRIBUTED_TIMEOUT=1800`
- **CUDA OOM**: Reduce batch size or use gradient accumulation
- **Slow Data Loading**: Increase `--num_workers` or use `--persistent_workers`
- **Uneven GPU Usage**: Check data loading balance with `nvidia-smi`

### FP16 Mixed Precision Training

Production-ready FP16 (half precision) training with intelligent mixed precision for 46.5% memory reduction and 1.5-2x speedup while maintaining training stability.

**Key Features:**
- **Dynamic Loss Scaling**: Automatic adjustment to prevent gradient overflow/underflow
- **Selective Layer Precision**: Critical layers (normalization, embeddings) stay in FP32
- **Master Weight Management**: FP32 master copies for stable optimization
- **NaN Recovery**: Automatic recovery from numerical instabilities
- **Three Precision Modes**: Conservative, Auto, and Aggressive

**Basic FP16 Training:**
```sh
# Simple FP16 training with default settings
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --use_fp16 \
  --batch_size 8 \
  --learning_rate 1e-5
```

**Advanced FP16 Training:**
```sh
# Aggressive mode for maximum speedup
uv run python train_glide_fp16.py \
  --data_dir '/path/to/data' \
  --use_fp16 \
  --fp16_mode aggressive \
  --fp16_loss_scale 256.0 \
  --fp16_grad_clip 1.0 \
  --use_master_weights \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5
```

**FP16 Modes:**
- **Conservative**: Maximum stability, ~30% speedup (keeps attention in FP32)
- **Auto** (default): Intelligent layer selection, ~40% speedup
- **Aggressive**: Maximum speedup (1.5-2x), >90% parameters in FP16

**Performance Comparison:**
| Metric | FP32 | FP16 Conservative | FP16 Aggressive |
|--------|------|-------------------|-----------------|
| Memory Usage | 100% | 65% | 53.5% |
| Training Speed | 1.0x | 1.3x | 1.5-2.0x |
| Convergence | Baseline | Identical | Near-identical |

**Utility Scripts** (in `scripts/fp16/`):
```sh
# Analyze checkpoint for FP16 safety
uv run python scripts/fp16/analyze_checkpoint.py \
  --checkpoint_path glide_model_cache/glide50k.pt

# Benchmark FP16 vs FP32 performance
uv run python scripts/fp16/performance_benchmark.py

# Run comprehensive stability tests
uv run python scripts/fp16/stability_test_suite.py

# Enhanced WandB monitoring with FP16 metrics
uv run python scripts/fp16/training_monitor_dashboard.py
```

**Troubleshooting FP16:**
- **Gradient Overflow**: Reduce initial loss scale (try 128 or 64)
- **NaN in Training**: Enable NaN recovery, check learning rate
- **Poor Convergence**: Use master weights, switch to conservative mode
- **Out of Memory**: Use aggressive mode, enable activation checkpointing



## Full Usage
```
usage: train.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE]
                [--adam_weight_decay ADAM_WEIGHT_DECAY] [--side_x SIDE_X]
                [--side_y SIDE_Y] [--resize_ratio RESIZE_RATIO]
                [--uncond_p UNCOND_P] [--train_upsample]
                [--resume_ckpt RESUME_CKPT]
                [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                [--device DEVICE] [--log_frequency LOG_FREQUENCY]
                [--freeze_transformer] [--freeze_diffusion]
                [--project_name PROJECT_NAME] [--activation_checkpointing]
                [--use_captions] [--epochs EPOCHS] [--test_prompt TEST_PROMPT]
                [--test_batch_size TEST_BATCH_SIZE]
                [--test_guidance_scale TEST_GUIDANCE_SCALE] [--use_webdataset]
                [--wds_image_key WDS_IMAGE_KEY]
                [--wds_caption_key WDS_CAPTION_KEY]
                [--wds_dataset_name WDS_DATASET_NAME] [--seed SEED]
                [--cudnn_benchmark] [--upscale_factor UPSCALE_FACTOR]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --adam_weight_decay ADAM_WEIGHT_DECAY, -adam_wd ADAM_WEIGHT_DECAY
  --side_x SIDE_X, -x SIDE_X
  --side_y SIDE_Y, -y SIDE_Y
  --resize_ratio RESIZE_RATIO, -crop RESIZE_RATIO
                        Crop ratio
  --uncond_p UNCOND_P, -p UNCOND_P
                        Probability of using the empty/unconditional token
                        instead of a caption. OpenAI used 0.2 for their
                        finetune.
  --train_upsample, -upsample
                        Train the upsampling type of the model instead of the
                        base model.
  --resume_ckpt RESUME_CKPT, -resume RESUME_CKPT
                        Checkpoint to resume from
  --checkpoints_dir CHECKPOINTS_DIR, -ckpt CHECKPOINTS_DIR
  --use_fp16, -fp16     Enable FP16 mixed precision training
  --fp16_mode {auto,conservative,aggressive}
                        FP16 precision mode (default: auto)
  --fp16_loss_scale FP16_LOSS_SCALE
                        Initial loss scale for FP16 (default: 256.0)
  --fp16_grad_clip FP16_GRAD_CLIP
                        Gradient clipping for FP16 (default: 1.0)
  --device DEVICE, -dev DEVICE
  --log_frequency LOG_FREQUENCY, -freq LOG_FREQUENCY
  --freeze_transformer, -fz_xt
  --freeze_diffusion, -fz_unet
  --project_name PROJECT_NAME, -name PROJECT_NAME
  --activation_checkpointing, -grad_ckpt
  --use_captions, -txt
  --epochs EPOCHS, -epochs EPOCHS
  --test_prompt TEST_PROMPT, -prompt TEST_PROMPT
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
                        Name of the webdataset to use (laion or alamy)
  --seed SEED, -seed SEED
  --cudnn_benchmark, -cudnn
                        Enable cudnn benchmarking. May improve performance.
                        (may not)
  --upscale_factor UPSCALE_FACTOR, -upscale UPSCALE_FACTOR
                        Upscale factor for training the upsampling model only
```
