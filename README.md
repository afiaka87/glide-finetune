# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Finetune GLIDE-text2im on your own image-text dataset.

--- 

- finetune the upscaler as well.
- drop-in support for laion and alamy.


## Installation

```sh
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS/Linux with homebrew: brew install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync  # Installs all dependencies from pyproject.toml
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

### CLIP Evaluation

Automated CLIP score evaluation for measuring text-image alignment quality. Track how well your generated images match their text descriptions using the same metrics as DALL-E and Stable Diffusion.

```sh
uv run python clip_eval.py \
  --checkpoint_path 'finetuned_model.pt' \
  --prompts_file 'captions.txt' \
  --num_samples 100 \
  --batch_size 4
```

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
