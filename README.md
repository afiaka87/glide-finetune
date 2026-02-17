# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Fine-tune and evaluate GLIDE text-to-image diffusion models with a modern CLI interface.

--- 

## Features

- 🎨 **Modern CLI**: Clean command-line interface built with Typer
- 🚀 **Advanced Samplers**: Euler, Euler-A, DPM++, PLMS, and DDIM
- 🎯 **CLIP Re-ranking**: Generate multiple candidates and select the best
- 📊 **WebDataset Support**: Train on large-scale datasets like LAION and synthetic datasets
- 🔧 **LoRA Support**: Parameter-efficient fine-tuning
- 📈 **W&B Integration**: Automatic experiment tracking
- ⚡ **Performance**: Gradient accumulation, BF16/FP16 mixed precision, torch.compile support
- 🆕 **BF16 Support**: Stable mixed-precision training with bfloat16 (recommended over FP16)
- 🧪 **Latent Diffusion**: Experimental latent-space training with a frozen VAE and CLIP encoder

## Installation

### Using uv (Recommended)
```bash
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
uv sync
uv pip install -e .
```

### Using pip
```bash
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Generate Images

```bash
# Generate a single image
glide eval generate base.pt sr.pt \
  --prompt "a serene mountain landscape at sunset" \
  --cfg 4.0 \
  --seed 42

# Generate from prompt file with CLIP re-ranking
glide eval generate base.pt sr.pt \
  --prompt-file prompts.txt \
  --clip-rerank \
  --clip-candidates 32 \
  --clip-top-k 8
```

### Train Models

```bash
# Train base model (64x64)
glide train base /path/to/dataset \
  --batch-size 4 \
  --lr 1e-4 \
  --wandb my-project

# Train upsampler (64→256)
glide train upsampler /path/to/dataset \
  --upscale 4 \
  --lr 5e-5 \
  --wandb my-upsampler
```

## CLI Usage

The GLIDE CLI provides two main commands: `train` and `eval`.

### Training Commands

#### Train Base Model (64x64)
```bash
glide train base /path/to/dataset \
  --batch-size 4 \
  --lr 1e-4 \
  --epochs 100 \
  --uncond-p 0.2 \  # Classifier-free guidance
  --wandb my-project \
  --fp16 \  # Mixed precision
  --grad-ckpt  # Gradient checkpointing
```

#### Train with LoRA (Efficient Fine-tuning)
```bash
glide train base /path/to/dataset \
  --lora \
  --lora-rank 8 \
  --lora-alpha 32 \
  --freeze-transformer \
  --wandb lora-finetune
```

#### Train on WebDataset (LAION)
```bash
glide train base /mnt/laion/*.tar \
  --webdataset \
  --batch-size 8 \
  --lr 1e-4 \
  --precision bf16 \
  --grad-ckpt
```

#### BF16 Mixed Precision Training (Recommended)
```bash
# BF16 provides better stability than FP16
python train_glide.py \
  --data_dir /path/to/dataset \
  --precision bf16 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --activation_checkpointing
```

### Evaluation Commands

#### Generate with Advanced Samplers
```bash
# DPM++ for better quality with fewer steps
glide eval generate base.pt sr.pt \
  --prompt "a masterpiece painting" \
  --sampler dpm++ \
  --base-steps 20 \
  --sr-steps 15

# Euler for fast generation
glide eval generate base.pt sr.pt \
  --prompt "futuristic city" \
  --sampler euler \
  --cfg 3.0
```

#### CLIP Re-ranking for Quality
```bash
glide eval generate base.pt sr.pt \
  --prompt-file artistic_prompts.txt \
  --clip-rerank \
  --clip-model ViT-L-14/laion2b_s32b_b82k \
  --clip-candidates 64 \
  --clip-top-k 4
```

#### Compare Models
```bash
glide eval compare \
  model1_base.pt model1_sr.pt \
  model2_base.pt model2_sr.pt \
  "a test prompt" \
  --seed 42
```

## Advanced Features

### Samplers
- **euler**: Fast deterministic ODE solver
- **euler_a**: Euler with ancestral sampling
- **dpm++**: DPM-Solver++ for fewer steps
- **plms**: Pseudo Linear Multi-Step
- **ddim**: Denoising Diffusion Implicit Models

### CLIP Re-ranking
Generate multiple candidates and select the best using CLIP:
- Supports OpenCLIP models (ViT-L-14/laion2b_s32b_b82k recommended)
- Memory-efficient GPU offloading
- Batch processing for speed

### Performance Optimizations
- **Gradient Accumulation**: Larger effective batch sizes
- **Mixed Precision**: FP16/BF16 training
- **Gradient Checkpointing**: Trade compute for memory
- **torch.compile**: Optimized inference
- **LoRA**: Parameter-efficient fine-tuning

## Latent Diffusion Mode (Experimental)

There is experimental support for training GLIDE in latent space using a frozen Stable Diffusion VAE. Instead of denoising 64x64 pixel images directly, the model operates on 32x32 latent representations and can produce 256x256 outputs after VAE decoding. A frozen OpenCLIP ViT-L/14 encoder replaces the orignal GLIDE text transformer for conditioning.

You can bootstrap a latent model from an existing pixel-space checkpoint (e.g. laionide-v3-base) using `--init_from_pixel`. This transfers the UNet backbone and text transformer weights — only the input/output convolutions need to be adapted for the different channel counts. New latent channels are zero-initialized so the model starts from a reasonable point rather than random noise.

### Latent Training
```bash
python train_glide.py \
  --latent_mode \
  --init_from_pixel laionide-v3-base.pt \
  --data_dir "/path/to/webdataset/{0000000..0000999}.tar" \
  --use_webdataset \
  --wds_dataset_name datacomp-real \
  --wds_image_key jpg \
  --wds_caption_key txt \
  --use_captions \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --precision bf16 \
  --uncond_p 0.2
```

### Latent Mode Arguments

| Argument | Default | Desription |
|----------|---------|------------|
| `--latent_mode` | off | Enable latent diffusion training |
| `--init_from_pixel` | `""` | Path to a pixel-space checkpoint for weight transfer |
| `--vae_model` | `stabilityai/sd-vae-ft-mse` | HuggingFace VAE model |
| `--clip_model_name` | `ViT-L-14` | OpenCLIP model architecture |
| `--clip_pretrained` | `laion2b_s32b_b82k` | OpenCLIP pretrained weights |

### How Weight Transfer Works

The pixel-space and latent-space models share most of their architecture — the text transformer, UNet middle blocks, and attention layers are all identical. Only the first and last convolutions differ:

- **Input conv**: pixel expects 3 RGB channels, latent expects 4 VAE channels. The 3 pretrained channels are copied and the 4th is zero-initialized.
- **Output conv**: pixel produces 6 channels (3 epsilon + 3 variance), latent produces 8 (4 epsilon + 4 variance). Epsilon and variance groups are mapped separately so the semantic split is preserved.

## Legacy Script Usage

The original training scripts are still available:

### Train Base Model (Traditional)
```bash
python train_glide.py \
  --data_dir '/path/to/dataset' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.2 \
  --wandb_project_name 'my_project'
```

### Train Upsampler (Traditional)
```bash
python train_glide.py \
  --data_dir '/path/to/dataset' \
  --train_upsample \
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0
```


## Full Usage
```
usage: train_glide.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                      [--learning_rate LEARNING_RATE]
                      [--adam_weight_decay ADAM_WEIGHT_DECAY]
                      [--ema_rate EMA_RATE] [--side_x SIDE_X]
                      [--side_y SIDE_Y] [--resize_ratio RESIZE_RATIO]
                      [--random_hflip] [--uncond_p UNCOND_P]
                      [--train_upsample] [--resume_ckpt RESUME_CKPT]
                      [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                      [--precision {fp32,fp16,bf16}] [--device DEVICE]
                      [--sample_interval SAMPLE_INTERVAL]
                      [--freeze_transformer] [--freeze_diffusion]
                      [--reinit_transformer] [--random_init]
                      [--wandb_project_name WANDB_PROJECT_NAME]
                      [--activation_checkpointing]
                      [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                      [--use_captions] [--epochs EPOCHS]
                      [--test_batch_size TEST_BATCH_SIZE]
                      [--test_guidance_scale TEST_GUIDANCE_SCALE]
                      [--use_webdataset] [--wds_image_key WDS_IMAGE_KEY]
                      [--wds_caption_key WDS_CAPTION_KEY]
                      [--wds_dataset_name WDS_DATASET_NAME]
                      [--wds_captions_jsonl WDS_CAPTIONS_JSONL] [--seed SEED]
                      [--cudnn_benchmark] [--upscale_factor UPSCALE_FACTOR]
                      [--use_sr_eval] [--sr_model_path SR_MODEL_PATH]
                      [--prompt_file PROMPT_FILE]
                      [--sample_batch_size SAMPLE_BATCH_SIZE]
                      [--eval_base_sampler {standard,euler,euler_a,dpm++}]
                      [--eval_sr_sampler {standard,euler,euler_a,dpm++}]
                      [--eval_base_sampler_steps EVAL_BASE_SAMPLER_STEPS]
                      [--eval_sr_sampler_steps EVAL_SR_SAMPLER_STEPS]
                      [--num_workers NUM_WORKERS]
                      [--wds_buffer_size WDS_BUFFER_SIZE]
                      [--wds_initial_prefetch WDS_INITIAL_PREFETCH]
                      [--wds_debug] [--skip_tar_validation]
                      [--no_cache_validation] [--clear_validation_cache]
                      [--validation_workers VALIDATION_WORKERS] [--use_lora]
                      [--lora_rank LORA_RANK] [--lora_alpha LORA_ALPHA]
                      [--lora_dropout LORA_DROPOUT]
                      [--lora_target_mode {attention,mlp,all,minimal}]
                      [--lora_save_steps LORA_SAVE_STEPS]
                      [--lora_resume LORA_RESUME]
                      [--save_checkpoint_interval SAVE_CHECKPOINT_INTERVAL]
                      [--eval_interval EVAL_INTERVAL]
                      [--reference_stats REFERENCE_STATS] [--latent_mode]
                      [--vae_model VAE_MODEL]
                      [--clip_model_name CLIP_MODEL_NAME]
                      [--clip_pretrained CLIP_PRETRAINED]
                      [--init_from_pixel INIT_FROM_PIXEL]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --adam_weight_decay ADAM_WEIGHT_DECAY, -adam_wd ADAM_WEIGHT_DECAY
  --ema_rate EMA_RATE   EMA decay rate (GLIDE uses 0.9999)
  --side_x SIDE_X, -x SIDE_X
  --side_y SIDE_Y, -y SIDE_Y
  --resize_ratio RESIZE_RATIO, -crop RESIZE_RATIO
                        Crop ratio
  --random_hflip        Apply random horizontal flip augmentation during
                        training (50% probability)
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
  --use_fp16, -fp16     [Deprecated] Use --precision fp16 instead
  --precision {fp32,fp16,bf16}
                        Precision for training: fp32 (default), fp16
                        (unstable), bf16 (recommended for mixed precision)
  --device DEVICE, -dev DEVICE
  --sample_interval SAMPLE_INTERVAL, -sample_freq SAMPLE_INTERVAL
                        Frequency of sampling images for evaluation (defaults
                        to 500)
  --freeze_transformer, -fz_xt
  --freeze_diffusion, -fz_unet
  --reinit_transformer  Reinitialize transformer/text encoder from scratch
                        (use with --freeze_diffusion to train only text
                        encoder)
  --random_init         Skip loading any pretrained weights, train from random
                        initialization
  --wandb_project_name WANDB_PROJECT_NAME, -wname WANDB_PROJECT_NAME
                        Project name for wandb logging
  --activation_checkpointing, -grad_ckpt
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS, -grad_acc GRADIENT_ACCUMULATION_STEPS
                        Number of gradient accumulation steps (effective batch
                        size = batch_size * gradient_accumulation_steps)
  --use_captions, -txt
  --epochs EPOCHS, -epochs EPOCHS
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
                        Name of the webdataset to use (laion, alamy, simple,
                        synthetic, datacomp-synthetic, or datacomp-real)
  --wds_captions_jsonl WDS_CAPTIONS_JSONL
                        Path to external JSONL captions file (required for
                        datacomp-synthetic dataset)
  --seed SEED, -seed SEED
  --cudnn_benchmark, -cudnn
                        Enable cudnn benchmarking. May improve performance.
                        (may not)
  --upscale_factor UPSCALE_FACTOR, -upscale UPSCALE_FACTOR
                        Upscale factor for training the upsampling model only
  --use_sr_eval         Use full pipeline (base + superres) for evaluation
                        sampling during training.
  --sr_model_path SR_MODEL_PATH
                        Path to the super-resolution model checkpoint.
  --prompt_file PROMPT_FILE
                        Path to file containing prompts for evaluation
                        (one per line)
  --sample_batch_size SAMPLE_BATCH_SIZE
                        Number of prompts to generate images for at each
                        sample interval
  --eval_base_sampler {standard,euler,euler_a,dpm++}
                        Sampler for base model evaluation
  --eval_sr_sampler {standard,euler,euler_a,dpm++}
                        Sampler for super-resolution evaluation
  --eval_base_sampler_steps EVAL_BASE_SAMPLER_STEPS
                        Diffusion steps for base model evaluation (default: 30)
  --eval_sr_sampler_steps EVAL_SR_SAMPLER_STEPS
                        Diffusion steps for SR evaluation (default: 17)
  --num_workers NUM_WORKERS
                        Number of dataloader workers (default: 4)
  --wds_buffer_size WDS_BUFFER_SIZE
                        WebDataset shuffle buffer size (default: 1000)
  --wds_initial_prefetch WDS_INITIAL_PREFETCH
                        WebDataset initial prefetch size (default: 10)
  --wds_debug           Enable debug printing for WebDataset loading
  --skip_tar_validation Skip validation of tar files
  --no_cache_validation Force re-validation of all tar files
  --clear_validation_cache
                        Clear the validation cache before starting
  --validation_workers VALIDATION_WORKERS
                        Parallel workers for tar validation (default: auto)
  --use_lora            Enable LoRA for efficient fine-tuning
  --lora_rank LORA_RANK
                        Rank of LoRA decomposition (default: 4)
  --lora_alpha LORA_ALPHA
                        LoRA scaling parameter (default: 32)
  --lora_dropout LORA_DROPOUT
                        Dropout for LoRA layers (default: 0.1)
  --lora_target_mode {attention,mlp,all,minimal}
                        Which modules to apply LoRA to (default: attention)
  --lora_save_steps LORA_SAVE_STEPS
                        Save LoRA adapter every N steps (default: 1000)
  --lora_resume LORA_RESUME
                        Path to resume LoRA adapter from
  --save_checkpoint_interval SAVE_CHECKPOINT_INTERVAL
                        Save full checkpoint every N steps (default: 5000)
  --eval_interval EVAL_INTERVAL
                        Compute FID/KID every N steps (default: 5000, 0 to
                        disable)
  --reference_stats REFERENCE_STATS
                        Path to pre-computed reference stats for FID/KID
  --latent_mode         Enable latent diffusion mode (32x32 latent space via
                        frozen VAE, 256x256 pixel output)
  --vae_model VAE_MODEL
                        HuggingFace model name for the frozen VAE (latent
                        mode only)
  --clip_model_name CLIP_MODEL_NAME
                        OpenCLIP model name for frozen CLIP encoder (latent
                        mode only)
  --clip_pretrained CLIP_PRETRAINED
                        OpenCLIP pretrained weights name (latent mode only)
  --init_from_pixel INIT_FROM_PIXEL
                        Path to a pixel-space checkpoint for weight transfer
                        to latent model
```
