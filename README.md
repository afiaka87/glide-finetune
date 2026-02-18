# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Fine-tune and evaluate GLIDE text-to-image diffusion models with a modern CLI interface.

--- 

## Features

- Multiple samplers: Euler, Euler-A, DPM++, PLMS, DDIM
- CLIP re-ranking: generate N candidates, keep the best K
- WebDataset support for large-scale training (LAION, DataComp, etc.)
- BF16/FP16 mixed precision, gradient accumulation, gradient checkpointing, torch.compile
- W&B logging
- Experimental latent diffusion mode (frozen SD 1.5 VAE + OpenCLIP encoder)

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

## Latent Diffusion Mode (Experimental)

Experimental support for training GLIDE in latent space. Instead of denoising 64x64 pixel images directly, the model operates on 32x32 latent representations (via a frozen SD 1.5 VAE) and produces 256x256 outputs after decoding. A frozen OpenCLIP ViT-L-14 encoder replaces the GLIDE text transformer for conditioning.

### Latent Training
```bash
python train_glide.py \
  --latent_mode \
  --data_dir "/path/to/webdataset/*.tar" \
  --use_webdataset \
  --wds_dataset_name datacomp-clip \
  --wds_image_key jpg \
  --use_captions \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --precision bf16 \
  --uncond_p 0.2
```

### Latent Mode Arguments

| Argument | Default | Description |
|----------|---------|------------|
| `--latent_mode` | off | Enable latent diffusion training |
| `--vae_model` | `stabilityai/sd-vae-ft-mse` | HuggingFace VAE model |
| `--clip_model_name` | `ViT-L-14` | OpenCLIP model architecture |
| `--clip_pretrained` | `laion2b_s32b_b82k` | OpenCLIP pretrained weights |

### Model Initialization (`--init`)

| Value | Description |
|---|---|
| *(empty)* | Auto: `pretrained` for pixel mode, `scratch` for latent mode |
| `pretrained` | Load OpenAI pretrained weights (pixel mode only) |
| `scratch` | Random initialization |
| `checkpoint:<path>` | Resume from a saved checkpoint |
| `pixel-transfer:<path>` | Transfer pixel-space weights to latent model (latent mode only) |

### Training Scope (`--train`)

| Value | Description |
|---|---|
| `all` | Train everything (default) |
| `unet` | Train only UNet, freeze text encoder |
| `unet-scratch` | Reinit UNet from random, freeze text encoder |
| `transformer` | Train only text encoder, freeze UNet (keeps encoder_kv trainable) |
| `transformer-scratch` | Reinit text encoder from random, freeze UNet |

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

Run `python train_glide.py --help` for the complete argument list. Key arguments are documented above.
