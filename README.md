# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Fine-tune and evaluate GLIDE text-to-image diffusion models.

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

### Train Base Model (64x64)
```bash
python train_glide.py \
  --data_dir /path/to/dataset \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --uncond_p 0.2 \
  --precision bf16 \
  --wandb_project_name my-project
```

### Train on WebDataset (LAION)
```bash
python train_glide.py \
  --data_dir "/mnt/laion/*.tar" \
  --use_webdataset \
  --wds_image_key jpg \
  --wds_caption_key txt \
  --use_captions \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --precision bf16 \
  --activation_checkpointing
```

### Train Upsampler (64->256)
```bash
python train_glide.py \
  --data_dir /path/to/dataset \
  --train_upsample \
  --upscale_factor 4 \
  --uncond_p 0.0 \
  --precision bf16
```

### Evaluate / Generate
```bash
python evaluate_glide.py \
  --prompt_file eval_captions.txt \
  --base_model checkpoints/base.pt \
  --sr_model checkpoints/sr.pt \
  --use_clip_rerank \
  --clip_candidates 32 \
  --clip_top_k 8 \
  --sampler euler \
  --cfg 4.0
```

## Features

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
- **Gradient Accumulation**: Larger effective batch sizes (`--gradient_accumulation_steps`)
- **Mixed Precision**: BF16 recommended (`--precision bf16`)
- **Gradient Checkpointing**: Trade compute for memory (`--activation_checkpointing`)
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

## Full Usage

Run `python train_glide.py --help` for the complete argument list. Key arguments are documented above.
