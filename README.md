# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Fine-tune and evaluate GLIDE text-to-image diffusion models.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12+.

```bash
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
uv sync
uv pip install -e .
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

## Performance Optimization Flags

The following flags enable GPU performance optimizations that are **disabled by default**. They were removed from the default training path because `torch.compile` was found to produce subtly wrong gradients through the mixed BF16/FP32 casting in pixel-space GLIDE models, causing training outputs to slowly degrade into abstract blobs over time. The other flags are safe individually but are opt-in for consistency.

| Flag | Description |
|---|---|
| `--use_compile` | Enable `torch.compile`. **Not recommended** for pixel-space BF16 models — causes gradient issues through mixed-precision ResBlocks. Safe for latent models which have explicit FP32 protections. |
| `--use_tf32` | Enable TF32 for matmul and cuDNN. Reduces FP32 precision from 23-bit to 10-bit mantissa. |
| `--use_channels_last` | Enable `channels_last` memory format for model weights and activations. |
| `--use_fused_adam` | Enable fused AdamW optimizer (single-kernel parameter update). |

## Testing

```bash
# Run all tests (requires GPU and pretrained weights)
uv run pytest tests/ -m slow -v

# Run only the fast tests (no GPU needed)
uv run pytest tests/ -v
```

### Test Suites

- **`tests/test_training_regression.py`** -- Regression tests that train for a small number of steps on synthetic data and verify loss decreases, outputs are not degenerate, and no NaN/Inf values appear.
- **`tests/test_training_sanity.py`** -- Ablation tests parameterized by feature flags (`torch.compile`, TF32, `channels_last`, fused AdamW). Used to isolate which optimizations cause training degradation.
- **`tests/test_latent_core.py`** -- Unit tests for the latent diffusion model architecture.

## Full Usage

Run `python train_glide.py --help` for the complete argument list. Key arguments are documented above.
