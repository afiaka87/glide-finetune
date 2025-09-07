# GLIDE CLI Documentation

The GLIDE CLI provides a clean, modern interface for training and evaluating GLIDE text-to-image models.

## Installation

```bash
# Install with uv
uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

The CLI is organized into two main commands: `train` and `eval`, each with their own subcommands.

```bash
# Show main help
glide --help

# Show version
glide --version
```

## Training Commands

### Train Base Model (64x64)

Train the base text-to-image model with classifier-free guidance:

```bash
glide train base /path/to/dataset \
  --batch-size 4 \
  --lr 1e-4 \
  --epochs 100 \
  --wandb my-project \
  --fp16 \
  --grad-ckpt
```

Key options:
- `--width`, `--height`: Image dimensions (default: 64x64)
- `--uncond-p`: Unconditional probability for CFG (default: 0.2)
- `--webdataset`: Use WebDataset format for large-scale training
- `--lora`: Enable LoRA for efficient finetuning
- `--freeze-transformer`: Freeze transformer weights
- `--freeze-diffusion`: Freeze diffusion model weights

### Train Upsampler Model (64→256)

Train the super-resolution model:

```bash
glide train upsampler /path/to/dataset \
  --upscale 4 \
  --batch-size 2 \
  --lr 5e-5 \
  --wandb my-upsampler-project
```

Key options:
- `--upscale`: Upscaling factor (default: 4)
- Input dimensions set with `--width`, `--height`
- Output will be input × upscale factor

## Evaluation Commands

### Generate Images

Generate images from text prompts:

```bash
# From command line prompts
glide eval generate base.pt sr.pt \
  --prompt "a cat playing piano" \
  --prompt "a dog reading newspaper" \
  --cfg 4.0 \
  --sampler euler

# From prompt file
glide eval generate base.pt sr.pt \
  --prompt-file prompts.txt \
  --output my-outputs \
  --format png
```

### With CLIP Re-ranking

Generate multiple candidates and select the best using CLIP:

```bash
glide eval generate base.pt sr.pt \
  --prompt-file prompts.txt \
  --clip-rerank \
  --clip-model ViT-L-14/laion2b_s32b_b82k \
  --clip-candidates 32 \
  --clip-top-k 8
```

### Batch Evaluation

Process multiple prompts with configurable samples per prompt:

```bash
glide eval batch base.pt sr.pt prompts.txt \
  --num-samples 4 \
  --grid \
  --wandb eval-project
```

### Compare Models

Compare outputs from multiple model checkpoints:

```bash
glide eval compare \
  model1_base.pt model1_sr.pt \
  model2_base.pt model2_sr.pt \
  "a beautiful landscape" \
  --seed 42
```

## Common Options

### Samplers
- `euler`: Euler ODE solver (default, fast)
- `euler_a`: Euler ancestral (stochastic)
- `dpm++`: DPM-Solver++ (fewer steps needed)
- `plms`: Pseudo Linear Multi-Step
- `ddim`: Denoising Diffusion Implicit Models

### Performance Options
- `--fp16`: Use FP16 mixed precision
- `--compile`: Use torch.compile for faster inference
- `--batch-size`: Process multiple items at once

### Output Options
- `--output`: Output directory
- `--format`: Output format (jpg/png)
- `--grid`: Save grid of all outputs
- `--seed`: Fix random seed for reproducibility

## Examples

### Quick Test
```bash
# Generate a single image
glide eval generate base.pt sr.pt \
  --prompt "a majestic mountain landscape" \
  --cfg 3.0 \
  --seed 42
```

### Production Training
```bash
# Train on LAION with all optimizations
glide train base /mnt/laion/*.tar \
  --webdataset \
  --batch-size 8 \
  --lr 1e-4 \
  --fp16 \
  --grad-ckpt \
  --lora \
  --lora-rank 8 \
  --wandb laion-finetune \
  --checkpoint-dir ./checkpoints \
  --save-interval 5000
```

### High-Quality Generation
```bash
# Generate with CLIP re-ranking for best quality
glide eval generate base.pt sr.pt \
  --prompt-file artistic_prompts.txt \
  --clip-rerank \
  --clip-candidates 64 \
  --clip-top-k 4 \
  --sampler dpm++ \
  --cfg 5.0 \
  --output ./best_outputs
```

## Shell Completion

Enable tab completion for your shell:

```bash
# Bash
glide --install-completion

# Show completion script (for manual installation)
glide --show-completion
```

## Tips

1. **Memory Management**: Use `--grad-ckpt` and `--fp16` to reduce memory usage during training
2. **WebDataset**: For large datasets, use tar files with `--webdataset` flag
3. **CLIP Re-ranking**: Generate more candidates (32-64) and keep fewer (4-8) for best quality
4. **Samplers**: Start with `euler` for speed, try `dpm++` for quality with fewer steps
5. **Batch Size**: Larger batch sizes improve training stability but require more memory

## Troubleshooting

- **OOM Errors**: Reduce batch size, enable gradient checkpointing, or use LoRA
- **Slow Training**: Enable FP16, use gradient accumulation, or freeze some layers
- **Poor Quality**: Increase CFG scale, use CLIP re-ranking, or try different samplers
- **WebDataset Issues**: Ensure tar files are properly formatted with correct keys

For more details on any command:
```bash
glide [command] [subcommand] --help
```