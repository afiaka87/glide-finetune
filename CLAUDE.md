# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup

```bash
# Install dependencies with uv (Python 3.12+)
uv sync
uv pip install -e .

# Install glide-text2im dependency (crowsonkb fork required)
git clone https://github.com/crowsonkb/glide-text2im
cd glide-text2im && pip install -e . && cd ..
uv pip install -e glide-text2im
```

### Linting and Type Checking

```bash
# Run ruff linting
uv run ruff check .
uv run ruff format .

# Run mypy type checking
uv run mypy .
```

### Training Commands

```bash
# Train base model (64x64) on LAION dataset
bash run_train_glide.sh

# Alternative: Direct training command
uv run python train_glide.py \
  --data_dir /mnt/usb_nvme_2tb/Data/laion-2b-en-aesthetic-subset \
  --use_webdataset \
  --wds_caption_key txt \
  --wds_image_key jpg \
  --wds_dataset_name laion \
  --use_captions \
  --batch_size 4 \
  --learning_rate 3e-4 \
  --adam_weight_decay 0.01 \
  --precision bf16 \
  --uncond_p 0.2 \
  --gradient_accumulation_steps 1 \
  --wandb_project_name 'glide-laion-finetune'

# Upsampler training (64→256)
uv run python train_glide.py \
  --data_dir /path/to/dataset \
  --train_upsample True \
  --upscale_factor 4 \
  --uncond_p 0.0 \
  --precision bf16

```

### Evaluation and Generation

```bash
# Batch evaluation with CLIP re-ranking
uv run python evaluate_glide.py \
  --prompt_file eval_captions_persons_aesthetic.txt \
  --base_model checkpoints/base.pt \
  --sr_model checkpoints/sr.pt \
  --use_clip_rerank \
  --clip_candidates 32 \
  --clip_top_k 8 \
  --sampler euler \
  --cfg 4.0

# Interactive Gradio interface
uv run python gradio_app.py
# Access at http://localhost:7860
```

## Architecture

### Training Pipeline

The repository implements GLIDE (Guided Language to Image Diffusion) fine-tuning with two distinct training modes:

1. **Base Model** (`train_upsample=False`): 64x64 text-to-image generation with classifier-free guidance
   - Uses `uncond_p=0.2` to randomly replace captions with empty tokens for CFG training
   - Generates low-resolution images from text prompts
   - Default hyperparameters: AdamW optimizer with lr=3e-4, weight_decay=0.01, EMA=0.9999

2. **Upsampler Model** (`train_upsample=True`): 64→256 super-resolution with prompt awareness
   - Uses `uncond_p=0.0` - always conditioned on text
   - Upscales base model outputs to high resolution
   - Typically uses same optimizer settings as base model

### Key Implementation Details

#### WebDataset Loading (for LAION-scale training)
- Default dataset location: `/mnt/usb_nvme_2tb/Data/laion-2b-en-aesthetic-subset`
- Alternative location: `~/laion2b_en_aesthetic_wds/`
- Each tar contains ~10,000 image-text pairs
- WebDataset with `resampled=True` creates infinite iterators requiring manual epoch tracking
- Dataloader wraps WebDataset iterator: `DataLoader(wds.WebDataset(...).batched(batch_size))`
- Steps per epoch: `(num_tars * samples_per_tar) // batch_size`
- Automatic tar file validation with caching in `./cache/valid_tars_*.json`

#### Gradient Accumulation and Training Loop
- Loss scaled by `1/accumulation_steps` during backward pass
- Optimizer step only after accumulating gradients: `if (step + 1) % grad_acc_steps == 0`
- Global step increments only on weight updates, not every batch
- WandB logs metrics after optimizer.step() with averaged loss
- LR scheduler steps after weight updates only

#### Memory Optimization
- **BF16 Training** (`--precision bf16`): Recommended for stability over FP16
- **Gradient Checkpointing** (`--activation_checkpointing`): Trade compute for memory
- **CLIP Re-ranking**: Offloads GLIDE to CPU during CLIP scoring

### Module Structure

```
glide_finetune/
├── glide_finetune.py      # Core training loop with wandb integration
├── glide_util.py          # Model loading, tokenization, sampling
├── loader.py              # Standard image-caption dataset loader
├── wds_loader.py          # WebDataset loader for LAION/Alamy
├── train_util.py          # Training utilities, wandb setup
├── fp16_util.py           # Mixed precision (FP16/BF16) utilities
├── enhanced_samplers.py   # Euler, Euler-A, DPM++ implementations
├── clip_rerank.py         # CLIP-based quality selection
└── cli_utils.py           # CLI utilities for evaluation and generation
```

### Sampling Methods

Available samplers with performance characteristics:
- **PLMS** (default): Baseline quality/speed balance
- **Euler**: 15-20% faster than PLMS, deterministic
- **Euler Ancestral** (`euler_a`): Stochastic variant with eta parameter
- **DPM++**: Advanced solver, good quality with 20-30 steps
- **DDIM**: Flexible deterministic/stochastic via eta

Usage in code:
```python
from glide_finetune.glide_util import sample
samples = sample(model, options, 64, 64,
                 prompt="...", sampler="euler",
                 sampler_eta=0.0, dpm_order=2)
```

## Critical Configuration Notes

### WebDataset Training
- Token mismatch: LAION uses 128 tokens, GLIDE expects 77 - handled in tokenizer
- Empty batches: Ensure glob patterns are expanded and tar files accessible
- Epoch boundaries: Manually track with `samples_seen // dataset_size`

### Training Parameters
- `uncond_p`: 0.2 for base model, 0.0 for upsampler (critical for proper CFG)
- `gradient_accumulation_steps`: Effective batch = batch_size * grad_acc_steps
- `save_checkpoint_interval`: Default 2500 steps (based on global_step/weight updates)
- `sample_interval`: Generate sample images every N steps (default 250)
- `learning_rate`: Default 3e-4 (GLIDE paper uses 1e-4)
- `adam_weight_decay`: Default 0.01 (GLIDE paper uses 0.0)

### CLIP Re-ranking
- Model format: Use `ViT-L-14` (hyphen) not `ViT-L/14` (slash) for OpenCLIP
- Memory management: Automatic GPU offloading during ranking phase
- Output structure: Images saved as `image_rank_001.jpg` with metadata.json

## Common Issues and Solutions

### WebDataset Issues
```bash
# Problem: Empty epoch_metrics, no batches yielded
# Solution: Check tar file accessibility and glob expansion
ls -la /mnt/usb_nvme_2tb/Data/laion-2b-en-aesthetic-subset/*.tar | head -5

# Problem: Steps per epoch incorrect
# Solution: Calculate as (num_tars * 10000) // batch_size for LAION
```

### Training Issues
```bash
# Problem: OOM errors
# Solution: Use gradient accumulation or reduce batch size
--gradient_accumulation_steps 4 --batch_size 1

# Problem: FP16 training unstable
# Solution: Use BF16 instead
--precision bf16

# Problem: Checkpoints not saving
# Solution: Ensure save_checkpoint_interval aligns with global_step increments
```

### CLIP Re-ranking Issues
```bash
# Problem: "Model ViT-L/14 not found"
# Solution: Use correct OpenCLIP format
--clip_model "ViT-L-14"  # Correct
--clip_model "ViT-L/14"  # Wrong
```

## Key Training Scripts

### Main Scripts
- `train_glide.py`: Primary training script with WebDataset support and tar validation
- `evaluate_glide.py`: Batch generation with CLIP re-ranking and Rich UI
- `gradio_app.py`: Interactive web interface for generation
- `run_train_glide.sh`: Configured training script for LAION dataset

### Utility Scripts
- `test_bf16.py`: Test BF16 mixed precision training
- `example_samplers.py`: Demonstrate different sampling methods
- `quick_test_samplers.py`: Quick sampler testing

### Evaluation Prompts
- `eval_captions_persons_aesthetic.txt`: Human-focused evaluation prompts for testing model quality