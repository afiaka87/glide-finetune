# Script Consolidation Migration Guide

This guide helps you migrate from the old individual scripts to the new consolidated scripts.

## What Changed

We've consolidated 13 scripts into 3 unified scripts:
- 6 precompute scripts → `precompute-clip-embeddings.sh`
- 7 finetune scripts → `finetune-glide.sh` and `finetune-glide-clip.sh`

Old scripts have been moved to `.scratch/scripts/` for reference.

## Migration Examples

### Precompute Scripts

| Old Script | New Command |
|------------|-------------|
| `precompute-clip-birds.sh` | `./scripts/precompute-clip-embeddings.sh --dataset birds` |
| `precompute-clip-cc12m.sh` | `./scripts/precompute-clip-embeddings.sh --dataset cc12m` |
| `precompute-clip-laion.sh` | `./scripts/precompute-clip-embeddings.sh --dataset laion --speed standard` |
| `precompute-clip-laion-fast.sh` | `./scripts/precompute-clip-embeddings.sh --dataset laion --speed fast` |
| `precompute-clip-laion-prefetch.sh` | `./scripts/precompute-clip-embeddings.sh --dataset laion --speed prefetch` |
| `precompute-clip-laion-ultra-fast.sh` | `./scripts/precompute-clip-embeddings.sh --dataset laion --speed ultra-fast` |

### Finetune Scripts

| Old Script | New Command |
|------------|-------------|
| `run-finetune.sh` | `./scripts/finetune-glide.sh --data-dir /path/to/data` |
| `run-finetune-laion.sh` | `./scripts/finetune-glide.sh --dataset laion --config-preset laion` |
| `run-finetune-cc12m.sh` | `./scripts/finetune-glide.sh --dataset cc12m --config-preset cc12m` |
| `run-finetune-laion-clip.sh` | `./scripts/finetune-glide-clip.sh --dataset laion --phase 1` |
| `run-finetune-birds-3phase.sh 1` | `./scripts/finetune-glide-clip.sh --dataset birds --phase 1` |
| `run-finetune-laion-clip-3phase.sh 2` | `./scripts/finetune-glide-clip.sh --dataset laion --phase 2` |
| `run-finetune-laion-synthetic-clip-3phase.sh 3` | `./scripts/finetune-glide-clip.sh --dataset laion-synthetic --phase 3` |

## Key Improvements

1. **Unified Interface**: All scripts now use consistent command-line arguments
2. **Dataset Presets**: Common datasets have built-in paths and configurations
3. **Auto-resume**: CLIP training phases automatically resume from previous phase
4. **Better Defaults**: Speed modes and batch sizes are auto-configured
5. **Help System**: Use `--help` on any script to see all options

## Custom Datasets

For custom datasets, specify paths explicitly:

```bash
# Precompute
./scripts/precompute-clip-embeddings.sh \
    --data-dir /custom/path/to/webdataset \
    --output-dir /custom/path/to/clip_cache \
    --clip-model ViT-L/14

# Regular training
./scripts/finetune-glide.sh \
    --data-dir /custom/path/to/dataset \
    --checkpoint-dir ./my-checkpoints

# CLIP training
./scripts/finetune-glide-clip.sh \
    --data-dir /custom/path/to/webdataset \
    --clip-cache-dir /custom/path/to/clip_cache \
    --phase 1
```

## Backward Compatibility

The old scripts are preserved in `.scratch/scripts/` if you need to:
- Reference old configurations
- Compare behavior
- Temporarily use old scripts

## Getting Help

Each script has comprehensive help:
```bash
./scripts/precompute-clip-embeddings.sh --help
./scripts/finetune-glide.sh --help
./scripts/finetune-glide-clip.sh --help
```