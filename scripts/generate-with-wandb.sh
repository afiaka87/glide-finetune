#!/bin/bash
# Generate images with GLIDE using CLIP re-ranking and wandb logging

# Usage: ./scripts/generate-with-wandb.sh [prompt]

# Set default prompt if none provided
if [ -z "$1" ]; then
    PROMPT_ARG="--prompt_file examples/trippy_prompts_32.txt"
else
    PROMPT_ARG="--prompt \"$1\""
fi

# Run inference with wandb logging
uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --num_samples 16 \
    --batch_size 8 \
    --sampler euler \
    --steps 30 \
    --guidance_scale 3.0 \
    --clip_model ViT-L/14 ViT-B/32 RN50x4 ViT-B-32/laion2b_s34b_b79k \
    --use_esrgan \
    --compile \
    --compile_clip \
    --amp \
    --save_all \
    --wandb \
    --wandb_project glide-inference \
    $PROMPT_ARG