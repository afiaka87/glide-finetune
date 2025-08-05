#\!/bin/bash
# Full ensemble generation with ESRGAN upscaling and CLIP re-ranking

uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --prompt "a majestic golden retriever sitting in a sunlit meadow" \
    --num_samples 16 \
    --sampler euler \
    --steps 30 \
    --guidance_scale 3.0 \
    --use_esrgan \
    --clip_model ViT-L/14 ViT-B/32 ViT-B-32/laion2b_s34b_b79k ViT-L-14/laion2b_s32b_b82k \
    --save_all \
    --seed 42
EOF < /dev/null
