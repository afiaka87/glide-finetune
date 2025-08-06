#!/bin/bash
# Generate images from a prompt file with CLIP re-ranking

# Default to trippy prompts if no argument provided
PROMPT_FILE="${1:-examples/trippy_prompts_32.txt}"

echo "🎨 Generating images from prompt file: $PROMPT_FILE"
echo

# Check if file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    echo
    echo "Available prompt files:"
    ls -1 examples/*prompts*.txt
    exit 1
fi

# Count prompts
NUM_PROMPTS=$(grep -c . "$PROMPT_FILE")
echo "Found $NUM_PROMPTS prompts"
echo

# Run inference with all optimizations
uv run python inference_clip_rerank.py \
    --prompt_file "$PROMPT_FILE" \
    --num_samples 16 \
    --sampler euler \
    --steps 30 \
    --guidance_scale 3.0 \
    --clip_model ViT-L/14 ViT-B/32 \
    --compile \
    --compile_clip \
    --compile_mode reduce-overhead \
    --amp \
    --batch_size 8

echo
echo "✅ Generation complete!"
echo "Results saved to outputs/inference/