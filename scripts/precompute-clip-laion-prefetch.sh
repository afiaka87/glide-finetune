#!/bin/bash

# Precompute CLIP text embeddings for LAION dataset with prefetching for maximum GPU utilization
# This script uses producer-consumer pattern to prefetch tar files while GPU processes embeddings

# Set the base directory for LAION dataset
LAION_DATA_DIR="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"

# Output directory for CLIP embeddings cache
OUTPUT_DIR="./laion-synthetic-clip_cache"

# CLIP model to use
CLIP_MODEL="ViT-B/32"

# Processing parameters
BATCH_SIZE=8192
PREFETCH_SIZE=8  # Number of tar files to prefetch ahead

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "🚀 LAION CLIP Precomputation with Prefetching"
echo "=================================================="
echo "📁 Dataset: $LAION_DATA_DIR"
echo "💾 Output: $OUTPUT_DIR"  
echo "🤖 Model: $CLIP_MODEL"
echo "📦 Batch size: $BATCH_SIZE"
echo "🔄 Prefetch size: $PREFETCH_SIZE"
echo "=================================================="

# Make the script executable if it isn't already
chmod +x scripts/precompute_clip_webdataset_embeddings_prefetch.py

# Run the prefetch precompute script with optimizations
uv run python scripts/precompute_clip_webdataset_embeddings_prefetch.py \
    --tar_urls "$LAION_DATA_DIR/*.tar" \
    --cache_dir "$OUTPUT_DIR" \
    --clip_model_name "$CLIP_MODEL" \
    --caption_key "txt" \
    --batch_size $BATCH_SIZE \
    --prefetch_size $PREFETCH_SIZE \
    --normalize \
    --device cuda \
    --verbose

echo "✅ Precomputation complete!"