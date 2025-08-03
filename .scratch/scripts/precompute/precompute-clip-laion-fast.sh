#!/usr/bin/env bash
set -euo pipefail

# Fast script to precompute CLIP embeddings for LAION dataset
# Uses optimizations from clip-retrieval for 5-10x speedup

# Get script directory for relative path resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
LAION_DATA_DIR="/mnt/usb_nvme_2tb/Data/laion400m-dat-release/"
OUTPUT_DIR="/mnt/usb_nvme_2tb/Data/laion400m-dat-release/clip_cache"
CLIP_MODEL="ViT-B/32"  # Options: ViT-B/32, ViT-L/14, RN50
BATCH_SIZE=1024
NUM_WORKERS=8

echo "================================================"
echo "Fast CLIP Embedding Precomputation for LAION"
echo "================================================"
echo "Data directory: $LAION_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "================================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build tar list with proper expansion
TAR_LIST=$(printf "%s," "$LAION_DATA_DIR"/*.tar | sed 's/,$//')

if [ -z "$TAR_LIST" ] || [ "$TAR_LIST" = "$LAION_DATA_DIR/*.tar" ]; then
    echo "ERROR: No .tar files found in $LAION_DATA_DIR"
    exit 1
fi

echo "Found tar files: $(echo "$TAR_LIST" | tr ',' '\n' | wc -l)"

# Run the optimized precompute script
uv run python "$SCRIPT_DIR/precompute_clip_webdataset_embeddings_fast.py" \
    --tar_urls "$TAR_LIST" \
    --cache_dir "$OUTPUT_DIR" \
    --clip_model_name "$CLIP_MODEL" \
    --caption_key "txt" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS

echo "================================================"
echo "Precomputation complete!"
echo "CLIP embeddings saved to: $OUTPUT_DIR"
echo "================================================"