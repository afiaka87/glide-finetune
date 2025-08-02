#!/bin/bash
# Script to precompute CLIP embeddings for LAION dataset
# This speeds up training by 5-10x by avoiding on-the-fly CLIP encoding
# Note: For faster processing, use precompute-clip-laion-fast.sh instead

# Configuration
LAION_DATA_DIR="/mnt/usb_nvme_2tb/Data/laion400m-dat-release/"
OUTPUT_DIR="/mnt/usb_nvme_2tb/Data/laion400m-dat-release/clip_cache"
CLIP_MODEL="ViT-B/32"  # Options: ViT-B/32, ViT-L/14, RN50
BATCH_SIZE=32  # Smaller batch size for the original script
NUM_WORKERS=8

echo "================================================"
echo "Precomputing CLIP embeddings for LAION dataset"
echo "================================================"
echo "Data directory: $LAION_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "================================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the precompute script
uv run python scripts/precompute_clip_webdataset_embeddings.py \
    --tar_urls "$LAION_DATA_DIR/*.tar" \
    --cache_dir "$OUTPUT_DIR" \
    --clip_model_name "$CLIP_MODEL" \
    --caption_key "txt" \
    --batch_size $BATCH_SIZE

echo "================================================"
echo "Precomputation complete!"
echo "CLIP embeddings saved to: $OUTPUT_DIR"
echo "================================================"