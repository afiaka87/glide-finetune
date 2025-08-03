#!/usr/bin/env bash
set -euo pipefail

# Precompute CLIP embeddings for captioned-birds-wds dataset
# This speeds up training by 5-10x by avoiding on-the-fly CLIP encoding

# Get script directory for relative path resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
DATA_DIR="/mnt/usb_nvme_2tb/Data/CC12M/mehdi_split_tars"
OUTPUT_DIR="/mnt/usb_nvme_2tb/Data/CC12M/mehdi_split_tars/clip_cache"
CLIP_MODEL="ViT-L/14"  # Match the model used in training scripts

# Choose which version to run (comment/uncomment as needed)

# Option 1: Standard version (reliable, moderate speed)
echo "================================================"
echo "Precomputing CLIP embeddings for Birds dataset"
echo "================================================"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "================================================"

mkdir -p "$OUTPUT_DIR"

#uv run python "$SCRIPT_DIR/precompute_clip_webdataset_embeddings.py" \
#    --tar_urls "$DATA_DIR/*.tar" \
#    --cache_dir "$OUTPUT_DIR" \
#    --clip_model_name "$CLIP_MODEL" \
#    --caption_key "txt" \
#    --batch_size 32 \
#    --device cuda

# Option 2: Fast version (5-10x faster, uses optimizations)
# Uncomment below to use instead of standard version
# uv run python "$SCRIPT_DIR/precompute_clip_webdataset_embeddings_fast.py" \
#     --tar_urls "$DATA_DIR/*.tar" \
#     --cache_dir "$OUTPUT_DIR" \
#     --clip_model_name "$CLIP_MODEL" \
#     --caption_key "txt" \
#     --batch_size 1024 \
#     --num_workers 8 \
#     --device cuda

# Option 3: Ultra-fast version (maximum speed, requires PyTorch 2.0+)
# Uncomment below to use instead
uv run python "$SCRIPT_DIR/precompute_clip_webdataset_embeddings_ultra_fast.py" \
    --tar_urls "$DATA_DIR/*.tar" \
    --cache_dir "$OUTPUT_DIR" \
    --clip_model_name "$CLIP_MODEL" \
    --caption_key "txt" \
    --batch_size 2048 \
    --num_workers 12 \
    --max_concurrent_tars 1 \
    --device cuda

echo "================================================"
echo "Precomputation complete!"
echo "CLIP embeddings saved to: $OUTPUT_DIR"
echo "================================================"
