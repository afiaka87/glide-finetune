#!/usr/bin/env bash
set -euo pipefail

# Ultra-fast script to precompute CLIP embeddings for LAION dataset
# Uses advanced optimizations: torch.compile, BF16, larger batches, concurrent processing

# Get script directory for relative path resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
LAION_DATA_DIR="/home/sam/Data/captioned-birds-wds"
OUTPUT_DIR="/home/sam/Data/captioned-birds-wds/clip_cache"
CLIP_MODEL="ViT-B/32"  # Options: ViT-B/32, ViT-L/14, RN50
BATCH_SIZE=8192  # Much larger batch size for better GPU utilization
NUM_WORKERS=24   # More workers for data loading
MAX_CONCURRENT_TARS=4  # Process 2 tar files concurrently

echo "================================================"
echo "Ultra-Fast CLIP Embedding Precomputation"
echo "================================================"
echo "Data directory: $LAION_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Concurrent tars: $MAX_CONCURRENT_TARS"
echo "================================================"

# Check if PyTorch 2.0+ is available for torch.compile
echo "Checking PyTorch version..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set environment variables for maximum performance
export OMP_NUM_THREADS=1  # Prevent CPU oversubscription
export CUDA_LAUNCH_BLOCKING=0  # Enable async CUDA operations

# Build tar list with proper expansion
TAR_LIST=$(printf "%s," "$LAION_DATA_DIR"/*.tar | sed 's/,$//')

if [ -z "$TAR_LIST" ] || [ "$TAR_LIST" = "$LAION_DATA_DIR/*.tar" ]; then
    echo "ERROR: No .tar files found in $LAION_DATA_DIR"
    exit 1
fi

echo "Found tar files: $(echo "$TAR_LIST" | tr ',' '\n' | wc -l)"

# Run the ultra-fast precompute script
uv run python "$SCRIPT_DIR/precompute_clip_webdataset_embeddings_ultra_fast.py" \
    --tar_urls "$TAR_LIST" \
    --cache_dir "$OUTPUT_DIR" \
    --clip_model_name "$CLIP_MODEL" \
    --caption_key "txt" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_concurrent_tars $MAX_CONCURRENT_TARS \
    --device "cuda"

echo "================================================"
echo "Precomputation complete!"
echo "CLIP embeddings saved to: $OUTPUT_DIR"
echo "================================================"