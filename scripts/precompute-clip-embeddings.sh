#!/usr/bin/env bash
set -euo pipefail

# Unified CLIP Embeddings Precomputation Script
# Uses ultra-fast implementation for maximum performance
#
# Usage:
#   ./scripts/precompute-clip-embeddings.sh [OPTIONS]
#
# Options:
#   --dataset DATASET    Dataset name (laion, cc12m, birds, custom) [default: custom]
#   --data-dir PATH      Path to WebDataset tar files [required for custom]
#   --output-dir PATH    Output directory for CLIP cache [default: ./clip_cache]
#   --clip-model MODEL   CLIP model name [default: ViT-B/32]
#   --batch-size SIZE    Batch size for processing [default: 2048]
#   --num-workers N      Number of workers [default: 12]
#   --help               Show this help message

# Get script directory for relative path resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
DATASET="laion"
DATA_DIR="/mnt/usb_nvme_2tb/Data/laion400m-dat-release/"
OUTPUT_DIR="/mnt/usb_nvme_2tb/Data/laion400m-dat-release/clip_cache"
CLIP_MODEL="ViT-B/32"
BATCH_SIZE="2048"
NUM_WORKERS="12"
CAPTION_KEY="txt"
IMAGE_KEY="jpg"
MAX_CONCURRENT_TARS="1"

# Dataset presets
declare -A DATASET_PATHS=(
    ["laion"]="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
    ["cc12m"]="/mnt/usb_nvme_2tb/Data/CC12M/mehdi_split_tars"
    ["birds"]="/home/sam/Data/captioned-birds-wds"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --clip-model)
            CLIP_MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | grep -E "Usage:|Options:" -A 20 | grep -v "^$" | cut -c3-
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set data directory based on dataset if not explicitly provided
if [ -z "$DATA_DIR" ]; then
    if [ "$DATASET" != "custom" ] && [ -n "${DATASET_PATHS[$DATASET]}" ]; then
        DATA_DIR="${DATASET_PATHS[$DATASET]}"
    else
        echo "ERROR: --data-dir is required for custom datasets"
        exit 1
    fi
fi

# Always use ultra-fast implementation
PYTHON_SCRIPT="precompute_clip_webdataset_embeddings_ultra_fast.py"

# Set output directory to include dataset name if using default
if [ "$OUTPUT_DIR" = "./clip_cache" ] && [ "$DATASET" != "custom" ]; then
    OUTPUT_DIR="${DATA_DIR}/clip_cache"
fi

# Display configuration
echo "================================================"
echo "CLIP Embeddings Precomputation (Ultra-Fast)"
echo "================================================"
echo "Dataset: $DATASET"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "================================================"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build tar list
if [[ "$DATA_DIR" == *".tar" ]]; then
    # If DATA_DIR already contains .tar pattern, use it directly
    TAR_LIST="$DATA_DIR"
    echo "Found tar files: 1"
else
    # Create a temporary file to store tar file list
    TAR_LIST_FILE=$(mktemp /tmp/tar_list.XXXXXX)
    
    # Find all tar files and write to temporary file
    find "$DATA_DIR" -name "*.tar" -type f > "$TAR_LIST_FILE"
    
    NUM_TARS=$(wc -l < "$TAR_LIST_FILE")
    if [ "$NUM_TARS" -eq 0 ]; then
        echo "ERROR: No .tar files found in $DATA_DIR"
        rm -f "$TAR_LIST_FILE"
        exit 1
    fi
    
    echo "Found tar files: $NUM_TARS"
    
    # Convert file list to comma-separated string for the script
    TAR_LIST=$(tr '\n' ',' < "$TAR_LIST_FILE" | sed 's/,$//')
fi

# Build the command - use tar list file if available
if [ -n "${TAR_LIST_FILE:-}" ] && [ -f "$TAR_LIST_FILE" ]; then
    CMD="uv run python \"$SCRIPT_DIR/$PYTHON_SCRIPT\" \
        --tar_urls_file \"$TAR_LIST_FILE\" \
        --cache_dir \"$OUTPUT_DIR\" \
        --clip_model_name \"$CLIP_MODEL\" \
        --caption_key \"$CAPTION_KEY\" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --max_concurrent_tars $MAX_CONCURRENT_TARS \
        --device cuda"
else
    CMD="uv run python \"$SCRIPT_DIR/$PYTHON_SCRIPT\" \
        --tar_urls \"$TAR_LIST\" \
        --cache_dir \"$OUTPUT_DIR\" \
        --clip_model_name \"$CLIP_MODEL\" \
        --caption_key \"$CAPTION_KEY\" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --max_concurrent_tars $MAX_CONCURRENT_TARS \
        --device cuda"
fi

# Execute the command
echo "Starting precomputation..."
echo "================================================"
eval "$CMD"

# Clean up temporary file if it exists
if [ -n "${TAR_LIST_FILE:-}" ] && [ -f "$TAR_LIST_FILE" ]; then
    rm -f "$TAR_LIST_FILE"
fi

echo "================================================"
echo "Precomputation complete!"
echo "CLIP embeddings saved to: $OUTPUT_DIR"
echo "================================================"