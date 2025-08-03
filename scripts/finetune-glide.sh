#!/usr/bin/env bash
set -euo pipefail

# Unified GLIDE Fine-tuning Script (Regular Training)
# Consolidates regular fine-tuning scripts into a single configurable interface
#
# Usage:
#   ./scripts/finetune-glide.sh [OPTIONS]
#
# Options:
#   --dataset DATASET        Dataset name (laion, cc12m, custom) [default: custom]
#   --data-dir PATH          Path to dataset [required for custom]
#   --checkpoint-dir PATH    Directory for saving checkpoints [default: ./checkpoints]
#   --config-preset PRESET   Configuration preset (default, laion, cc12m) [default: default]
#   --batch-size SIZE        Batch size for training [default: 8]
#   --epochs N               Number of epochs [default: 20]
#   --learning-rate LR       Learning rate [default: 1e-4]
#   --resume PATH            Resume from checkpoint
#   --eval-prompts FILE      File containing evaluation prompts
#   --project-name NAME      W&B project name [default: glide-finetune]
#   --help                   Show this help message

# Get script directory for relative path resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
DATASET="custom"
DATA_DIR=""
CHECKPOINT_DIR="./checkpoints"
CONFIG_PRESET="default"
BATCH_SIZE=8
EPOCHS=20
LEARNING_RATE="1e-4"
RESUME_CHECKPOINT=""
EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/people_captions_16.txt"
PROJECT_NAME="glide-finetune"
USE_WEBDATASET=0

# Dataset presets
declare -A DATASET_PATHS=(
    ["laion"]="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
    ["cc12m"]="/mnt/usb_nvme_2tb/Data/CC12M/mehdi_split_tars"
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
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --config-preset)
            CONFIG_PRESET="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --eval-prompts)
            EVAL_PROMPTS="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
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
        USE_WEBDATASET=1
    else
        echo "ERROR: --data-dir is required for custom datasets"
        exit 1
    fi
else
    # Check if data directory contains tar files to determine if it's webdataset
    if ls "$DATA_DIR"/*.tar >/dev/null 2>&1; then
        USE_WEBDATASET=1
    fi
fi

# Apply configuration presets
case $CONFIG_PRESET in
    default)
        # Default settings already set above
        ;;
    laion)
        BATCH_SIZE=8
        LEARNING_RATE="7e-5"
        WARMUP_STEPS=1000
        CHECKPOINT_DIR="/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune"
        PROJECT_NAME="glide-finetune-laion-nostalgia"
        EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/people_captions_16.txt"
        EXTRA_ARGS="--laion_no_filter"
        ;;
    cc12m)
        BATCH_SIZE=8
        LEARNING_RATE="1e-4"
        WARMUP_STEPS=4000
        CHECKPOINT_DIR="/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune"
        PROJECT_NAME="glide-finetune-cc12m"
        EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/robust_prompts_16.txt"
        EXTRA_ARGS="--use_esrgan"
        ;;
    *)
        echo "ERROR: Invalid config preset: $CONFIG_PRESET"
        echo "Valid presets: default, laion, cc12m"
        exit 1
        ;;
esac

# Display configuration
echo "================================================"
echo "GLIDE Fine-tuning Configuration"
echo "================================================"
echo "Dataset: $DATASET"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Config preset: $CONFIG_PRESET"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Project name: $PROJECT_NAME"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
fi
echo "================================================"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Build the base command
CMD="uv run python train_glide.py \
    --data_dir '$DATA_DIR' \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --adam_weight_decay 0.01 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --device cuda \
    --checkpoints_dir '$CHECKPOINT_DIR' \
    --activation_checkpointing \
    --use_captions \
    --project_name '$PROJECT_NAME' \
    --eval_prompts_file '$EVAL_PROMPTS' \
    --test_batch_size 1 \
    --test_guidance_scale 4.0 \
    --test_steps 20 \
    --sampler 'dpm++_2m_karras' \
    --log_frequency 100 \
    --sample_interval 1000 \
    --use_8bit_adam \
    --use_tf32 \
    --cudnn_benchmark"

# Add WebDataset specific options if needed
if [ $USE_WEBDATASET -eq 1 ]; then
    CMD="$CMD \
        --use_webdataset \
        --wds_dataset_name 'webdataset' \
        --wds_caption_key 'txt' \
        --wds_image_key 'jpg'"
fi

# Add warmup steps if defined
if [ -n "${WARMUP_STEPS:-}" ]; then
    CMD="$CMD --warmup_steps $WARMUP_STEPS --warmup_type linear"
fi

# Add extra arguments if defined
if [ -n "${EXTRA_ARGS:-}" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_ckpt '$RESUME_CHECKPOINT'"
fi

# Execute the command
echo "Starting training..."
echo "================================================"
eval "$CMD"

echo "================================================"
echo "Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "================================================"