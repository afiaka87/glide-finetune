#!/usr/bin/env bash
set -euo pipefail

# Unified GLIDE Fine-tuning Script with CLIP Adapters
# Supports three-phase training and all CLIP-related configurations
#
# Usage:
#   ./scripts/finetune-glide-clip.sh [OPTIONS]
#
# Options:
#   --dataset DATASET        Dataset name (laion, laion-synthetic, birds, custom) [default: custom]
#   --phase PHASE            Training phase (1, 2, or 3) [default: 1]
#   --data-dir PATH          Path to WebDataset tar files [required for custom]
#   --clip-cache-dir PATH    Path to CLIP embeddings cache [default: auto-detect]
#   --checkpoint-dir PATH    Base directory for checkpoints [default: ./checkpoints-clip]
#   --clip-model MODEL       CLIP model name [default: ViT-B/32]
#   --batch-size SIZE        Batch size [default: auto based on phase]
#   --epochs N               Number of epochs [default: auto based on phase]
#   --resume PATH            Resume from specific checkpoint
#   --eval-prompts FILE      File containing evaluation prompts
#   --project-name NAME      W&B project name [default: glide-clip-finetune]
#   --test-mode N            Run in test mode for N steps (disables W&B)
#   --help                   Show this help message

# Get script directory for relative path resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
DATASET="custom"
PHASE=1
DATA_DIR=""
CLIP_CACHE_DIR=""
CHECKPOINT_BASE="./checkpoints-clip"
CLIP_MODEL="ViT-B/32"
BATCH_SIZE=""
EPOCHS=""
RESUME_CHECKPOINT=""
EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/people_captions_16.txt"
PROJECT_NAME="glide-clip-finetune"
TEST_MODE=0

# Dataset presets
declare -A DATASET_PATHS=(
    ["laion"]="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
    ["laion-synthetic"]="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
    ["birds"]="/home/sam/Data/captioned-birds-wds"
)

declare -A DATASET_CLIP_CACHE=(
    ["laion"]="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds/clip_cache"
    ["laion-synthetic"]="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds/clip_cache"
    ["birds"]="/home/sam/Data/captioned-birds-wds/clip_cache"
)

declare -A DATASET_PROMPTS=(
    ["laion-synthetic"]="/home/sam/GitHub/glide-finetune/examples/trippy_prompts_16.txt"
    ["birds"]="/home/sam/GitHub/glide-finetune/examples/birds_prompts_16.txt"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --clip-cache-dir)
            CLIP_CACHE_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_BASE="$2"
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
        --epochs)
            EPOCHS="$2"
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
        --test-mode)
            TEST_MODE="$2"
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

# Validate phase
if [[ ! "$PHASE" =~ ^[123]$ ]]; then
    echo "ERROR: Invalid phase: $PHASE. Please use 1, 2, or 3."
    exit 1
fi

# Set data directory based on dataset if not explicitly provided
if [ -z "$DATA_DIR" ]; then
    if [ "$DATASET" != "custom" ] && [ -n "${DATASET_PATHS[$DATASET]+x}" ]; then
        DATA_DIR="${DATASET_PATHS[$DATASET]}"
    else
        echo "ERROR: --data-dir is required for custom datasets"
        exit 1
    fi
fi

# Set CLIP cache directory if not explicitly provided
if [ -z "$CLIP_CACHE_DIR" ]; then
    if [ "$DATASET" != "custom" ] && [ -n "${DATASET_CLIP_CACHE[$DATASET]+x}" ]; then
        CLIP_CACHE_DIR="${DATASET_CLIP_CACHE[$DATASET]}"
    else
        # Try to auto-detect
        CLIP_CACHE_DIR="${DATA_DIR}/clip_cache"
    fi
fi

# Set evaluation prompts based on dataset if available
if [ "$DATASET" != "custom" ] && [ -n "${DATASET_PROMPTS[$DATASET]+x}" ]; then
    EVAL_PROMPTS="${DATASET_PROMPTS[$DATASET]}"
fi

# Set phase-specific parameters
case $PHASE in
    1)
        PHASE_NAME="adapter_only"
        ADAPTER_LR="1e-5"
        MAIN_LR=0
        : ${BATCH_SIZE:=4}
        : ${EPOCHS:=10}
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase1"
        WARMUP_STEPS=5000
        ;;
    2)
        PHASE_NAME="adapter_gates"
        ADAPTER_LR="5e-6"
        MAIN_LR=0
        : ${BATCH_SIZE:=10}
        : ${EPOCHS:=5}
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase2"
        WARMUP_STEPS=1000
        # Auto-resume from phase 1 if no resume checkpoint specified
        if [ -z "$RESUME_CHECKPOINT" ] && [ -f "$CHECKPOINT_BASE/phase1/latest.pt" ]; then
            RESUME_CHECKPOINT="$CHECKPOINT_BASE/phase1/latest.pt"
        fi
        ;;
    3)
        PHASE_NAME="full"
        ADAPTER_LR="1e-6"
        MAIN_LR="1e-7"
        : ${BATCH_SIZE:=6}
        : ${EPOCHS:=10}
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase3"
        WARMUP_STEPS=2000
        # Auto-resume from phase 2 if no resume checkpoint specified
        if [ -z "$RESUME_CHECKPOINT" ] && [ -f "$CHECKPOINT_BASE/phase2/latest.pt" ]; then
            RESUME_CHECKPOINT="$CHECKPOINT_BASE/phase2/latest.pt"
        fi
        ;;
esac

# Display configuration
echo "================================================"
echo "GLIDE + CLIP Three-Phase Training"
echo "================================================"
echo "Dataset: $DATASET"
echo "Phase: $PHASE ($PHASE_NAME)"
echo "Data directory: $DATA_DIR"
echo "CLIP cache: $CLIP_CACHE_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Adapter LR: $ADAPTER_LR"
if [ $PHASE -eq 3 ]; then
    echo "Main LR: $MAIN_LR"
fi
if [ $TEST_MODE -gt 0 ]; then
    echo "TEST MODE: Will stop after $TEST_MODE steps"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
fi
echo "================================================"

# Verify CLIP cache exists
CLIP_MODEL_DIR=$(echo "$CLIP_MODEL" | tr '/' '-')
if [ ! -d "$CLIP_CACHE_DIR/$CLIP_MODEL_DIR" ]; then
    echo "ERROR: CLIP cache not found at $CLIP_CACHE_DIR/$CLIP_MODEL_DIR"
    echo "Please run precompute-clip-embeddings.sh first"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Build the command
CMD="uv run python train_glide.py \
    --data_dir '$DATA_DIR' \
    --use_webdataset \
    --wds_dataset_name 'webdataset' \
    --wds_caption_key 'txt' \
    --wds_image_key 'jpg' \
    --laion_no_filter \
    --use_captions \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --adam_weight_decay 0.01 \
    --use_8bit_adam \
    --warmup_steps $WARMUP_STEPS \
    --warmup_type linear \
    --device cuda \
    --activation_checkpointing \
    --cudnn_benchmark \
    --use_tf32 \
    --eval_prompts_file '$EVAL_PROMPTS' \
    --test_batch_size 1 \
    --test_guidance_scale 4.0 \
    --test_steps 20 \
    --sampler 'dpm++_2m_karras' \
    --checkpoints_dir '$CHECKPOINT_DIR' \
    --project_name '$PROJECT_NAME-phase$PHASE' \
    --log_frequency 1 \
    --sample_interval 100 \
    --use_clip \
    --clip_model_name '$CLIP_MODEL' \
    --use_clip_cache \
    --clip_cache_dir '$CLIP_CACHE_DIR' \
    --adapter_training_phase '$PHASE_NAME' \
    --adapter_lr $ADAPTER_LR \
    --adapter_warmup_steps $WARMUP_STEPS \
    --adapter_wd 1e-2 \
    --adapter_beta2 0.98 \
    --adapter_grad_clip 0.5 \
    --main_grad_clip 2.0 \
    --kl_loss_interval 100 \
    --kl_loss_weight 0.01 \
    --stability_threshold 10.0 \
    --dry_run_interval 2000 \
    --dry_run_samples 5 \
    --early_stop_threshold 0.1 \
    --early_stop_patience 2000 \
    --baseline_eval_interval 1000"

# Add learning rate for phase 3
if [ $PHASE -eq 3 ]; then
    CMD="$CMD --learning_rate $MAIN_LR"
fi

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ -f "$RESUME_CHECKPOINT" ]; then
        echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
        echo "NOTE: If resuming from a non-CLIP checkpoint, missing CLIP weights will be randomly initialized"
        CMD="$CMD --resume_ckpt '$RESUME_CHECKPOINT'"
    else
        echo "WARNING: Resume checkpoint not found: $RESUME_CHECKPOINT"
        echo "Starting fresh training..."
    fi
fi

# Add test mode if enabled
if [ $TEST_MODE -gt 0 ]; then
    CMD="$CMD --test_run $TEST_MODE"
fi

# Execute the command
echo "Starting training..."
echo "================================================"
eval "$CMD"

echo "================================================"
if [ $TEST_MODE -gt 0 ]; then
    echo "TEST MODE: Phase $PHASE test complete!"
else
    echo "Phase $PHASE training complete!"
fi
if [ $PHASE -lt 3 ]; then
    echo "To continue with phase $((PHASE + 1)), run:"
    echo "./scripts/finetune-glide-clip.sh --dataset $DATASET --phase $((PHASE + 1))"
fi
echo "================================================"