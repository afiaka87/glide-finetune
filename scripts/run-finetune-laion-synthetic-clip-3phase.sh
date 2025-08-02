#!/bin/bash
# Advanced three-phase training script for GLIDE with CLIP adapters on LAION Synthetic
# Uses existing CLIP cache at ./laion-synthetic-clip_cache
# Phase 1: Adapter only -> Phase 2: Adapter + Gates -> Phase 3: Full model
#
# Usage:
#   ./run-finetune-laion-synthetic-clip-3phase.sh [PHASE] [RESUME_CHECKPOINT] [TEST_MODE]
#
# Arguments:
#   PHASE: 1, 2, or 3 (default: 1)
#   RESUME_CHECKPOINT: Path to checkpoint to resume from (optional)
#   TEST_MODE: Number of steps to run in test mode (0=disabled, default: 0)
#              When > 0, wandb is disabled and training stops after N steps
#
# Examples:
#   ./run-finetune-laion-synthetic-clip-3phase.sh 1              # Run phase 1
#   ./run-finetune-laion-synthetic-clip-3phase.sh 2              # Run phase 2 (auto-resumes from phase 1)
#   ./run-finetune-laion-synthetic-clip-3phase.sh 1 "" 100       # Test mode: phase 1, 100 steps only

# Paths
LAION_DATA_DIR="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
CLIP_CACHE_DIR="./clip_cache"  # Use the local cache on faster drive
CHECKPOINT_BASE="/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune-clip-synthetic"
EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/people_captions_16.txt"

# CLIP Configuration (must match the cached model)
CLIP_MODEL="ViT-B/32"

# Parse command line arguments
PHASE=${1:-1}  # Default to phase 1
RESUME_FROM=${2:-""}  # Optional resume checkpoint
TEST_MODE=${3:-0}  # Optional test mode (0=off, N=stop after N steps)

echo "================================================"
echo "GLIDE + CLIP Three-Phase Training on LAION Synthetic"
echo "Using CLIP cache: $CLIP_CACHE_DIR"
echo "Running Phase: $PHASE"
if [ $TEST_MODE -gt 0 ]; then
    echo "TEST MODE ENABLED: Will stop after $TEST_MODE steps"
fi
echo "================================================"

# Verify CLIP cache exists
if [ ! -d "$CLIP_CACHE_DIR/ViT-B-32" ]; then
    echo "ERROR: CLIP cache not found at $CLIP_CACHE_DIR/ViT-B-32"
    echo "Please run precompute script first or check the path"
    exit 1
fi

# Set phase-specific parameters
case $PHASE in
    1)
        echo "Phase 1: Training CLIP adapter only"
        ADAPTER_PHASE="adapter_only"
        ADAPTER_LR=1e-5
        MAIN_LR=0  # Not used in phase 1
        if [ $TEST_MODE -gt 0 ]; then
            BATCH_SIZE=2  # Even smaller batch for test mode
        else
            BATCH_SIZE=4  # Reduced from 12 to avoid OOM
        fi
        EPOCHS=10  # Roughly 10k iterations
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase1"
        WARMUP_STEPS=5000
        ;;
    2)
        echo "Phase 2: Training adapter + attention gates"
        ADAPTER_PHASE="adapter_gates"
        ADAPTER_LR=5e-6
        MAIN_LR=0  # Not used in phase 2
        BATCH_SIZE=10
        EPOCHS=5  # Roughly 5k iterations
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase2"
        WARMUP_STEPS=1000
        # Default to resuming from phase 1 if no checkpoint specified
        if [ -z "$RESUME_FROM" ]; then
            RESUME_FROM="$CHECKPOINT_BASE/phase1/latest.pt"
        fi
        ;;
    3)
        echo "Phase 3: Full model fine-tuning"
        ADAPTER_PHASE="full"
        ADAPTER_LR=1e-6
        MAIN_LR=1e-7
        BATCH_SIZE=6
        EPOCHS=10  # Roughly 10k iterations
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase3"
        WARMUP_STEPS=2000
        # Default to resuming from phase 2 if no checkpoint specified
        if [ -z "$RESUME_FROM" ]; then
            RESUME_FROM="$CHECKPOINT_BASE/phase2/latest.pt"
        fi
        ;;
    *)
        echo "Invalid phase: $PHASE. Please use 1, 2, or 3."
        exit 1
        ;;
esac

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Build the command
CMD="uv run python train_glide.py \
    --data_dir '$LAION_DATA_DIR' \
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
    --project_name 'glide-clip-laion-synthetic-phase$PHASE' \
    --log_frequency 100 \
    --sample_interval 1000 \
    --use_clip \
    --clip_model_name '$CLIP_MODEL' \
    --use_clip_cache \
    --clip_cache_dir '$CLIP_CACHE_DIR' \
    --adapter_training_phase '$ADAPTER_PHASE' \
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
if [ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM" ]; then
    echo "Resuming from checkpoint: $RESUME_FROM"
    CMD="$CMD --resume_ckpt '$RESUME_FROM'"
fi

# Add test mode if enabled
if [ $TEST_MODE -gt 0 ]; then
    CMD="$CMD --test_run $TEST_MODE"
fi

# Print cache info
echo "================================================"
echo "CLIP Cache Information:"
echo "Path: $CLIP_CACHE_DIR"
echo "Model: $CLIP_MODEL"
if [ -f "$CLIP_CACHE_DIR/ViT-B-32/metadata.json" ]; then
    echo "Cache metadata found"
fi
echo "================================================"

echo "Starting training..."
echo "================================================"

# Execute the command
eval $CMD

echo "================================================"
if [ $TEST_MODE -gt 0 ]; then
    echo "TEST MODE: Phase $PHASE test complete!"
else
    echo "Phase $PHASE training complete!"
fi
if [ $PHASE -lt 3 ]; then
    echo "To continue with phase $((PHASE + 1)), run:"
    echo "./scripts/run-finetune-laion-synthetic-clip-3phase.sh $((PHASE + 1))"
fi
echo "================================================"