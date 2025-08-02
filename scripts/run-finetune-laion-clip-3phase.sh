#!/bin/bash
# Advanced three-phase training script for GLIDE with CLIP adapters on LAION
# Phase 1: Adapter only -> Phase 2: Adapter + Gates -> Phase 3: Full model

# Paths
LAION_DATA_DIR="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
CLIP_CACHE_DIR="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/clip_cache"
CHECKPOINT_BASE="/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune-clip"
EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/people_captions_16.txt"

# CLIP Configuration
CLIP_MODEL="ViT-B/32"

# Parse command line arguments
PHASE=${1:-1}  # Default to phase 1
RESUME_FROM=${2:-""}  # Optional resume checkpoint

echo "================================================"
echo "GLIDE + CLIP Three-Phase Training on LAION"
echo "Running Phase: $PHASE"
echo "================================================"

# Set phase-specific parameters
case $PHASE in
    1)
        echo "Phase 1: Training CLIP adapter only"
        ADAPTER_PHASE="adapter_only"
        ADAPTER_LR=1e-5
        MAIN_LR=0  # Not used in phase 1
        BATCH_SIZE=12
        NUM_ITERATIONS=10000
        CHECKPOINT_DIR="$CHECKPOINT_BASE/phase1"
        WARMUP_STEPS=5000
        ;;
    2)
        echo "Phase 2: Training adapter + attention gates"
        ADAPTER_PHASE="adapter_gates"
        ADAPTER_LR=5e-6
        MAIN_LR=0  # Not used in phase 2
        BATCH_SIZE=10
        NUM_ITERATIONS=5000
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
        NUM_ITERATIONS=10000
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
    --wds_dataset_name 'laion' \
    --wds_caption_key 'txt' \
    --wds_image_key 'jpg' \
    --laion_no_filter \
    --use_captions \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --batch_size $BATCH_SIZE \
    --num_iterations $NUM_ITERATIONS \
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
    --project_name 'glide-clip-laion-phase$PHASE' \
    --log_frequency 100 \
    --sample_interval 1000 \
    --checkpoint_interval 2500 \
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
    --baseline_eval_interval 1000 \
    --use_wandb"

# Add learning rate for phase 3
if [ $PHASE -eq 3 ]; then
    CMD="$CMD --learning_rate $MAIN_LR"
fi

# Add resume checkpoint if provided
if [ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM" ]; then
    echo "Resuming from checkpoint: $RESUME_FROM"
    CMD="$CMD --resume_ckpt '$RESUME_FROM'"
fi

echo "================================================"
echo "Starting training..."
echo "================================================"

# Execute the command
eval $CMD

echo "================================================"
echo "Phase $PHASE training complete!"
if [ $PHASE -lt 3 ]; then
    echo "To continue with phase $((PHASE + 1)), run:"
    echo "./scripts/run-finetune-laion-clip-3phase.sh $((PHASE + 1))"
fi
echo "================================================"