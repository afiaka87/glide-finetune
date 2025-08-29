#!/bin/bash
set -e

# ============================================
# Multi-GPU Frozen Transformer Training
# Train only the UNet/Diffusion while keeping transformer frozen
# ============================================

# Multi-GPU Configuration
NUM_GPUS=2                        # Number of GPUs to use
MIXED_PRECISION="no"              # Mixed precision: "no", "fp16", "bf16"

# Data Configuration
DATA_DIR="/path/to/your/dataset"  # UPDATE: Path to your image dataset
USE_WEBDATASET=false              # Set to true if using .tar files

# Model Configuration
FREEZE_TRANSFORMER=true            # Freeze the text encoder
FREEZE_DIFFUSION=false            # Keep UNet trainable

# Training Hyperparameters
BATCH_SIZE=4                      # Batch size PER GPU
LEARNING_RATE=5e-5                # Lower LR for finetuning
NUM_EPOCHS=10                     # Number of training epochs
GRADIENT_ACCUMULATION_STEPS=4     # Gradient accumulation steps

# Environment Variables for Multi-GPU Training
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="online"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Set CUDA devices based on NUM_GPUS
if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$NUM_GPUS" = "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$NUM_GPUS" = "4" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Save Directory
SAVE_DIRECTORY="./checkpoints/frozen_transformer_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIRECTORY"

echo "Starting Multi-GPU Frozen Transformer Training"
echo "============================================="
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Mode: Frozen Transformer (training UNet only)"
echo "  Data: $DATA_DIR"
echo "  Batch Size: $BATCH_SIZE per GPU (total: $((BATCH_SIZE * NUM_GPUS)))"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Build the command
CMD="uv run accelerate launch"
CMD="$CMD --num_processes $NUM_GPUS"

if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="$CMD --multi_gpu"
fi

if [ "$MIXED_PRECISION" != "no" ]; then
    CMD="$CMD --mixed_precision $MIXED_PRECISION"
fi

CMD="$CMD train.py"
CMD="$CMD --data_dir \"$DATA_DIR\""
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --save_directory \"$SAVE_DIRECTORY\""
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD="$CMD --freeze_transformer"  # Key flag for frozen transformer mode
CMD="$CMD --use_captions"
CMD="$CMD --seed 42"

if [ "$USE_WEBDATASET" = true ]; then
    CMD="$CMD --use_webdataset"
fi

echo "Command: $CMD"
echo ""

# Execute training
eval $CMD

echo ""
echo "Training completed! Checkpoints saved to: $SAVE_DIRECTORY"