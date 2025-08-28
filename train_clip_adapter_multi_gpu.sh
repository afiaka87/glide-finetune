#!/bin/bash
set -e

# ============================================
# Multi-GPU CLIP Adapter Training Configuration
# ============================================

# Multi-GPU Configuration
NUM_GPUS=4                        # Number of GPUs to use (set to "auto" to use all available)
ACCELERATE_CONFIG=""              # Optional: Path to accelerate config file (leave empty for auto)
MIXED_PRECISION="no"             # Mixed precision: "no", "fp16", "bf16"
GRADIENT_CHECKPOINTING_STEPS=1   # Gradient checkpointing every N layers (1 = all layers)

# Data Configuration
DATA_DIR="/mnt/t7_2tb/Data/laion400m-dat-release/0{0000..2931}.tar"  # UPDATE: Path to your image dataset
USE_CAPTIONS=true                  # Use text files alongside images
USE_WEBDATASET=true               # Set to true if using .tar files
WDS_DATASET_NAME="laion"         # For WebDataset: "laion", "synthetic", etc.
WDS_IMAGE_KEY="jpg"               # Image file extension in tar files
WDS_CAPTION_KEY="txt"             # Caption file extension in tar files

# Model Configuration
USE_CLIP_ADAPTER=true              # Enable CLIP adapter
CLIP_ADAPTER_ONLY=true            # Train only adapter (freeze base model)
CLIP_FEATURES_PATH=""             # Optional: Path to precomputed CLIP features
USE_FP16=false                    # Mixed precision training (use MIXED_PRECISION above for multi-GPU)
FP16_MODE="auto"                  # FP16 mode: "auto", "aggressive", "conservative"

# Training Hyperparameters
BATCH_SIZE=12                      # Batch size PER GPU (total = BATCH_SIZE * NUM_GPUS)
LEARNING_RATE=1e-4               # Learning rate for adapter
NUM_EPOCHS=100                    # Number of training epochs
GRADIENT_ACCUMULATION_STEPS=4    # Gradient accumulation steps
GRADIENT_CLIPPING=1.0             # Gradient clipping value
WEIGHT_DECAY=0.01                 # Weight decay for AdamW

# Dataset Configuration
IMAGE_SIZE=64                     # Image size (64 for base, 256 for upsampler)
UNCOND_P=0.2                      # Unconditional probability for classifier-free guidance
RESIZE_RATIO=0.75                 # Random crop ratio
TRIM_WHITE_PADDING=true           # Remove white padding from images
WHITE_THRESH=245                  # Threshold for white padding removal

# Sampling Configuration (only runs on main process)
SAMPLE_FREQUENCY=1000              # Sample every N steps
EVAL_BATCH_SIZE=8                 # Number of samples per conditioning mode
GUIDANCE_SCALE=3.0                # Classifier-free guidance scale
NUM_STEPS=20                      # DDIM sampling steps
SAMPLER="dpm++"                   # Sampler: "plms", "ddim", "euler", "dpm++"
EVAL_PROMPT_FILE="captions_8.txt" # Optional: File with prompts for evaluation

# Checkpointing Configuration
SAVE_DIRECTORY="./checkpoints/clip_adapter_multi_gpu_$(date +%Y%m%d_%H%M%S)"
SAVE_FREQUENCY=1000              # Save checkpoint every N steps
RESUME_CHECKPOINT=""             # Optional: Path to resume from
SKIP_SAMPLES=0                   # Number of samples to skip (for resumption)

# System Configuration
DEVICE="cuda"                    # Device: "cuda" or "cpu"
SEED=42                          # Random seed (0 for non-deterministic)
NUM_WORKERS=12                   # DataLoader workers PER GPU
WANDB_PROJECT="clip-adapter-multi-gpu"  # Optional: W&B project name
WANDB_RUN_NAME=""                # Optional: W&B run name
LOG_LEVEL="INFO"                 # Logging level: DEBUG, INFO, WARNING, ERROR

# Memory Optimization
ACTIVATION_CHECKPOINTING=true    # Enable gradient checkpointing

# LAION Filtering
DISABLE_LAION_FILTERS=true       # Set to true to disable all LAION quality/NSFW/similarity filters

# ============================================
# Auto-detect GPU count if set to "auto"
# ============================================

if [ "$NUM_GPUS" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPUs"
    else
        echo "ERROR: nvidia-smi not found, cannot auto-detect GPU count"
        exit 1
    fi
fi

# ============================================
# Build Command Arguments
# ============================================

echo "Starting Multi-GPU CLIP Adapter Training"
echo "========================================="
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Data Directory: $DATA_DIR"
echo "  Batch Size: $BATCH_SIZE per GPU (total: $((BATCH_SIZE * NUM_GPUS)))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Device: $DEVICE"
echo "  Save Directory: $SAVE_DIRECTORY"
echo "  Mixed Precision: $MIXED_PRECISION"
echo ""

# Create save directory
mkdir -p "$SAVE_DIRECTORY"

# Build the accelerate launch command
ACCELERATE_CMD="uv run accelerate launch"

# Add accelerate configuration
if [ -n "$ACCELERATE_CONFIG" ] && [ -f "$ACCELERATE_CONFIG" ]; then
    # Use provided accelerate config file (must be an actual accelerate config, not clip_adapter_config.yaml)
    ACCELERATE_CMD="$ACCELERATE_CMD --config_file \"$ACCELERATE_CONFIG\""
else
    # Use default multi-GPU configuration
    ACCELERATE_CMD="$ACCELERATE_CMD --num_processes $NUM_GPUS"
    
    # Only add --multi_gpu if we have more than 1 GPU
    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCELERATE_CMD="$ACCELERATE_CMD --multi_gpu"
    fi
    
    if [ "$MIXED_PRECISION" != "no" ]; then
        ACCELERATE_CMD="$ACCELERATE_CMD --mixed_precision $MIXED_PRECISION"
    fi
    
    if [ "$GRADIENT_CHECKPOINTING_STEPS" -gt 0 ]; then
        ACCELERATE_CMD="$ACCELERATE_CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
    fi
fi

# Build the training command
CMD="$ACCELERATE_CMD train.py"
CMD="$CMD --data_dir \"$DATA_DIR\""
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --save_directory \"$SAVE_DIRECTORY\""
CMD="$CMD --save_frequency $SAVE_FREQUENCY"
CMD="$CMD --sample_frequency $SAMPLE_FREQUENCY"
CMD="$CMD --eval_batch_size $EVAL_BATCH_SIZE"
CMD="$CMD --test_guidance_scale $GUIDANCE_SCALE"
CMD="$CMD --num_steps $NUM_STEPS"
CMD="$CMD --sampler $SAMPLER"
CMD="$CMD --device $DEVICE"
CMD="$CMD --seed $SEED"
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD="$CMD --grad_clip $GRADIENT_CLIPPING"
CMD="$CMD --adam_weight_decay $WEIGHT_DECAY"
CMD="$CMD --side_x $IMAGE_SIZE"
CMD="$CMD --side_y $IMAGE_SIZE"
CMD="$CMD --uncond_p $UNCOND_P"
CMD="$CMD --resize_ratio $RESIZE_RATIO"

# Add boolean flags
if [ "$USE_CAPTIONS" = true ]; then
    CMD="$CMD --use_captions"
fi

if [ "$USE_CLIP_ADAPTER" = true ]; then
    CMD="$CMD --use_clip_adapter"
fi

if [ "$CLIP_ADAPTER_ONLY" = true ]; then
    CMD="$CMD --clip_adapter_only"
fi

# Note: For multi-GPU, we use accelerate's mixed precision instead of the custom FP16
# if [ "$USE_FP16" = true ]; then
#     CMD="$CMD --use_fp16"
#     CMD="$CMD --fp16_mode $FP16_MODE"
# fi

if [ "$USE_WEBDATASET" = true ]; then
    CMD="$CMD --use_webdataset"
    if [ -n "$WDS_DATASET_NAME" ]; then
        CMD="$CMD --wds_dataset_name \"$WDS_DATASET_NAME\""
    fi
    if [ -n "$WDS_IMAGE_KEY" ]; then
        CMD="$CMD --wds_image_key \"$WDS_IMAGE_KEY\""
    fi
    if [ -n "$WDS_CAPTION_KEY" ]; then
        CMD="$CMD --wds_caption_key \"$WDS_CAPTION_KEY\""
    fi
fi

if [ "$TRIM_WHITE_PADDING" = true ]; then
    CMD="$CMD --trim_white_padding"
    CMD="$CMD --white_thresh $WHITE_THRESH"
fi

if [ "$ACTIVATION_CHECKPOINTING" = true ]; then
    CMD="$CMD --activation_checkpointing"
fi

if [ "$DISABLE_LAION_FILTERS" = true ]; then
    CMD="$CMD --disable_laion_filters"
fi

# Add optional parameters
if [ -n "$CLIP_FEATURES_PATH" ]; then
    CMD="$CMD --clip_features_path \"$CLIP_FEATURES_PATH\""
fi

if [ -n "$EVAL_PROMPT_FILE" ]; then
    CMD="$CMD --eval_prompt_file \"$EVAL_PROMPT_FILE\""
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_ckpt \"$RESUME_CHECKPOINT\""
fi

if [ $SKIP_SAMPLES -gt 0 ]; then
    CMD="$CMD --skip_samples $SKIP_SAMPLES"
fi

if [ -n "$WANDB_PROJECT" ]; then
    CMD="$CMD --project_name \"$WANDB_PROJECT\""
fi

# ============================================
# Pre-flight Checks
# ============================================

# Check if accelerate is installed
if ! uv run python -c "import accelerate" 2>/dev/null; then
    echo "ERROR: 'accelerate' package not found. Install with: uv add accelerate"
    exit 1
fi

# Check if data directory exists (only for non-WebDataset)
if [ "$USE_WEBDATASET" = false ] && [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Check if resume checkpoint exists if specified
if [ -n "$RESUME_CHECKPOINT" ] && [ ! -f "$RESUME_CHECKPOINT" ] && [ ! -d "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: Resume checkpoint does not exist: $RESUME_CHECKPOINT"
    exit 1
fi

# Check if eval prompt file exists if specified
if [ -n "$EVAL_PROMPT_FILE" ] && [ ! -f "$EVAL_PROMPT_FILE" ]; then
    echo "WARNING: Evaluation prompt file does not exist: $EVAL_PROMPT_FILE"
fi

# Check CUDA availability if using GPU
if [ "$DEVICE" = "cuda" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: CUDA requested but nvidia-smi not found."
        exit 1
    fi
    
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
        echo "ERROR: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
        exit 1
    fi
fi

# ============================================
# Display GPU Information
# ============================================

if [ "$DEVICE" = "cuda" ] && command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
fi

# ============================================
# Launch Training
# ============================================

echo "Launching multi-GPU training with command:"
echo "$CMD"
echo ""

# Save configuration to file
CONFIG_FILE="$SAVE_DIRECTORY/training_config.txt"
echo "Multi-GPU Training Configuration - $(date)" > "$CONFIG_FILE"
echo "==========================================" >> "$CONFIG_FILE"
echo "GPUs: $NUM_GPUS" >> "$CONFIG_FILE"
echo "Mixed Precision: $MIXED_PRECISION" >> "$CONFIG_FILE"
echo "Total Batch Size: $((BATCH_SIZE * NUM_GPUS))" >> "$CONFIG_FILE"
echo "" >> "$CONFIG_FILE"
echo "Command:" >> "$CONFIG_FILE"
echo "$CMD" >> "$CONFIG_FILE"
echo "" >> "$CONFIG_FILE"
echo "Environment:" >> "$CONFIG_FILE"
echo "  Python: $(uv run python --version 2>&1)" >> "$CONFIG_FILE"
echo "  PyTorch: $(uv run python -c 'import torch; print(f"torch=={torch.__version__}")' 2>/dev/null || echo 'N/A')" >> "$CONFIG_FILE"
echo "  Accelerate: $(uv run accelerate --version 2>&1)" >> "$CONFIG_FILE"
echo "  CUDA: $(uv run python -c 'import torch; print(f"cuda={torch.cuda.is_available()}")' 2>/dev/null || echo 'N/A')" >> "$CONFIG_FILE"
if [ "$DEVICE" = "cuda" ]; then
    echo "  GPUs:" >> "$CONFIG_FILE"
    nvidia-smi --query-gpu=index,name --format=csv >> "$CONFIG_FILE" 2>/dev/null || echo '    N/A' >> "$CONFIG_FILE"
fi

# Execute the training command
eval $CMD

# Save exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "Multi-GPU training completed successfully!"
    echo "Checkpoints saved to: $SAVE_DIRECTORY"
else
    echo ""
    echo "Multi-GPU training failed with exit status: $EXIT_STATUS"
    echo "Check logs in: $SAVE_DIRECTORY"
    echo ""
    echo "Troubleshooting tips:"
    echo "  - Check if all GPUs are available: nvidia-smi"
    echo "  - Try reducing batch size or enabling gradient checkpointing"
    echo "  - Check accelerate configuration: accelerate config"
    echo "  - Run with fewer GPUs to isolate issues"
fi

exit $EXIT_STATUS