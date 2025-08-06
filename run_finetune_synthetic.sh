#!/bin/bash

# Fine-tune GLIDE base model on synthetic DALLE-3 dataset
# Dataset: /mnt/t7_2tb/Data/synthetic/synthetic-dataset-1m-dalle3-high-quality-captions/data
# 
# Enabled optimizations:
# - TF32 for better stability
# - 8-bit Adam optimizer
# - torch.compile with reduce-overhead mode
# - Activation checkpointing
# - ESRGAN upscaling
# - CuDNN benchmarking
# - Gradient accumulation
# 
# NOT using:
# - LoRA
# - CLIP adapter

# Dataset path
DATA_DIR="/mnt/t7_2tb/Data/synthetic/synthetic-dataset-1m-dalle3-high-quality-captions/data"

# # Training configuration
# BATCH_SIZE=4
# LEARNING_RATE=1e-5
# EPOCHS=20
# RESUME_CKPT=""
# CHECKPOINT_DIR=""
# GRADIENT_ACCUMULATION_STEPS=4

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo "=========================================="
echo "GLIDE Base Model Fine-tuning"
echo "=========================================="
echo "Dataset: $DATA_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "=========================================="
echo ""
#--resume_ckpt '/home/sam/GitHub/glide-finetune/glide_epoch3_step144174.pt' \

#    --resume_ckpt '/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune/0037/interrupted_checkpoint_epoch1_step70751.pt' \
# --test_prompt "a beautiful landscape with mountains and a lake at sunset" \
# Run training with all specified optimizations
# plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras
uv run python train_glide.py \
    --data_dir '/mnt/t7_2tb/Data/synthetic/synthetic-dataset-1m-dalle3-high-quality-captions/data' \
    --resume_ckpt '/mnt/t7_2tb/Checkpoints/glide-finetune-synthetic/0006/emergency_checkpoint_epoch0_step18086_20250805_010646.pt' \
    --use_webdataset \
    --wds_image_key "jpg" \
    --wds_caption_key "json" \
    --wds_dataset_name "webdataset" \
    --batch_size 8 \
    --learning_rate 7e-5 \
    --warmup_type cosine \
    --warmup_steps 2000 \
    --epochs 2 \
    --checkpoints_dir '/mnt/t7_2tb/Checkpoints/glide-finetune-synthetic' \
    --use_tf32 \
    --use_8bit_adam \
    --activation_checkpointing \
    --cudnn_benchmark \
    --gradient_accumulation_steps 8 \
    --laion_no_filter \
    --uncond_p 0.2 \
    --log_frequency 1 \
    --sample_interval 100 \
    --eval_prompts_file '/home/sam/GitHub/glide-finetune/examples/trippy_prompts_32.txt' \
    --test_guidance_scale 4.0 \
    --test_steps 10 \
    --sampler "dpm++_2m_karras" \
    --project_name "glide-synthetic-dalle3-base" \
    --use_captions \
    --seed 42
