#!/bin/bash

# Example script for fine-tuning GLIDE base model
# Update paths and parameters as needed

uv run python train_glide.py \
    --data_dir '~/datasets/coco-style-dataset' \
    --batch_size 2 \
    --learning_rate 0.0001 \
    --adam_weight_decay 0.1 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 0.8 \
    --uncond_p 0.2 \
    --use_fp16 \
    --device cuda \
    --checkpoints_dir './finetune_checkpoints' \
    --activation_checkpointing \
    --use_captions \
    --project_name 'glide-finetune-wandb' \
    --epochs 40 \
    --test_prompt 'a beautiful landscape' \
    --test_guidance_scale 4.0 \
    --test_batch_size 1 \
    --log_frequency 100 \
    --sampler 'plms' \
    --test_steps 100