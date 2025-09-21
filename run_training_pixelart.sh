#!/bin/bash

uv run python train_glide.py \
    --data_dir pixelart-v7-large-no-menus \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --resize_ratio 1.0 \
    --uncond_p 1.0 \
    --device cuda \
    --log_frequency 1 \
    --sample_interval 50 \
    --freeze_transformer \
    --wandb_project_name glide_pixelart_nomenus \
    --activation_checkpointing \
    --epochs 10 \
    --prompt_file data/generated-captions-1k.txt \
    --use_sr_eval \
    --precision bf16