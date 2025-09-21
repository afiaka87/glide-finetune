#!/bin/bash

uv run python train_glide.py \
    --use_webdataset \
    --wds_image_key 'jpg' \
    --wds_caption_key 'txt' \
    --data_dir /home/sam/Claude/laion_filter_pixelart/retro_and_pixelart-wds \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --resize_ratio 1.0 \
    --use_captions \
    --uncond_p 0.2 \
    --device cuda \
    --log_frequency 1 \
    --sample_interval 50 \
    --wandb_project_name glide_retro_pixelart_games \
    --activation_checkpointing \
    --epochs 20 \
    --prompt_file data/generated-captions-1k.txt \
    --use_sr_eval \
    --precision bf16


#--freeze_transformer \
#--freeze_diffusion \
