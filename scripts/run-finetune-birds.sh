#!/bin/bash

# Run script for fine-tuning GLIDE on the captioned birds dataset

uv run python train_glide.py \
    --data_dir '/home/sam/Data/captioned-birds-wds' \
    --use_webdataset \
    --wds_caption_key 'txt' \
    --wds_image_key 'png' \
    --wds_dataset_name 'laion' \
    --batch_size 4 \
    --learning_rate 1e-04 \
    --adam_weight_decay 0.0 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 0.8 \
    --uncond_p 0.2 \
    --use_captions \
    --device cuda \
    --checkpoints_dir './finetune_checkpoints/birds' \
    --project_name 'glide-finetune-birds' \
    --epochs 20 \
    --test_prompt 'a beautiful colorful bird perched on a branch' \
    --test_batch_size 1 \
    --test_guidance_scale 4.0 \
    --log_frequency 100 \
    --use_8bit_adam \
    --activation_checkpointing \
    --use_tf32