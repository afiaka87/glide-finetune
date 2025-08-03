#!/bin/bash

# Example script for fine-tuning GLIDE base model
# This showcases various training options available

# For standard image-caption dataset (folder structure)
uv run python train_glide.py \
    --data_dir './path/to/dataset' \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.01 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 0.8 \
    --uncond_p 0.2 \
    --device cuda \
    --checkpoints_dir './finetune_checkpoints' \
    --activation_checkpointing \
    --use_captions \
    --project_name 'glide-finetune' \
    --epochs 10 \
    --test_prompt 'a beautiful landscape' \
    --test_guidance_scale 4.0 \
    --test_batch_size 1 \
    --log_frequency 100 \
    --sample_interval 1000 \
    --sampler 'dpm++_2m_karras' \
    --test_steps 50 \
    --use_8bit_adam \
    --use_tf32 \
    --warmup_steps 1000

# For WebDataset format (uncomment and modify as needed)
# uv run python train_glide.py \
#     --data_dir './path/to/webdataset/tars' \
#     --use_webdataset \
#     --wds_dataset_name 'webdataset' \
#     --wds_caption_key 'txt' \
#     --wds_image_key 'jpg' \
#     --batch_size 4 \
#     --learning_rate 1e-4 \
#     --use_8bit_adam \
#     --use_tf32 \
#     --checkpoints_dir './finetune_checkpoints'