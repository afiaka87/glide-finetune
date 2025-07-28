#!/bin/bash

# Run script for fine-tuning GLIDE on the captioned LAION dataset

uv run python train_glide.py \
    --data_dir '/home/sam/Data/laion400m-dat-release' \
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
    --freeze_diffusion \
    --batch_size 8 \
    --epochs 20 \
    --learning_rate 7e-05 \
    --adam_weight_decay 0.01 \
    --use_8bit_adam \
    --warmup_steps 1000 \
    --warmup_type linear \
    --device cuda \
    --activation_checkpointing \
    --cudnn_benchmark \
    --use_tf32 \
    --test_prompt 'a male mannequin dressed in a black leather jacket and gray pleated trousers' \
    --test_batch_size 1 \
    --test_guidance_scale 4.0 \
    --test_steps 100 \
    --sampler plms \
    --checkpoints_dir './finetune_checkpoints/laion' \
    --project_name 'glide-finetune-laion' \
    --log_frequency 100
