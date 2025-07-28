#!/bin/bash

# Run script for fine-tuning GLIDE on the captioned birds dataset

uv run python train_glide.py \
    --data_dir '/home/sam/Data/laion400m-dat-release' \
    --use_webdataset \
    --wds_caption_key 'txt' \
    --wds_image_key 'jpg' \
    --wds_dataset_name 'laion' \
    --batch_size 8 \
    --learning_rate 7e-05 \
    --adam_weight_decay 0.01 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --use_captions \
    --device cuda \
    --checkpoints_dir './finetune_checkpoints/laion' \
    --project_name 'glide-finetune-laion' \
    --epochs 20 \
    --test_prompt 'a male mannequin dressed in a black leather jacket and gray pleated trousers' \
    --test_batch_size 4 \
    --test_guidance_scale 4.0 \
    --test_steps 100
    --log_frequency 500 \
    --use_8bit_adam \
    --activation_checkpointing \
    --cudnn_benchmark \
    --use_tf32 \
    --use_fp16 \
    --freeze_transformer \
    --sampler plms
