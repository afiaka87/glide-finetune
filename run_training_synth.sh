#!/bin/bash

uv run python train_glide.py \
    --data_dir "/mnt/t7_2tb/Data/synthetic/synthetic-dataset-1m-dalle3-high-quality-captions/data/*.tar" \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_caption_key json \
    --wds_dataset_name synthetic \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --device cuda \
    --log_frequency 1 \
    --sample_interval 100 \
    --freeze_diffusion \
    --wandb_project_name glide_synth_dalle3 \
    --activation_checkpointing \
    --epochs 10 \
    --prompt_file data/generated-captions-1k.txt \
    --sample_batch_size 8 \
    --use_sr_eval \
    --gradient_accumulation_steps 4 \
    --use_captions \
    --num_workers 4 \
    --wds_buffer_size 1000 \
    --save_checkpoint_interval 5000 \
    --precision bf16