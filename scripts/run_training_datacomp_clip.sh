#!/bin/bash

# GLIDE Fine-tuning on DataComp-10M with best-of CLIP caption selection
# Uses the caption (original or generated) with the higher CLIP score per sample.
# Requires the merged captions JSONL with both original_caption_clip_score
# and generated_caption_clip_score fields.

echo "Starting GLIDE training on DataComp-10M (best CLIP caption)..."
echo "Dataset: /home/sam/Data/datacomp-proper-wds/"
echo "Captions: /home/sam/Data/datacomp-proper-wds/captions/datacomp-10m-captions.jsonl"
echo ""

export WANDB_API_KEY='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'

uv run python train_glide.py \
    --data_dir "/home/sam/Data/datacomp-proper-wds/*.tar" \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_dataset_name datacomp-clip \
    --wds_captions_jsonl /home/sam/Data/datacomp-proper-wds/captions/datacomp-10m-captions.jsonl \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.0 \
    --ema_rate 0.9999 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --checkpoints_dir ./checkpoints \
    --precision bf16 \
    --log_frequency 10 \
    --sample_interval 2500 \
    --wandb_project_name 'glide-datacomp-clip' \
    --activation_checkpointing \
    --gradient_accumulation_steps 4 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 16 \
    --eval_base_sampler "euler_a" \
    --eval_sr_sampler "euler" \
    --eval_base_sampler_steps 30 \
    --eval_sr_sampler_steps 20 \
    --test_guidance_scale 4.0 \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
    --save_checkpoint_interval 5000 \
    --prompt_file data/generated-captions-32.txt \
    --use_sr_eval \
    --validation_workers 8 \
    --random_init
