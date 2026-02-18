#!/bin/bash

# GLIDE Latent-mode Fine-tuning on DataComp-10M with best-of CLIP caption selection
# Latent diffusion: 256x256 images encoded to 32x32 latents via frozen SD 1.5 VAE.
# Text conditioning via frozen OpenCLIP ViT-L-14 instead of GLIDE's text transformer.
# Training from random init.

echo "Starting GLIDE LATENT training on DataComp-10M (best CLIP caption)..."
echo "Dataset: /home/sam/Data/datacomp10m/"
echo "Captions: /home/sam/Data/datacomp10m/captions/datacomp-10m-captions.jsonl"
echo ""

uv run python train_glide.py \
    --data_dir "/home/sam/Data/datacomp10m/*.tar" \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_dataset_name datacomp-clip \
    --wds_captions_jsonl /home/sam/Data/datacomp10m/captions/datacomp-10m-captions.jsonl \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --adam_weight_decay 0.01 \
    --ema_rate 0.9999 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --checkpoints_dir ./checkpoints \
    --precision bf16 \
    --sample_interval 5000 \
    --wandb_project_name 'glide-latent-test' \
    --gradient_accumulation_steps 4 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 16 \
    --eval_base_sampler "euler" \
    --eval_sr_sampler "euler" \
    --eval_base_sampler_steps 30 \
    --eval_sr_sampler_steps 20 \
    --test_guidance_scale 4.0 \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
    --save_checkpoint_interval 10000 \
    --prompt_file data/generated-captions-32.txt \
    --validation_workers 8 \
    --latent_mode \
    --random_init \
    --skip_tar_validation \
    --clip_threshold 0.3 \
    --resume_ckpt checkpoints/ema_0.9999_0x060000.pt
