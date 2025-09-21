#!/bin/bash

# GLIDE Fine-tuning with EMA (Exponential Moving Average) on LAION-2B
# This script trains with EMA weights for improved sample quality
# EMA models typically produce better samples than the raw training weights

echo "================================================================================"
echo "GLIDE Training with EMA (Exponential Moving Average)"
echo "================================================================================"
echo ""
echo "Dataset: LAION-2B English Aesthetic"
echo "EMA Rate: 0.9999 (GLIDE paper specification)"
echo ""
echo "VRAM OVERHEAD WARNING:"
echo "  - EMA requires maintaining TWO copies of the model in memory"
echo "  - Base GLIDE 64x64 model: ~500M parameters = ~1GB in FP32"
echo "  - With BF16: ~500MB per model copy"
echo "  - Total EMA overhead: +500MB VRAM for BF16, +1GB for FP32"
echo "  - Recommendation: Reduce batch size if OOM occurs"
echo ""
echo "EMA Benefits:"
echo "  - Significantly better sample quality"
echo "  - More stable training"
echo "  - Reduces noise in generated images"
echo "  - Standard practice in diffusion model training"
echo ""
echo "Note: EMA checkpoints saved as 'ema_0.9999_*.pt' for inference"
echo "================================================================================"
echo ""

# Main training command with EMA
uv run python train_glide.py \
    --data_dir /mnt/usb_nvme_2tb/Data/laion-2b-en-aesthetic-subset \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.0 \
    --ema_rate 0.9999 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --checkpoints_dir ./checkpoints_ema \
    --precision bf16 \
    --log_frequency 10 \
    --sample_interval 500 \
    --wandb_project_name 'glide-laion-ema' \
    --activation_checkpointing \
    --gradient_accumulation_steps 1 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 8 \
    --eval_base_sampler "euler_a" \
    --eval_sr_sampler "euler_a" \
    --eval_base_sampler_steps 25 \
    --eval_sr_sampler_steps 25 \
    --test_guidance_scale 4.0 \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_caption_key txt \
    --wds_dataset_name laion \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
    --save_checkpoint_interval 2500 \
    --prompt_file eval_captions_persons_aesthetic.txt \
    --use_sr_eval \
    --validation_workers 8 \
    --random_hflip

echo ""
echo "================================================================================"
echo "Training Configuration Summary:"
echo "================================================================================"
echo "Model Configuration:"
echo "  - Base resolution: 64x64"
echo "  - EMA decay rate: 0.9999 (GLIDE paper)"
echo "  - Mixed precision: BF16"
echo "  - Activation checkpointing: Enabled (saves VRAM)"
echo ""
echo "Optimizer Settings (GLIDE paper):"
echo "  - Optimizer: AdamW"
echo "  - Learning rate: 1e-4"
echo "  - Weight decay: 0.0"
echo "  - Betas: (0.9, 0.999) [PyTorch defaults]"
echo "  - No gradient clipping"
echo ""
echo "Training Settings:"
echo "  - Batch size: 4"
echo "  - Gradient accumulation: 1 step"
echo "  - Unconditional probability: 0.2 (for CFG)"
echo "  - Random horizontal flip: Enabled"
echo ""
echo "Checkpointing:"
echo "  - Directory: ./checkpoints_ema/"
echo "  - Save interval: 2500 steps"
echo "  - EMA checkpoints: ema_0.9999_*.pt"
echo "  - Regular checkpoints: glide-ft-*.pt"
echo ""
echo "Monitoring:"
echo "  - WandB project: glide-laion-ema"
echo "  - Sample generation: Every 500 steps"
echo "  - SR evaluation: Enabled (64x64 and 256x256)"
echo ""
echo "================================================================================"
echo ""

# Optional: Memory-saving configurations for limited VRAM
echo "Memory-Constrained Alternative Commands:"
echo ""
echo "For 16GB VRAM (reduce batch size):"
echo "  Add: --batch_size 2 --gradient_accumulation_steps 2"
echo ""
echo "For 12GB VRAM (aggressive memory saving):"
echo "  Add: --batch_size 1 --gradient_accumulation_steps 4"
echo "  Consider: --precision fp32 --ema_rate 0 (disable EMA)"
echo ""
echo "To disable EMA entirely (saves ~500MB-1GB VRAM):"
echo "  Set: --ema_rate 0"
echo ""