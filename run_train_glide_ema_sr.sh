#!/bin/bash

# GLIDE Super-Resolution Fine-tuning with EMA (Exponential Moving Average) on LAION-2B
# This script trains the 64→256 upsampling model with EMA weights for improved quality
# EMA models typically produce better super-resolution results than raw training weights

echo "================================================================================"
echo "GLIDE SUPER-RESOLUTION Training with EMA (Exponential Moving Average)"
echo "================================================================================"
echo ""
echo "Dataset: LAION-2B English Aesthetic"
echo "Model: Super-Resolution (64x64 → 256x256)"
echo "Starting checkpoint: glide_model_cache/upsample.pt"
echo "EMA Rate: 0.9999 (GLIDE paper specification)"
echo ""
echo "VRAM OVERHEAD WARNING:"
echo "  - EMA requires maintaining TWO copies of the upsampler model in memory"
echo "  - Base GLIDE upsampler: ~500M parameters = ~1GB in FP32"
echo "  - With BF16: ~500MB per model copy"
echo "  - Total EMA overhead: +500MB VRAM for BF16, +1GB for FP32"
echo "  - SR models use more VRAM due to 256x256 resolution"
echo "  - Recommendation: Use batch_size=2 with grad_accumulation=8"
echo ""
echo "EMA Benefits for Super-Resolution:"
echo "  - Significantly sharper upsampled images"
echo "  - More stable texture generation"
echo "  - Reduces artifacts in high-frequency details"
echo "  - Better preservation of fine details from 64x64 input"
echo ""
echo "Note: EMA checkpoints saved as 'ema_0.9999_sr_*.pt' for inference"
echo "================================================================================"
echo ""

# Generate base images for evaluation if they don't exist
if [ ! -d "data/images/base_64x64" ] || [ -z "$(ls -A data/images/base_64x64 2>/dev/null)" ]; then
    echo "Generating 32 base images for SR evaluation..."
    echo "This ensures consistent evaluation during training"
    echo ""
    uv run python generate_base_images.py \
        --base_model_path glide_model_cache/base.pt \
        --prompt_file data/generated-captions-1k.txt \
        --output_dir data/images/base_64x64 \
        --num_images 32 \
        --batch_size 4 \
        --guidance_scale 4.0 \
        --sampler euler \
        --sampler_steps 50 \
        --seed 42 \
        --use_bf16
    echo ""
    echo "Base images generated successfully!"
    echo "================================================================================"
    echo ""
else
    echo "Base images already exist in data/images/base_64x64, skipping generation."
    echo ""
fi

# Main training command with EMA for super-resolution
uv run python train_glide.py \
    --data_dir /mnt/usb_nvme_2tb/Data/laion-2b-en-aesthetic-subset \
    --resume_ckpt glide_model_cache/upsample.pt \
    --train_upsample \
    --upscale_factor 4 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --adam_weight_decay 0.0 \
    --ema_rate 0.9999 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.0 \
    --checkpoints_dir ./checkpoints_ema_sr \
    --precision bf16 \
    --log_frequency 10 \
    --sample_interval 500 \
    --wandb_project_name 'glide-laion-ema-sr' \
    --activation_checkpointing \
    --gradient_accumulation_steps 8 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 4 \
    --eval_sr_sampler "euler" \
    --eval_sr_sampler_steps 27 \
    --test_guidance_scale 0.0 \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_caption_key txt \
    --wds_dataset_name laion \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
    --save_checkpoint_interval 2500 \
    --prompt_file 'data/generated-captions-1k.txt' \
    --validation_workers 8 \
    --eval_sr_base_images data/images/base_64x64

echo ""
echo "================================================================================"
echo "Training Configuration Summary:"
echo "================================================================================"
echo "Model Configuration:"
echo "  - Model type: Upsampler (Super-Resolution)"
echo "  - Base resolution: 64x64 → 256x256"
echo "  - Starting from: glide_model_cache/upsample.pt"
echo "  - EMA decay rate: 0.9999 (GLIDE paper)"
echo "  - Mixed precision: BF16"
echo "  - Activation checkpointing: Enabled (saves VRAM)"
echo ""
echo "Optimizer Settings (SR-specific):"
echo "  - Optimizer: AdamW"
echo "  - Learning rate: 5e-5 (10x lower than base model for stability)"
echo "  - Weight decay: 0.0 (GLIDE paper)"
echo "  - Betas: (0.9, 0.999) [PyTorch defaults]"
echo "  - No gradient clipping"
echo ""
echo "Training Settings:"
echo "  - Batch size: 2 (reduced due to 256x256 resolution)"
echo "  - Gradient accumulation: 8 steps (effective batch size: 16)"
echo "  - Unconditional probability: 0.0 (always conditioned on text for SR)"
echo ""
echo "Checkpointing:"
echo "  - Directory: ./checkpoints_ema_sr/"
echo "  - Save interval: 2500 steps"
echo "  - EMA checkpoints: ema_0.9999_sr_*.pt"
echo "  - Regular checkpoints: glide-sr-ft-*.pt"
echo ""
echo "Evaluation:"
echo "  - Fixed base images: data/images/base_64x64/"
echo "  - SR outputs saved: data/images/sr_256x256/"
echo "  - Sampler: Euler (deterministic, fast for SR)"
echo "  - Steps: 27 (optimal for SR quality/speed)"
echo ""
echo "Monitoring:"
echo "  - WandB project: glide-laion-ema-sr"
echo "  - Sample generation: Every 500 steps"
echo "  - Fixed input image for consistent evaluation"
echo ""
echo "================================================================================"
echo ""

# Optional: Memory-saving configurations for limited VRAM
echo "Memory-Constrained Alternative Commands:"
echo ""
echo "For 16GB VRAM (further reduce batch size):"
echo "  Add: --batch_size 1 --gradient_accumulation_steps 16"
echo ""
echo "For 12GB VRAM (aggressive memory saving):"
echo "  Add: --batch_size 1 --gradient_accumulation_steps 16"
echo "  Consider: --precision fp32 --ema_rate 0 (disable EMA)"
echo ""
echo "To disable EMA entirely (saves ~500MB-1GB VRAM):"
echo "  Set: --ema_rate 0"
echo ""
echo "Key differences from base model EMA training:"
echo "  - Lower batch size (2 vs 4) due to 256x256 resolution"
echo "  - Higher gradient accumulation (8 vs 1) to maintain effective batch size"
echo "  - Lower learning rate (5e-5 vs 1e-4) for SR stability"
echo "  - No classifier-free guidance (uncond_p=0.0 vs 0.2)"
echo "  - Different checkpoint directory (./checkpoints_ema_sr vs ./checkpoints_ema)"
echo "  - Fixed evaluation images for consistent SR quality tracking"
echo ""