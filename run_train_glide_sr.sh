#!/bin/bash

# GLIDE Upsampler Fine-tuning on LAION-2B English Aesthetic Dataset
# This script trains the 64→256 super-resolution model with text conditioning
# Dataset location: ~/laion2b_en_aesthetic_wds/

echo "Starting GLIDE UPSAMPLER training on LAION-2B English Aesthetic dataset..."
echo "Dataset: ~/laion2b_en_aesthetic_wds/"
echo "Model: Super-resolution (64x64 → 256x256)"
echo "Note: Corrupted/incomplete tar files will be automatically skipped"
echo "      Validation results are cached in ./cache/ for faster restarts"
echo ""

uv run python train_glide.py \
    --data_dir ~/laion2b_en_aesthetic_wds/ \
    --train_upsample \
    --upscale_factor 4 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --adam_weight_decay 0.0 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.0 \
    --checkpoints_dir ./checkpoints_sr \
    --precision bf16 \
    --log_frequency 1 \
    --sample_interval 50 \
    --wandb_project_name 'glide-laion-upsampler' \
    --activation_checkpointing \
    --gradient_accumulation_steps 8 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 8 \
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
    --save_checkpoint_interval 5000 \
    --prompt_file 'data/generated-captions-1k.txt' \
    --validation_workers 32

# Note: Upsampler evaluation will generate 256x256 images during training
# These will be logged to WandB for quality monitoring

# Optional flags (uncomment as needed):
# --base_model_path path/to/base_64x64.pt \  # Provide base model for joint evaluation
# --image_to_upsample path/to/test_64x64.png \  # Fixed low-res image for consistent eval
# --skip_tar_validation \           # Skip tar file validation (faster startup if all files are known to be valid)
# --no_cache_validation \           # Force re-validation of all tar files (ignore cache)
# --clear_validation_cache \        # Clear the validation cache before starting
# --validation_workers 8 \          # Set number of parallel workers for tar validation (default: auto)
# --wds_debug \                     # Enable debug output for WebDataset
# --freeze_transformer \            # Freeze transformer weights (train only diffusion model)
# --freeze_diffusion \              # Freeze diffusion weights (train only transformer)
# --resume_ckpt path/to/checkpoint.pt \  # Resume from checkpoint
# --use_lora \                      # Enable LoRA for parameter-efficient training
# --lora_rank 8 \                   # LoRA rank
# --lora_alpha 32 \                 # LoRA alpha scaling
# --lora_target_mode attention \    # LoRA target modules

echo ""
echo "Training configuration:"
echo "  - Model type: Upsampler (Super-Resolution)"
echo "  - Resolution: 64x64 → 256x256"
echo "  - Batch size: 2"
echo "  - Gradient accumulation: 8 steps (effective batch size: 16)"
echo "  - Learning rate: 5e-5 (lower than base model)"
echo "  - Mixed precision: BF16"
echo "  - Unconditional probability: 0.0 (always conditioned on text)"
echo "  - Dataset: LAION-2B English Aesthetic"
echo "  - Data processing: Original → 256x256 → 64x64 (proper downsampling)"
echo "  - Validation: Parallel tar file validation with caching"
echo "  - Cache location: ./cache/valid_tars_*.json"
echo "  - Evaluation: Generates 256x256 upsampled images"
echo "  - WandB logging: 256x256 galleries and metrics"
echo ""
echo "Key differences from base model training:"
echo "  - Lower batch size (2 vs 32) due to 256x256 resolution"
echo "  - Higher gradient accumulation (8 vs 1) to maintain effective batch size"
echo "  - Lower learning rate (5e-5 vs 3e-4) for stability"
echo "  - No classifier-free guidance (uncond_p=0.0 vs 0.2)"
echo "  - Different checkpoint directory (./checkpoints_sr)"
echo "  - Different WandB project name (glide-laion-upsampler)"
