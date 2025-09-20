#!/bin/bash

# GLIDE Fine-tuning on LAION-2B English Aesthetic Dataset
# This script trains the 64x64 base model with classifier-free guidance
# Dataset location: ~/laion2b_en_aesthetic_wds/

echo "Starting GLIDE training on LAION-2B English Aesthetic dataset..."
echo "Dataset: ~/laion2b_en_aesthetic_wds/"
echo "Note: Corrupted/incomplete tar files will be automatically skipped"
echo "      Validation results are cached in ./cache/ for faster restarts"
echo ""

uv run python train_glide.py \
    --data_dir ~/laion2b_en_aesthetic_wds/ \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --adam_weight_decay 0.0 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --checkpoints_dir ./checkpoints \
    --precision bf16 \
    --log_frequency 1 \
    --sample_interval 50 \
    --wandb_project_name 'glide-laion-finetune' \
    --activation_checkpointing \
    --gradient_accumulation_steps 1 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 32 \
    --eval_base_sampler "euler_a" \
    --eval_sr_sampler "euler_a" \
    --eval_base_sampler_steps 30 \
    --eval_sr_sampler_steps 27 \
    --test_guidance_scale 4.0 \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_caption_key txt \
    --wds_dataset_name laion \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
    --save_checkpoint_interval 5000 \
    --prompt_file eval_captions_persons_aesthetic.txt \
    --use_sr_eval \
    --validation_workers 32

# Note: SR evaluation will generate both 64x64 and 256x256 images during training
# Both resolutions will be logged to WandB for comparison

# Optional flags (uncomment as needed):
# --sr_model_path path/to/custom_sr.pt \  # Use custom SR model instead of default
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
echo "  - Batch size: 4"
echo "  - Gradient accumulation: 4 steps (effective batch size: 16)"
echo "  - Learning rate: 1e-4"
echo "  - Mixed precision: BF16"
echo "  - Unconditional probability: 0.2 (for classifier-free guidance)"
echo "  - Dataset: LAION-2B English Aesthetic"
echo "  - Validation: Parallel tar file validation with caching"
echo "  - Cache location: ./cache/valid_tars_*.json"
echo "  - SR Evaluation: Enabled (generates both 64x64 and 256x256 images)"
echo "  - WandB logging: Both resolutions will be logged separately"
