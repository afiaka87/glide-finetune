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
    --data_dir /mnt/usb_nvme_2tb/Data/laion-2b-en-aesthetic-subset \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.0 \
    --ema_rate 0.9999 \
    --side_x 64 \
    --side_y 64 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --checkpoints_dir ./checkpoints \
    --precision bf16 \
    --log_frequency 1 \
    --sample_interval 250 \
    --wandb_project_name 'glide-laion-finetune' \
    --activation_checkpointing \
    --gradient_accumulation_steps 1 \
    --use_captions \
    --epochs 10 \
    --sample_batch_size 16 \
    --eval_base_sampler "euler_a" \
    --eval_sr_sampler "euler" \
    --eval_base_sampler_steps 30 \
    --eval_sr_sampler_steps 20 \
    --test_guidance_scale 4.0 \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_caption_key txt \
    --wds_dataset_name laion \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
<<<<<<< Updated upstream
    --save_checkpoint_interval 2500 \
    --prompt_file 'data/generated-captions-1k.txt' \
    --use_sr_eval \
    --validation_workers 8
=======
    --save_checkpoint_interval 5000 \
    --prompt_file eval_captions_persons_aesthetic.txt \
    --validation_workers 32
>>>>>>> Stashed changes

#--use_sr_eval \
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
echo "Training configuration (GLIDE paper hyperparameters):"
echo "  - Batch size: 4"
echo "  - Gradient accumulation: 1 step"
echo "  - Learning rate: 1e-4 (GLIDE paper value)"
echo "  - Weight decay: 0.0 (GLIDE paper value)"
echo "  - EMA rate: 0.9999 (GLIDE paper value)"
echo "  - Optimizer: AdamW with default betas (0.9, 0.999)"
echo "  - Gradient clipping: None (as per GLIDE)"
echo "  - Mixed precision: BF16"
echo "  - Unconditional probability: 0.2 (for classifier-free guidance)"
echo "  - Dataset: LAION-2B English Aesthetic"
echo "  - Validation: Parallel tar file validation with caching"
echo "  - Cache location: ./cache/valid_tars_*.json"
echo "  - SR Evaluation: Enabled (generates both 64x64 and 256x256 images)"
echo "  - WandB logging: Both resolutions will be logged separately"
echo "  - EMA checkpoints: Saved as ema_0.9999_*.pt files"
