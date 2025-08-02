#!/bin/bash
# Run script for fine-tuning GLIDE on LAION dataset with CLIP adapters
# Uses pre-computed CLIP embeddings for faster training

# Paths
LAION_DATA_DIR="/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds"
CLIP_CACHE_DIR="./laion-synthetic-clip_cache"
CHECKPOINT_DIR="/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune-clip"
EVAL_PROMPTS="/home/sam/GitHub/glide-finetune/examples/people_captions_16.txt"

# CLIP Configuration
CLIP_MODEL="ViT-B/32"  # Should match the model used in precompute-clip-laion.sh
ADAPTER_LR=1e-5
ADAPTER_WARMUP=10000
ADAPTER_PHASE="adapter_only"  # Options: adapter_only, adapter_gates, full

# Training Configuration
BATCH_SIZE=8
LEARNING_RATE=1e-6  # Lower than usual when using CLIP adapters
EPOCHS=20
WARMUP_STEPS=2000

echo "================================================"
echo "Fine-tuning GLIDE with CLIP adapters on LAION"
echo "================================================"
echo "Data: $LAION_DATA_DIR"
echo "CLIP cache: $CLIP_CACHE_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Adapter phase: $ADAPTER_PHASE"
echo "Adapter LR: $ADAPTER_LR"
echo "Main LR: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE"
echo "================================================"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Run training with CLIP adapters
uv run python train_glide.py \
    --data_dir "$LAION_DATA_DIR" \
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
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --adam_weight_decay 0.01 \
    --use_8bit_adam \
    --warmup_steps $WARMUP_STEPS \
    --warmup_type linear \
    --device cuda \
    --activation_checkpointing \
    --cudnn_benchmark \
    --use_tf32 \
    --eval_prompts_file "$EVAL_PROMPTS" \
    --test_batch_size 1 \
    --test_guidance_scale 4.0 \
    --test_steps 20 \
    --sampler 'dpm++_2m_karras' \
    --checkpoints_dir "$CHECKPOINT_DIR" \
    --project_name 'glide-finetune-laion-clip' \
    --log_frequency 100 \
    --sample_interval 1000 \
    --use_clip \
    --clip_model_name "$CLIP_MODEL" \
    --use_clip_cache \
    --clip_cache_dir "$CLIP_CACHE_DIR" \
    --adapter_training_phase "$ADAPTER_PHASE" \
    --adapter_lr $ADAPTER_LR \
    --adapter_warmup_steps $ADAPTER_WARMUP \
    --adapter_wd 1e-2 \
    --adapter_beta2 0.98 \
    --adapter_grad_clip 0.5 \
    --main_grad_clip 2.0 \
    --kl_loss_interval 100 \
    --kl_loss_weight 0.01 \
    --stability_threshold 10.0 \
    --dry_run_interval 2000 \
    --dry_run_samples 5 \
    --early_stop_threshold 0.1 \
    --early_stop_patience 2000 \
    --baseline_eval_interval 1000 

# Add resume checkpoint if you want to continue from a previous run
# --resume_ckpt "$CHECKPOINT_DIR/latest.pt" \
#--checkpoint_interval 2500 \
