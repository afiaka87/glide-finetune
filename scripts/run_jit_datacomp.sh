#!/bin/bash

# JiT-B/4 Rectified Flow Training on DataComp-Proper (~10M images)
#
# Model: JiT-B (768-dim, 12 layers, 12 heads, patch_size=4, ~463M params)
# Data:  ~/Data/datacomp-proper-wds/ (1000 tar files, ~376GB, jpg+txt+json)
# GPU:   RTX 4070 12GB — batch_size=32 uses ~6.4GB, leaving headroom
#
# Effective batch size: 32 * 4 = 128
# Epoch length: auto (num_tars * 10000 = ~10M samples per epoch)
# Training: x-prediction + v-loss (rectified flow)
# CFG dropout: 10% (handled by RectifiedFlow, not dataloader)
# Optimizer: Adam(lr=2e-4, betas=(0.9, 0.95), wd=0)
#
# Usage:
#   ./scripts/run_jit_datacomp.sh                    # fresh start
#   ./scripts/run_jit_datacomp.sh checkpoint:path.pt  # resume from checkpoint

set -euo pipefail

INIT="${1:-resume:checkpoints_jit/0006/training_state_00017351.pt}"

echo "=== JiT-B/4 Rectified Flow Training ==="
echo "Init: ${INIT}"
echo "Data: ~/Data/datacomp-proper-wds/"
echo ""

exec uv run python train_glide.py \
    --jit_mode \
    --jit_size B \
    --data_dir "$HOME/Data/datacomp-proper-wds/*.tar" \
    --use_webdataset \
    --wds_image_key jpg \
    --wds_caption_key txt \
    --wds_dataset_name datacomp-real \
    --use_captions \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --precision bf16 \
    --cfg_drop_prob 0.1 \
    --time_mu -0.8 \
    --time_sigma 0.8 \
    --ema_rate 0 \
    --max_grad_norm 1.0 \
    --loss_spike_threshold 5.0 \
    --checkpoints_dir ./checkpoints_jit \
    --save_checkpoint_interval 5000 \
    --sample_interval 500 \
    --eval_interval 0 \
    --prompt_file data/generated-captions-32.txt \
    --sample_batch_size 16 \
    --test_guidance_scale 4.0 \
    --jit_sampler euler \
    --jit_sampler_steps 50 \
    --wandb_project_name jit-training \
    --epochs 20 \
    --seed 42 \
    --cudnn_benchmark \
    --num_workers 8 \
    --wds_buffer_size 1000 \
    --random_hflip \
    --color_jitter 0.1 \
    --skip_tar_validation \
    --glide_text_encoder_path ./laionide-v3-base.pt \
    --init "${INIT}"
