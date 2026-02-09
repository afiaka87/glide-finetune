#!/usr/bin/env bash
set -euo pipefail

# Fine-tune laionide-v3-base on datacomp-small

export TORCH_LOGS="graph_breaks,recompiles"

DATA_DIR="$HOME/Data/datacomp-small/shards"
CHECKPOINTS_DIR="./glide_checkpoints/laionide-v3-datacomp"
RESUME_CKPT="./laionide-v3-base.pt"
WANDB_PROJECT="glide-datacomp-small"

uv run python train_glide.py \
    --data_dir "$DATA_DIR" \
    --resume_ckpt "$RESUME_CKPT" \
    --use_webdataset \
    --wds_dataset_name simple \
    --wds_image_key jpg \
    --wds_caption_key txt \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --precision bf16 \
    --uncond_p 0.2 \
    --use_captions \
    --epochs 20 \
    --cudnn_benchmark \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --save_checkpoint_interval 5000 \
    --sample_interval 500 \
    --eval_interval 0 \
    --log_frequency 100 \
    --num_workers 16 \
    --wds_buffer_size 1000 \
    --wandb_project_name "$WANDB_PROJECT" \
    --prompt_file eval_captions.txt \
    --seed 42 \
    "$@"
