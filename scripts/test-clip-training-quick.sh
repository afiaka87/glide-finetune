#!/bin/bash
# Quick test script for CLIP training - minimal overhead

echo "Quick CLIP training test - 2 steps only"

uv run python train_glide.py \
    --data_dir '/mnt/9_1T_HDD_OLDER/DATASETS/Laion_Synthetic/laion_synth_5m_wds' \
    --use_webdataset \
    --wds_dataset_name 'webdataset' \
    --wds_caption_key 'txt' \
    --wds_image_key 'jpg' \
    --laion_no_filter \
    --batch_size 2 \
    --test_run 2 \
    --use_clip \
    --clip_model_name 'ViT-B/32' \
    --use_clip_cache \
    --clip_cache_dir './clip_cache' \
    --adapter_training_phase 'adapter_only' \
    --use_8bit_adam \
    --use_tf32 \
    --device cuda \
    --checkpoints_dir '/tmp/test-checkpoints'

echo "Test complete!"