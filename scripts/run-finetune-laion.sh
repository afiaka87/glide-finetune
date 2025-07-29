#!/bin/bash
# Run script for fine-tuning GLIDE on the captioned LAION dataset
#  Model: /mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune/0007/interrupted_checkpoint_epoch0_step15893.pt
#  Optimizer: /mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune/0007/interrupted_checkpoint_epoch0_step15893.optimizer.pt
#  Metadata: /mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune/0007/interrupted_checkpoint_epoch0_step15893.json
#--sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}

uv run python train_glide.py \
    --resume '/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune/0007/interrupted_checkpoint_epoch0_step15893.pt' \
    --data_dir '/home/sam/Data/laion400m-dat-release' \
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
    --freeze_diffusion \
    --batch_size 8 \
    --epochs 20 \
    --learning_rate 1e-04 \
    --adam_weight_decay 0.01 \
    --use_8bit_adam \
    --warmup_steps 2000 \
    --warmup_type linear \
    --device cuda \
    --activation_checkpointing \
    --cudnn_benchmark \
    --use_tf32 \
    --eval_prompts_file '/home/sam/GitHub/glide-finetune/examples/eval_prompts_32.txt' \
    --test_batch_size 1 \
    --test_guidance_scale 4.0 \
    --test_steps 20 \
    --sampler 'dpm++_2m_karras' \
    --checkpoints_dir '/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune' \
    --project_name 'glide-finetune-laion-nostalgia' \
    --log_frequency 5 \
    --sample_interval 100 \
    --use_esrgan \
    --resume_ckpt '/mnt/9_1T_HDD_OLDER/Checkpoints/glide-finetune/0023/interrupted_checkpoint_epoch3_step144174.pt'
