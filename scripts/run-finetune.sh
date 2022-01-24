python ../glide-finetune.py \
    --uncond_p 1.0 \
    --data_dir '' \
    --batch_size 4 \
    --grad_acc 2 \
    --dropout 0.1 \
    --checkpoints_dir 'my_checkpoints' \
    --resume_ckpt '' \
    --device 'cuda' \
    --project_name 'glide-finetune'
