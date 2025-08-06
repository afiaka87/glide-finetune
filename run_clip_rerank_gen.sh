# usage: inference_clip_rerank.py [-h] [--checkpoint CHECKPOINT] [--model_type {base,upsample,base-inpaint,upsample-inpaint}] [--prompt PROMPT |
#                                 --prompt_file PROMPT_FILE] [--num_samples NUM_SAMPLES] [--batch_size BATCH_SIZE]
#                                 [--sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}] [--steps STEPS] [--guidance_scale GUIDANCE_SCALE]
#                                 [--seed SEED] [--clip_model CLIP_MODEL [CLIP_MODEL ...]] [--clip_cache_dir CLIP_CACHE_DIR] [--list_models] [--use_esrgan]
#                                 [--esrgan_model {RealESRGAN_x4plus,RealESRGAN_x4plus_anime_6B}] [--esrgan_cache_dir ESRGAN_CACHE_DIR]
#                                 [--output_dir OUTPUT_DIR] [--save_all] [--device DEVICE] [--use_fp16] [--compile] [--compile_clip]
#                                 [--compile_mode {default,reduce-overhead,max-autotune}] [--compile_cache_dir COMPILE_CACHE_DIR] [--amp]

# Generate images with GLIDE and re-rank with CLIP

# options:
#   -h, --help            show this help message and exit
#   --checkpoint CHECKPOINT
#                         Path to model checkpoint
#   --model_type {base,upsample,base-inpaint,upsample-inpaint}
#                         Type of model to load
#   --prompt PROMPT       Text prompt for generation
#   --prompt_file PROMPT_FILE
#                         File containing line-separated prompts (default: examples/trippy_prompts_32.txt)
#   --num_samples NUM_SAMPLES
#                         Number of images to generate (default: 16)
#   --batch_size BATCH_SIZE
#                         Batch size for generation
#   --sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}
#                         Sampler to use (default: euler)
#   --steps STEPS         Number of sampling steps (default: 30)
#   --guidance_scale GUIDANCE_SCALE
#                         Classifier-free guidance scale
#   --seed SEED           Random seed for reproducibility
#   --clip_model CLIP_MODEL [CLIP_MODEL ...]
#                         CLIP model(s) for ranking. Multiple models will be ensembled.
#   --clip_cache_dir CLIP_CACHE_DIR
#                         Directory to cache CLIP models
#   --list_models         List available CLIP models and exit
#   --use_esrgan          Use ESRGAN to upsample images before ranking (64x64 -> 256x256)
#   --esrgan_model {RealESRGAN_x4plus,RealESRGAN_x4plus_anime_6B}
#                         ESRGAN model to use
#   --esrgan_cache_dir ESRGAN_CACHE_DIR
#                         Directory to cache ESRGAN models
#   --output_dir OUTPUT_DIR
#                         Directory to save outputs
#   --save_all            Save all generated images (not just the best)
#   --device DEVICE       Device to use (cuda/cpu)
#   --use_fp16            Use FP16 precision for model weights
#   --compile             Use torch.compile for GLIDE model (requires PyTorch 2.0+)
#   --compile_clip        Also compile CLIP models for faster ranking
#   --compile_mode {default,reduce-overhead,max-autotune}
#                         Torch compile mode (reduce-overhead is fastest for inference)
#   --compile_cache_dir COMPILE_CACHE_DIR
#                         Directory to cache compiled models (speeds up subsequent runs)
#   --amp                 Use Automatic Mixed Precision (AMP) for faster inference

uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --prompt_file examples/conceptual_prompts_64.txt \
    --num_samples 64 \
    --sampler dpm++_2m_karras \
    --steps 15 \
    --guidance_scale 9.0 \
    --use_esrgan \
    --esrgan_model RealESRGAN_x4plus \
    --clip_model ViT-L/14 ViT-B/32 ViT-B-32/laion2b_s34b_b79k ViT-L-14/laion2b_s32b_b82k \
    --save_all \
    --seed 43 \
    --compile \
    --compile_clip \
    --compile_mode reduce-overhead \
    --save_all \
    --batch_size 16 \
    --wandb \
    --wandb_project 'clip-glide-finetune'
