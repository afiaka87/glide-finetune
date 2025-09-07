# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Fine-tune and evaluate GLIDE text-to-image diffusion models with a modern CLI interface.

--- 

## Features

- 🎨 **Modern CLI**: Clean command-line interface built with Typer
- 🚀 **Advanced Samplers**: Euler, Euler-A, DPM++, PLMS, and DDIM
- 🎯 **CLIP Re-ranking**: Generate multiple candidates and select the best
- 📊 **WebDataset Support**: Train on large-scale datasets like LAION
- 🔧 **LoRA Support**: Parameter-efficient fine-tuning
- 📈 **W&B Integration**: Automatic experiment tracking
- ⚡ **Performance**: Gradient accumulation, mixed precision, torch.compile support

## Installation

### Using uv (Recommended)
```bash
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
uv sync
uv pip install -e .
```

### Using pip
```bash
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Generate Images

```bash
# Generate a single image
glide eval generate base.pt sr.pt \
  --prompt "a serene mountain landscape at sunset" \
  --cfg 4.0 \
  --seed 42

# Generate from prompt file with CLIP re-ranking
glide eval generate base.pt sr.pt \
  --prompt-file prompts.txt \
  --clip-rerank \
  --clip-candidates 32 \
  --clip-top-k 8
```

### Train Models

```bash
# Train base model (64x64)
glide train base /path/to/dataset \
  --batch-size 4 \
  --lr 1e-4 \
  --wandb my-project

# Train upsampler (64→256)
glide train upsampler /path/to/dataset \
  --upscale 4 \
  --lr 5e-5 \
  --wandb my-upsampler
```

## CLI Usage

The GLIDE CLI provides two main commands: `train` and `eval`.

### Training Commands

#### Train Base Model (64x64)
```bash
glide train base /path/to/dataset \
  --batch-size 4 \
  --lr 1e-4 \
  --epochs 100 \
  --uncond-p 0.2 \  # Classifier-free guidance
  --wandb my-project \
  --fp16 \  # Mixed precision
  --grad-ckpt  # Gradient checkpointing
```

#### Train with LoRA (Efficient Fine-tuning)
```bash
glide train base /path/to/dataset \
  --lora \
  --lora-rank 8 \
  --lora-alpha 32 \
  --freeze-transformer \
  --wandb lora-finetune
```

#### Train on WebDataset (LAION)
```bash
glide train base /mnt/laion/*.tar \
  --webdataset \
  --batch-size 8 \
  --lr 1e-4 \
  --fp16 \
  --grad-ckpt
```

### Evaluation Commands

#### Generate with Advanced Samplers
```bash
# DPM++ for better quality with fewer steps
glide eval generate base.pt sr.pt \
  --prompt "a masterpiece painting" \
  --sampler dpm++ \
  --base-steps 20 \
  --sr-steps 15

# Euler for fast generation
glide eval generate base.pt sr.pt \
  --prompt "futuristic city" \
  --sampler euler \
  --cfg 3.0
```

#### CLIP Re-ranking for Quality
```bash
glide eval generate base.pt sr.pt \
  --prompt-file artistic_prompts.txt \
  --clip-rerank \
  --clip-model ViT-L-14/laion2b_s32b_b82k \
  --clip-candidates 64 \
  --clip-top-k 4
```

#### Compare Models
```bash
glide eval compare \
  model1_base.pt model1_sr.pt \
  model2_base.pt model2_sr.pt \
  "a test prompt" \
  --seed 42
```

## Advanced Features

### Samplers
- **euler**: Fast deterministic ODE solver
- **euler_a**: Euler with ancestral sampling
- **dpm++**: DPM-Solver++ for fewer steps
- **plms**: Pseudo Linear Multi-Step
- **ddim**: Denoising Diffusion Implicit Models

### CLIP Re-ranking
Generate multiple candidates and select the best using CLIP:
- Supports OpenCLIP models (ViT-L-14/laion2b_s32b_b82k recommended)
- Memory-efficient GPU offloading
- Batch processing for speed

### Performance Optimizations
- **Gradient Accumulation**: Larger effective batch sizes
- **Mixed Precision**: FP16/BF16 training
- **Gradient Checkpointing**: Trade compute for memory
- **torch.compile**: Optimized inference
- **LoRA**: Parameter-efficient fine-tuning

## Legacy Script Usage

The original training scripts are still available:

### Train Base Model (Traditional)
```bash
python train_glide.py \
  --data_dir '/path/to/dataset' \
  --train_upsample False \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.2 \
  --wandb_project_name 'my_project'
```

### Train Upsampler (Traditional)
```bash
python train_glide.py \
  --data_dir '/path/to/dataset' \
  --train_upsample True \
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0
```


## Full Usage
```
usage: train.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE]
                [--adam_weight_decay ADAM_WEIGHT_DECAY] [--side_x SIDE_X]
                [--side_y SIDE_Y] [--resize_ratio RESIZE_RATIO]
                [--uncond_p UNCOND_P] [--train_upsample]
                [--resume_ckpt RESUME_CKPT]
                [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                [--device DEVICE] [--log_frequency LOG_FREQUENCY]
                [--freeze_transformer] [--freeze_diffusion]
                [--project_name PROJECT_NAME] [--activation_checkpointing]
                [--use_captions] [--epochs EPOCHS] [--test_prompt TEST_PROMPT]
                [--test_batch_size TEST_BATCH_SIZE]
                [--test_guidance_scale TEST_GUIDANCE_SCALE] [--use_webdataset]
                [--wds_image_key WDS_IMAGE_KEY]
                [--wds_caption_key WDS_CAPTION_KEY]
                [--wds_dataset_name WDS_DATASET_NAME] [--seed SEED]
                [--cudnn_benchmark] [--upscale_factor UPSCALE_FACTOR]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --adam_weight_decay ADAM_WEIGHT_DECAY, -adam_wd ADAM_WEIGHT_DECAY
  --side_x SIDE_X, -x SIDE_X
  --side_y SIDE_Y, -y SIDE_Y
  --resize_ratio RESIZE_RATIO, -crop RESIZE_RATIO
                        Crop ratio
  --uncond_p UNCOND_P, -p UNCOND_P
                        Probability of using the empty/unconditional token
                        instead of a caption. OpenAI used 0.2 for their
                        finetune.
  --train_upsample, -upsample
                        Train the upsampling type of the model instead of the
                        base model.
  --resume_ckpt RESUME_CKPT, -resume RESUME_CKPT
                        Checkpoint to resume from
  --checkpoints_dir CHECKPOINTS_DIR, -ckpt CHECKPOINTS_DIR
  --use_fp16, -fp16
  --device DEVICE, -dev DEVICE
  --log_frequency LOG_FREQUENCY, -freq LOG_FREQUENCY
  --freeze_transformer, -fz_xt
  --freeze_diffusion, -fz_unet
  --project_name PROJECT_NAME, -name PROJECT_NAME
  --activation_checkpointing, -grad_ckpt
  --use_captions, -txt
  --epochs EPOCHS, -epochs EPOCHS
  --test_prompt TEST_PROMPT, -prompt TEST_PROMPT
  --test_batch_size TEST_BATCH_SIZE, -tbs TEST_BATCH_SIZE
                        Batch size used for model eval, not training.
  --test_guidance_scale TEST_GUIDANCE_SCALE, -tgs TEST_GUIDANCE_SCALE
                        Guidance scale used during model eval, not training.
  --use_webdataset, -wds
                        Enables webdataset (tar) loading
  --wds_image_key WDS_IMAGE_KEY, -wds_img WDS_IMAGE_KEY
                        A 'key' e.g. 'jpg' used to access the image in the
                        webdataset
  --wds_caption_key WDS_CAPTION_KEY, -wds_cap WDS_CAPTION_KEY
                        A 'key' e.g. 'txt' used to access the caption in the
                        webdataset
  --wds_dataset_name WDS_DATASET_NAME, -wds_name WDS_DATASET_NAME
                        Name of the webdataset to use (laion or alamy)
  --seed SEED, -seed SEED
  --cudnn_benchmark, -cudnn
                        Enable cudnn benchmarking. May improve performance.
                        (may not)
  --upscale_factor UPSCALE_FACTOR, -upscale UPSCALE_FACTOR
                        Upscale factor for training the upsampling model only
```
