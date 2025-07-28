# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Finetune GLIDE-text2im on your own image-text dataset.

--- 

## Features

- Finetune both base model (64x64) and upsampler (64x64 → 256x256)
- Memory-efficient 8-bit AdamW optimizer support
- TensorFloat-32 (TF32) support for faster training on Ampere GPUs
- WebDataset support with custom filtering modes
- Gradient checkpointing for reduced memory usage
- Early stopping for testing and integration
- Built-in Weights & Biases (wandb) logging
- Drop-in support for LAION and Alamy datasets
- Modern diffusion samplers with memory-optimized implementations


## Installation

```sh
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# To run commands, use uv run:
# uv run python train_glide.py --help
```

## Example usage

### Finetune the base model

The base model should be tuned for "classifier free guidance". This means you want to randomly replace captions with an unconditional (empty) token about 20% of the time. This is controlled by the argument `--uncond_p`, which is set to 0.2 by default and should only be disabled for the upsampler.

```sh
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --resize_ratio 1.0 \
  --uncond_p 0.2 \
  --checkpoints_dir 'my_local_checkpoint_directory'

# With memory optimizations
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --use_8bit_adam \
  --activation_checkpointing \
  --checkpoints_dir './finetune_checkpoints'
```

### Finetune the prompt-aware super-res model (stage 2 of generating)

Note that the `--side_x` and `--side_y` args here should still be 64. They are scaled to 256 after mutliplying by the upscaling factor (4, by default.)

```sh
uv run python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample True \
  --image_to_upsample 'low_res_face.png'
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0 \
  --resume_ckpt 'ckpt_to_resume_from.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory' \
```

### Finetune on WebDataset (LAION, Alamy, or custom)

The project supports WebDataset format for efficient large-scale training.

```sh
# For LAION dataset (applies metadata filtering)
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion'

# For custom WebDataset (no filtering, faster loading)
uv run python train_glide.py \
  --data_dir '/folder/with/tars/' \
  --use_webdataset \
  --wds_caption_key 'txt' \
  --wds_image_key 'png' \
  --wds_dataset_name 'webdataset'
```

## Diffusion Samplers

The project includes several modern diffusion sampling algorithms:

### Available Samplers

- **`plms`** (default): Pseudo Linear Multi-Step method
  - ✓ Stable and well-tested with GLIDE
  - ✓ Good quality at reasonable step counts
  - ✗ Not the fastest option available

- **`ddim`**: Denoising Diffusion Implicit Models
  - ✓ Deterministic when eta=0 (reproducible results)
  - ✓ Good for testing and comparisons
  - ✗ Can be slower than newer methods

- **`euler`**: Euler method (first-order ODE solver)
  - ✓ Fast generation
  - ✓ Good quality with 20-50 steps
  - ✓ Low memory usage
  - ✗ Less stable at very low step counts

- **`euler_a`**: Euler Ancestral (adds noise at each step)
  - ✓ More variation and "creativity" in outputs
  - ✓ Good for exploration and diverse results
  - ✗ Non-convergent (more steps ≠ better quality)
  - ✗ Results vary even with same seed

- **`dpm++_2m`**: DPM++ 2nd order multistep
  - ✓ Excellent quality/speed balance
  - ✓ Stable at low step counts (10-20 steps)
  - ✗ Slightly higher memory usage than Euler

- **`dpm++_2m_karras`**: DPM++ 2M with Karras noise schedule
  - ✓ Best quality at very low step counts (10-15 steps)
  - ✓ Improved color and detail preservation
  - ✓ Recommended for fast inference
  - ✗ Slightly higher computational cost

### Usage Example

```sh
# Train with Euler sampler for faster iteration during development
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --sampler euler \
  --test_guidance_scale 3.0 \
  --test_steps 50

# Train with DPM++ 2M Karras for best quality evaluation
uv run python train_glide.py \
  --data_dir '/path/to/data' \
  --sampler dpm++_2m_karras \
  --test_guidance_scale 4.0 \
  --test_steps 20
```


## Full Usage
```
usage: train_glide.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                      [--learning_rate LEARNING_RATE]
                      [--adam_weight_decay ADAM_WEIGHT_DECAY]
                      [--side_x SIDE_X] [--side_y SIDE_Y]
                      [--resize_ratio RESIZE_RATIO] [--uncond_p UNCOND_P]
                      [--train_upsample] [--resume_ckpt RESUME_CKPT]
                      [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                      [--device DEVICE] [--log_frequency LOG_FREQUENCY]
                      [--freeze_transformer] [--freeze_diffusion]
                      [--project_name PROJECT_NAME]
                      [--activation_checkpointing] [--use_captions]
                      [--epochs EPOCHS] [--test_prompt TEST_PROMPT]
                      [--test_batch_size TEST_BATCH_SIZE]
                      [--test_guidance_scale TEST_GUIDANCE_SCALE]
                      [--use_webdataset] [--wds_image_key WDS_IMAGE_KEY]
                      [--wds_caption_key WDS_CAPTION_KEY]
                      [--wds_dataset_name WDS_DATASET_NAME] [--seed SEED]
                      [--cudnn_benchmark] [--upscale_factor UPSCALE_FACTOR]
                      [--image_to_upsample IMAGE_TO_UPSAMPLE]
                      [--use_8bit_adam] [--use_tf32] [--early_stop EARLY_STOP]
                      [--sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}]
                      [--test_steps TEST_STEPS]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
                        Path to dataset directory
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size for training (default: 1)
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate (default: 2e-5)
  --adam_weight_decay ADAM_WEIGHT_DECAY, -adam_wd ADAM_WEIGHT_DECAY
                        Adam weight decay (default: 0.0)
  --side_x SIDE_X, -x SIDE_X
                        Width of training images (default: 64)
  --side_y SIDE_Y, -y SIDE_Y
                        Height of training images (default: 64)
  --resize_ratio RESIZE_RATIO, -crop RESIZE_RATIO
                        Crop ratio for training (default: 0.8)
  --uncond_p UNCOND_P, -p UNCOND_P
                        Probability of using empty/unconditional token instead
                        of caption. OpenAI used 0.2 for their finetune.
  --train_upsample, -upsample
                        Train the upsampling type of the model instead of the
                        base model.
  --resume_ckpt RESUME_CKPT, -resume RESUME_CKPT
                        Checkpoint to resume from
  --checkpoints_dir CHECKPOINTS_DIR, -ckpt CHECKPOINTS_DIR
                        Directory to save checkpoints
  --use_fp16, -fp16     Use mixed precision training
  --device DEVICE, -dev DEVICE
                        Device to use (e.g., cuda, cpu)
  --log_frequency LOG_FREQUENCY, -freq LOG_FREQUENCY
                        How often to log training progress
  --freeze_transformer, -fz_xt
                        Freeze transformer weights during training
  --freeze_diffusion, -fz_unet
                        Freeze diffusion model weights during training
  --project_name PROJECT_NAME, -name PROJECT_NAME
                        Weights & Biases project name
  --activation_checkpointing, -grad_ckpt
                        Enable gradient checkpointing to save memory
  --use_captions, -txt  Use captions during training
  --epochs EPOCHS, -epochs EPOCHS
                        Number of epochs to train (default: 20)
  --test_prompt TEST_PROMPT, -prompt TEST_PROMPT
                        Prompt to use for generating test images
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
                        Name of the webdataset to use (laion, alamy, or
                        webdataset for no filtering)
  --seed SEED, -seed SEED
                        Random seed for reproducibility
  --cudnn_benchmark, -cudnn
                        Enable cudnn benchmarking. May improve performance.
  --upscale_factor UPSCALE_FACTOR, -upscale UPSCALE_FACTOR
                        Upscale factor for training the upsampling model only
  --image_to_upsample IMAGE_TO_UPSAMPLE, -lowres IMAGE_TO_UPSAMPLE
                        Path to low-res image for upsampling visualization
  --use_8bit_adam, -8bit
                        Use 8-bit AdamW optimizer to save memory (requires
                        bitsandbytes)
  --use_tf32, -tf32     Enable TF32 on Ampere GPUs for faster training with
                        slightly reduced precision
  --early_stop EARLY_STOP
                        Stop training after this many steps (0 = disabled).
                        Useful for testing.
  --sampler {plms,ddim,euler,euler_a,dpm++_2m,dpm++_2m_karras}
                        Sampler to use for generating test images during
                        training. Options: plms (default) - stable, original
                        GLIDE sampler; ddim - deterministic when eta=0, good
                        for reproducibility; euler - fast first-order solver,
                        good quality; euler_a - euler with added noise, more
                        variation but non-convergent; dpm++_2m - second-order
                        solver, good quality/speed balance; dpm++_2m_karras -
                        dpm++_2m with improved schedule for low step counts
  --test_steps TEST_STEPS
                        Number of sampling steps for test image generation
                        (default: 100)
```
