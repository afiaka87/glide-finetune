# glide-finetune

[colab](https://github.com/eliohead/glide-finetune-colab)

Finetune GLIDE-text2im on your own image-text dataset.

--- 

- finetune the upscaler as well.
- drop-in support for laion and alamy.


## Installation

```sh
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
python3 -m venv .venv # create a virtual environment to keep global install clean.
source .venv/bin/activate
(.venv) # optionally install pytorch manually for your own specific env first...
(.venv) python -m pip install -r requirements.txt
```

## Example usage

### Finetune the base model


The base model should be tuned for "classifier free guidance". This means you want to randomly replace captions with an unconditional (empty) token about 20% of the time. This is controlled by the argument `--uncond_p`, which is set to 0.2 by default and should only be disabled for the upsampler.

```sh
python train_glide.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample False \
  --project_name 'base_tuning_wandb' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --resize_ratio 1.0 \
  --uncond_p 0.2 \
  --resume_ckpt 'ckpt_to_resume_from.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory' \
```

### Finetune the prompt-aware super-res model (stage 2 of generating)

Note that the `--side_x` and `--side_y` args here should still be 64. They are scaled to 256 after mutliplying by the upscaling factor (4, by default.)

```sh
python train_glide.py \
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

### Finetune on LAION or alamy (webdataset)

I have written data loaders for both LAION2B and Alamy. Other webdatasets may require custom caption/image keys.

```sh
python train_glide.py \
  --data_dir '/folder/with/tars/in/it/' \
  --wds_caption_key 'txt' \
  --wds_image_key 'jpg' \
  --wds_dataset_name 'laion' \
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
