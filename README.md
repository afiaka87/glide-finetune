# glide-finetune

[![Neverix Finetuning Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/neverix/glide-tuning)
Thanks to neverix from the EleutherAI discord for all the help with this!

Finetune the base 64 px GLIDE-text2im model from OpenAI on your own image-text dataset.

---

## Installation

```sh
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
python3 -m venv .venv # create a virtual environment to keep global install clean.
source .venv/bin/activate
(.venv) # optionally install pytorch manually for your own specific env first...
(.venv) python -m pip install -r requirements.txt
```

## Usage
```sh
usage: glide-finetune.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                         [--learning_rate LEARNING_RATE] [--dropout DROPOUT]
                         [--side_x SIDE_X] [--side_y SIDE_Y]
                         [--resize_ratio RESIZE_RATIO] [--uncond_p UNCOND_P]
                         [--resume_ckpt RESUME_CKPT]
                         [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                         [--device DEVICE] [--log_frequency LOG_FREQUENCY]
                         [--freeze_transformer] [--freeze_diffusion]
                         [--project_name PROJECT_NAME]
                         [--activation_checkpointing] [--use_captions]
                         [--epochs EPOCHS] [--test_prompt TEST_PROMPT]
                         [--test_batch_size TEST_BATCH_SIZE]
                         [--test_guidance_scale TEST_GUIDANCE_SCALE]
                         [--use_sgd]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --dropout DROPOUT, -drop DROPOUT
  --side_x SIDE_X, -x SIDE_X
  --side_y SIDE_Y, -y SIDE_Y
  --resize_ratio RESIZE_RATIO, -crop RESIZE_RATIO
  --uncond_p UNCOND_P, -p UNCOND_P
  --resume_ckpt RESUME_CKPT, -resume RESUME_CKPT
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
  --use_sgd, -sgd

```