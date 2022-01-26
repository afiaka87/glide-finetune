# glide-finetune

[![Neverix Finetuning Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/neverix/glide-tuning)

Finetune the base 64 px GLIDE-text2im model from OpenAI on your own image-text dataset.

Presently has all sorts of issues that are challenging to debug. 

Thanks to neverix from the EleutherAI discord for all the help with this!

---

## Checkpoints

- [glide-eleu-30k-steps.pt](https://www.dropbox.com/s/6htem47spd9cxzv/glide-eleu-30k-steps.pt) - GLIDE (small filtered) finetuned on generations from EleutherAI's #faraday-cage for 30,000 iterations with a batch size of 4.

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
                         [--timestep_respacing TIMESTEP_RESPACING]
                         [--side_x SIDE_X] [--side_y SIDE_Y]
                         [--resize_ratio RESIZE_RATIO] [--uncond_p UNCOND_P]
                         [--resume_ckpt RESUME_CKPT]
                         [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                         [--device DEVICE] [--log_frequency LOG_FREQUENCY]
                         [--freeze_transformer] [--freeze_diffusion]
                         [--project_name PROJECT_NAME]
                         [--activation_checkpointing] [--use_captions]
                         [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -data DATA_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --dropout DROPOUT, -drop DROPOUT
  --timestep_respacing TIMESTEP_RESPACING, -respace TIMESTEP_RESPACING
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

```
