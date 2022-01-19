# glide-finetune

Finetune the base 64 px GLIDE-text2im model from OpenAI on your own image-text dataset.

## Installation
```sh
git clone https://github.com/afiaka87/GLIDE-Finetune.git
cd GLIDE-Finetune/
python3 -m venv .venv # create a virtual environment to keep global install clean.
source .venv/bin/activate
(.venv) # optionally install pytorch manually for your own specific env first...
(.venv) python -m pip install -r requirements.txt
```

## Usage
```sh
(.venv) python glide-finetune.py 
    --data_dir=./data \
    --batch_size=1 \
    --grad_acc=1 \
    --guidance_scale=4.0 \
    --learning_rate=2e-5 \
    --dropout=0.1 \
    --timestep_respacing=1000 \
    --side_x=64 \
    --side_y=64 \
    --resume_ckpt='' \
    --checkpoints_dir='./glide_checkpoints/' \
    --use_fp16 \
    --device='' \
    --freeze_transformer \
    --freeze_diffusion \
    --weight_decay=0.0 \
    --project_name='glide-finetune'
```


## Known issues:
- batching isn't handled in the dataloader
- NaN/Inf errors
- some of the code is messy, needs refactoring.