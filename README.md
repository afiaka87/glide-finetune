# glide-finetune

Finetune the base 64 px GLIDE-text2im model from OpenAI on your own image-text dataset.

Presently has all sorts of issues that are challenging to debug. 

## Installation


```sh
sudo apt install libopenmpi-dev # On macOS: brew install mpich
git clone https://github.com/afiaka87/glide-finetune.git
cd glide-finetune/
python3 -m venv .venv # create a virtual environment to keep global install clean.
source .venv/bin/activate
(.venv) # optionally install pytorch manually for your own specific env first...
(.venv) python -m pip install -r requirements.txt
```
### bitsandbytes
```sh
# choices: {cuda92, cuda 100, cuda101, cuda102, cuda110, cuda111, cuda113}
# replace XXX with the respective number
pip install bitsandbytes-cudaXXX
```
If you prefer to use `torch.optim.Adam`, you can just change `bnb` to `th` in the code.

## Usage
```sh
usage: glide-finetune.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                        [--grad_acc GRAD_ACC]
                        [--guidance_scale GUIDANCE_SCALE]
                        [--learning_rate LEARNING_RATE] [--dropout DROPOUT]
                        [--timestep_respacing TIMESTEP_RESPACING]
                        [--side_x SIDE_X] [--side_y SIDE_Y]
                        [--resize_ratio RESIZE_RATIO] [--uncond_p UNCOND_P]
                        [--resume_ckpt RESUME_CKPT]
                        [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                        [--device DEVICE] [--freeze_transformer]
                        [--freeze_diffusion] [--weight_decay WEIGHT_DECAY]
                        [--project_name PROJECT_NAME]
```
