import PIL
from typing import Tuple
import wandb
import torch as th
from glide_text2im.download import load_checkpoint
import os
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def pred_to_pil(pred: th.Tensor) -> PIL.Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return PIL.Image.fromarray(reshaped.numpy())


def load_base_model(glide_path:str='', use_fp16:bool=False, dropout: float=0.1, freeze_transformer: bool = False, freeze_diffusion: bool = False):
    """
    Loads a base model and returns it and its options. Optionally specify custom checkpoint `glide_path`.

    Args:
        glide_path: The path to the checkpoint to load.
        use_fp16: Whether to use fp16.
        dropout: The dropout rate to use.
        timestep_respacing: The timestep respacing to use.
        freeze_transformer: Whether to freeze the transformer for captions.
        freeze_diffusion: Whether to freeze the diffusion for images.
    """
    options = model_and_diffusion_defaults()
    options['use_fp16'] = use_fp16
    options['dropout'] = dropout
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if use_fp16:
        glide_model.convert_to_fp16()
    if freeze_transformer:
        glide_model.requires_grad_(True)
        glide_model.transformer.requires_grad_(False) # freeze transformer
    elif freeze_diffusion:
        glide_model.requires_grad_(False) # freeze everything,
        glide_model.transformer.requires_grad_(True) # then unfreeze transformer
    else:
        glide_model.requires_grad_(True) # unfreeze everything
    if len(glide_path) > 0: # user provided checkpoint
        assert os.path.exists(glide_path), 'glide path does not exist'
        weights = th.load(glide_path, map_location='cpu')
        glide_model.load_state_dict(weights)
    else: # use default checkpoint from openai
        glide_model.load_state_dict(load_checkpoint('base', 'cpu'))
    return glide_model, glide_diffusion, options



def wandb_setup(
    batch_size: int,
    grad_acc: int,
    side_x: int,
    side_y: int,
    learning_rate: float,
    use_fp16: bool,
    device: str,
    data_dir: str,
    base_dir: str,
    project_name: str = 'glide-text2im-finetune',
):
    return wandb.init(
        project=project_name,
        config={
            "batch_size": batch_size,
            "grad_acc": grad_acc,
            "side_x": side_x,
            "side_y": side_y,
            "learning_rate": learning_rate,
            "use_fp16": use_fp16,
            "device": device,
            "data_dir": data_dir,
            "base_dir": base_dir,
        },
    )
