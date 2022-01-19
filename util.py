import wandb
import torch as th
from glide_text2im.download import load_checkpoint
import os
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
#     """
#     Compute training losses for a single timestep.

#     :param model: the model to evaluate loss on.
#     :param x_start: the [N x C x ...] tensor of inputs.
#     :param t: a batch of timestep indices.
#     :param model_kwargs: if not None, a dict of extra keyword arguments to
#         pass to the model. This can be used for conditioning.
#     :param noise: if specified, the specific Gaussian noise to try to remove.
#     :return: a dict with the key "loss" containing a tensor of shape [N].
#                 Some me

def load_base_model(glide_path:str='', use_fp16:bool=False, dropout: float=0.1, timestep_respacing: str = '1000', freeze_transformer: bool = False, freeze_diffusion: bool = False):
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
    options['timestep_respacing'] = timestep_respacing
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if use_fp16:
        glide_model.convert_to_fp16()
    if len(glide_path) > 0: # user provided checkpoint
        assert os.path.exists(glide_path), 'glide path does not exist'
        weights = th.load(glide_path, map_location='cpu')
        glide_model.load_state_dict(weights)
    else: # use default checkpoint from openai
        glide_model.load_state_dict(load_checkpoint('base', 'cpu'))
    if freeze_transformer:
        glide_model.requires_grad_(True)
        glide_model.transformer.requires_grad_(False) # freeze transformer
    elif freeze_diffusion:
        glide_model.requires_grad_(False) # freeze everything,
        glide_model.transformer.requires_grad_(True) # then unfreeze transformer
    else:
        glide_model.requires_grad_(True) # unfreeze everything
    return glide_model, glide_diffusion, options


def prompt_to_model_kwargs(model: th.nn.Module, options: dict, prompt: str = '', _batch_size: int = 1, device: str = 'cpu') -> dict:
    """
    Tokenizes a prompt and returns a dictionary of model keyword args in the formate GLIDE expects.

    Args:
        model: The instance of a model to use. Must have a tokenize method.
        options: The default or modified options for the model.
        prompt: The prompt to tokenize.
        _batch_size: The batch size to use. The kwargs will use double this value to support unconditional/classifier-free guidance.
        device: The device to use. cpu or cuda.
    """
    prompt = prompt.lower()
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options['text_ctx'])
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])
    return dict(
        tokens=th.tensor(
            [tokens] * _batch_size + 
            [uncond_tokens] * _batch_size, 
            device=device
        ),
        mask=th.tensor(
            [mask] * _batch_size + 
            [uncond_mask] * _batch_size,
            dtype=th.bool,
            device=device
        ),
    )

def wandb_setup(
    batch_size: int,
    grad_acc: int,
    side_x: int,
    side_y: int,
    learning_rate: float,
    guidance_scale: float,
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
            "guidance_scale": guidance_scale,
            "use_fp16": use_fp16,
            "device": device,
            "data_dir": data_dir,
            "base_dir": base_dir,
        },
    )
