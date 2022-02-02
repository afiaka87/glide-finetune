import numpy as np
from PIL import Image
import wandb
import torch as th
from glide_text2im.download import load_checkpoint
import os
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def pred_to_pil(pred: th.Tensor) -> Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def load_base_model(
    glide_path:str='',
    use_fp16:bool=False,
    dropout: float=0.1,
    timestep_respacing: str='1000',
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,):

    options = model_and_diffusion_defaults()
    options['use_fp16'] = use_fp16
    options['dropout'] = dropout
    options['timestep_respacing'] = timestep_respacing
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    # if use_fp16:
    #     glide_model.convert_to_fp16()
    # if freeze_transformer:
    #     glide_model.requires_grad_(True)
    #     glide_model.transformer.requires_grad_(False) # freeze transformer
    # elif freeze_diffusion:
    #     glide_model.requires_grad_(False) # freeze everything,
    #     glide_model.transformer.requires_grad_(True) # then unfreeze transformer
    #     glide_model.transformer_proj.requires_grad_(True)
    #     glide_model.token_embedding.requires_grad_(True)
    #     glide_model.positional_embedding.requires_grad_(True)
    #     glide_model.padding_embedding.requires_grad_(True)
    #     glide_model.final_ln.requires_grad_(True)
    # else:
    if activation_checkpointing:
        glide_model.use_checkpoint = True
    glide_model.requires_grad_(True) # unfreeze everything
    if len(glide_path) > 0: # user provided checkpoint
        assert os.path.exists(glide_path), 'glide path does not exist'
        weights = th.load(glide_path, map_location='cpu')
        glide_model.load_state_dict(weights)
    else: # use default checkpoint from openai
        glide_model.load_state_dict(load_checkpoint('base', 'cpu'))
    return glide_model, glide_diffusion, options


# Sample from the base model.
def sample(model, eval_diffusion, options, side_x, side_y, prompt='', batch_size=1, guidance_scale=4, device='cpu'):
    model.del_cache()

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    with th.inference_mode():
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            beta = eval_diffusion.betas[int(ts.flatten()[0].item() / options["diffusion_steps"] * len(eval_diffusion.betas))]
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            current_prediction_pil = pred_to_pil((x_t - eps * (beta ** 0.5))[:batch_size])
            current_prediction_pil.save('current_prediction.png')
            return th.cat([eps, rest], dim=1)

        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    model.del_cache()
    return samples



def wandb_setup(
    batch_size: int,
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
            "side_x": side_x,
            "side_y": side_y,
            "learning_rate": learning_rate,
            "use_fp16": use_fp16,
            "device": device,
            "data_dir": data_dir,
            "base_dir": base_dir,
        },
    )


"""Exponential moving average for PyTorch. Adapted from
https://www.zijianhu.com/post/pytorch/ema/.
"""

from copy import deepcopy

from torch import nn


class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.model = model
        self.decay = decay
        self.average = deepcopy(self.model)
        for param in self.average.parameters():
            param.detach_()

    @th.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError('Update should only be called during training')

        model_params = dict(self.model.named_parameters())
        average_params = dict(self.average.named_parameters())
        assert model_params.keys() == average_params.keys()

        for name, param in model_params.items():
            average_params[name].mul_(self.decay)
            average_params[name].add_((1 - self.decay) * param)

        model_buffers = dict(self.model.named_buffers())
        average_buffers = dict(self.average.named_buffers())
        assert model_buffers.keys() == average_buffers.keys()

        for name, buffer in model_buffers.items():
            average_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.average(*args, **kwargs)