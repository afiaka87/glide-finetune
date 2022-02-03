import os

import torch as th
import wandb
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from PIL import Image


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
    glide_path: str = "",
    use_fp16: bool = False,
    dropout: float = 0.1,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
):

    options = model_and_diffusion_defaults()
    options["use_fp16"] = use_fp16
    options["dropout"] = dropout
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True
    if freeze_transformer:
        glide_model.requires_grad_(False)
        glide_model.time_embed.requires_grad_(True)
        glide_model.input_blocks.requires_grad_(True)
        glide_model.middle_blocks.requires_grad_(True)
        glide_model.output_blocks.requires_grad_(True)
    elif freeze_diffusion:
        glide_model.requires_grad_(False)
        glide_model.transformer.requires_grad_(True)  # then unfreeze transformer
        glide_model.transformer_proj.requires_grad_(True)
        glide_model.token_embedding.requires_grad_(True)
        glide_model.positional_embedding.requires_grad_(True)
        glide_model.padding_embedding.requires_grad_(True)
        glide_model.final_ln.requires_grad_(True)
        glide_model.time_embed.requires_grad_(True)
    else:
        glide_model.requires_grad_(True)  # unfreeze everything
    if len(glide_path) > 0:  # user provided checkpoint
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(glide_path, map_location="cpu")
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        glide_model.load_state_dict(
            load_checkpoint("base", "cpu")
        )  # always load to cpu, saves memory
    return glide_model, glide_diffusion, options


# Sample from the base model.
@th.no_grad()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt="",
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
):
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    # Create the text tokens to feed to the model.
    tokens = glide_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(
        tokens, glide_options["text_ctx"]
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
        [], glide_options["text_ctx"]
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
            model_out = glide_model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            beta = eval_diffusion.betas[
                int(
                    ts.flatten()[0].item()
                    / glide_options["diffusion_steps"]
                    * len(eval_diffusion.betas)
                )
            ]
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            current_prediction_pil = pred_to_pil(
                (x_t - eps * (beta ** 0.5))[:batch_size]
            )
            current_prediction_pil.save("current_prediction.png")
            return th.cat([eps, rest], dim=1)

        samples = eval_diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
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
    project_name: str = "glide-text2im-finetune",
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