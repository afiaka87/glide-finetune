# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.
import os
from typing import Tuple

import torch as th
from glide_finetune.util import pred_to_pil
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from glide_text2im.tokenizer.bpe import Encoder


def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        tokens = th.tensor(tokens)  # + uncond_tokens)
        mask = th.tensor(mask, dtype=th.bool)  # + uncond_mask, dtype=th.bool)
        return tokens, mask


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
    glide_model.requires_grad_(True)  # unfreeze everything
    if activation_checkpointing:
        glide_model.use_checkpoint = True

    glide_model.requires_grad_(True)
    if freeze_transformer:
        glide_model.transformer.requires_grad_(False)
        glide_model.transformer_proj.requires_grad_(False)
        glide_model.token_embedding.requires_grad_(False)
        glide_model.padding_embedding.requires_grad_(False)
        glide_model.positional_embedding.requires_grad_(False)
    if freeze_diffusion:
        glide_model.out.requires_grad_(False)
        glide_model.input_blocks.requires_grad_(False)
        glide_model.middle_block.requires_grad_(False)
        glide_model.output_blocks.requires_grad_(False)
    if len(glide_path) > 0:  # user provided checkpoint
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(glide_path, map_location="cpu")
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        glide_model.load_state_dict(
            load_checkpoint("base", "cpu")
        )  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    return glide_model, glide_diffusion, options


# Sample from the base model.
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
    glide_model.del_cache()
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
                (x_t - eps * (beta**0.5))[:batch_size]
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
    glide_model.del_cache()
    glide_model.train()
    return samples
