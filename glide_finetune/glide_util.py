## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
from typing import Tuple, Literal, Optional

import PIL
import numpy as np
import torch as th
from glide_finetune.train_util import pred_to_pil
from glide_finetune.enhanced_samplers import enhance_diffusion
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]


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


def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
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
            load_checkpoint(model_type, "cpu")
        )  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    return glide_model, glide_diffusion, options

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

# Sample from the base model.

@th.inference_mode()
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
    upsample_enabled=False,
    image_to_upsample='',
    upsample_temp=0.997,
    sampler: Literal["plms", "ddim", "euler", "euler_a", "dpm++"] = "plms",
    sampler_eta: float = 0.0,
    dpm_order: int = 2,
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
    uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask( [], glide_options["text_ctx"])

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        )
    )

    def cfg_model_fn(x_t, ts, **kwargs):
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

    model_fn = cfg_model_fn # so we use CFG for the base model.
    if upsample_enabled:
        assert image_to_upsample != '', "You must specify a path to an image to upsample."
        low_res_samples = read_image(image_to_upsample, size=(side_x, side_y))
        model_kwargs['low_res'] = low_res_samples
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_kwargs['noise'] = noise
        model_fn = glide_model # just use the base model, no need for CFG.

    # Choose the appropriate sampler
    if sampler == "plms":
        samples = eval_diffusion.plms_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    elif sampler == "ddim":
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            eta=sampler_eta,
        )[:batch_size]
    elif sampler == "euler":
        # Enhance diffusion with new samplers
        enhance_diffusion(eval_diffusion)
        samples = eval_diffusion.euler_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            eta=0.0,  # Euler is deterministic
        )[:batch_size]
    elif sampler == "euler_a":
        # Enhance diffusion with new samplers
        enhance_diffusion(eval_diffusion)
        samples = eval_diffusion.euler_ancestral_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            eta=sampler_eta if sampler_eta > 0 else 1.0,  # Default to stochastic
        )[:batch_size]
    elif sampler == "dpm++":
        # Enhance diffusion with new samplers
        enhance_diffusion(eval_diffusion)
        samples = eval_diffusion.dpm_solver_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            eta=0.0,  # DPM++ is deterministic
            order=dpm_order,
        )[:batch_size]
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
        
    glide_model.del_cache()
    return samples


@th.inference_mode()
def sample_with_superres(
    base_model,
    base_options,
    upsampler_model,
    upsampler_options,
    prompt="",
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    base_respacing="30",  # Fast sampling for base
    upsampler_respacing="17",  # Fast sampling for upsampler
    upsample_temp=0.997,
    sampler: Literal["plms", "ddim", "euler", "euler_a", "dpm++"] = "euler",  # Default to Euler
    sampler_eta: float = 0.0,
    dpm_order: int = 2,
):
    """
    Generate samples using the full pipeline: base model (64x64) -> upsampler (256x256).
    
    This function follows the approach from the GLIDE notebook, generating a 64x64
    image with the base model and then upscaling it to 256x256 with the upsampler.
    """
    # First, generate 64x64 samples with the base model
    base_model.del_cache()
    base_diffusion = create_gaussian_diffusion(
        steps=base_options["diffusion_steps"],
        noise_schedule=base_options["noise_schedule"],
        timestep_respacing=base_respacing,
    )
    
    # Create the text tokens to feed to the base model
    tokens = base_model.tokenizer.encode(prompt)
    tokens, mask = base_model.tokenizer.padded_tokens_and_mask(
        tokens, base_options["text_ctx"]
    )
    
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = base_model.tokenizer.padded_tokens_and_mask(
        [], base_options["text_ctx"]
    )
    
    # Pack the tokens together into model kwargs for base model
    base_model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        )
    )
    
    # Classifier-free guidance function for base model
    def base_cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = base_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)
    
    # Sample from base model using the selected sampler
    if sampler == "plms":
        base_samples = base_diffusion.plms_sample_loop(
            base_cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,  # Don't show progress for base model
            model_kwargs=base_model_kwargs,
            cond_fn=None,
        )[:batch_size]
    elif sampler == "ddim":
        base_samples = base_diffusion.ddim_sample_loop(
            base_cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=base_model_kwargs,
            cond_fn=None,
            eta=sampler_eta,
        )[:batch_size]
    elif sampler == "euler":
        enhance_diffusion(base_diffusion)
        base_samples = base_diffusion.euler_sample_loop(
            base_cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=base_model_kwargs,
            cond_fn=None,
            eta=0.0,
        )[:batch_size]
    elif sampler == "euler_a":
        enhance_diffusion(base_diffusion)
        base_samples = base_diffusion.euler_ancestral_sample_loop(
            base_cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=base_model_kwargs,
            cond_fn=None,
            eta=sampler_eta if sampler_eta > 0 else 1.0,
        )[:batch_size]
    elif sampler == "dpm++":
        enhance_diffusion(base_diffusion)
        base_samples = base_diffusion.dpm_solver_sample_loop(
            base_cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=base_model_kwargs,
            cond_fn=None,
            eta=0.0,
            order=dpm_order,
        )[:batch_size]
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    
    base_model.del_cache()
    
    # Now upsample the 64x64 samples to 256x256
    upsampler_model.del_cache()
    upsampler_diffusion = create_gaussian_diffusion(
        steps=upsampler_options["diffusion_steps"],
        noise_schedule=upsampler_options["noise_schedule"],
        timestep_respacing=upsampler_respacing,
    )
    
    # Prepare tokens for upsampler (same prompt)
    up_tokens = upsampler_model.tokenizer.encode(prompt)
    up_tokens, up_mask = upsampler_model.tokenizer.padded_tokens_and_mask(
        up_tokens, upsampler_options["text_ctx"]
    )
    
    # Create the model conditioning dict for upsampler
    upsampler_model_kwargs = dict(
        # Low-res image to upsample (normalize to [-1, 1])
        low_res=((base_samples + 1) * 127.5).round() / 127.5 - 1,
        # Text tokens
        tokens=th.tensor([up_tokens] * batch_size, device=device),
        mask=th.tensor([up_mask] * batch_size, dtype=th.bool, device=device),
    )
    
    # Sample from the upsampler (no CFG needed for upsampler)
    up_shape = (batch_size, 3, 256, 256)
    
    # Use PLMS for upsampler as it's fast and works well
    upsampled_samples = upsampler_diffusion.plms_sample_loop(
        upsampler_model,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=False,  # Don't show progress for upsampler
        model_kwargs=upsampler_model_kwargs,
        cond_fn=None,
    )[:batch_size]
    
    upsampler_model.del_cache()
    
    return upsampled_samples