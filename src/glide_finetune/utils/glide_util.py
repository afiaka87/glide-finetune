## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .freeze_utils import FreezeSummary

import sys

import numpy as np
import PIL
import torch as th
from torch import nn

from .train_util import pred_to_pil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "glide-text2im"))

from glide_finetune.enhanced_samplers import enhance_glide_diffusion

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

# Initialize logger
logger = get_logger("glide_finetune.glide_util")

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]


def get_uncond_tokens_mask(tokenizer: Encoder) -> tuple[th.Tensor, th.Tensor]:
    # Handle both API styles
    if hasattr(tokenizer, 'padded_tokens_and_mask'):
        uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
        return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)
    else:
        # SimpleTokenizer returns (tokens, length) not (tokens, mask)
        uncond_tokens, text_len = tokenizer.padded_tokens_and_len([], 128)
        # Create mask from length - True for valid tokens, False for padding
        uncond_mask = [i < text_len for i in range(128)]
        return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> tuple[th.Tensor, th.Tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    tokens = tokenizer.encode(prompt)
    
    # Handle both API styles
    if hasattr(tokenizer, 'padded_tokens_and_mask'):
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        mask = th.tensor(mask, dtype=th.bool)
    else:
        # SimpleTokenizer returns (tokens, length) not (tokens, mask)
        tokens, text_len = tokenizer.padded_tokens_and_len(tokens, context_len)
        # Create mask from length - True for valid tokens, False for padding
        mask = th.tensor([i < text_len for i in range(context_len)], dtype=th.bool)
    
    tokens = th.tensor(tokens)
    return tokens, mask


def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
) -> tuple[nn.Module, Any, dict[str, Any]]:
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

    # Don't set requires_grad here - let the freeze functions handle it properly
    # The old implementation was incomplete and didn't handle eval/train modes
    if len(glide_path) > 0:  # user provided checkpoint
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(glide_path, map_location="cpu", weights_only=False)
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        # Handle distributed downloading properly
        if accelerator is not None:
            if accelerator.is_main_process:
                # Main process downloads the weights
                weights = load_checkpoint(model_type, "cpu")
                glide_model.load_state_dict(weights)
            # All processes wait for main to finish
            accelerator.wait_for_everyone()
            if not accelerator.is_main_process:
                # Other processes load from cache (already downloaded)
                weights = load_checkpoint(model_type, "cpu")
                glide_model.load_state_dict(weights)
        else:
            # Single process - load normally
            glide_model.load_state_dict(
                load_checkpoint(model_type, "cpu")
            )  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        logger.info("Converted to fp16, likely gradients will explode")

    # Apply proper freezing after model is loaded
    _apply_freeze_settings(glide_model, freeze_transformer, freeze_diffusion)

    return glide_model, glide_diffusion, options


def _apply_freeze_settings(
    model: nn.Module, freeze_transformer_flag: bool, freeze_diffusion_flag: bool
) -> "FreezeSummary":
    """Apply freeze settings with proper eval/train modes using the new freeze policy."""
    from .freeze_utils import apply_freeze_policy

    # Apply the freeze policy with mutual exclusivity check
    return apply_freeze_policy(
        model,
        freeze_transformer=freeze_transformer_flag,
        freeze_diffusion=freeze_diffusion_flag,
    )

    # The summary is already logged by apply_freeze_policy


def read_image(path: str, shape: tuple[int, int]) -> th.Tensor:
    pil_img = PIL.Image.open(path).convert("RGB")
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


# Sample from the base model.


@th.inference_mode()
def sample_with_conditioning(
    glide_model: nn.Module,
    glide_options: dict[str, Any],
    side_x: int,
    side_y: int,
    prompt: str = "",
    clip_embeddings: th.Tensor | None = None,
    batch_size: int = 1,
    guidance_scale: float = 4,
    device: str | th.device = "cpu",
    prediction_respacing: str = "100",
    upsample_enabled: bool = False,
    image_to_upsample: str = "",
    upsample_temp: float = 0.997,
    sampler: str = "plms",
    num_steps: int | None = None,
    eta: float = 0.0,
    use_swinir: bool = False,
    swinir_model_type: str = "classical_sr_x4",
) -> th.Tensor:
    """Sample from GLIDE with optional CLIP conditioning.
    
    Args:
        glide_model: GLIDE model
        glide_options: Model options
        side_x: Image width
        side_y: Image height
        prompt: Text prompt (can be empty for CLIP-only conditioning)
        clip_embeddings: Optional CLIP embeddings for conditioning
        batch_size: Batch size
        guidance_scale: Classifier-free guidance scale
        device: Device to run on
        prediction_respacing: Timestep respacing
        upsample_enabled: Whether to use upsampling
        image_to_upsample: Path to image to upsample
        upsample_temp: Temperature for upsampling
        sampler: Sampling method
        num_steps: Number of sampling steps
        eta: DDIM eta parameter
        use_swinir: Whether to use SwinIR
        swinir_model_type: SwinIR model type
        
    Returns:
        Generated image tensor
    """
    glide_model.del_cache()

    # If num_steps is provided, override prediction_respacing
    if num_steps is not None:
        prediction_respacing = str(num_steps)

    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    
    # Create the text tokens to feed to the model.
    tokens = glide_model.tokenizer.encode(prompt) if prompt else []
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, glide_options["text_ctx"])

    # For true unconditional generation (guidance_scale <= 0), skip CFG entirely
    if guidance_scale <= 0:
        # Only compute unconditional output
        full_batch_size = batch_size
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
            [], glide_options["text_ctx"]
        )
        
        model_kwargs = {
            "tokens": th.tensor([uncond_tokens] * batch_size, device=device),
            "mask": th.tensor([uncond_mask] * batch_size, dtype=th.bool, device=device),
        }
        
        # Add zero CLIP embeddings for unconditional
        if clip_embeddings is not None:
            clip_embeddings = clip_embeddings.to(device)
            uncond_clip = th.zeros_like(clip_embeddings)
            model_kwargs["clip_embeddings"] = uncond_clip
        
        # Use model directly without CFG
        model_fn = glide_model
    else:
        # Standard CFG setup for guided generation
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
            [], glide_options["text_ctx"]
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = {
            "tokens": th.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device),
            "mask": th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        }
        
        # Add CLIP embeddings if provided
        if clip_embeddings is not None:
            # Duplicate for classifier-free guidance (conditioned and unconditioned)
            clip_embeddings = clip_embeddings.to(device)
            # For unconditioned, we use zero embeddings
            uncond_clip = th.zeros_like(clip_embeddings)
            model_kwargs["clip_embeddings"] = th.cat([clip_embeddings, uncond_clip], dim=0)

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
            # Apply CFG formula
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            # Don't save intermediate predictions during sampling
            return th.cat([eps, rest], dim=1)

        model_fn = cfg_model_fn  # so we use CFG for the base model.
    if upsample_enabled:
        assert image_to_upsample != "", "You must specify a path to an image to upsample."
        low_res_samples = read_image(image_to_upsample, size=(side_x, side_y))
        model_kwargs["low_res"] = low_res_samples
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_kwargs["noise"] = noise
        model_fn = glide_model  # just use the base model, no need for CFG.

    # Determine number of steps from prediction_respacing
    try:
        actual_num_steps = int(prediction_respacing)
    except ValueError:
        actual_num_steps = num_steps if num_steps is not None else 50  # Default fallback

    # Choose sampling method (rest is identical to original sample function)
    if sampler.lower() in ["euler", "euler_discrete"]:
        # Add enhanced samplers to diffusion instance
        enhance_glide_diffusion(eval_diffusion)
        samples = eval_diffusion.euler_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
            num_steps=actual_num_steps,
        )[:batch_size]
    elif sampler.lower() in ["euler_a", "euler_ancestral"]:
        enhance_glide_diffusion(eval_diffusion)
        samples = eval_diffusion.euler_ancestral_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
            num_steps=actual_num_steps,
        )[:batch_size]
    elif sampler.lower() in ["dpm++", "dpmpp", "dpm_plus_plus"]:
        enhance_glide_diffusion(eval_diffusion)
        samples = eval_diffusion.dpm_solver_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
            num_steps=actual_num_steps,
        )[:batch_size]
    elif sampler.lower() == "ddim":
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
        )[:batch_size]
    elif sampler.lower() == "plms":
        samples = eval_diffusion.plms_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
        )[:batch_size]
    else:
        # Default to p_sample_loop
        samples = eval_diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
        )[:batch_size]

    # Handle SwinIR upsampling if requested
    if use_swinir and not upsample_enabled:
        from glide_finetune.swinir.swinir_upsample import load_swinir_model, apply_swinir_sr
        swinir_model = load_swinir_model(swinir_model_type, device=device)
        samples = apply_swinir_sr(samples, swinir_model)

    return samples


def sample(
    glide_model: nn.Module,
    glide_options: dict[str, Any],
    side_x: int,
    side_y: int,
    prompt: str = "",
    batch_size: int = 1,
    guidance_scale: float = 4,
    device: str | th.device = "cpu",
    prediction_respacing: str = "100",
    upsample_enabled: bool = False,
    image_to_upsample: str = "",
    upsample_temp: float = 0.997,
    sampler: str = "plms",
    num_steps: int | None = None,
    eta: float = 0.0,
    use_swinir: bool = False,
    swinir_model_type: str = "classical_sr_x4",
) -> th.Tensor:
    glide_model.del_cache()

    # If num_steps is provided, override prediction_respacing
    if num_steps is not None:
        prediction_respacing = str(num_steps)

    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    # Create the text tokens to feed to the model.
    tokens = glide_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, glide_options["text_ctx"])

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
        [], glide_options["text_ctx"]
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = {
        "tokens": th.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device),
        "mask": th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    }

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
        current_prediction_pil = pred_to_pil((x_t - eps * (beta**0.5))[:batch_size])
        current_prediction_pil.save("current_prediction.png")
        return th.cat([eps, rest], dim=1)

    model_fn = cfg_model_fn  # so we use CFG for the base model.
    if upsample_enabled:
        assert image_to_upsample != "", "You must specify a path to an image to upsample."
        low_res_samples = read_image(image_to_upsample, size=(side_x, side_y))
        model_kwargs["low_res"] = low_res_samples
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_kwargs["noise"] = noise
        model_fn = glide_model  # just use the base model, no need for CFG.

    # Determine number of steps from prediction_respacing
    try:
        actual_num_steps = int(prediction_respacing)
    except ValueError:
        actual_num_steps = num_steps if num_steps is not None else 50  # Default fallback

    # Choose sampling method
    if sampler.lower() in ["euler", "euler_discrete"]:
        # Add enhanced samplers to diffusion instance
        enhance_glide_diffusion(eval_diffusion)
        samples = eval_diffusion.euler_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
            num_steps=actual_num_steps,
        )[:batch_size]

    elif sampler.lower() in ["euler_a", "euler_ancestral"]:
        # Add enhanced samplers to diffusion instance
        enhance_glide_diffusion(eval_diffusion)
        samples = eval_diffusion.euler_ancestral_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
            num_steps=actual_num_steps,
        )[:batch_size]

    elif sampler.lower() in ["dpm++", "dpm_solver", "dpmpp"]:
        # Add enhanced samplers to diffusion instance
        enhance_glide_diffusion(eval_diffusion)
        samples = eval_diffusion.dpm_solver_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
            num_steps=actual_num_steps,
            order=2,
        )[:batch_size]

    elif sampler.lower() == "ddim":
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            eta=eta,
        )[:batch_size]

    elif sampler.lower() == "plms":
        samples = eval_diffusion.plms_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]

    else:
        # Default to PLMS if unknown sampler
        logger.warning(f"Warning: Unknown sampler '{sampler}', falling back to PLMS")
        samples = eval_diffusion.plms_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    glide_model.del_cache()

    # Apply SwinIR upscaling if requested
    if use_swinir:
        from glide_finetune.swinir_upscaler import create_swinir_upscaler

        try:
            upscaler = create_swinir_upscaler(
                model_type=swinir_model_type,
                device=device,
                use_fp16=glide_model.dtype == th.float16,
            )
            # Upscale from 64x64 to 256x256 (4x)
            samples = upscaler(samples)
            logger.info(
                f"SwinIR upscaling applied: {side_x}x{side_y} -> {samples.shape[-1]}x{samples.shape[-2]}"
            )
        except Exception as e:
            logger.warning(f"Warning: SwinIR upscaling failed: {e}")
            logger.info("Returning original resolution samples")

    return samples
