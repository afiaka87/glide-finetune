## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
from typing import Tuple, Literal, Optional, Union

import PIL
import numpy as np
import torch as th
from glide_finetune.enhanced_samplers import enhance_diffusion
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_latent,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]


def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.Tensor, th.Tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)  # type: ignore
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        tokens_tensor = th.tensor(tokens)  # + uncond_tokens)
        mask_tensor = th.tensor(mask, dtype=th.bool)  # + uncond_mask, dtype=th.bool)
        return tokens_tensor, mask_tensor


def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    precision: str = "fp32",  # "fp32", "fp16", "bf16"
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
    random_init: bool = False,
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    # Handle legacy use_fp16 parameter
    if use_fp16:
        precision = "fp16"
    options["use_fp16"] = precision == "fp16"
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True

    # Start with all parameters trainable
    glide_model.requires_grad_(True)

    # Freeze transformer components if requested
    if freeze_transformer:
        # Freeze the main transformer
        if hasattr(glide_model, "transformer"):
            glide_model.transformer.requires_grad_(False)

        # Freeze transformer projection layer
        if hasattr(glide_model, "transformer_proj"):
            glide_model.transformer_proj.requires_grad_(False)

        # Freeze token embedding
        if hasattr(glide_model, "token_embedding"):
            glide_model.token_embedding.requires_grad_(False)

        # Freeze positional and padding embeddings (these are Parameters, not Modules)
        if hasattr(glide_model, "padding_embedding"):
            glide_model.padding_embedding.requires_grad = False
        if hasattr(glide_model, "positional_embedding"):
            glide_model.positional_embedding.requires_grad = False

        # Freeze final layer norm (part of transformer output)
        if hasattr(glide_model, "final_ln"):
            glide_model.final_ln.requires_grad_(False)

    # Freeze diffusion/UNet components if requested
    if freeze_diffusion:
        # Freeze time embedding layers
        if hasattr(glide_model, "time_embed"):
            glide_model.time_embed.requires_grad_(False)

        # Freeze input blocks
        if hasattr(glide_model, "input_blocks"):
            glide_model.input_blocks.requires_grad_(False)

        # Freeze middle block
        if hasattr(glide_model, "middle_block"):
            glide_model.middle_block.requires_grad_(False)

        # Freeze output blocks
        if hasattr(glide_model, "output_blocks"):
            glide_model.output_blocks.requires_grad_(False)

        # Freeze final output projection
        if hasattr(glide_model, "out"):
            glide_model.out.requires_grad_(False)

        # Unfreeze cross-attention encoder_kv layers so the UNet can adapt
        # to new text representations when training the text encoder
        for name, param in glide_model.named_parameters():
            if "encoder_kv" in name:
                param.requires_grad = True

        # Note: label_emb doesn't exist in base model but might in other variants

    if random_init:
        print("Using random initialization (no pretrained weights)")
    elif len(glide_path) > 0:  # user provided checkpoint
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(glide_path, map_location="cpu")
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        if any(k.startswith("_orig_mod.") for k in weights):
            weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        glide_model.load_state_dict(
            load_checkpoint(model_type, th.device("cpu"))
        )  # always load to cpu, saves memory

    # Print parameter freeze status
    if freeze_transformer or freeze_diffusion:
        _print_freeze_status(glide_model, freeze_transformer, freeze_diffusion)

    if precision == "fp16":
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    elif precision == "bf16":
        glide_model.convert_to_bf16()
        print("Converted to bf16 for stable mixed precision training")
    return glide_model, glide_diffusion, options


def _print_freeze_status(model, freeze_transformer, freeze_diffusion):
    """Print detailed information about which parameters are frozen."""

    def count_params(module_or_param):
        if isinstance(module_or_param, th.nn.Parameter):
            return module_or_param.numel()
        return sum(p.numel() for p in module_or_param.parameters())

    def count_trainable_params(module_or_param):
        if isinstance(module_or_param, th.nn.Parameter):
            return module_or_param.numel() if module_or_param.requires_grad else 0
        return sum(p.numel() for p in module_or_param.parameters() if p.requires_grad)

    # Categorize parameters
    transformer_components = {
        "transformer": None,
        "transformer_proj": None,
        "token_embedding": None,
        "padding_embedding": None,
        "positional_embedding": None,
        "final_ln": None,
    }

    diffusion_components = {
        "time_embed": None,
        "input_blocks": None,
        "middle_block": None,
        "output_blocks": None,
        "out": None,
    }

    # Collect existing components
    for comp_name in transformer_components:
        if hasattr(model, comp_name):
            transformer_components[comp_name] = getattr(model, comp_name)

    for comp_name in diffusion_components:
        if hasattr(model, comp_name):
            diffusion_components[comp_name] = getattr(model, comp_name)

    # Count parameters
    print("\n" + "=" * 60)
    print("PARAMETER FREEZE STATUS")
    print("=" * 60)

    # Transformer components
    print("\n📝 TRANSFORMER COMPONENTS:")
    trans_total = 0
    trans_trainable = 0
    for name, component in transformer_components.items():
        if component is not None:
            total = count_params(component)
            trainable = count_trainable_params(component)
            trans_total += total
            trans_trainable += trainable
            status = "❄️ FROZEN" if trainable == 0 else "🔥 TRAINABLE"
            print(
                f"  {name:20s}: {total:>12,} params | {trainable:>12,} trainable | {status}"
            )

    # Diffusion components
    print("\n🎨 DIFFUSION/UNET COMPONENTS:")
    diff_total = 0
    diff_trainable = 0
    for name, component in diffusion_components.items():
        if component is not None:
            total = count_params(component)
            trainable = count_trainable_params(component)
            diff_total += total
            diff_trainable += trainable
            status = "❄️ FROZEN" if trainable == 0 else "🔥 TRAINABLE"
            print(
                f"  {name:20s}: {total:>12,} params | {trainable:>12,} trainable | {status}"
            )

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print(
        f"  Transformer: {trans_total:>15,} total | {trans_trainable:>15,} trainable ({100 * trans_trainable / trans_total if trans_total > 0 else 0:.1f}%)"
    )
    print(
        f"  Diffusion:   {diff_total:>15,} total | {diff_trainable:>15,} trainable ({100 * diff_trainable / diff_total if diff_total > 0 else 0:.1f}%)"
    )

    total_params = trans_total + diff_total
    trainable_params = trans_trainable + diff_trainable
    print("  " + "-" * 56)
    print(
        f"  TOTAL:       {total_params:>15,} total | {trainable_params:>15,} trainable ({100 * trainable_params / total_params if total_params > 0 else 0:.1f}%)"
    )

    if freeze_transformer and trans_trainable > 0:
        print(
            "\n⚠️  WARNING: freeze_transformer=True but some transformer params are still trainable!"
        )
    if freeze_diffusion and diff_trainable > 0:
        print(
            "\n⚠️  WARNING: freeze_diffusion=True but some diffusion params are still trainable!"
        )

    print("=" * 60 + "\n")


def load_model_with_lora(
    glide_path: str = "",
    use_fp16: bool = False,
    precision: str = "fp32",  # "fp32", "fp16", "bf16"
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_mode: str = "attention",
    lora_resume: str = "",
    device: Union[str, th.device] = "cpu",
):
    """
    Load GLIDE model with optional LoRA support.

    Args:
        Standard load_model arguments plus:
        use_lora: Enable LoRA adaptation
        lora_rank: Rank of LoRA decomposition
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout for LoRA layers
        lora_target_mode: Which modules to target
        lora_resume: Path to resume LoRA adapter from
        device: Device to load model on

    Returns:
        model, diffusion, options (model is PEFT model if LoRA enabled)
    """
    # Load base model
    glide_model, glide_diffusion, options = load_model(
        glide_path=glide_path,
        use_fp16=use_fp16,
        precision=precision,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type=model_type,
    )

    if use_lora:
        from glide_finetune.lora_wrapper import (
            apply_lora_to_glide,
            load_lora_checkpoint,
        )

        # If resuming from LoRA checkpoint
        if lora_resume and os.path.exists(lora_resume):
            print(f"Loading LoRA adapter from {lora_resume}")
            glide_model = load_lora_checkpoint(
                glide_model, lora_resume, device=device, is_trainable=True
            )
        else:
            # Apply new LoRA configuration
            print(
                f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}, mode={lora_target_mode}"
            )
            glide_model = apply_lora_to_glide(
                glide_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_mode=lora_target_mode,
                verbose=True,
            )
            glide_model = glide_model.to(device)

    return glide_model, glide_diffusion, options


def load_latent_model(
    glide_path: str = "",
    precision: str = "fp32",
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    init_from_pixel: str = "",
):
    """Load a LatentText2ImUNet model for latent-space diffusion.

    Args:
        glide_path: Path to a latent-mode checkpoint to resume from.
        precision: "fp32", "fp16", or "bf16".
        freeze_transformer: Freeze the GLIDE text transformer.
        freeze_diffusion: Freeze the UNet diffusion backbone.
        activation_checkpointing: Enable gradient checkpointing.
        init_from_pixel: Path to a pixel-space checkpoint for weight transfer.
    """
    options = model_and_diffusion_defaults_latent()
    options["use_fp16"] = precision == "fp16"
    glide_model, glide_diffusion = create_model_and_diffusion(**options)

    if activation_checkpointing:
        glide_model.use_checkpoint = True

    glide_model.requires_grad_(True)

    if glide_path and os.path.exists(glide_path):
        weights = th.load(glide_path, map_location="cpu")
        if any(k.startswith("_orig_mod.") for k in weights):
            weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
        glide_model.load_state_dict(weights)
    elif init_from_pixel and os.path.exists(init_from_pixel):
        from glide_finetune.latent_util import init_latent_from_pixel

        pixel_weights = th.load(init_from_pixel, map_location="cpu")
        init_latent_from_pixel(glide_model, pixel_weights)
    else:
        print(
            "Warning: No checkpoint provided for latent model. Using random initialization."
        )

    # Freeze transformer if requested
    if freeze_transformer:
        for attr in ("transformer", "transformer_proj", "token_embedding", "final_ln"):
            if hasattr(glide_model, attr):
                getattr(glide_model, attr).requires_grad_(False)
        for attr in ("padding_embedding", "positional_embedding"):
            if hasattr(glide_model, attr):
                getattr(glide_model, attr).requires_grad = False

    # Freeze diffusion backbone if requested
    if freeze_diffusion:
        for attr in (
            "time_embed",
            "input_blocks",
            "middle_block",
            "output_blocks",
            "out",
        ):
            if hasattr(glide_model, attr):
                getattr(glide_model, attr).requires_grad_(False)
        # Keep cross-attention encoder_kv trainable
        for name, param in glide_model.named_parameters():
            if "encoder_kv" in name:
                param.requires_grad = True

    if freeze_transformer or freeze_diffusion:
        _print_freeze_status(glide_model, freeze_transformer, freeze_diffusion)

    if precision == "fp16":
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    elif precision == "bf16":
        glide_model.convert_to_bf16()
        print("Converted to bf16 for stable mixed precision training")

    return glide_model, glide_diffusion, options


def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert("RGB")
    pil_img = pil_img.resize(shape, resample=PIL.Image.Resampling.BICUBIC)
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
    image_to_upsample="",
    upsample_temp=0.997,
    sampler: Literal["plms", "ddim", "euler", "euler_a", "dpm++"] = "plms",
    sampler_eta: float = 0.0,
    dpm_order: int = 2,
    latent_mode: bool = False,
    vae=None,
    clip_encoder=None,
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

    # For latent mode, add CLIP embeddings to model_kwargs
    if latent_mode and clip_encoder is not None:
        clip_emb = clip_encoder.encode_text_batch([prompt] * batch_size).to(device)
        uncond_clip_emb = th.zeros_like(clip_emb)
        model_kwargs["clip_emb"] = th.cat([clip_emb, uncond_clip_emb], dim=0)

    num_channels = 4 if latent_mode else 3
    sample_shape = (full_batch_size, num_channels, side_y, side_x)

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        C = x_t.shape[1]
        eps, rest = model_out[:, :C], model_out[:, C:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    model_fn = cfg_model_fn  # so we use CFG for the base model.
    if upsample_enabled:
        assert image_to_upsample != "", (
            "You must specify a path to an image to upsample."
        )
        low_res_samples = read_image(image_to_upsample, shape=(side_x, side_y))
        # Repeat low_res_samples to match full_batch_size (which is batch_size * 2 for CFG)
        # Even though we're not using CFG for upsampling, the tokens are still doubled
        if low_res_samples.shape[0] == 1:
            low_res_samples = low_res_samples.repeat(full_batch_size, 1, 1, 1)
        model_kwargs["low_res"] = low_res_samples.to(device)
        # For upsampling, the output resolution is 256x256
        output_side = side_x * 4  # Upsampling factor is 4 (64->256)
        # The noise will be generated by the sampler at the correct resolution
        model_fn = glide_model  # just use the base model, no need for CFG.
        # Update side_x and side_y to output dimensions for the sampling loop
        side_x = output_side
        side_y = output_side

    # Choose the appropriate sampler
    if sampler == "plms":
        samples = eval_diffusion.plms_sample_loop(
            model_fn,
            sample_shape,
            device=device,
            clip_denoised=not latent_mode,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    elif sampler == "ddim":
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            sample_shape,
            device=device,
            clip_denoised=not latent_mode,
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
            sample_shape,
            device=device,
            clip_denoised=not latent_mode,
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
            sample_shape,
            device=device,
            clip_denoised=not latent_mode,
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
            sample_shape,
            device=device,
            clip_denoised=not latent_mode,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            eta=0.0,  # DPM++ is deterministic
            order=dpm_order,
        )[:batch_size]
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    glide_model.del_cache()
    if latent_mode and vae is not None:
        samples = vae.decode(samples)
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
    sampler: Literal[
        "plms", "ddim", "euler", "euler_a", "dpm++"
    ] = "euler",  # Default to Euler
    base_sampler: Optional[
        Literal["plms", "ddim", "euler", "euler_a", "dpm++"]
    ] = None,  # Sampler for base model (defaults to sampler)
    upsampler_sampler: Optional[
        Literal["plms", "ddim", "euler", "euler_a", "dpm++"]
    ] = None,  # Sampler for upsampler (defaults to sampler)
    sampler_eta: float = 0.0,
    dpm_order: int = 2,
):
    """
    Generate samples using the full pipeline: base model (64x64) -> upsampler (256x256).

    This function follows the approach from the GLIDE notebook, generating a 64x64
    image with the base model and then upscaling it to 256x256 with the upsampler.

    Args:
        sampler: Default sampler to use for both models if base_sampler/upsampler_sampler not specified
        base_sampler: Specific sampler for base model (overrides sampler)
        upsampler_sampler: Specific sampler for upsampler (overrides sampler)
    """
    # Use specific samplers if provided, otherwise fall back to default sampler
    base_sampler = base_sampler or sampler
    upsampler_sampler = upsampler_sampler or sampler
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
        ),
    )

    # Classifier-free guidance function for base model
    def base_cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = base_model(combined, ts, **kwargs)
        C = x_t.shape[1]
        eps, rest = model_out[:, :C], model_out[:, C:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    # Sample from base model using the selected sampler
    if base_sampler == "plms":
        base_samples = base_diffusion.plms_sample_loop(
            base_cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=False,  # Don't show progress for base model
            model_kwargs=base_model_kwargs,
            cond_fn=None,
        )[:batch_size]
    elif base_sampler == "ddim":
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
    elif base_sampler == "euler":
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
    elif base_sampler == "euler_a":
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
    elif base_sampler == "dpm++":
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
        raise ValueError(f"Unknown sampler: {base_sampler}")

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

    # Sample from upsampler using the selected sampler
    if upsampler_sampler == "plms":
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
    elif upsampler_sampler == "ddim":
        upsampled_samples = upsampler_diffusion.ddim_sample_loop(
            upsampler_model,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=upsampler_model_kwargs,
            cond_fn=None,
            eta=sampler_eta,
        )[:batch_size]
    elif upsampler_sampler == "euler":
        enhance_diffusion(upsampler_diffusion)
        upsampled_samples = upsampler_diffusion.euler_sample_loop(
            upsampler_model,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=upsampler_model_kwargs,
            cond_fn=None,
            eta=0.0,
        )[:batch_size]
    elif upsampler_sampler == "euler_a":
        enhance_diffusion(upsampler_diffusion)
        upsampled_samples = upsampler_diffusion.euler_ancestral_sample_loop(
            upsampler_model,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=upsampler_model_kwargs,
            cond_fn=None,
            eta=sampler_eta if sampler_eta > 0 else 1.0,
        )[:batch_size]
    elif upsampler_sampler == "dpm++":
        enhance_diffusion(upsampler_diffusion)
        upsampled_samples = upsampler_diffusion.dpm_solver_sample_loop(
            upsampler_model,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=upsampler_model_kwargs,
            cond_fn=None,
            eta=0.0,
            order=dpm_order,
        )[:batch_size]
    else:
        raise ValueError(f"Unknown sampler: {upsampler_sampler}")

    upsampler_model.del_cache()

    return upsampled_samples
