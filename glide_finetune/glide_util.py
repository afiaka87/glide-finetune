## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
from typing import Tuple

import numpy as np
import PIL.Image
import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

from glide_finetune.train_util import pred_to_pil

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]


def get_uncond_tokens_mask(tokenizer: Encoder) -> Tuple[th.Tensor, th.Tensor]:
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.Tensor, th.Tensor]:
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
    torch_compile: bool = False,
    compile_mode: str = "default",
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

    # Start with all parameters trainable
    glide_model.requires_grad_(True)

    # Freeze transformer components (text processing)
    if freeze_transformer:
        # Core transformer and embeddings
        if hasattr(glide_model, "transformer") and glide_model.transformer is not None:
            glide_model.transformer.requires_grad_(False)
        if (
            hasattr(glide_model, "transformer_proj")
            and glide_model.transformer_proj is not None
        ):
            glide_model.transformer_proj.requires_grad_(False)
        if (
            hasattr(glide_model, "token_embedding")
            and glide_model.token_embedding is not None
        ):
            glide_model.token_embedding.requires_grad_(False)
        if (
            hasattr(glide_model, "positional_embedding")
            and glide_model.positional_embedding is not None
        ):
            glide_model.positional_embedding.requires_grad = False
        if (
            hasattr(glide_model, "padding_embedding")
            and glide_model.padding_embedding is not None
        ):
            glide_model.padding_embedding.requires_grad = False
        # Final layer norm is part of transformer output processing
        if hasattr(glide_model, "final_ln") and glide_model.final_ln is not None:
            glide_model.final_ln.requires_grad_(False)

    # Freeze diffusion/UNet components
    if freeze_diffusion:
        # UNet blocks
        if (
            hasattr(glide_model, "input_blocks")
            and glide_model.input_blocks is not None
        ):
            glide_model.input_blocks.requires_grad_(False)
        if (
            hasattr(glide_model, "middle_block")
            and glide_model.middle_block is not None
        ):
            glide_model.middle_block.requires_grad_(False)
        if (
            hasattr(glide_model, "output_blocks")
            and glide_model.output_blocks is not None
        ):
            glide_model.output_blocks.requires_grad_(False)
        if hasattr(glide_model, "out") and glide_model.out is not None:
            glide_model.out.requires_grad_(False)
        # Note: We intentionally keep time_embed trainable as it's needed for
        # adapting the diffusion process to new domains
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

    # Report freezing status
    if freeze_transformer or freeze_diffusion:
        total_params = sum(p.numel() for p in glide_model.parameters())
        trainable_params = sum(
            p.numel() for p in glide_model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        print("\nModel parameter summary:")
        print(f"  Total parameters: {total_params:,}")
        trainable_pct = trainable_params / total_params * 100
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.1f}%)")
        frozen_pct = frozen_params / total_params * 100
        print(f"  Frozen parameters: {frozen_params:,} ({frozen_pct:.1f}%)")

        if freeze_transformer:
            print("  ✓ Transformer components frozen (text processing)")
        if freeze_diffusion:
            print("  ✓ Diffusion/UNet components frozen (image generation backbone)")

    # Apply torch.compile if requested
    if torch_compile:
        print(f"\nApplying torch.compile with mode='{compile_mode}'...")
        try:
            import torch

            if hasattr(torch, "compile"):
                glide_model = torch.compile(glide_model, mode=compile_mode)
                print(
                    f"✓ Model compiled successfully with torch.compile (mode={compile_mode})"
                )
            else:
                msg = ("⚠️  torch.compile not available (requires PyTorch 2.0+), "
                       "skipping compilation")
                print(msg)
        except Exception as e:
            print(f"⚠️  Failed to compile model: {e}")
            print("   Continuing without compilation...")

    return glide_model, glide_diffusion, options


def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert("RGB")
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
    image_to_upsample="",
    upsample_temp=0.997,
    sampler_name="plms",
    sampler_kwargs=None,
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
        assert image_to_upsample != "", (
            "You must specify a path to an image to upsample."
        )
        low_res_samples = read_image(image_to_upsample, shape=(side_x, side_y))
        model_kwargs["low_res"] = low_res_samples
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_kwargs["noise"] = noise
        model_fn = glide_model  # just use the base model, no need for CFG.

    # Use the new sampler system
    from glide_finetune.samplers import SamplerRegistry

    sampler_kwargs = sampler_kwargs or {}
    shape = (full_batch_size, 3, side_y, side_x)

    # Get sampler class and instantiate
    sampler_class = SamplerRegistry.get_sampler(sampler_name)
    sampler = sampler_class(
        diffusion=eval_diffusion,
        model_fn=model_fn,
        shape=shape,
        device=device,
        clip_denoised=True,
        model_kwargs=model_kwargs,
    )

    # Get number of steps from respacing
    num_steps = int(prediction_respacing) if prediction_respacing.isdigit() else 100

    # Sample
    samples = sampler.sample(num_steps=num_steps, progress=True, **sampler_kwargs)[
        :batch_size
    ]

    glide_model.del_cache()
    return samples
