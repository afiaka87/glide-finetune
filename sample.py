#!/usr/bin/env python3
"""
Sample images using GLIDE base model and upscale with SwinIR
"""

import os
import sys
from typing import Literal

import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
from transformers.models.swin2sr import Swin2SRForImageSuperResolution

# Add glide-text2im to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "glide-text2im"))

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class UpscaleSR:
    def __init__(
        self,
        scale: Literal[2, 4, 8] = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        repo = f"caidas/swin2SR-classical-sr-x{scale}-64"
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = (
            Swin2SRForImageSuperResolution.from_pretrained(repo)
            .to(device=device, dtype=dtype)
            .eval()
        )

    @torch.inference_mode()
    def __call__(self, imgs_bchw: torch.Tensor) -> torch.Tensor:
        # imgs in [-1,1], Bx3x64x64 -> returns [-1,1], Bx3x256x256
        B, C, H, W = imgs_bchw.shape
        imgs01 = (imgs_bchw * 0.5 + 0.5).clamp(0, 1)

        # Process with padding disabled to get exact 4x upscale
        inputs = self.proc(
            images=imgs01, do_rescale=False, do_pad=False, return_tensors="pt"
        ).to(self.model.device)
        with torch.autocast(
            device_type=self.model.device.type,
            enabled=self.model.dtype == torch.float16,
        ):
            out = self.model(**inputs).reconstruction  # Bx3x(H*4)x(W*4) in [0,1]

        # Ensure output is exactly 4x the input size
        _, _, out_H, out_W = out.shape
        if out_H != H * 4 or out_W != W * 4:
            # Crop to exact size if needed
            out = out[:, :, : H * 4, : W * 4]

        return out.clamp(0, 1) * 2 - 1


def sample_glide_and_upscale(
    prompt: str = "a beautiful landscape with mountains and trees",
    batch_size: int = 1,
    guidance_scale: float = 4.0,
    device: str = "cuda",
    num_steps: int = 100,
    output_path: str = "output.png",
    save_intermediate: bool = True,
):
    """
    Sample from GLIDE base model and upscale with SwinIR

    Args:
        prompt: Text prompt for generation
        batch_size: Number of images to generate
        guidance_scale: Classifier-free guidance scale
        device: Device to run on
        num_steps: Number of diffusion steps
        output_path: Path to save final image
        save_intermediate: Whether to save 64x64 intermediate image
    """

    # Load GLIDE base model
    print("Loading GLIDE base model...")
    has_cuda = torch.cuda.is_available()
    device = torch.device(device if has_cuda else "cpu")

    # Create model and diffusion
    options = model_and_diffusion_defaults()
    options["use_fp16"] = has_cuda

    model, diffusion = create_model_and_diffusion(**options)
    model.eval()

    # Load checkpoint from glide_model_cache
    checkpoint_path = "glide_model_cache/base.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        print("Loading checkpoint from OpenAI...")
        model.load_state_dict(load_checkpoint("base", device))

    if has_cuda:
        model.convert_to_fp16()
    model.to(device)

    # Create tokens for prompt
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])

    # Create classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options["text_ctx"]
    )

    # Pack the tokens together into model kwargs
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),
    )

    # Create a classifier-free guidance sampler
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # Sample from the base model
    print(f"Sampling from GLIDE with prompt: '{prompt}'")
    print(f"Steps: {num_steps}, Guidance scale: {guidance_scale}")

    # Create evaluation diffusion with fewer steps
    eval_diffusion = create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=str(num_steps),
    )

    # Sample
    samples = eval_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, 64, 64),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    # Save 64x64 intermediate result if requested
    if save_intermediate:
        # Convert to PIL image
        sample_64 = samples[0].cpu()
        sample_64 = (sample_64 * 0.5 + 0.5).clamp(0, 1)
        sample_64 = (sample_64 * 255).to(torch.uint8)
        sample_64 = sample_64.permute(1, 2, 0).numpy()
        img_64 = Image.fromarray(sample_64)
        intermediate_path = output_path.replace(".png", "_64x64.png")
        img_64.save(intermediate_path)
        print(f"Saved 64x64 image to {intermediate_path}")

    # Upscale with SwinIR
    print("Upscaling with SwinIR (64x64 -> 256x256)...")
    upscaler = UpscaleSR(
        scale=4, device=device, dtype=torch.float16 if has_cuda else torch.float32
    )
    samples_256 = upscaler(samples)

    # Save final 256x256 result
    sample_256 = samples_256[0].cpu()
    sample_256 = (sample_256 * 0.5 + 0.5).clamp(0, 1)
    sample_256 = (sample_256 * 255).to(torch.uint8)
    sample_256 = sample_256.permute(1, 2, 0).numpy()
    img_256 = Image.fromarray(sample_256)
    img_256.save(output_path)
    print(f"Saved 256x256 image to {output_path}")

    return samples_256


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample from GLIDE and upscale with SwinIR"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful sunset over the ocean",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output path for generated image",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Do not save intermediate 64x64 image",
    )

    args = parser.parse_args()

    sample_glide_and_upscale(
        prompt=args.prompt,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        device=args.device,
        num_steps=args.num_steps,
        output_path=args.output,
        save_intermediate=not args.no_intermediate,
    )


if __name__ == "__main__":
    main()
