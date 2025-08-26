#!/usr/bin/env python3
"""
Sample images using GLIDE models with comprehensive sampling options.

Supports base model sampling, upsampling, and various post-processing options
including SwinIR super-resolution.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union, cast, assert_never

import numpy as np
import torch
from PIL import Image

# Add glide-text2im to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "glide-text2im"))

# Import enhanced samplers
from glide_finetune.enhanced_samplers import enhance_glide_diffusion

# Import unified model loader
from glide_finetune.model_loader import (
    ModelInfo,
    ModelLoadConfig,
    ModelType,
    UnifiedModelLoader,
)

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger
from glide_text2im.model_creation import create_gaussian_diffusion

# Initialize logger
logger = get_logger("glide_finetune.sample")


# Type definitions
DeviceType = Union[str, torch.device]
SamplerType = Literal["plms", "ddim", "euler", "euler_a", "dpm++"]
UpscalerType = Literal["swinir", "swin2sr", "none"]


@dataclass
class SamplingConfig:
    """Configuration for sampling."""
    
    prompt: str = "a beautiful sunset over the ocean"
    batch_size: int = 1
    guidance_scale: float = 4.0
    num_steps: int = 100
    sampler: SamplerType = "plms"
    eta: float = 0.0
    seed: int | None = None
    device: DeviceType = "cuda"
    use_fp16: bool = True


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    
    model_type: ModelType = "base"
    checkpoint_path: str | None = None
    use_openai_checkpoint: bool = True


@dataclass
class OutputConfig:
    """Configuration for output saving."""
    
    output_dir: str = "./outputs"
    output_prefix: str = "sample"
    save_intermediate: bool = True
    save_grid: bool = False
    image_format: str = "png"


@dataclass 
class UpscaleConfig:
    """Configuration for upscaling."""
    
    enable_upscale: bool = True
    upscaler_type: UpscalerType = "swinir"
    upscale_factor: int = 4
    upscale_temp: float = 0.997


class SwinIRUpscaler:
    """SwinIR-based upscaler for 64x64 -> 256x256."""
    
    def __init__(
        self,
        scale: int = 4,
        device: DeviceType = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        try:
            from transformers import AutoImageProcessor
            from transformers.models.swin2sr import Swin2SRForImageSuperResolution
        except ImportError as e:
            msg = "Please install transformers: uv add transformers"
            raise ImportError(msg) from e
        
        repo = f"caidas/swin2SR-classical-sr-x{scale}-64"
        self.scale = scale
        self.proc = AutoImageProcessor.from_pretrained(repo)  # type: ignore[no-untyped-call]
        self.model = (
            Swin2SRForImageSuperResolution.from_pretrained(repo)
            .to(device=device, dtype=dtype)
            .eval()
        )
    
    @torch.inference_mode()
    def __call__(self, imgs_bchw: torch.Tensor) -> torch.Tensor:
        """
        Upscale images using SwinIR.
        
        Args:
            imgs_bchw: Images in [-1,1], shape (B, 3, H, W)
            
        Returns:
            Upscaled images in [-1,1], shape (B, 3, H*scale, W*scale)
        """
        batch, channel, height, width = imgs_bchw.shape
        imgs01 = (imgs_bchw * 0.5 + 0.5).clamp(0, 1)
        
        # Process with padding disabled to get exact upscale
        inputs = self.proc(
            images=imgs01,
            do_rescale=False,
            do_pad=False,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.autocast(
            device_type=self.model.device.type,
            enabled=self.model.dtype == torch.float16,
        ):
            out = self.model(**inputs).reconstruction
        
        # Ensure output is exactly scale*input size
        _, _, out_height, out_width = out.shape
        if out_height != height * self.scale or out_width != width * self.scale:
            out = out[:, :, : height * self.scale, : width * self.scale]
        
        result: torch.Tensor = out.clamp(0, 1) * 2 - 1
        return result


class GlideSampler:
    """Main GLIDE sampling class with comprehensive options."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        sampling_config: SamplingConfig,
        upscale_config: UpscaleConfig,
        output_config: OutputConfig,
    ):
        self.model_config = model_config
        self.sampling_config = sampling_config
        self.upscale_config = upscale_config
        self.output_config = output_config
        
        # Load model
        self.model_info = self._load_model()
        
        # Create upscaler if needed
        self.upscaler: SwinIRUpscaler | None = None
        if upscale_config.enable_upscale and upscale_config.upscaler_type != "none":
            self.upscaler = self._create_upscaler()
        
        # Set seed if provided
        if sampling_config.seed is not None:
            self._set_seed(sampling_config.seed)
    
    def _load_model(self) -> ModelInfo:
        """Load the GLIDE model."""
        logger.info(f"Loading {self.model_config.model_type} model...")
        
        ModelLoadConfig(
            model_type=self.model_config.model_type,
            checkpoint_path=self.model_config.checkpoint_path,
            use_fp16=self.sampling_config.use_fp16,
            device=self.sampling_config.device,
            use_openai_checkpoint=self.model_config.use_openai_checkpoint,
        )
        
        return UnifiedModelLoader.load_for_inference(
            model_type=self.model_config.model_type,
            checkpoint_path=self.model_config.checkpoint_path,
            device=self.sampling_config.device,
            use_fp16=self.sampling_config.use_fp16,
        )
    
    def _create_upscaler(self) -> SwinIRUpscaler:
        """Create the upscaler."""
        logger.info(f"Creating {self.upscale_config.upscaler_type} upscaler...")
        
        if self.upscale_config.upscaler_type in ["swinir", "swin2sr"]:
            return SwinIRUpscaler(
                scale=self.upscale_config.upscale_factor,
                device=self.sampling_config.device,
                dtype=torch.float16 if self.sampling_config.use_fp16 else torch.float32,
            )
        msg = f"Unknown upscaler type: {self.upscale_config.upscaler_type}"
        raise ValueError(msg)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")
    
    def sample(
        self,
        prompts: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Sample images from the model.
        
        Args:
            prompts: List of prompts. If None, uses config prompt.
            
        Returns:
            Generated images tensor.
        """
        if prompts is None:
            prompts = [self.sampling_config.prompt] * self.sampling_config.batch_size
        
        batch_size = len(prompts)
        model = self.model_info.model
        options = self.model_info.options
        tokenizer = self.model_info.tokenizer
        
        # Determine image size based on model type
        if "upsample" in self.model_config.model_type:
            img_size = 256  # Upsampling models output 256x256
        else:
            img_size = 64  # Base models output 64x64
        
        # Create tokens for prompts
        all_tokens = []
        all_masks = []
        
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            tokens, mask = tokenizer.padded_tokens_and_mask(
                tokens, options["text_ctx"]
            )
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        # Create unconditional tokens for classifier-free guidance
        uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask(
            [], options["text_ctx"]
        )
        
        # Stack tokens and masks
        tokens_tensor = torch.tensor(
            all_tokens + [uncond_tokens] * batch_size,
            device=self.sampling_config.device,
        )
        mask_tensor = torch.tensor(
            all_masks + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=self.sampling_config.device,
        )
        
        model_kwargs = {
            "tokens": tokens_tensor,
            "mask": mask_tensor,
        }
        
        # Create diffusion for sampling
        eval_diffusion = create_gaussian_diffusion(
            steps=options["diffusion_steps"],
            noise_schedule=options["noise_schedule"],
            timestep_respacing=str(self.sampling_config.num_steps),
        )
        
        # Add enhanced samplers if needed
        if self.sampling_config.sampler in ["euler", "euler_a", "dpm++"]:
            enhance_glide_diffusion(eval_diffusion)  # type: ignore[no-untyped-call]
        
        # Create model function with classifier-free guidance
        def model_fn(x_t: torch.Tensor, ts: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            half = x_t[:batch_size]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, batch_size, dim=0)
            half_eps = uncond_eps + self.sampling_config.guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        
        # Sample using the selected method
        logger.info(f"Sampling with {self.sampling_config.sampler} sampler...")
        logger.info(f"Steps: {self.sampling_config.num_steps}, Guidance: {self.sampling_config.guidance_scale}")
        
        full_batch_size = batch_size * 2
        shape = (full_batch_size, 3, img_size, img_size)
        
        if self.sampling_config.sampler == "plms":
            samples = eval_diffusion.plms_sample_loop(
                model_fn,
                shape,
                device=self.sampling_config.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:batch_size]
        
        elif self.sampling_config.sampler == "ddim":
            samples = eval_diffusion.ddim_sample_loop(
                model_fn,
                shape,
                device=self.sampling_config.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=self.sampling_config.eta,
            )[:batch_size]
        
        elif self.sampling_config.sampler == "euler":
            samples = eval_diffusion.euler_sample_loop(
                model_fn,
                shape,
                device=self.sampling_config.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=self.sampling_config.eta,
                num_steps=self.sampling_config.num_steps,
            )[:batch_size]
        
        elif self.sampling_config.sampler == "euler_a":
            samples = eval_diffusion.euler_ancestral_sample_loop(
                model_fn,
                shape,
                device=self.sampling_config.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=self.sampling_config.eta,
                num_steps=self.sampling_config.num_steps,
            )[:batch_size]
        
        elif self.sampling_config.sampler == "dpm++":
            samples = eval_diffusion.dpm_solver_sample_loop(
                model_fn,
                shape,
                device=self.sampling_config.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=self.sampling_config.eta,
                num_steps=self.sampling_config.num_steps,
                order=2,
            )[:batch_size]
        
        else:
            assert_never(self.sampling_config.sampler)
        
        return cast("torch.Tensor", samples)
    
    def upscale(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Upscale samples if configured.
        
        Args:
            samples: Input samples to upscale.
            
        Returns:
            Upscaled samples or original if upscaling disabled.
        """
        if self.upscaler is None:
            return samples
        
        logger.info(f"Upscaling {samples.shape[-1]}x{samples.shape[-2]} -> "
                   f"{samples.shape[-1] * self.upscale_config.upscale_factor}x"
                   f"{samples.shape[-2] * self.upscale_config.upscale_factor}...")
        
        return self.upscaler(samples)
    
    def save_samples(
        self,
        samples: torch.Tensor,
        prompts: list[str],
        suffix: str = "",
    ) -> list[str]:
        """
        Save samples to disk.
        
        Args:
            samples: Generated samples tensor.
            prompts: List of prompts used.
            suffix: Optional suffix for filenames.
            
        Returns:
            List of saved file paths.
        """
        # Create output directory
        output_dir = Path(self.output_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, (sample, prompt) in enumerate(zip(samples, prompts, strict=False)):
            # Convert to PIL image
            sample_cpu = sample.cpu()
            sample_01 = (sample_cpu * 0.5 + 0.5).clamp(0, 1)
            sample_uint8 = (sample_01 * 255).to(torch.uint8)
            sample_np = sample_uint8.permute(1, 2, 0).numpy()
            img = Image.fromarray(sample_np)
            
            # Create filename
            prompt_slug = prompt[:50].replace(" ", "_").replace("/", "_")
            filename = f"{self.output_config.output_prefix}_{i:03d}_{prompt_slug}{suffix}.{self.output_config.image_format}"
            filepath = output_dir / filename
            
            # Save image
            img.save(filepath)
            saved_paths.append(str(filepath))
            logger.info(f"Saved: {filepath}")
        
        return saved_paths
    
    def run(self, prompts: list[str] | None = None) -> tuple[torch.Tensor, list[str]]:
        """
        Run the complete sampling pipeline.
        
        Args:
            prompts: Optional list of prompts.
            
        Returns:
            Tuple of (final samples tensor, saved file paths)
        """
        # Sample from model
        samples = self.sample(prompts)
        
        # Get actual prompts used
        if prompts is None:
            prompts = [self.sampling_config.prompt] * self.sampling_config.batch_size
        
        # Save intermediate if requested
        if self.output_config.save_intermediate and self.upscaler is not None:
            self.save_samples(samples, prompts, suffix="_intermediate")
        
        # Upscale if configured
        if self.upscaler is not None:
            samples = self.upscale(samples)
        
        # Save final samples
        saved_paths = self.save_samples(samples, prompts)
        
        return samples, saved_paths


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser with organized groups."""
    parser = argparse.ArgumentParser(
        description="Sample images from GLIDE models with various options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-type",
        type=str,
        choices=["base", "upsample", "base-inpaint", "upsample-inpaint"],
        default="base",
        help="Type of model to use",
    )
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses OpenAI checkpoint if not provided)",
    )
    model_group.add_argument(
        "--no-openai-checkpoint",
        action="store_true",
        help="Do not fall back to OpenAI checkpoint",
    )
    
    # Sampling arguments
    sampling_group = parser.add_argument_group("Sampling Configuration")
    sampling_group.add_argument(
        "--prompt",
        type=str,
        default="a beautiful sunset over the ocean",
        help="Text prompt for generation",
    )
    sampling_group.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    sampling_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    sampling_group.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    sampling_group.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion steps",
    )
    sampling_group.add_argument(
        "--sampler",
        type=str,
        choices=["plms", "ddim", "euler", "euler_a", "dpm++"],
        default="plms",
        help="Sampling method to use",
    )
    sampling_group.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Eta parameter for DDIM/Euler samplers",
    )
    sampling_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    # Upscaling arguments
    upscale_group = parser.add_argument_group("Upscaling Configuration")
    upscale_group.add_argument(
        "--no-upscale",
        action="store_true",
        help="Disable upscaling",
    )
    upscale_group.add_argument(
        "--upscaler",
        type=str,
        choices=["swinir", "swin2sr", "none"],
        default="swinir",
        help="Upscaler to use",
    )
    upscale_group.add_argument(
        "--upscale-factor",
        type=int,
        default=4,
        help="Upscaling factor",
    )
    upscale_group.add_argument(
        "--upscale-temp",
        type=float,
        default=0.997,
        help="Temperature for upsampling noise",
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs",
    )
    output_group.add_argument(
        "--output-prefix",
        type=str,
        default="sample",
        help="Prefix for output filenames",
    )
    output_group.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Do not save intermediate (pre-upscale) images",
    )
    output_group.add_argument(
        "--save-grid",
        action="store_true",
        help="Save images as a grid",
    )
    output_group.add_argument(
        "--image-format",
        type=str,
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Output image format",
    )
    
    # System arguments
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    system_group.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 (use FP32)",
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load prompts
    prompts: list[str]
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = [args.prompt] * args.batch_size
    
    # Create configurations
    model_config = ModelConfig(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        use_openai_checkpoint=not args.no_openai_checkpoint,
    )
    
    sampling_config = SamplingConfig(
        prompt=args.prompt,
        batch_size=len(prompts),
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        sampler=args.sampler,
        eta=args.eta,
        seed=args.seed,
        device=args.device,
        use_fp16=not args.no_fp16,
    )
    
    upscale_config = UpscaleConfig(
        enable_upscale=not args.no_upscale,
        upscaler_type=args.upscaler,
        upscale_factor=args.upscale_factor,
        upscale_temp=args.upscale_temp,
    )
    
    output_config = OutputConfig(
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        save_intermediate=not args.no_intermediate,
        save_grid=args.save_grid,
        image_format=args.image_format,
    )
    
    # Create sampler and run
    sampler = GlideSampler(
        model_config=model_config,
        sampling_config=sampling_config,
        upscale_config=upscale_config,
        output_config=output_config,
    )
    
    samples, saved_paths = sampler.run(prompts)
    
    logger.info(f"Generated {len(saved_paths)} images")
    logger.info("Done!")


if __name__ == "__main__":
    main()