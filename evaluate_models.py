#!/usr/bin/env python3
"""
Comprehensive evaluation script for comparing GLIDE models.

This script evaluates base GLIDE and CLIP-adapter enhanced GLIDE models across 
diverse prompts testing various aspects of vision and perception, with results 
organized in WandB for easy comparison.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Add glide-text2im to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "glide-text2im"))

# Import GLIDE utilities
from glide_finetune.clip_compute import CLIPFeatureComputer
from glide_finetune.utils.glide_util import (
    load_model,
    sample,
    sample_with_conditioning,
)
from glide_finetune.glide_finetune import create_image_grid
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    model_and_diffusion_defaults,
)

# Initialize logger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Prompts - Excellent diverse set covering vision/perception
# ============================================================================

EVALUATION_PROMPTS = {
    "people_portraits": [
        "A portrait of an elderly woman with kind eyes and silver hair, warm lighting",
        "A group of diverse friends laughing together at a coffee shop",
        "A child playing with soap bubbles in a sunny park, joy and wonder",
        "Business professionals having a meeting in a modern glass office",
        "A musician playing guitar on a street corner at sunset",
    ],
    "animals_nature": [
        "A majestic lion resting under an acacia tree at golden hour",
        "Colorful tropical fish swimming through a vibrant coral reef",
        "A misty mountain landscape with pine trees and morning fog",
        "A butterfly delicately landing on a wildflower in a meadow",
        "A family of elephants walking across the African savanna",
    ],
    "architecture_interiors": [
        "A futuristic glass skyscraper reflecting clouds and sky",
        "A cozy cottage with thatched roof in the English countryside",
        "Ancient temple ruins overgrown with vines and moss",
        "Modern minimalist living room with natural light and plants",
        "A grand library with tall bookshelves and reading lamps",
    ],
    "objects_still_life": [
        "A steaming cup of coffee with latte art on wooden table",
        "Fresh fruits and vegetables at a farmer's market stall",
        "Vintage camera and photographs scattered on a desk",
        "A bouquet of sunflowers in a ceramic vase by a window",
        "High-tech gadgets and devices on a clean white surface",
    ],
    "abstract_artistic": [
        "Swirling colors and brushstrokes in the style of Van Gogh",
        "Geometric patterns in bold primary colors, Bauhaus inspired",
        "Watercolor painting of emotions blending together",
        "Digital art with neon lights and cyberpunk aesthetic",
        "Abstract expressionist painting with dynamic energy",
    ],
    "complex_compositions": [
        "A busy farmer's market with vendors, customers, and produce",
        "A thunderstorm approaching over golden wheat fields",
        "An astronaut floating in space above Earth's blue marble",
        "A steampunk workshop filled with gears, pipes, and inventions",
        "A fantasy castle on a floating island with waterfalls",
    ],
    "lighting_atmosphere": [
        "Golden hour light streaming through autumn forest trees",
        "Northern lights dancing across a starry Arctic sky",
        "Dramatic storm clouds with lightning over the ocean",
        "Soft morning mist over a peaceful lake at dawn",
        "City lights reflecting on wet streets after rain",
    ],
    "text_typography": [
        "An old bookstore with shelves full of antique books",
        "Neon signs glowing in a Tokyo street at night",
        "Handwritten love letter on aged parchment paper",
        "Street art graffiti on an urban brick wall",
        "Menu board at a French café with elegant calligraphy",
    ],
}


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Model paths - properly named now
    base_glide_path: Optional[str] = None  # Base GLIDE model
    glide_clip_adapter_path: Optional[str] = None  # GLIDE with CLIP adapter
    clip_model_name: str = "ViT-B/32"  # CLIP model for evaluation metrics
    
    # Sampling parameters
    batch_size: int = 8  # Number of samples per prompt
    guidance_scale: float = 3.0
    num_steps: int = 30
    sampler: str = "euler"
    eta: float = 0.0
    seed: int = 42
    
    # Output settings
    output_dir: str = "./evaluation_results"
    save_individual: bool = True
    save_grids: bool = True
    
    # WandB settings
    wandb_project: str = "glide-evaluation"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = False
    
    # Evaluation settings
    categories: List[str] = field(default_factory=lambda: list(EVALUATION_PROMPTS.keys()))
    max_prompts_per_category: Optional[int] = None


class ModelEvaluator:
    """Handles model evaluation and comparison."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models (will be loaded later)
        self.base_glide = None
        self.glide_with_adapter = None
        self.diffusion = None
        self.options = None
        
        # CLIP model for evaluation metrics (not for generation)
        self.clip_evaluator = None
        
        # Initialize WandB if enabled
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize WandB run for logging."""
        run_name = self.config.wandb_run_name or f"evaluation_{int(time.time())}"
        
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config={
                "batch_size": self.config.batch_size,
                "guidance_scale": self.config.guidance_scale,
                "num_steps": self.config.num_steps,
                "sampler": self.config.sampler,
                "categories": self.config.categories,
                "base_glide": self.config.base_glide_path,
                "glide_clip_adapter": self.config.glide_clip_adapter_path,
                "clip_evaluator": self.config.clip_model_name,
            },
        )
        logger.info(f"WandB run initialized: {run_name}")
    
    def load_models(self):
        """Load GLIDE models and CLIP evaluator."""
        logger.info("Loading models...")
        
        # Load base GLIDE model
        if self.config.base_glide_path:
            logger.info(f"Loading base GLIDE from {self.config.base_glide_path}")
            self.base_glide = self._load_glide_model(
                self.config.base_glide_path, 
                has_clip_adapter=False
            )
        else:
            logger.info("Loading default OpenAI GLIDE model")
            self.base_glide, self.diffusion, self.options = load_model(
                use_fp16=self.config.use_fp16,
                model_type="base"
            )
        
        # Load GLIDE with CLIP adapter
        if self.config.glide_clip_adapter_path:
            logger.info(f"Loading GLIDE with CLIP adapter from {self.config.glide_clip_adapter_path}")
            self.glide_with_adapter = self._load_glide_model(
                self.config.glide_clip_adapter_path,
                has_clip_adapter=True
            )
        else:
            logger.warning("No GLIDE-CLIP adapter path provided, skipping adapter evaluation")
        
        # Initialize CLIP evaluator for metrics (not for generation)
        logger.info(f"Initializing CLIP evaluator model: {self.config.clip_model_name}")
        self.clip_evaluator = CLIPFeatureComputer(
            clip_model_name=self.config.clip_model_name,
            device=self.device
        )
        
        # Initialize CLIP model for adapter conditioning (always ViT-B/32 to match training)
        if self.glide_with_adapter:
            logger.info("Initializing CLIP ViT-B/32 for adapter conditioning")
            self.clip_adapter_encoder = CLIPFeatureComputer(
                clip_model_name="ViT-B/32",  # Must match what adapter was trained with
                device=self.device
            )
        else:
            self.clip_adapter_encoder = None
        
        # Move models to device
        self.base_glide = self.base_glide.to(self.device)
        if self.glide_with_adapter:
            self.glide_with_adapter = self.glide_with_adapter.to(self.device)
        
        # If we haven't loaded diffusion yet, create it
        if self.diffusion is None:
            _, self.diffusion, self.options = load_model(
                use_fp16=self.config.use_fp16,
                model_type="base"
            )
        
        logger.info("Models loaded successfully")
    
    def _load_glide_model(
        self, 
        checkpoint_path: str, 
        has_clip_adapter: bool = False
    ) -> nn.Module:
        """Load a GLIDE model from checkpoint."""
        # Load checkpoint to check if it has adapter weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Check if checkpoint contains CLIP adapter weights
        checkpoint_has_adapter = any(key.startswith("clip_adapter.") for key in state_dict.keys())
        
        if checkpoint_has_adapter:
            # Need to create model with adapter architecture BEFORE loading weights
            from glide_text2im.model_creation import (
                model_and_diffusion_defaults,
                create_model_and_diffusion,
            )
            from glide_text2im.download import load_checkpoint
            from glide_finetune.clip_adapter import integrate_clip_adapter_to_model
            
            # Get options for base model
            options = model_and_diffusion_defaults()
            options["use_fp16"] = self.config.use_fp16
            
            # Create model and diffusion
            model, diffusion = create_model_and_diffusion(**options)
            
            # Load OpenAI base weights first
            model.load_state_dict(load_checkpoint("base", "cpu"))
            
            # Move to device before adding adapter
            model = model.to(self.device)
            
            # Add CLIP adapter architecture
            model = integrate_clip_adapter_to_model(
                model,
                clip_model_name="ViT-B/32",  # Default CLIP model used in training
                device=self.device,
            )
            logger.info("Added CLIP adapter architecture to model")
            
            # Now load the full checkpoint with adapter weights
            model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint with CLIP adapter from: {checkpoint_path}")
            
            # Store diffusion and options if not already stored
            if self.diffusion is None:
                self.diffusion = diffusion
                self.options = options
        else:
            # Standard model without adapter
            model, diffusion, options = load_model(
                use_fp16=self.config.use_fp16,
                model_type="base"
            )
            
            # Store diffusion and options if not already stored
            if self.diffusion is None:
                self.diffusion = diffusion
                self.options = options
            
            # Load state dict
            model.load_state_dict(state_dict)
            logger.info(f"Loaded standard GLIDE model from: {checkpoint_path}")
        
        # Verify CLIP adapter is present if expected
        if has_clip_adapter:
            if not hasattr(model, 'clip_adapter') or model.clip_adapter is None:
                logger.warning("CLIP adapter not found in model despite has_clip_adapter=True")
            else:
                logger.info("CLIP adapter successfully integrated into model")
        
        return model
    
    def generate_samples_batch(
        self,
        model: nn.Module,
        prompts: List[str],
        use_clip_conditioning: bool = False,
        seed_offset: int = 0,
    ) -> Dict[str, List[Image.Image]]:
        """Generate one sample for each prompt in a single batch for efficiency.
        
        Returns:
            Dictionary mapping prompt to list containing single generated image
        """
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed + seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed + seed_offset)
        
        # Check if model has CLIP adapter for conditioning
        has_clip_adapter = hasattr(model, 'clip_adapter') and model.clip_adapter is not None
        
        results = {}
        
        # Process prompts in batches
        batch_size = min(self.config.batch_size, 8)  # Max 8 prompts at once
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            current_batch_size = len(batch_prompts)
            
            if use_clip_conditioning and has_clip_adapter and self.clip_adapter_encoder:
                # Get CLIP embeddings for all prompts in batch using the adapter's CLIP model
                clip_embeddings = self.clip_adapter_encoder.compute_text_features(batch_prompts)
                
                # Generate with CLIP conditioning - use special batched version
                batch_samples = self._sample_batch_with_conditioning(
                    glide_model=model,
                    glide_options=self.options,
                    prompts=batch_prompts,
                    clip_embeddings=clip_embeddings,
                    guidance_scale=self.config.guidance_scale,
                    device=self.device,
                    prediction_respacing=str(self.config.num_steps),
                    sampler=self.config.sampler,
                    eta=self.config.eta,
                )
            else:
                # Generate with text-only conditioning
                batch_samples = self._sample_batch(
                    glide_model=model,
                    glide_options=self.options,
                    prompts=batch_prompts,
                    guidance_scale=self.config.guidance_scale,
                    device=self.device,
                    prediction_respacing=str(self.config.num_steps),
                    sampler=self.config.sampler,
                    eta=self.config.eta,
                )
            
            # Convert to PIL images and store results
            for j, prompt in enumerate(batch_prompts):
                img_tensor = batch_samples[j:j+1]
                img = self._tensor_to_pil(img_tensor)
                results[prompt] = [img]  # Single image per prompt
            
            # Clear cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def generate_samples(
        self,
        model: nn.Module,
        prompt: str,
        num_samples: int = 8,
        use_clip_conditioning: bool = False,
        seed_offset: int = 0,
    ) -> List[Image.Image]:
        """Generate samples for a given prompt."""
        samples = []
        
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed + seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed + seed_offset)
        
        # Check if model has CLIP adapter for conditioning
        has_clip_adapter = hasattr(model, 'clip_adapter') and model.clip_adapter is not None
        
        # Generate in batches if necessary
        remaining = num_samples
        batch_idx = 0
        
        while remaining > 0:
            current_batch = min(remaining, 4)  # Max 4 at a time for memory
            
            if use_clip_conditioning and has_clip_adapter and self.clip_evaluator:
                # Get CLIP embeddings for the prompt
                clip_embeddings = self.clip_evaluator.compute_text_features([prompt])
                clip_embeddings = clip_embeddings.repeat(current_batch, 1)
                
                # Generate with CLIP conditioning
                batch_samples = sample_with_conditioning(
                    glide_model=model,
                    glide_options=self.options,
                    prompt=prompt,
                    clip_embeddings=clip_embeddings,
                    batch_size=current_batch,
                    guidance_scale=self.config.guidance_scale,
                    device=self.device,
                    prediction_respacing=str(self.config.num_steps),
                    sampler=self.config.sampler,
                    eta=self.config.eta,
                    side_x=64,
                    side_y=64,
                )
            else:
                # Generate with text-only conditioning
                batch_samples = sample(
                    glide_model=model,
                    glide_options=self.options,
                    prompt=prompt,
                    batch_size=current_batch,
                    guidance_scale=self.config.guidance_scale,
                    device=self.device,
                    prediction_respacing=str(self.config.num_steps),
                    sampler=self.config.sampler,
                    eta=self.config.eta,
                    side_x=64,
                    side_y=64,
                )
            
            # Convert to PIL images
            for i in range(batch_samples.shape[0]):
                img_tensor = batch_samples[i:i+1]
                img = self._tensor_to_pil(img_tensor)
                samples.append(img)
            
            remaining -= current_batch
            batch_idx += 1
            
            # Clear cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Return only the first half (conditional samples)
        return samples[:batch_size][:num_samples]
    
    def _sample_batch(
        self,
        glide_model: nn.Module,
        glide_options: dict,
        prompts: List[str],
        guidance_scale: float = 4.0,
        device: str = "cuda",
        prediction_respacing: str = "100",
        sampler: str = "plms",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Sample multiple prompts in a single batch (text-only conditioning)."""
        from glide_text2im.model_creation import create_gaussian_diffusion
        
        batch_size = len(prompts)
        
        # Create diffusion
        eval_diffusion = create_gaussian_diffusion(
            steps=glide_options["diffusion_steps"],
            noise_schedule=glide_options["noise_schedule"],
            timestep_respacing=prediction_respacing,
        )
        
        # Tokenize all prompts
        all_tokens = []
        all_masks = []
        for prompt in prompts:
            tokens = glide_model.tokenizer.encode(prompt) if prompt else []
            tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, glide_options["text_ctx"])
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        # Also prepare unconditional tokens for CFG
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask([], glide_options["text_ctx"])
        
        # Stack tokens for batch processing with CFG
        model_kwargs = {
            "tokens": torch.tensor(all_tokens + [uncond_tokens] * batch_size, device=device),
            "mask": torch.tensor(all_masks + [uncond_mask] * batch_size, dtype=torch.bool, device=device),
        }
        
        # CFG model function
        def cfg_model_fn(x_t, ts, **kwargs):
            # x_t is already doubled for CFG (batch_size * 2)
            # Split it to get conditional and unconditional parts
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = glide_model(combined, ts, **kwargs)
            epsilon, rest = model_out[:, :3], model_out[:, 3:]
            conditioned_epsilon, unconditional_epsilon = torch.split(epsilon, len(epsilon) // 2, dim=0)
            half_epsilon = unconditional_epsilon + guidance_scale * (conditioned_epsilon - unconditional_epsilon)
            epsilon = torch.cat([half_epsilon, half_epsilon], dim=0)
            return torch.cat([epsilon, rest], dim=1)
        
        # Sample using the specified sampler
        glide_model.del_cache()
        
        # For CFG, we need doubled batch size
        full_batch_size = batch_size * 2
        
        # Choose sampling method based on sampler parameter
        noise = torch.randn((full_batch_size, 3, 64, 64), device=device)
        
        # Get actual number of steps from prediction_respacing
        try:
            actual_num_steps = int(prediction_respacing)
        except ValueError:
            actual_num_steps = 50  # Default fallback
        
        if sampler.lower() in ["euler", "euler_discrete"]:
            # Add enhanced samplers to diffusion instance
            from glide_finetune.enhanced_samplers import enhance_glide_diffusion
            enhance_glide_diffusion(eval_diffusion)
            samples = eval_diffusion.euler_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
                num_steps=actual_num_steps,
            )
        elif sampler.lower() in ["euler_a", "euler_ancestral"]:
            from glide_finetune.enhanced_samplers import enhance_glide_diffusion
            enhance_glide_diffusion(eval_diffusion)
            samples = eval_diffusion.euler_ancestral_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
                num_steps=actual_num_steps,
            )
        elif sampler.lower() in ["dpm++", "dpmpp", "dpm_plus_plus"]:
            from glide_finetune.enhanced_samplers import enhance_glide_diffusion
            enhance_glide_diffusion(eval_diffusion)
            samples = eval_diffusion.dpm_solver_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
                num_steps=actual_num_steps,
            )
        elif sampler.lower() == "ddim":
            samples = eval_diffusion.ddim_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
            )
        elif sampler.lower() == "plms":
            samples = eval_diffusion.plms_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
            )
        else:
            # Default to p_sample_loop
            logger.warning(f"Unknown sampler '{sampler}', falling back to p_sample_loop")
            samples = eval_diffusion.p_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
            )
        
        # Return only the first half (conditional samples)
        return samples[:batch_size]
    
    def _sample_batch_with_conditioning(
        self,
        glide_model: nn.Module,
        glide_options: dict,
        prompts: List[str],
        clip_embeddings: torch.Tensor,
        guidance_scale: float = 4.0,
        device: str = "cuda",
        prediction_respacing: str = "100",
        sampler: str = "plms",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Sample multiple prompts in a single batch with CLIP conditioning."""
        from glide_text2im.model_creation import create_gaussian_diffusion
        
        batch_size = len(prompts)
        
        # Create diffusion
        eval_diffusion = create_gaussian_diffusion(
            steps=glide_options["diffusion_steps"],
            noise_schedule=glide_options["noise_schedule"],
            timestep_respacing=prediction_respacing,
        )
        
        # Tokenize all prompts
        all_tokens = []
        all_masks = []
        for prompt in prompts:
            tokens = glide_model.tokenizer.encode(prompt) if prompt else []
            tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, glide_options["text_ctx"])
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        # Also prepare unconditional tokens for CFG
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask([], glide_options["text_ctx"])
        
        # Stack tokens for batch processing with CFG
        model_kwargs = {
            "tokens": torch.tensor(all_tokens + [uncond_tokens] * batch_size, device=device),
            "mask": torch.tensor(all_masks + [uncond_mask] * batch_size, dtype=torch.bool, device=device),
        }
        
        # Add CLIP embeddings for CFG (conditioned and unconditioned)
        uncond_clip = torch.zeros_like(clip_embeddings)
        model_kwargs["clip_embeddings"] = torch.cat([clip_embeddings, uncond_clip], dim=0)
        
        # CFG model function
        def cfg_model_fn(x_t, ts, **kwargs):
            # x_t is already doubled for CFG (batch_size * 2)
            # Split it to get conditional and unconditional parts
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = glide_model(combined, ts, **kwargs)
            epsilon, rest = model_out[:, :3], model_out[:, 3:]
            conditioned_epsilon, unconditional_epsilon = torch.split(epsilon, len(epsilon) // 2, dim=0)
            half_epsilon = unconditional_epsilon + guidance_scale * (conditioned_epsilon - unconditional_epsilon)
            epsilon = torch.cat([half_epsilon, half_epsilon], dim=0)
            return torch.cat([epsilon, rest], dim=1)
        
        # Sample using the specified sampler
        glide_model.del_cache()
        
        # For CFG, we need doubled batch size
        full_batch_size = batch_size * 2
        
        # Choose sampling method based on sampler parameter
        noise = torch.randn((full_batch_size, 3, 64, 64), device=device)
        
        # Get actual number of steps from prediction_respacing
        try:
            actual_num_steps = int(prediction_respacing)
        except ValueError:
            actual_num_steps = 50  # Default fallback
        
        if sampler.lower() in ["euler", "euler_discrete"]:
            # Add enhanced samplers to diffusion instance
            from glide_finetune.enhanced_samplers import enhance_glide_diffusion
            enhance_glide_diffusion(eval_diffusion)
            samples = eval_diffusion.euler_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
                num_steps=actual_num_steps,
            )
        elif sampler.lower() in ["euler_a", "euler_ancestral"]:
            from glide_finetune.enhanced_samplers import enhance_glide_diffusion
            enhance_glide_diffusion(eval_diffusion)
            samples = eval_diffusion.euler_ancestral_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
                num_steps=actual_num_steps,
            )
        elif sampler.lower() in ["dpm++", "dpmpp", "dpm_plus_plus"]:
            from glide_finetune.enhanced_samplers import enhance_glide_diffusion
            enhance_glide_diffusion(eval_diffusion)
            samples = eval_diffusion.dpm_solver_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
                num_steps=actual_num_steps,
            )
        elif sampler.lower() == "ddim":
            samples = eval_diffusion.ddim_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                eta=eta,
            )
        elif sampler.lower() == "plms":
            samples = eval_diffusion.plms_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
            )
        else:
            # Default to p_sample_loop
            logger.warning(f"Unknown sampler '{sampler}', falling back to p_sample_loop")
            samples = eval_diffusion.p_sample_loop(
                cfg_model_fn,
                (full_batch_size, 3, 64, 64),
                noise=noise,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
            )
        
        # Return only the first half (conditional samples)
        return samples[:batch_size]
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image."""
        # Tensor is [1, 3, H, W] in range [-1, 1]
        tensor = tensor.squeeze(0)  # Remove batch dimension
        tensor = (tensor + 1) / 2  # Scale to [0, 1]
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu()
        
        # Convert to numpy and transpose to HWC
        array = tensor.numpy()
        array = np.transpose(array, (1, 2, 0))
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array)
    
    def calculate_clip_score(
        self,
        images: List[Image.Image],
        prompt: str,
    ) -> float:
        """Calculate CLIP score for prompt-image alignment using evaluation CLIP model."""
        if not self.clip_evaluator:
            return 0.0
        
        # Get text features
        text_features = self.clip_evaluator.compute_text_features([prompt])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Prepare image transform for CLIP
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        
        scores = []
        for img in images:
            # Transform image for CLIP
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_evaluator.clip_model.encode_image(img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            score = F.cosine_similarity(text_features, image_features).item()
            scores.append(score)
        
        return np.mean(scores)
    
    def calculate_diversity_score(
        self,
        images: List[Image.Image],
    ) -> float:
        """Calculate diversity score within a batch of images."""
        if len(images) < 2:
            return 0.0
        
        if not self.clip_evaluator:
            # Use pixel-level diversity
            return self._calculate_pixel_diversity(images)
        
        # Use CLIP feature diversity
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        
        features = []
        for img in images:
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_evaluator.clip_model.encode_image(img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
        
        # Calculate pairwise distances
        features = torch.cat(features, dim=0)
        distances = []
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = 1 - F.cosine_similarity(
                    features[i:i+1], 
                    features[j:j+1]
                ).item()
                distances.append(dist)
        
        return np.mean(distances)
    
    def _calculate_pixel_diversity(self, images: List[Image.Image]) -> float:
        """Calculate pixel-level diversity for images."""
        # Convert to tensors
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        tensors = [transform(img) for img in images]
        tensors = torch.stack(tensors)
        
        # Calculate pairwise L2 distances
        distances = []
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                dist = torch.norm(tensors[i] - tensors[j]).item()
                distances.append(dist)
        
        return np.mean(distances)
    
    def evaluate_category(
        self,
        category: str,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """Evaluate models on a category of prompts using batched generation."""
        logger.info(f"\nEvaluating category: {category}")
        category_results = {
            "category": category,
            "prompts": [],
        }
        
        # Limit prompts if specified
        if self.config.max_prompts_per_category:
            prompts = prompts[:self.config.max_prompts_per_category]
        
        # Generate all samples for this category in batches
        logger.info(f"  Generating base GLIDE samples for {len(prompts)} prompts...")
        base_results = self.generate_samples_batch(
            self.base_glide,
            prompts,
            use_clip_conditioning=False,
            seed_offset=0,
        )
        
        adapter_results = {}
        if self.glide_with_adapter:
            logger.info(f"  Generating GLIDE+adapter samples for {len(prompts)} prompts...")
            adapter_results = self.generate_samples_batch(
                self.glide_with_adapter,
                prompts,
                use_clip_conditioning=True,
                seed_offset=1000,  # Different seed for variety
            )
        
        # Process results for each prompt
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Processing {category}")):
            prompt_results = {
                "prompt": prompt,
                "base_glide_samples": base_results.get(prompt, []),
                "glide_adapter_samples": adapter_results.get(prompt, []),
                "base_glide_metrics": {},
                "glide_adapter_metrics": {},
            }
            
            # Calculate base model metrics
            base_samples = prompt_results["base_glide_samples"]
            if base_samples:
                # For single sample per prompt, diversity is N/A
                prompt_results["base_glide_metrics"] = {
                    "clip_score": self.calculate_clip_score(base_samples, prompt),
                    "diversity": 0.0,  # Single sample, no diversity
                    "num_samples": len(base_samples),
                }
            
            # Calculate adapter model metrics
            adapter_samples = prompt_results["glide_adapter_samples"]
            if adapter_samples:
                prompt_results["glide_adapter_metrics"] = {
                    "clip_score": self.calculate_clip_score(adapter_samples, prompt),
                    "diversity": 0.0,  # Single sample, no diversity
                    "num_samples": len(adapter_samples),
                }
            
            # Save individual images if requested
            if self.config.save_individual:
                self._save_individual_images(
                    category, 
                    prompt_idx, 
                    prompt,
                    base_samples,
                    adapter_samples,
                )
            
            # Create and save comparison grid
            if self.config.save_grids:
                self._save_comparison_grid(
                    category,
                    prompt_idx,
                    prompt,
                    base_samples,
                    adapter_samples,
                )
            
            category_results["prompts"].append(prompt_results)
        
        return category_results
    
    def _save_individual_images(
        self,
        category: str,
        prompt_idx: int,
        prompt: str,
        base_samples: List[Image.Image],
        adapter_samples: List[Image.Image],
    ):
        """Save individual sample images."""
        category_dir = self.output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Save base GLIDE samples
        base_dir = category_dir / f"prompt_{prompt_idx:03d}_base_glide"
        base_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(base_samples):
            img.save(base_dir / f"sample_{i:02d}.png")
        
        # Save GLIDE+adapter samples
        if adapter_samples:
            adapter_dir = category_dir / f"prompt_{prompt_idx:03d}_glide_adapter"
            adapter_dir.mkdir(exist_ok=True)
            
            for i, img in enumerate(adapter_samples):
                img.save(adapter_dir / f"sample_{i:02d}.png")
        
        # Save prompt text
        with open(category_dir / f"prompt_{prompt_idx:03d}.txt", "w") as f:
            f.write(prompt)
    
    def _save_comparison_grid(
        self,
        category: str,
        prompt_idx: int,
        prompt: str,
        base_samples: List[Image.Image],
        adapter_samples: List[Image.Image],
    ):
        """Save comparison grid of samples."""
        grids_dir = self.output_dir / "grids" / category
        grids_dir.mkdir(parents=True, exist_ok=True)
        
        # Create grid for base GLIDE
        if base_samples:
            base_grid = create_image_grid(base_samples, rows=2, cols=4)
            base_grid.save(grids_dir / f"prompt_{prompt_idx:03d}_base_glide_grid.png")
        
        # Create grid for GLIDE+adapter
        if adapter_samples:
            adapter_grid = create_image_grid(adapter_samples, rows=2, cols=4)
            adapter_grid.save(grids_dir / f"prompt_{prompt_idx:03d}_glide_adapter_grid.png")
        
        # Create side-by-side comparison if both available
        if base_samples and adapter_samples:
            # Stack grids horizontally
            comparison_grid = Image.new(
                "RGB",
                (base_grid.width + adapter_grid.width, base_grid.height),
            )
            comparison_grid.paste(base_grid, (0, 0))
            comparison_grid.paste(adapter_grid, (base_grid.width, 0))
            comparison_grid.save(
                grids_dir / f"prompt_{prompt_idx:03d}_comparison.png"
            )
    
    def log_to_wandb(self, results: Dict[str, Any]):
        """Log evaluation results to WandB."""
        if not self.config.use_wandb or not self.wandb_run:
            return
        
        logger.info("Logging results to WandB...")
        
        # Create comparison table with metrics
        columns = [
            "Category",
            "Prompt",
            "Model",
            "CLIP_Score",
            "Diversity",
            "Sample_1",
            "Sample_2", 
            "Sample_3",
            "Sample_4",
            "Sample_5",
            "Sample_6",
            "Sample_7",
            "Sample_8",
        ]
        
        comparison_table = wandb.Table(columns=columns)
        
        # Track aggregate metrics
        base_clip_scores = []
        adapter_clip_scores = []
        base_diversity_scores = []
        adapter_diversity_scores = []
        
        for category_results in results["categories"]:
            category = category_results["category"]
            
            for prompt_data in category_results["prompts"]:
                prompt = prompt_data["prompt"]
                
                # Add base GLIDE row
                if prompt_data["base_glide_samples"]:
                    base_metrics = prompt_data.get("base_glide_metrics", {})
                    clip_score = base_metrics.get("clip_score", 0.0)
                    diversity = base_metrics.get("diversity", 0.0)
                    
                    base_clip_scores.append(clip_score)
                    base_diversity_scores.append(diversity)
                    
                    row = [category, prompt, "Base GLIDE", f"{clip_score:.3f}", f"{diversity:.3f}"]
                    for img in prompt_data["base_glide_samples"]:
                        row.append(wandb.Image(img))
                    # Pad if fewer than 8 samples
                    while len(row) < len(columns):
                        row.append(None)
                    comparison_table.add_data(*row)
                
                # Add GLIDE+adapter row
                if prompt_data.get("glide_adapter_samples"):
                    adapter_metrics = prompt_data.get("glide_adapter_metrics", {})
                    clip_score = adapter_metrics.get("clip_score", 0.0)
                    diversity = adapter_metrics.get("diversity", 0.0)
                    
                    adapter_clip_scores.append(clip_score)
                    adapter_diversity_scores.append(diversity)
                    
                    row = [category, prompt, "GLIDE+Adapter", f"{clip_score:.3f}", f"{diversity:.3f}"]
                    for img in prompt_data["glide_adapter_samples"]:
                        row.append(wandb.Image(img))
                    # Pad if fewer than 8 samples
                    while len(row) < len(columns):
                        row.append(None)
                    comparison_table.add_data(*row)
        
        # Log the comparison table
        self.wandb_run.log({"evaluation/comparison_table": comparison_table})
        
        # Log aggregate metrics
        if base_clip_scores:
            self.wandb_run.log({
                "metrics/base_glide_clip_score_mean": np.mean(base_clip_scores),
                "metrics/base_glide_clip_score_std": np.std(base_clip_scores),
                "metrics/base_glide_diversity_mean": np.mean(base_diversity_scores),
                "metrics/base_glide_diversity_std": np.std(base_diversity_scores),
            })
        
        if adapter_clip_scores:
            self.wandb_run.log({
                "metrics/glide_adapter_clip_score_mean": np.mean(adapter_clip_scores),
                "metrics/glide_adapter_clip_score_std": np.std(adapter_clip_scores),
                "metrics/glide_adapter_diversity_mean": np.mean(adapter_diversity_scores),
                "metrics/glide_adapter_diversity_std": np.std(adapter_diversity_scores),
            })
            
            # Log improvement metrics
            if base_clip_scores:
                clip_improvement = np.mean(adapter_clip_scores) - np.mean(base_clip_scores)
                diversity_change = np.mean(adapter_diversity_scores) - np.mean(base_diversity_scores)
                
                self.wandb_run.log({
                    "metrics/clip_score_improvement": clip_improvement,
                    "metrics/diversity_change": diversity_change,
                })
        
        # Log category grids
        for category_results in results["categories"]:
            category = category_results["category"]
            grids_dir = self.output_dir / "grids" / category
            
            if grids_dir.exists():
                for grid_path in grids_dir.glob("*_comparison.png"):
                    img = Image.open(grid_path)
                    prompt_idx = int(grid_path.stem.split("_")[1])
                    prompt = category_results["prompts"][prompt_idx]["prompt"]
                    
                    self.wandb_run.log({
                        f"grids/{category}/{prompt_idx:03d}": wandb.Image(
                            img,
                            caption=prompt,
                        )
                    })
        
        logger.info("Results logged to WandB")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        logger.info("Starting model evaluation...")
        
        # Load models
        self.load_models()
        
        # Prepare results structure
        results = {
            "config": self.config.__dict__,
            "categories": [],
            "timestamp": time.time(),
        }
        
        # Evaluate each category
        for category in self.config.categories:
            if category not in EVALUATION_PROMPTS:
                logger.warning(f"Unknown category: {category}")
                continue
            
            prompts = EVALUATION_PROMPTS[category]
            category_results = self.evaluate_category(category, prompts)
            results["categories"].append(category_results)
        
        # Log to WandB
        self.log_to_wandb(results)
        
        # Save results summary
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            # Convert images to paths for JSON serialization
            json_results = self._prepare_json_results(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")
        
        # Close WandB run
        if self.wandb_run:
            self.wandb_run.finish()
        
        return results
    
    def _prepare_json_results(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization."""
        json_results = {
            "config": results["config"],
            "timestamp": results["timestamp"],
            "categories": [],
        }
        
        for category in results["categories"]:
            json_category = {
                "category": category["category"],
                "prompts": [],
            }
            
            for prompt_data in category["prompts"]:
                json_prompt = {
                    "prompt": prompt_data["prompt"],
                    "base_glide_samples_count": len(prompt_data["base_glide_samples"]),
                    "glide_adapter_samples_count": len(prompt_data.get("glide_adapter_samples", [])),
                    "base_glide_metrics": prompt_data.get("base_glide_metrics", {}),
                    "glide_adapter_metrics": prompt_data.get("glide_adapter_metrics", {}),
                }
                json_category["prompts"].append(json_prompt)
            
            json_results["categories"].append(json_category)
        
        return json_results


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare GLIDE models"
    )
    
    # Model arguments - properly named
    parser.add_argument(
        "--base-glide",
        type=str,
        default=None,
        help="Path to base GLIDE model checkpoint",
    )
    parser.add_argument(
        "--glide-clip-adapter",
        type=str,
        default=None,
        help="Path to GLIDE model with CLIP adapter checkpoint",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use for evaluation metrics (default: ViT-B/32)",
    )
    
    # Sampling arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of samples per prompt (default: 8)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of sampling steps (default: 30)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["plms", "ddim", "euler", "euler_a", "dpm++", "euler_discrete", "euler_ancestral", "dpmpp"],
        help="Sampling method (default: euler)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-save-individual",
        action="store_true",
        help="Don't save individual images",
    )
    parser.add_argument(
        "--no-save-grids",
        action="store_true",
        help="Don't save image grids",
    )
    
    # WandB arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="glide-evaluation",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=list(EVALUATION_PROMPTS.keys()),
        help="Categories to evaluate",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum prompts per category",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        base_glide_path=args.base_glide,
        glide_clip_adapter_path=args.glide_clip_adapter,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        sampler=args.sampler,
        seed=args.seed,
        output_dir=args.output_dir,
        save_individual=not args.no_save_individual,
        save_grids=not args.no_save_grids,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        categories=args.categories,
        max_prompts_per_category=args.max_prompts,
        device=args.device,
        use_fp16=args.fp16,
    )
    
    # Run evaluation
    evaluator = ModelEvaluator(config)
    results = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    # Calculate overall metrics
    all_base_clip_scores = []
    all_adapter_clip_scores = []
    all_base_diversity = []
    all_adapter_diversity = []
    
    for category_results in results["categories"]:
        category = category_results["category"]
        num_prompts = len(category_results["prompts"])
        print(f"\n{category}: {num_prompts} prompts evaluated")
        
        for i, prompt_data in enumerate(category_results["prompts"]):
            prompt = prompt_data["prompt"]
            base_count = len(prompt_data["base_glide_samples"])
            adapter_count = len(prompt_data.get("glide_adapter_samples", []))
            
            base_metrics = prompt_data.get("base_glide_metrics", {})
            adapter_metrics = prompt_data.get("glide_adapter_metrics", {})
            
            if base_metrics:
                all_base_clip_scores.append(base_metrics.get("clip_score", 0))
                all_base_diversity.append(base_metrics.get("diversity", 0))
            
            if adapter_metrics:
                all_adapter_clip_scores.append(adapter_metrics.get("clip_score", 0))
                all_adapter_diversity.append(adapter_metrics.get("diversity", 0))
            
            print(f"  {i+1}. {prompt[:50]}...")
            print(f"     Base GLIDE: {base_count} samples", end="")
            if base_metrics:
                print(f" (CLIP: {base_metrics.get('clip_score', 0):.3f}, Div: {base_metrics.get('diversity', 0):.3f})", end="")
            print()
            
            if adapter_count > 0:
                print(f"     GLIDE+Adapter: {adapter_count} samples", end="")
                if adapter_metrics:
                    print(f" (CLIP: {adapter_metrics.get('clip_score', 0):.3f}, Div: {adapter_metrics.get('diversity', 0):.3f})", end="")
                print()
    
    # Print overall metrics
    print("\n" + "-" * 60)
    print("OVERALL METRICS")
    print("-" * 60)
    
    if all_base_clip_scores:
        print(f"\nBase GLIDE Model:")
        print(f"  CLIP Score: {np.mean(all_base_clip_scores):.3f} ± {np.std(all_base_clip_scores):.3f}")
        print(f"  Diversity:  {np.mean(all_base_diversity):.3f} ± {np.std(all_base_diversity):.3f}")
    
    if all_adapter_clip_scores:
        print(f"\nGLIDE with CLIP Adapter:")
        print(f"  CLIP Score: {np.mean(all_adapter_clip_scores):.3f} ± {np.std(all_adapter_clip_scores):.3f}")
        print(f"  Diversity:  {np.mean(all_adapter_diversity):.3f} ± {np.std(all_adapter_diversity):.3f}")
        
        if all_base_clip_scores:
            print(f"\nImprovement with Adapter:")
            print(f"  CLIP Score: {np.mean(all_adapter_clip_scores) - np.mean(all_base_clip_scores):+.3f}")
            print(f"  Diversity:  {np.mean(all_adapter_diversity) - np.mean(all_base_diversity):+.3f}")
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {config.output_dir}")
    if config.use_wandb:
        print(f"WandB project: {config.wandb_project}")
    print("=" * 60)


if __name__ == "__main__":
    main()