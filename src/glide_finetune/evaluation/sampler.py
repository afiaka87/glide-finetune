"""
Sampling utilities for evaluation with consistent seed management.
"""

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import nn
from glide_text2im.tokenizer.bpe import Encoder

from glide_finetune.model_loader import ModelInfo, UnifiedModelLoader
from glide_finetune.utils.glide_util import sample
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.train_util import pred_to_pil

logger = get_logger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for sampling during evaluation."""

    batch_size: int = 4
    guidance_scale: float = 3.5
    prediction_respacing: str = "50"
    sampler: str = "plms"
    device: str | torch.device = "cuda"
    image_size: int = 64
    seed_offset: int = 1000
    use_fp16: bool = False


class GlideSampler:
    """Wrapper for GLIDE sampling that handles model switching for evaluation."""

    def __init__(
        self,
        sampling_config: SamplingConfig,
        base_model_path: str | None = None,
    ) -> None:
        """
        Initialize GLIDE sampler for evaluation.
        
        Args:
            sampling_config: Sampling configuration
            base_model_path: Path to base model checkpoint (None for OpenAI default)
        """
        self.config = sampling_config
        self.device = (
            sampling_config.device
            if isinstance(sampling_config.device, torch.device)
            else torch.device(sampling_config.device)
        )

        # Models
        self.current_model_info: ModelInfo | None = None
        self.base_model_info: ModelInfo | None = None

        # Load base model
        self._load_base_model(base_model_path)

    def _load_base_model(self, base_model_path: str | None) -> None:
        """Load the base GLIDE model for comparison."""
        logger.info("Loading base model for comparison...")

        if base_model_path:
            logger.info(f"Loading base model from {base_model_path}")
        else:
            logger.info("Loading base model from OpenAI checkpoint")

        self.base_model_info = UnifiedModelLoader.load_for_inference(
            model_type="base",
            checkpoint_path=base_model_path,
            device=self.device,
            use_fp16=self.config.use_fp16,
            verbose=False,
        )

        # Ensure model is in eval mode
        self.base_model_info.model.eval()

    def set_current_model(
        self,
        model: nn.Module,
        diffusion: Any,
        options: dict[str, Any],
    ) -> None:
        """
        Set the current fine-tuned model for comparison.
        
        Args:
            model: The fine-tuned model
            diffusion: Diffusion instance
            options: Model options dictionary
        """
        # Get tokenizer from model
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = Encoder()

        self.current_model_info = ModelInfo(
            model=model,
            diffusion=diffusion,
            options=options,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            model_type="base",
        )

        # Ensure model is in eval mode
        model.eval()

    def set_current_model_from_info(self, model_info: ModelInfo) -> None:
        """
        Set current model from ModelInfo object.
        
        Args:
            model_info: ModelInfo object containing model and metadata
        """
        self.current_model_info = model_info
        model_info.model.eval()

    @torch.inference_mode()
    def sample_current(
        self,
        prompts: list[str],
        seeds: list[int] | None = None,
    ) -> list[Image.Image]:
        """
        Sample from the current fine-tuned model.
        
        Args:
            prompts: List of text prompts
            seeds: Optional list of random seeds
            
        Returns:
            List of PIL images
        """
        if self.current_model_info is None:
            msg = "Current model not set. Call set_current_model() first."
            raise ValueError(msg)

        return self._sample_with_model(
            prompts,
            seeds,
            self.current_model_info,
        )

    @torch.inference_mode()
    def sample_base(
        self,
        prompts: list[str],
        seeds: list[int] | None = None,
    ) -> list[Image.Image]:
        """
        Sample from the base model.
        
        Args:
            prompts: List of text prompts
            seeds: Optional list of random seeds
            
        Returns:
            List of PIL images
        """
        if self.base_model_info is None:
            msg = "Base model not loaded"
            raise ValueError(msg)

        return self._sample_with_model(
            prompts,
            seeds,
            self.base_model_info,
        )

    def _sample_with_model(
        self,
        prompts: list[str],
        seeds: list[int] | None,
        model_info: ModelInfo,
    ) -> list[Image.Image]:
        """
        Sample images using the specified model.
        
        Args:
            prompts: List of text prompts
            seeds: Optional list of random seeds
            model_info: Model information object
            
        Returns:
            List of PIL images
        """
        images: list[Image.Image] = []

        # Use provided seeds or generate default
        if seeds is None:
            seeds = list(range(len(prompts)))

        for prompt, seed in zip(prompts, seeds, strict=False):
            # Set seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Sample single image
            samples = sample(
                glide_model=model_info.model,
                glide_options=dict(model_info.options),
                side_x=self.config.image_size,
                side_y=self.config.image_size,
                prompt=prompt,
                batch_size=1,
                guidance_scale=self.config.guidance_scale,
                device=str(self.device),
                prediction_respacing=self.config.prediction_respacing,
                sampler=self.config.sampler,
            )

            # Convert to PIL
            pil_images = pred_to_pil(samples)
            images.extend(pil_images)

        return images

    @torch.inference_mode()
    def sample_batch(
        self,
        prompts: list[str],
        num_variations: int = 1,
        base_seed: int = 0,
        use_current: bool = True,
    ) -> list[list[Image.Image]]:
        """
        Sample multiple variations for each prompt.
        
        Args:
            prompts: List of text prompts
            num_variations: Number of variations per prompt
            base_seed: Base seed for reproducibility
            use_current: If True, use current model; else use base model
            
        Returns:
            List of lists, where each inner list contains variations for one prompt
        """
        all_images = []

        for prompt_idx, prompt in enumerate(prompts):
            prompt_images = []

            for variation in range(num_variations):
                seed = base_seed + prompt_idx + variation * self.config.seed_offset

                if use_current:
                    images = self.sample_current([prompt], [seed])
                else:
                    images = self.sample_base([prompt], [seed])

                prompt_images.extend(images)

            all_images.append(prompt_images)

        return all_images
