# a backup, just in case.
"""
Enhanced evaluation module for GLIDE fine-tuning with CLIP scoring and base model comparison.
Based on the ChatGPT conversation requirements for win-rate metrics and CLIP evaluation.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open_clip

from .glide_util import load_model, sample, get_tokens_and_mask
from .train_util import pred_to_pil

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    prompts_file: Path
    base_model_path: str = ""  # Path to base model cache
    batch_size: int = 4  # Smaller batches for 64x64 images
    guidance_scale: float = 3.5
    prediction_respacing: str = "50"
    sampler: str = "plms"
    device: str = "cuda"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    variations: int = 1  # Number of variations per prompt
    seed_offset: int = 1000  # Offset between variations
    image_size: int = 64  # GLIDE base model outputs 64x64
    max_prompts: int = 64  # Use first 64 prompts


class ClipScorer:
    """CLIP scoring utility for text-image similarity."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @th.inference_mode()
    def score_images(self, prompts: List[str], images: List[Image.Image]) -> th.Tensor:
        """
        Calculate CLIP scores for text-image pairs.

        Args:
            prompts: List of text prompts
            images: List of PIL images

        Returns:
            Tensor of similarity scores [N]
        """
        assert len(prompts) == len(images), "Prompts and images must have same length"

        # Tokenize text
        text_tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Process images
        image_tensors = th.stack(
            [self.preprocess(img.convert("RGB")) for img in images]
        ).to(self.device)
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        similarities = (text_features * image_features).sum(dim=-1)
        return similarities


class GlideSampler:
    """Wrapper for GLIDE sampling that handles model switching."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.current_model = None
        self.current_diffusion = None
        self.current_options = None
        self.base_model = None
        self.base_diffusion = None
        self.base_options = None
        self._load_base_model()

    def _load_base_model(self):
        """Load the base GLIDE model from cache."""
        if self.config.base_model_path:
            base_path = self.config.base_model_path
        else:
            # Use default cached base model
            base_path = os.path.join(
                os.path.dirname(__file__), "..", "glide_model_cache", "base.pt"
            )

        if os.path.exists(base_path):
            print(f"Loading base model from {base_path}")
            self.base_model, self.base_diffusion, self.base_options = load_model(
                glide_path=base_path, model_type="base", use_fp16=False
            )
            self.base_model.to(self.config.device)
            self.base_model.eval()
        else:
            print(
                f"Loading base model from OpenAI checkpoint (no cache found at {base_path})"
            )
            self.base_model, self.base_diffusion, self.base_options = load_model(
                model_type="base", use_fp16=False
            )
            self.base_model.to(self.config.device)
            self.base_model.eval()

    def set_current_model(self, model, diffusion, options):
        """Set the current fine-tuned model for comparison."""
        self.current_model = model
        self.current_diffusion = diffusion
        self.current_options = options

    @th.inference_mode()
    def sample_current(self, prompts: List[str], seeds: List[int]) -> List[Image.Image]:
        """Sample from the current fine-tuned model."""
        if self.current_model is None:
            raise ValueError("Current model not set. Call set_current_model() first.")

        return self._sample_with_model(
            prompts,
            seeds,
            self.current_model,
            self.current_diffusion,
            self.current_options,
        )

    @th.inference_mode()
    def sample_base(self, prompts: List[str], seeds: List[int]) -> List[Image.Image]:
        """Sample from the base model."""
        return self._sample_with_model(
            prompts, seeds, self.base_model, self.base_diffusion, self.base_options
        )

    def _sample_with_model(
        self, prompts: List[str], seeds: List[int], model, diffusion, options
    ) -> List[Image.Image]:
        """Sample images using the specified model."""
        images = []

        for prompt, seed in zip(prompts, seeds):
            # Set seed for reproducibility
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

            # Sample single image
            samples = sample(
                glide_model=model,
                glide_options=options,
                side_x=self.config.image_size,
                side_y=self.config.image_size,
                prompt=prompt,
                batch_size=1,
                guidance_scale=self.config.guidance_scale,
                device=self.config.device,
                prediction_respacing=self.config.prediction_respacing,
                sampler=self.config.sampler,
            )

            # Convert to PIL
            pil_images = pred_to_pil(samples)
            images.extend(pil_images)

        return images


class EvaluationRunner:
    """Main evaluation runner that coordinates sampling and scoring."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sampler = GlideSampler(config)
        self.clip_scorer = ClipScorer(
            config.clip_model, config.clip_pretrained, config.device
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> List[str]:
        """Load prompts from file."""
        with open(self.config.prompts_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Take first max_prompts
        prompts = lines[: self.config.max_prompts]
        if len(prompts) < self.config.max_prompts:
            print(
                f"Warning: Only {len(prompts)} prompts available, expected {self.config.max_prompts}"
            )

        return prompts

    def _generate_seeds(self, base_seeds: List[int], variation: int) -> List[int]:
        """Generate seeds for a specific variation."""
        return [seed + variation * self.config.seed_offset for seed in base_seeds]

    def run_evaluation(
        self, current_model, current_diffusion, current_options
    ) -> Dict[str, Any]:
        """
        Run full evaluation comparing current model to base model.

        Returns:
            Dictionary with evaluation metrics including CLIP scores and win rates.
        """
        print(
            f"Running evaluation with {len(self.prompts)} prompts, {self.config.variations} variations"
        )

        # Set current model
        self.sampler.set_current_model(
            current_model, current_diffusion, current_options
        )

        # Base seeds (0 to num_prompts-1)
        base_seeds = list(range(len(self.prompts)))

        all_clip_scores_current = []
        all_clip_scores_base = []
        all_current_images = []
        all_base_images = []

        start_time = time.time()

        for variation in range(self.config.variations):
            print(f"Processing variation {variation + 1}/{self.config.variations}")

            # Generate seeds for this variation
            seeds = self._generate_seeds(base_seeds, variation)

            # Sample from both models with identical seeds
            print("  Sampling from current model...")
            current_images = self.sampler.sample_current(self.prompts, seeds)

            print("  Sampling from base model...")
            base_images = self.sampler.sample_base(self.prompts, seeds)

            # Calculate CLIP scores
            print("  Calculating CLIP scores...")
            clip_scores_current = self.clip_scorer.score_images(
                self.prompts, current_images
            )
            clip_scores_base = self.clip_scorer.score_images(self.prompts, base_images)

            all_clip_scores_current.append(clip_scores_current)
            all_clip_scores_base.append(clip_scores_base)
            all_current_images.extend(current_images)
            all_base_images.extend(base_images)

        # Aggregate results
        eval_time = time.time() - start_time
        metrics = self._calculate_metrics(all_clip_scores_current, all_clip_scores_base)
        metrics["evaluation_time"] = eval_time
        metrics["total_images_generated"] = len(all_current_images) + len(
            all_base_images
        )

        return metrics

    def _calculate_metrics(
        self, current_scores: List[th.Tensor], base_scores: List[th.Tensor]
    ) -> Dict[str, Any]:
        """Calculate aggregated metrics from CLIP scores."""
        # Stack all variations: [variations, prompts]
        current_scores_tensor = th.stack(current_scores)  # [V, P]
        base_scores_tensor = th.stack(base_scores)  # [V, P]

        # Average across variations for each prompt
        current_mean_per_prompt = current_scores_tensor.mean(dim=0)  # [P]
        base_mean_per_prompt = base_scores_tensor.mean(dim=0)  # [P]

        # Overall metrics
        clip_score_current = current_mean_per_prompt.mean().item()
        clip_score_base = base_mean_per_prompt.mean().item()
        clip_score_delta = clip_score_current - clip_score_base

        # Win rate: fraction of prompts where current > base
        wins = (current_mean_per_prompt > base_mean_per_prompt).float()
        win_rate = wins.mean().item()

        # Standard deviations
        clip_std_current = current_mean_per_prompt.std().item()
        clip_std_base = base_mean_per_prompt.std().item()

        # Per-variation statistics
        current_scores_all = th.cat(current_scores)  # [V*P]
        base_scores_all = th.cat(base_scores)  # [V*P]

        return {
            "clip_score_current": clip_score_current,
            "clip_score_base": clip_score_base,
            "clip_score_delta": clip_score_delta,
            "win_rate_vs_base": win_rate,
            "clip_std_current": clip_std_current,
            "clip_std_base": clip_std_base,
            "num_prompts": len(current_mean_per_prompt),
            "num_variations": len(current_scores),
            "wins_count": wins.sum().item(),
            "losses_count": (len(current_mean_per_prompt) - wins.sum()).item(),
            # Additional statistics
            "clip_score_current_all_samples": current_scores_all.mean().item(),
            "clip_score_base_all_samples": base_scores_all.mean().item(),
            "clip_std_current_all_samples": current_scores_all.std().item(),
            "clip_std_base_all_samples": base_scores_all.std().item(),
        }


def create_evaluation_runner(
    prompts_file: str,
    base_model_path: str = "",
    device: str = "cuda",
    variations: int = 1,
    guidance_scale: float = 3.5,
    **kwargs,
) -> EvaluationRunner:
    """
    Factory function to create an evaluation runner.

    Args:
        prompts_file: Path to file containing evaluation prompts
        base_model_path: Path to base model (empty for OpenAI default)
        device: Device to run on
        variations: Number of variations per prompt
        guidance_scale: Guidance scale for sampling
        **kwargs: Additional config parameters

    Returns:
        Configured EvaluationRunner instance
    """
    config = EvaluationConfig(
        prompts_file=Path(prompts_file),
        base_model_path=base_model_path,
        device=device,
        variations=variations,
        guidance_scale=guidance_scale,
        **kwargs,
    )

    return EvaluationRunner(config)


# Convenience function for quick evaluation
def run_clip_evaluation(
    current_model,
    current_diffusion,
    current_options,
    prompts_file: str = "experiments/captions/evaluation1.txt",
    base_model_path: str = "",
    variations: int = 1,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick evaluation function that can be called from training loops.

    Args:
        current_model: The fine-tuned GLIDE model
        current_diffusion: Diffusion instance for current model
        current_options: Options dict for current model
        prompts_file: Path to evaluation prompts
        base_model_path: Path to base model cache
        variations: Number of sampling variations per prompt
        device: Device to run evaluation on
        **kwargs: Additional configuration options

    Returns:
        Dictionary with evaluation metrics
    """
    runner = create_evaluation_runner(
        prompts_file=prompts_file,
        base_model_path=base_model_path,
        device=device,
        variations=variations,
        **kwargs,
    )

    return runner.run_evaluation(current_model, current_diffusion, current_options)
