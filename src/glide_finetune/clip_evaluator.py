"""
Enhanced evaluation module for GLIDE fine-tuning with CLIP scoring and base model comparison.
Based on the ChatGPT conversation requirements for win-rate metrics and CLIP evaluation.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as th
from PIL import Image

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

from .utils.glide_util import load_model, sample
from .utils.train_util import pred_to_pil

# Initialize logger
logger = get_logger("glide_finetune.clip_evaluator")

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("Warning: open_clip not available. CLIP evaluation will be disabled.")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
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
    enable_wandb_logging: bool = False  # Enable WandB image logging with CLIP scores
    wandb_project: str = "glide-clip-eval"  # WandB project name


class ClipScorer:
    """CLIP scoring utility for text-image similarity."""

    def __init__(
        self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"
    ):
        if not OPEN_CLIP_AVAILABLE:
            msg = "open_clip is required for CLIP evaluation"
            raise ImportError(msg)

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @th.inference_mode()
    def score_images(self, prompts: list[str], images: list[Image.Image]) -> th.Tensor:
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
        image_tensors = th.stack([self.preprocess(img.convert("RGB")) for img in images]).to(
            self.device
        )
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        return (text_features * image_features).sum(dim=-1)


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
            base_path = Path(__file__).parent / ".." / "glide_model_cache" / "base.pt"

        if base_path.exists():
            logger.info(f"Loading base model from {base_path}")
            self.base_model, self.base_diffusion, self.base_options = load_model(
                glide_path=base_path, model_type="base", use_fp16=False
            )
            self.base_model.to(self.config.device)
            self.base_model.eval()
        else:
            logger.info(f"Loading base model from OpenAI checkpoint (no cache found at {base_path})")
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
    def sample_current(self, prompts: list[str], seeds: list[int]) -> list[Image.Image]:
        """Sample from the current fine-tuned model."""
        if self.current_model is None:
            msg = "Current model not set. Call set_current_model() first."
            raise ValueError(msg)

        return self._sample_with_model(
            prompts, seeds, self.current_model, self.current_diffusion, self.current_options
        )

    @th.inference_mode()
    def sample_base(self, prompts: list[str], seeds: list[int]) -> list[Image.Image]:
        """Sample from the base model."""
        return self._sample_with_model(
            prompts, seeds, self.base_model, self.base_diffusion, self.base_options
        )

    def _sample_with_model(
        self, prompts: list[str], seeds: list[int], model, _diffusion, options
    ) -> list[Image.Image]:
        """Sample images using the specified model."""
        images: list[Image.Image] = []

        for prompt, seed in zip(prompts, seeds, strict=False):
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

            # Convert to PIL - pred_to_pil returns a single image
            pil_image = pred_to_pil(samples)
            images.append(pil_image)

        return images


class EvaluationRunner:
    """Main evaluation runner that coordinates sampling and scoring."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sampler = GlideSampler(config)

        if OPEN_CLIP_AVAILABLE:
            self.clip_scorer = ClipScorer(config.clip_model, config.clip_pretrained, config.device)
        else:
            msg = "CLIP evaluation requires open_clip package"
            raise ImportError(msg)

        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[str]:
        """Load prompts from file."""
        with Path(self.config.prompts_file).open(encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Take first max_prompts
        prompts = lines[: self.config.max_prompts]
        if len(prompts) < self.config.max_prompts:
            logger.info(
                f"Warning: Only {len(prompts)} prompts available, expected {self.config.max_prompts}"
            )

        return prompts

    def _generate_seeds(self, base_seeds: list[int], variation: int) -> list[int]:
        """Generate seeds for a specific variation."""
        return [seed + variation * self.config.seed_offset for seed in base_seeds]

    def _log_to_wandb(
        self,
        current_images: list[Image.Image],
        base_images: list[Image.Image],
        prompts: list[str],
        clip_scores_current: th.Tensor,
        clip_scores_base: th.Tensor,
        variation: int,
        step: int = 0,
    ):
        """Log images to WandB with CLIP scores in captions."""
        if not self.config.enable_wandb_logging or not WANDB_AVAILABLE or wandb is None:
            return

        wandb_images_current = []
        wandb_images_base = []

        for img_cur, img_base, prompt, score_cur, score_base in zip(
            current_images, base_images, prompts, clip_scores_current, clip_scores_base, strict=False
        ):
            # Create captions with CLIP scores
            caption_current = f"CLIP: {score_cur:.3f} | {prompt}"
            caption_base = f"CLIP: {score_base:.3f} | {prompt} (base)"

            wandb_images_current.append(wandb.Image(img_cur, caption=caption_current))
            wandb_images_base.append(wandb.Image(img_base, caption=caption_base))

        # Log images in batches to avoid overwhelming WandB
        wandb.log(
            {
                f"eval_images_current_v{variation}": wandb_images_current,
                f"eval_images_base_v{variation}": wandb_images_base,
            },
            step=step,
        )

    def run_evaluation(
        self, current_model, current_diffusion, current_options, step: int = 0
    ) -> dict[str, Any]:
        """
        Run full evaluation comparing current model to base model.

        Args:
            current_model: Current GLIDE model
            current_diffusion: Diffusion instance
            current_options: Model options dict
            step: Training step for WandB logging

        Returns:
            Dictionary with evaluation metrics including CLIP scores and win rates.
        """
        logger.info(
            f"Running evaluation with {len(self.prompts)} prompts, {self.config.variations} variations"
        )

        # Set current model
        self.sampler.set_current_model(current_model, current_diffusion, current_options)

        # Base seeds (0 to num_prompts-1)
        base_seeds = list(range(len(self.prompts)))

        all_clip_scores_current = []
        all_clip_scores_base = []
        all_current_images = []
        all_base_images = []

        start_time = time.time()

        for variation in range(self.config.variations):
            logger.info(f"Processing variation {variation + 1}/{self.config.variations}")

            # Generate seeds for this variation
            seeds = self._generate_seeds(base_seeds, variation)

            # Sample from both models with identical seeds
            logger.info("  Sampling from current model...")
            current_images = self.sampler.sample_current(self.prompts, seeds)

            logger.info("  Sampling from base model...")
            base_images = self.sampler.sample_base(self.prompts, seeds)

            # Calculate CLIP scores
            logger.info("  Calculating CLIP scores...")
            clip_scores_current = self.clip_scorer.score_images(self.prompts, current_images)
            clip_scores_base = self.clip_scorer.score_images(self.prompts, base_images)

            # Log to WandB with CLIP scores in captions
            self._log_to_wandb(
                current_images,
                base_images,
                self.prompts,
                clip_scores_current,
                clip_scores_base,
                variation,
                step,
            )

            all_clip_scores_current.append(clip_scores_current)
            all_clip_scores_base.append(clip_scores_base)
            all_current_images.extend(current_images)
            all_base_images.extend(base_images)

        # Aggregate results
        eval_time = time.time() - start_time
        metrics = self._calculate_metrics(all_clip_scores_current, all_clip_scores_base)
        metrics["evaluation_time"] = eval_time
        metrics["total_images_generated"] = len(all_current_images) + len(all_base_images)

        # Log summary metrics to WandB
        if self.config.enable_wandb_logging and WANDB_AVAILABLE and wandb is not None:
            wandb.log(
                {
                    "eval/clip_score_current": metrics["clip_score_current"],
                    "eval/clip_score_base": metrics["clip_score_base"],
                    "eval/clip_score_delta": metrics["clip_score_delta"],
                    "eval/win_rate_vs_base": metrics["win_rate_vs_base"],
                    "eval/evaluation_time": eval_time,
                },
                step=step,
            )

        return metrics

    def _calculate_metrics(
        self, current_scores: list[th.Tensor], base_scores: list[th.Tensor]
    ) -> dict[str, Any]:
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
            "wins_count": int(wins.sum().item()),
            "losses_count": int((len(current_mean_per_prompt) - wins.sum()).item()),
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
    enable_wandb_logging: bool = False,
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
        enable_wandb_logging: Enable WandB image logging with CLIP scores
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
        enable_wandb_logging=enable_wandb_logging,
        **kwargs,
    )

    return EvaluationRunner(config)


def run_clip_evaluation(
    current_model,
    current_diffusion,
    current_options,
    prompts_file: str = "experiments/captions/evaluation1.txt",
    base_model_path: str = "",
    variations: int = 1,
    device: str = "cuda",
    step: int = 0,
    enable_wandb_logging: bool = False,
    **kwargs,
) -> dict[str, Any]:
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
        step: Training step for WandB logging
        enable_wandb_logging: Enable WandB image logging with CLIP scores
        **kwargs: Additional configuration options

    Returns:
        Dictionary with evaluation metrics
    """
    runner = create_evaluation_runner(
        prompts_file=prompts_file,
        base_model_path=base_model_path,
        device=device,
        variations=variations,
        enable_wandb_logging=enable_wandb_logging,
        **kwargs,
    )

    return runner.run_evaluation(current_model, current_diffusion, current_options, step)
