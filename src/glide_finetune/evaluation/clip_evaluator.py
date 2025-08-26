"""
Main CLIP evaluation module for comparing fine-tuned models against base models.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from glide_finetune.model_loader import ModelInfo
from glide_finetune.utils.logging_utils import get_logger

from .sampler import GlideSampler, SamplingConfig
from .scorer import ClipScorer

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    prompts_file: str | Path
    base_model_path: str | None = None
    batch_size: int = 4
    guidance_scale: float = 3.5
    prediction_respacing: str = "50"
    sampler: str = "plms"
    device: str | torch.device = "cuda"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    variations: int = 1
    seed_offset: int = 1000
    image_size: int = 64
    max_prompts: int | None = 64
    use_fp16: bool = False


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Primary metrics
    clip_score_current: float
    clip_score_base: float
    clip_score_delta: float
    win_rate_vs_base: float

    # Statistics
    clip_std_current: float
    clip_std_base: float
    num_prompts: int
    num_variations: int
    wins_count: int
    losses_count: int

    # Additional metrics
    evaluation_time: float
    total_images_generated: int

    # Per-sample statistics
    clip_scores_current_all: list[float] = field(default_factory=list)
    clip_scores_base_all: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "clip_score_current": self.clip_score_current,
            "clip_score_base": self.clip_score_base,
            "clip_score_delta": self.clip_score_delta,
            "win_rate_vs_base": self.win_rate_vs_base,
            "clip_std_current": self.clip_std_current,
            "clip_std_base": self.clip_std_base,
            "num_prompts": self.num_prompts,
            "num_variations": self.num_variations,
            "wins_count": self.wins_count,
            "losses_count": self.losses_count,
            "evaluation_time": self.evaluation_time,
            "total_images_generated": self.total_images_generated,
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"CLIP Evaluation Results:\n"
            f"  Current Model: {self.clip_score_current:.4f} ± {self.clip_std_current:.4f}\n"
            f"  Base Model:    {self.clip_score_base:.4f} ± {self.clip_std_base:.4f}\n"
            f"  Delta:         {self.clip_score_delta:+.4f}\n"
            f"  Win Rate:      {self.win_rate_vs_base:.1%} ({self.wins_count}/{self.num_prompts})\n"
            f"  Samples:       {self.num_prompts} prompts × {self.num_variations} variations\n"
            f"  Time:          {self.evaluation_time:.1f}s"
        )


class ClipEvaluator:
    """Main evaluation runner that coordinates sampling and scoring."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = (
            config.device
            if isinstance(config.device, torch.device)
            else torch.device(config.device)
        )

        # Create sampling config
        sampling_config = SamplingConfig(
            batch_size=config.batch_size,
            guidance_scale=config.guidance_scale,
            prediction_respacing=config.prediction_respacing,
            sampler=config.sampler,
            device=self.device,
            image_size=config.image_size,
            seed_offset=config.seed_offset,
            use_fp16=config.use_fp16,
        )

        # Initialize components
        self.sampler = GlideSampler(sampling_config, config.base_model_path)
        self.clip_scorer = ClipScorer(
            config.clip_model,
            config.clip_pretrained,
            self.device,
        )

        # Load prompts
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[str]:
        """Load prompts from file."""
        prompts_path = Path(self.config.prompts_file)

        if not prompts_path.exists():
            msg = f"Prompts file not found: {prompts_path}"
            raise FileNotFoundError(msg)

        with open(prompts_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Limit to max_prompts if specified
        if self.config.max_prompts is not None:
            lines = lines[:self.config.max_prompts]
            if len(lines) < self.config.max_prompts:
                logger.warning(
                    f"Only {len(lines)} prompts available, "
                    f"expected {self.config.max_prompts}"
                )

        logger.info(f"Loaded {len(lines)} prompts from {prompts_path}")
        return lines

    def run_evaluation(
        self,
        current_model: nn.Module,
        current_diffusion: Any,
        current_options: dict[str, Any],
    ) -> EvaluationMetrics:
        """
        Run full evaluation comparing current model to base model.
        
        Args:
            current_model: The fine-tuned model
            current_diffusion: Diffusion instance
            current_options: Model options dictionary
            
        Returns:
            EvaluationMetrics object with results
        """
        logger.info(
            f"Running evaluation: {len(self.prompts)} prompts, "
            f"{self.config.variations} variations"
        )

        # Set current model
        self.sampler.set_current_model(current_model, current_diffusion, current_options)

        # Base seeds for reproducibility
        base_seeds = list(range(len(self.prompts)))

        all_clip_scores_current: list[torch.Tensor] = []
        all_clip_scores_base: list[torch.Tensor] = []

        start_time = time.time()

        for variation in range(self.config.variations):
            logger.info(f"Processing variation {variation + 1}/{self.config.variations}")

            # Generate seeds for this variation
            seeds = [
                seed + variation * self.config.seed_offset
                for seed in base_seeds
            ]

            # Sample from both models
            logger.info("  Sampling from current model...")
            current_images = self.sampler.sample_current(self.prompts, seeds)

            logger.info("  Sampling from base model...")
            base_images = self.sampler.sample_base(self.prompts, seeds)

            # Calculate CLIP scores
            logger.info("  Calculating CLIP scores...")
            clip_scores_current = self.clip_scorer.score_images(
                self.prompts, current_images, return_features=False
            )
            clip_scores_base = self.clip_scorer.score_images(
                self.prompts, base_images, return_features=False
            )

            assert isinstance(clip_scores_current, torch.Tensor)
            assert isinstance(clip_scores_base, torch.Tensor)
            all_clip_scores_current.append(clip_scores_current)
            all_clip_scores_base.append(clip_scores_base)

        # Calculate metrics
        eval_time = time.time() - start_time
        metrics = self._calculate_metrics(
            all_clip_scores_current,
            all_clip_scores_base,
            eval_time,
        )

        logger.info(str(metrics))
        return metrics

    def run_evaluation_from_model_info(
        self,
        current_model_info: ModelInfo,
    ) -> EvaluationMetrics:
        """
        Run evaluation using ModelInfo object.
        
        Args:
            current_model_info: ModelInfo object for current model
            
        Returns:
            EvaluationMetrics object with results
        """
        self.sampler.set_current_model_from_info(current_model_info)
        return self.run_evaluation(
            current_model_info.model,
            current_model_info.diffusion,
            dict(current_model_info.options),
        )

    def _calculate_metrics(
        self,
        current_scores: list[torch.Tensor],
        base_scores: list[torch.Tensor],
        eval_time: float,
    ) -> EvaluationMetrics:
        """
        Calculate aggregated metrics from CLIP scores.
        
        Args:
            current_scores: List of score tensors for current model
            base_scores: List of score tensors for base model
            eval_time: Time taken for evaluation
            
        Returns:
            EvaluationMetrics object
        """
        # Stack all variations: [variations, prompts]
        current_scores_tensor = torch.stack(current_scores)
        base_scores_tensor = torch.stack(base_scores)

        # Average across variations for each prompt
        current_mean_per_prompt = current_scores_tensor.mean(dim=0)
        base_mean_per_prompt = base_scores_tensor.mean(dim=0)

        # Overall metrics
        clip_score_current = current_mean_per_prompt.mean().item()
        clip_score_base = base_mean_per_prompt.mean().item()
        clip_score_delta = clip_score_current - clip_score_base

        # Win rate: fraction of prompts where current > base
        wins = (current_mean_per_prompt > base_mean_per_prompt).float()
        win_rate = wins.mean().item()
        wins_count = int(wins.sum().item())
        losses_count = len(current_mean_per_prompt) - wins_count

        # Standard deviations
        clip_std_current = current_mean_per_prompt.std().item()
        clip_std_base = base_mean_per_prompt.std().item()

        # Flatten all scores
        current_scores_all = torch.cat(current_scores)
        base_scores_all = torch.cat(base_scores)

        return EvaluationMetrics(
            clip_score_current=clip_score_current,
            clip_score_base=clip_score_base,
            clip_score_delta=clip_score_delta,
            win_rate_vs_base=win_rate,
            clip_std_current=clip_std_current,
            clip_std_base=clip_std_base,
            num_prompts=len(self.prompts),
            num_variations=self.config.variations,
            wins_count=wins_count,
            losses_count=losses_count,
            evaluation_time=eval_time,
            total_images_generated=len(current_scores_all) + len(base_scores_all),
            clip_scores_current_all=current_scores_all.cpu().tolist(),
            clip_scores_base_all=base_scores_all.cpu().tolist(),
        )


def create_evaluation_runner(
    prompts_file: str | Path,
    base_model_path: str | None = None,
    device: str | torch.device = "cuda",
    variations: int = 1,
    guidance_scale: float = 3.5,
    **kwargs: Any,
) -> ClipEvaluator:
    """
    Factory function to create an evaluation runner.
    
    Args:
        prompts_file: Path to file containing evaluation prompts
        base_model_path: Path to base model (None for OpenAI default)
        device: Device to run on
        variations: Number of variations per prompt
        guidance_scale: Guidance scale for sampling
        **kwargs: Additional config parameters
        
    Returns:
        Configured ClipEvaluator instance
    """
    config = EvaluationConfig(
        prompts_file=prompts_file,
        base_model_path=base_model_path,
        device=device,
        variations=variations,
        guidance_scale=guidance_scale,
        **kwargs,
    )

    return ClipEvaluator(config)


def run_clip_evaluation(
    current_model: nn.Module,
    current_diffusion: Any,
    current_options: dict[str, Any],
    prompts_file: str | Path = "experiments/captions/evaluation1.txt",
    base_model_path: str | None = None,
    variations: int = 1,
    device: str | torch.device = "cuda",
    **kwargs: Any,
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
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = create_evaluation_runner(
        prompts_file=prompts_file,
        base_model_path=base_model_path,
        device=device,
        variations=variations,
        **kwargs,
    )

    metrics = evaluator.run_evaluation(
        current_model,
        current_diffusion,
        current_options,
    )

    return metrics.to_dict()
