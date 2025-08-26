"""
Evaluation module for GLIDE fine-tuning.

Provides CLIP-based evaluation metrics and model comparison utilities.
"""

from .clip_evaluator import (
    ClipEvaluator,
    EvaluationConfig,
    EvaluationMetrics,
    create_evaluation_runner,
    run_clip_evaluation,
)
from .sampler import GlideSampler, SamplingConfig
from .scorer import ClipScorer

__all__ = [
    "ClipEvaluator",
    "ClipScorer",
    "EvaluationConfig",
    "EvaluationMetrics",
    "GlideSampler",
    "SamplingConfig",
    "create_evaluation_runner",
    "run_clip_evaluation",
]
