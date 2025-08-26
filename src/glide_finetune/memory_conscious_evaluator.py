"""
Memory-conscious CLIP evaluation system for GLIDE fine-tuning.
Designed for 12GB VRAM constraint with careful model loading/unloading.
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch as th
from PIL import Image

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

from .memory_manager import ModelMemoryManager, ModelStateManager, temporary_model_load
from .utils.glide_util import load_model, sample
from .utils.train_util import pred_to_pil

# Initialize logger
logger = get_logger("glide_finetune.memory_conscious_evaluator")

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class EvaluationPhase(Enum):
    """Phases of the memory-conscious evaluation process."""

    SAVING_TRAINING_STATE = "saving_training_state"
    SAMPLING_CURRENT = "sampling_current"
    CLIP_SCORING_CURRENT = "clip_scoring_current"
    LOADING_BASE_MODEL = "loading_base_model"
    SAMPLING_BASE = "sampling_base"
    CLIP_SCORING_BASE = "clip_scoring_base"
    CALCULATING_METRICS = "calculating_metrics"
    RESTORING_TRAINING_STATE = "restoring_training_state"
    COMPLETE = "complete"


@dataclass
class MemoryConstrainedEvalConfig:
    """Configuration for memory-constrained evaluation."""

    prompts_file: Path
    base_model_path: str = ""
    device: str = "cuda"
    variations: int = 1
    seed_offset: int = 1000
    max_prompts: int = 64
    batch_size: int = 1  # Process one image at a time for memory efficiency
    guidance_scale: float = 3.5
    prediction_respacing: str = "50"
    sampler: str = "plms"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    enable_wandb_logging: bool = False
    wandb_project: str = "glide-clip-eval"
    max_memory_usage_pct: float = 85.0  # Conservative limit for 12GB
    temp_dir: str = "/tmp/glide_eval"
    save_sample_images: bool = False  # Save images to disk for debugging


class MemoryConstrainedClipEvaluator:
    """
    CLIP evaluator that carefully manages GPU memory usage.

    Process:
    1. Save current training state to disk
    2. Sample from current model (one image at a time)
    3. Unload current model, load CLIP, score current samples, unload CLIP
    4. Load base model, sample from it, unload base model
    5. Load CLIP again, score base samples, unload CLIP
    6. Calculate win-rate metrics
    7. Restore training state
    """

    def __init__(self, config: MemoryConstrainedEvalConfig):
        self.config = config
        self.memory_manager = ModelMemoryManager(
            device=config.device, max_memory_usage_pct=config.max_memory_usage_pct
        )
        self.state_manager = ModelStateManager(config.temp_dir)

        # Create output directories
        self.eval_dir = Path(config.temp_dir) / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        if config.save_sample_images:
            self.samples_dir = self.eval_dir / "samples"
            self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Load prompts
        self.prompts = self._load_prompts()

        # State tracking
        self.current_phase = EvaluationPhase.SAVING_TRAINING_STATE
        self.evaluation_results = {}

        logger.info("🧪 Memory-Conscious CLIP Evaluator initialized")
        logger.info(f"   Prompts: {len(self.prompts)}")
        logger.info(f"   Variations: {config.variations}")
        logger.info(f"   Memory limit: {config.max_memory_usage_pct}%")
        logger.info(f"   Temp dir: {config.temp_dir}")

    def _load_prompts(self) -> list[str]:
        """Load evaluation prompts from file."""
        with open(self.config.prompts_file, encoding="utf-8") as f:
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

    def _save_sample_images(self, images: list[Image.Image], prefix: str, variation: int):
        """Save sample images to disk for debugging."""
        if not self.config.save_sample_images:
            return

        var_dir = self.samples_dir / f"variation_{variation}"
        var_dir.mkdir(exist_ok=True)

        for i, img in enumerate(images):
            img_path = var_dir / f"{prefix}_prompt_{i:03d}.png"
            img.save(img_path)

    def _load_clip_model(self):
        """Load CLIP model with memory management."""
        if not OPEN_CLIP_AVAILABLE:
            msg = "open_clip is required for CLIP evaluation"
            raise ImportError(msg)

        def clip_loader():
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model,
                pretrained=self.config.clip_pretrained,
                device=self.config.device,
            )
            tokenizer = open_clip.get_tokenizer(self.config.clip_model)
            return model, preprocess, tokenizer

        return self.memory_manager.load_model_safely(clip_loader, "clip_model")

    def _score_images_with_clip(
        self, prompts: list[str], images: list[Image.Image], phase_name: str
    ) -> th.Tensor:
        """Score images using CLIP with memory management."""
        logger.info(f"🔍 CLIP scoring for {phase_name}...")

        with temporary_model_load(self.memory_manager, self._load_clip_model, "clip_scorer") as (
            clip_model,
            preprocess,
            tokenizer,
        ):
            # Process in small batches to save memory
            all_scores = []
            batch_size = min(4, len(images))  # Small batches

            for i in range(0, len(images), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                batch_images = images[i : i + batch_size]

                # Tokenize text
                text_tokens = tokenizer(batch_prompts).to(self.config.device)
                with th.inference_mode():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Process images
                image_tensors = th.stack(
                    [preprocess(img.convert("RGB")) for img in batch_images]
                ).to(self.config.device)

                with th.inference_mode():
                    image_features = clip_model.encode_image(image_tensors)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Calculate similarities
                similarities = (text_features * image_features).sum(dim=-1)
                all_scores.append(similarities.cpu())

                # Clear intermediate tensors
                del text_tokens, text_features, image_tensors, image_features, similarities
                th.cuda.empty_cache()

            return th.cat(all_scores)

    def _sample_from_model(
        self, model, _diffusion, options, prompts: list[str], seeds: list[int], model_name: str
    ) -> list[Image.Image]:
        """Sample images from a model one at a time to save memory."""
        logger.info(f"🎨 Sampling from {model_name}...")

        images = []
        model.eval()

        for i, (prompt, seed) in enumerate(zip(prompts, seeds, strict=False)):
            logger.info(f"  Generating image {i + 1}/{len(prompts)}: {prompt[:50]}...")

            # Set seed for reproducibility
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

            # Sample single image
            with th.inference_mode():
                samples = sample(
                    glide_model=model,
                    glide_options=options,
                    side_x=64,
                    side_y=64,
                    prompt=prompt,
                    batch_size=1,
                    guidance_scale=self.config.guidance_scale,
                    device=self.config.device,
                    prediction_respacing=self.config.prediction_respacing,
                    sampler=self.config.sampler,
                )

            # Convert to PIL
            pil_image = pred_to_pil(samples)
            images.append(pil_image)

            # Clear cache after each sample
            th.cuda.empty_cache()

        return images

    def _load_base_model(self):
        """Load base GLIDE model."""

        def base_loader():
            if self.config.base_model_path and os.path.exists(self.config.base_model_path):
                logger.info(f"Loading base model from {self.config.base_model_path}")
                return load_model(
                    glide_path=self.config.base_model_path, model_type="base", use_fp16=False
                )
            logger.info("Loading base model from OpenAI checkpoint")
            return load_model(model_type="base", use_fp16=False)

        return self.memory_manager.load_model_safely(base_loader, "base_model")

    def run_evaluation(
        self,
        current_model,
        current_diffusion,
        current_options,
        optimizer: th.optim.Optimizer | None = None,
        step: int = 0,
    ) -> dict[str, Any]:
        """
        Run the complete memory-conscious evaluation process.

        Args:
            current_model: Current training model
            current_diffusion: Diffusion for current model
            current_options: Options for current model
            optimizer: Training optimizer (for state saving)
            step: Training step number

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("🚀 Starting memory-conscious CLIP evaluation...")
        start_time = time.time()

        # Initialize results storage
        all_current_images = []
        all_base_images = []
        all_current_scores = []
        all_base_scores = []

        base_seeds = list(range(len(self.prompts)))

        try:
            # Phase 1: Save training state
            self.current_phase = EvaluationPhase.SAVING_TRAINING_STATE
            logger.info(f"📁 Phase 1: {self.current_phase.value}")

            training_state_name = f"training_state_step_{step}"
            self.state_manager.save_model_state(current_model, optimizer, training_state_name)

            # Process each variation
            for variation in range(self.config.variations):
                logger.info(f"\n🔄 Processing variation {variation + 1}/{self.config.variations}")
                seeds = self._generate_seeds(base_seeds, variation)

                # Phase 2: Sample from current model
                self.current_phase = EvaluationPhase.SAMPLING_CURRENT
                logger.info(f"📁 Phase 2: {self.current_phase.value}")

                current_images = self._sample_from_model(
                    current_model,
                    current_diffusion,
                    current_options,
                    self.prompts,
                    seeds,
                    "current_model",
                )
                all_current_images.extend(current_images)
                self._save_sample_images(current_images, "current", variation)

                # Phase 3: Unload current model and score with CLIP
                self.current_phase = EvaluationPhase.CLIP_SCORING_CURRENT
                logger.info(f"📁 Phase 3: {self.current_phase.value}")

                # Temporarily move current model to CPU to free GPU memory
                current_model.cpu()
                self.memory_manager.clear_gpu_cache()

                current_scores = self._score_images_with_clip(
                    self.prompts, current_images, "current model"
                )
                all_current_scores.append(current_scores)

                # Move current model back to GPU for next variation (if any)
                if variation < self.config.variations - 1:
                    current_model.to(self.config.device)

                # Phase 4: Load base model and sample
                self.current_phase = EvaluationPhase.LOADING_BASE_MODEL
                logger.info(f"📁 Phase 4: {self.current_phase.value}")

                with temporary_model_load(
                    self.memory_manager, self._load_base_model, "base_model_temp"
                ) as (base_model, base_diffusion, base_options):
                    self.current_phase = EvaluationPhase.SAMPLING_BASE
                    logger.info(f"📁 Phase 5: {self.current_phase.value}")

                    base_images = self._sample_from_model(
                        base_model, base_diffusion, base_options, self.prompts, seeds, "base_model"
                    )
                    all_base_images.extend(base_images)
                    self._save_sample_images(base_images, "base", variation)

                # Phase 6: Score base images with CLIP
                self.current_phase = EvaluationPhase.CLIP_SCORING_BASE
                logger.info(f"📁 Phase 6: {self.current_phase.value}")

                base_scores = self._score_images_with_clip(self.prompts, base_images, "base model")
                all_base_scores.append(base_scores)

                # Log to WandB if enabled
                if self.config.enable_wandb_logging and WANDB_AVAILABLE and wandb is not None:
                    self._log_variation_to_wandb(
                        current_images, base_images, current_scores, base_scores, variation, step
                    )

            # Phase 7: Calculate final metrics
            self.current_phase = EvaluationPhase.CALCULATING_METRICS
            logger.info(f"📁 Phase 7: {self.current_phase.value}")

            metrics = self._calculate_metrics(all_current_scores, all_base_scores)

            # Phase 8: Restore training state
            self.current_phase = EvaluationPhase.RESTORING_TRAINING_STATE
            logger.info(f"📁 Phase 8: {self.current_phase.value}")

            # Restore training model state
            success = self.state_manager.restore_model_state(
                current_model, optimizer, training_state_name, self.config.device
            )

            if not success:
                logger.info("⚠️  Warning: Failed to restore training state")
            else:
                current_model.train()  # Return to training mode

            # Final metrics and logging
            eval_time = time.time() - start_time
            metrics["evaluation_time"] = eval_time
            metrics["total_images_generated"] = len(all_current_images) + len(all_base_images)

            self.current_phase = EvaluationPhase.COMPLETE

            logger.info("✅ Memory-conscious evaluation complete!")
            logger.info(f"   Evaluation time: {eval_time:.1f}s")
            logger.info(f"   CLIP Score Current: {metrics['clip_score_current']:.4f}")
            logger.info(f"   CLIP Score Base: {metrics['clip_score_base']:.4f}")
            logger.info(f"   Win Rate: {metrics['win_rate_vs_base']:.3f}")

            # Final memory report
            logger.info(self.memory_manager.memory_report())

            # Cleanup old states
            self.state_manager.cleanup_states(keep_latest=2)

            return metrics

        except Exception as e:
            logger.info(f"❌ Evaluation failed during {self.current_phase.value}: {e}")

            # Try to restore training state on failure
            try:
                self.state_manager.restore_model_state(
                    current_model, optimizer, training_state_name, self.config.device
                )
                current_model.train()
            except Exception:
                logger.info("❌ Failed to restore training state after error")

            raise e

    def _log_variation_to_wandb(
        self,
        current_images: list[Image.Image],
        base_images: list[Image.Image],
        current_scores: th.Tensor,
        base_scores: th.Tensor,
        variation: int,
        step: int,
    ):
        """Log variation results to WandB with CLIP scores."""
        if not WANDB_AVAILABLE or wandb is None:
            return

        wandb_images_current = []
        wandb_images_base = []

        for img_cur, img_base, prompt, score_cur, score_base in zip(
            current_images, base_images, self.prompts, current_scores, base_scores, strict=False
        ):
            caption_current = f"CLIP: {score_cur:.3f} | {prompt}"
            caption_base = f"CLIP: {score_base:.3f} | {prompt} (base)"

            wandb_images_current.append(wandb.Image(img_cur, caption=caption_current))
            wandb_images_base.append(wandb.Image(img_base, caption=caption_base))

        wandb.log(
            {
                f"eval_images_current_v{variation}": wandb_images_current,
                f"eval_images_base_v{variation}": wandb_images_base,
            },
            step=step,
        )

    def _calculate_metrics(
        self, current_scores: list[th.Tensor], base_scores: list[th.Tensor]
    ) -> dict[str, Any]:
        """Calculate aggregated metrics from CLIP scores."""
        # Stack all variations
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

        return {
            "clip_score_current": clip_score_current,
            "clip_score_base": clip_score_base,
            "clip_score_delta": clip_score_delta,
            "win_rate_vs_base": win_rate,
            "num_prompts": len(current_mean_per_prompt),
            "num_variations": len(current_scores),
            "wins_count": int(wins.sum().item()),
            "losses_count": int((len(current_mean_per_prompt) - wins.sum()).item()),
        }


def create_memory_conscious_evaluator(
    prompts_file: str,
    base_model_path: str = "",
    device: str = "cuda",
    variations: int = 1,
    max_memory_usage_pct: float = 85.0,
    **kwargs,
) -> MemoryConstrainedClipEvaluator:
    """
    Create a memory-conscious CLIP evaluator.

    Args:
        prompts_file: Path to evaluation prompts
        base_model_path: Path to base model
        device: Device to run on
        variations: Number of variations per prompt
        max_memory_usage_pct: Maximum GPU memory usage percentage
        **kwargs: Additional config options

    Returns:
        Configured evaluator
    """
    config = MemoryConstrainedEvalConfig(
        prompts_file=Path(prompts_file),
        base_model_path=base_model_path,
        device=device,
        variations=variations,
        max_memory_usage_pct=max_memory_usage_pct,
        **kwargs,
    )

    return MemoryConstrainedClipEvaluator(config)
