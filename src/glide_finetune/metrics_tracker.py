"""
Comprehensive metrics tracking for GLIDE training.
Inspired by VibeTune's metrics system with rolling averages and detailed logging.
Includes CLIP evaluation and base model comparison functionality.
"""

import time
import traceback
from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch as th
from PIL import Image

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

# Import memory-conscious evaluator (conditionally imported later if needed)
try:
    from .memory_conscious_evaluator import create_memory_conscious_evaluator
except ImportError:
    create_memory_conscious_evaluator = None  # type: ignore[assignment]

# Initialize logger
logger = get_logger("glide_finetune.metrics_tracker")


class RollingAverage:
    """Rolling average calculator for smooth metrics."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> None:
        """Add a new value to the rolling average."""
        self.values.append(value)

    @property
    def avg(self) -> float:
        """Get the current rolling average."""
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def count(self) -> int:
        """Get the number of values in the window."""
        return len(self.values)


class MetricsTracker:
    """Comprehensive metrics tracker for training monitoring."""

    def __init__(
        self,
        window_size: int = 100,
        enable_clip_eval: bool = False,
        clip_eval_config: dict[str, Any] | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.window_size = window_size
        self.rolling_metrics: dict[str, RollingAverage] = defaultdict(lambda: RollingAverage(window_size))
        self.step_times: deque[float] = deque(maxlen=window_size)
        self.optimizer_step_times: deque[float] = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.last_optimizer_step_time = time.time()

        # Gradient accumulation settings
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Cumulative counters - now tracking both iterations and optimizer steps
        self.total_iterations = 0  # Raw iteration count
        self.total_optimizer_steps = 0  # Actual optimizer steps
        self.total_samples_processed = 0  # Raw samples seen
        self.total_samples_effective = 0  # Samples per optimizer step

        # Backward compatibility aliases
        self.total_steps = 0  # Will track optimizer steps for compatibility
        self.total_samples = 0  # Will track effective samples for compatibility

        # CLIP evaluation setup
        self.enable_clip_eval = enable_clip_eval
        self.clip_evaluator: Any | None = None
        self.eval_milestones: list[int] = []
        self.completed_evals: set[int] = set()

        if enable_clip_eval:
            self._setup_clip_evaluation(clip_eval_config or {})

    def update_loss(self, loss: float) -> None:
        """Update loss metrics."""
        self.rolling_metrics["loss"].update(loss)

    def update_lr(self, lr: float) -> None:
        """Update learning rate."""
        self.rolling_metrics["lr"].update(lr)

    def update_grad_norm(self, model: th.nn.Module) -> None:
        """Calculate and update gradient norm."""
        total_norm = 0.0
        param_count = 0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1.0 / 2)
            self.rolling_metrics["grad_norm"].update(total_norm)

    def update_gradient_stats(self, model: th.nn.Module) -> None:
        """Calculate comprehensive gradient statistics."""
        total_norm = 0.0
        total_params = 0
        zero_grads = 0
        nan_grads = 0
        inf_grads = 0

        for p in model.parameters():
            if p.grad is not None:
                grad_data = p.grad.data

                # Count parameters with gradients
                total_params += grad_data.numel()

                # Check for zero gradients
                zero_grads += int((grad_data == 0).sum().item())

                # Check for NaN gradients
                nan_grads += int(th.isnan(grad_data).sum().item())

                # Check for infinite gradients
                inf_grads += int(th.isinf(grad_data).sum().item())

                # Calculate norm
                param_norm = grad_data.norm(2)
                total_norm += param_norm.item() ** 2

        if total_params > 0:
            # Global gradient norm
            global_grad_norm = total_norm**0.5
            self.rolling_metrics["grad_norm"].update(global_grad_norm)

            # Percentage statistics
            zero_grad_pct = (zero_grads / total_params) * 100
            nan_grad_pct = (nan_grads / total_params) * 100
            inf_grad_pct = (inf_grads / total_params) * 100

            self.rolling_metrics["grad_zero_pct"].update(zero_grad_pct)
            self.rolling_metrics["grad_nan_pct"].update(nan_grad_pct)
            self.rolling_metrics["grad_inf_pct"].update(inf_grad_pct)

            # Update individual counters for logging
            self.rolling_metrics["total_grad_params"].update(total_params)
            self.rolling_metrics["zero_grad_count"].update(zero_grads)
            self.rolling_metrics["nan_grad_count"].update(nan_grads)
            self.rolling_metrics["inf_grad_count"].update(inf_grads)

    def update_timing(self, is_optimizer_step: bool = False) -> None:
        """Update timing metrics.

        Args:
            is_optimizer_step: True if optimizer.step() was called this iteration
        """
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        self.total_iterations += 1

        # Track optimizer step timing separately
        if is_optimizer_step:
            optimizer_step_time = current_time - self.last_optimizer_step_time
            self.optimizer_step_times.append(optimizer_step_time)
            self.last_optimizer_step_time = current_time
            self.total_optimizer_steps += 1
            self.total_steps = self.total_optimizer_steps  # Compatibility

    def update_batch_size(self, batch_size: int, is_optimizer_step: bool = False) -> None:
        """Update batch size for throughput calculation.

        Args:
            batch_size: Size of the current batch
            is_optimizer_step: True if optimizer.step() was called this iteration
        """
        self.total_samples_processed += batch_size

        # Track effective samples (accounting for gradient accumulation)
        if is_optimizer_step:
            effective_batch_size = batch_size * self.gradient_accumulation_steps
            self.total_samples_effective += effective_batch_size
            self.total_samples = self.total_samples_effective  # Compatibility

    def check_evaluation_milestones(
        self, current_step: int, use_optimizer_steps: bool = True
    ) -> bool:
        """
        Check if current step is an evaluation milestone.

        Args:
            current_step: Current training step (optimizer steps by default)
            use_optimizer_steps: If True, use optimizer steps for milestones

        Returns:
            True if evaluation should be run at this step
        """
        if not self.enable_clip_eval or not self.eval_milestones:
            return False

        # Use optimizer steps by default for more accurate milestone tracking
        effective_step = self.total_optimizer_steps if use_optimizer_steps else current_step

        # Check if current step is a milestone and hasn't been evaluated yet
        for milestone in self.eval_milestones:
            if effective_step >= milestone and milestone not in self.completed_evals:
                self.completed_evals.add(milestone)
                logger.info(f"🎯 Reached evaluation milestone: optimizer step {milestone}")
                return True

        return False

    def maybe_run_evaluation(
        self, current_step: int, model: Any, diffusion: Any, options: Any, optimizer: Any | None = None
    ) -> dict[str, Any] | None:
        """
        Automatically run CLIP evaluation if current step is a milestone.

        Args:
            current_step: Current training step
            model: Current GLIDE model
            diffusion: Diffusion instance
            options: Model options dict
            optimizer: Training optimizer (optional)

        Returns:
            Evaluation results if evaluation was run, None otherwise
        """
        if self.check_evaluation_milestones(current_step):
            return self.run_clip_evaluation(model, diffusion, options, optimizer, current_step)
        return None

    def add_evaluation_milestone(self, step: int) -> None:
        """Add a new evaluation milestone."""
        if step not in self.eval_milestones:
            self.eval_milestones.append(step)
            self.eval_milestones.sort()
            logger.info(f"➕ Added evaluation milestone: step {step}")

    def remove_evaluation_milestone(self, step: int) -> None:
        """Remove an evaluation milestone."""
        if step in self.eval_milestones:
            self.eval_milestones.remove(step)
            logger.info(f"➖ Removed evaluation milestone: step {step}")

    def get_next_milestone(self, current_step: int) -> int | None:
        """Get the next evaluation milestone after current_step."""
        for milestone in self.eval_milestones:
            if milestone > current_step and milestone not in self.completed_evals:
                return milestone
        return None

    def update_aesthetic_score(self, images: list[Image.Image]) -> None:
        """Calculate and update aesthetic scores for generated images."""
        if not images:
            return

        scores: list[float] = []
        color_stats: dict[str, list[float]] = {"saturation": [], "brightness": [], "contrast": []}

        for img in images:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Calculate aesthetic score (simple heuristic-based)
            aesthetic_score = self._calculate_simple_aesthetic_score(img)
            scores.append(aesthetic_score)

            # Calculate color statistics
            img_array = np.array(img)
            hsv_img = th.tensor(img_array).float().permute(2, 0, 1) / 255.0

            # Convert RGB to HSV for saturation calculation
            hsv = self._rgb_to_hsv(hsv_img)

            saturation = hsv[1].mean().item()
            brightness = hsv[2].mean().item()
            contrast = hsv[2].std().item()

            color_stats["saturation"].append(saturation)
            color_stats["brightness"].append(brightness)
            color_stats["contrast"].append(contrast)

        # Update rolling averages
        if scores:
            avg_aesthetic = sum(scores) / len(scores)
            self.rolling_metrics["aesthetic_score"].update(avg_aesthetic)

        for stat_name, values in color_stats.items():
            if values:
                avg_value = sum(values) / len(values)
                self.rolling_metrics[f"color_{stat_name}"].update(avg_value)

    def _calculate_simple_aesthetic_score(self, img: Image.Image) -> float:
        """Calculate a simple aesthetic score based on image properties."""
        img_array = np.array(img)
        h, w, c = img_array.shape

        # Convert to float [0,1]
        img_float = img_array.astype(np.float32) / 255.0

        # Rule of thirds composition score
        thirds_score = self._rule_of_thirds_score(img_float)

        # Color harmony score
        harmony_score = self._color_harmony_score(img_float)

        # Sharpness/focus score
        sharpness_score = self._sharpness_score(img_float)

        # Brightness/contrast balance
        exposure_score = self._exposure_score(img_float)

        # Weighted combination
        aesthetic_score = (
            thirds_score * 0.2 + harmony_score * 0.3 + sharpness_score * 0.3 + exposure_score * 0.2
        )

        return float(aesthetic_score)

    def _rule_of_thirds_score(self, img: np.ndarray) -> float:
        """Score based on rule of thirds composition."""
        h, w = img.shape[:2]

        # Define rule of thirds lines
        h_lines = [h // 3, 2 * h // 3]
        v_lines = [w // 3, 2 * w // 3]

        # Calculate variance along these lines (higher = more interesting)
        score = 0.0
        for line in h_lines:
            if line < h:
                score += np.var(img[line, :])
        for line in v_lines:
            if line < w:
                score += np.var(img[:, line])

        return min(score / 4.0, 1.0)  # Normalize

    def _color_harmony_score(self, img: np.ndarray) -> float:
        """Score based on color harmony and distribution."""
        # Convert to HSV for better color analysis
        img_tensor = th.tensor(img).permute(2, 0, 1)
        hsv = self._rgb_to_hsv(img_tensor)

        # Hue distribution - penalize muddy/brown tones
        hue = hsv[0]
        hue_std = th.std(hue).item()

        # Saturation - prefer good saturation but not oversaturated
        saturation = hsv[1]
        sat_mean = th.mean(saturation).item()
        sat_score = 1.0 - abs(sat_mean - 0.6)  # Optimal around 0.6

        # Combine scores
        harmony_score = hue_std * 0.5 + sat_score * 0.5
        return min(harmony_score, 1.0)

    def _sharpness_score(self, img: np.ndarray) -> float:
        """Score based on image sharpness/focus."""
        # Convert to grayscale
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        # Calculate Laplacian variance (measure of sharpness)
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        # Apply convolution manually for simplicity
        h, w = gray.shape
        laplacian = np.zeros_like(gray)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                laplacian[i, j] = np.sum(gray[i - 1 : i + 2, j - 1 : j + 2] * laplacian_kernel)

        sharpness = np.var(laplacian)
        return float(min(sharpness * 1000, 1.0))  # Scale and cap

    def _exposure_score(self, img: np.ndarray) -> float:
        """Score based on exposure/brightness balance."""
        # Calculate brightness histogram
        brightness = np.mean(img, axis=2)

        # Penalize over/under exposure
        over_exposed = np.mean(brightness > 0.9)
        under_exposed = np.mean(brightness < 0.1)

        exposure_penalty = over_exposed + under_exposed

        # Prefer good contrast
        contrast = np.std(brightness)

        exposure_score = (1.0 - exposure_penalty) * min(float(contrast * 4), 1.0)
        return float(max(exposure_score, 0.0))

    def _rgb_to_hsv(self, rgb: th.Tensor) -> th.Tensor:
        """Convert RGB tensor to HSV."""
        # rgb: [3, H, W] in range [0, 1]
        r, g, b = rgb[0], rgb[1], rgb[2]

        max_val, max_idx = th.max(rgb, dim=0)
        min_val = th.min(rgb, dim=0)[0]
        diff = max_val - min_val

        # Hue calculation
        h = th.zeros_like(max_val)

        # Red is max
        mask = (max_idx == 0) & (diff != 0)
        h[mask] = ((g[mask] - b[mask]) / diff[mask]) % 6

        # Green is max
        mask = (max_idx == 1) & (diff != 0)
        h[mask] = ((b[mask] - r[mask]) / diff[mask]) + 2

        # Blue is max
        mask = (max_idx == 2) & (diff != 0)
        h[mask] = ((r[mask] - g[mask]) / diff[mask]) + 4

        h = h / 6.0  # Normalize to [0, 1]

        # Saturation calculation
        s = th.where(max_val != 0, diff / max_val, th.zeros_like(max_val))

        # Value is just the max
        v = max_val

        return th.stack([h, s, v])

    def _setup_clip_evaluation(self, config: dict[str, Any]) -> None:
        """Initialize memory-conscious CLIP evaluation runner."""
        try:
            if create_memory_conscious_evaluator is None:
                raise ImportError("Memory-conscious evaluator not available")
            
            # Default config for memory-conscious evaluation
            default_config = {
                "prompts_file": "experiments/captions/evaluation1.txt",
                "variations": 1,
                "guidance_scale": 3.5,
                "device": "cuda",
                "max_memory_usage_pct": 85.0,  # Conservative for 12GB VRAM
                "enable_wandb_logging": False,
                "save_sample_images": False,
            }
            default_config.update(config)

            # Extract milestone configuration
            self.eval_milestones = config.get(
                "eval_milestones", [1000, 2500, 5000, 10000, 15000, 25000]
            )

            self.clip_evaluator = create_memory_conscious_evaluator(**default_config)  # type: ignore[arg-type]
            logger.info(
                f"✅ Memory-conscious CLIP evaluation enabled with {len(self.clip_evaluator.prompts)} prompts"
            )
            logger.info(f"   Evaluation milestones: {self.eval_milestones}")

        except ImportError as e:
            logger.info(f"⚠️  CLIP evaluation disabled: {e}")
            self.enable_clip_eval = False
        except Exception as e:
            logger.info(f"⚠️  Failed to setup CLIP evaluation: {e}")
            self.enable_clip_eval = False

    def run_clip_evaluation(
        self, model: Any, diffusion: Any, options: Any, optimizer: Any | None = None, step: int = 0
    ) -> dict[str, Any] | None:
        """
        Run memory-conscious CLIP evaluation comparing current model to base model.

        Args:
            model: Current GLIDE model
            diffusion: Diffusion instance
            options: Model options dict
            optimizer: Training optimizer (for state saving)
            step: Training step number

        Returns:
            Dict with evaluation metrics or None if evaluation disabled
        """
        if not self.enable_clip_eval or self.clip_evaluator is None:
            return None

        try:
            logger.info("🔍 Running memory-conscious CLIP evaluation...")

            # Run the memory-conscious evaluation
            eval_results = self.clip_evaluator.run_evaluation(
                model, diffusion, options, optimizer, step
            )

            # Update rolling metrics with CLIP results
            self.rolling_metrics["clip_score_current"].update(eval_results["clip_score_current"])
            self.rolling_metrics["clip_score_base"].update(eval_results["clip_score_base"])
            self.rolling_metrics["clip_score_delta"].update(eval_results["clip_score_delta"])
            self.rolling_metrics["win_rate_vs_base"].update(eval_results["win_rate_vs_base"])
            self.rolling_metrics["clip_eval_time"].update(eval_results["evaluation_time"])

            logger.info(
                f"✅ Memory-conscious CLIP evaluation complete in {eval_results['evaluation_time']:.1f}s"
            )
            logger.info(
                f"   CLIP Score: {eval_results['clip_score_current']:.4f} (Δ{eval_results['clip_score_delta']:+.4f})"
            )
            logger.info(
                f"   Win Rate: {eval_results['win_rate_vs_base']:.3f} ({eval_results['wins_count']}/{eval_results['num_prompts']} wins)"
            )

            return eval_results  # type: ignore[no-any-return]

        except Exception as e:
            logger.info(f"⚠️  Memory-conscious CLIP evaluation failed: {e}")
            traceback.print_exc()

            # Ensure model is back in training mode and on correct device
            try:
                model.train()
                model.to(options.get("device", "cuda"))
            except Exception:
                pass

            return None

    def update_clip_metrics(self, clip_results: dict[str, Any] | None) -> None:
        """Update rolling metrics with CLIP evaluation results."""
        if clip_results is None:
            return

        for key in [
            "clip_score_current",
            "clip_score_base",
            "clip_score_delta",
            "win_rate_vs_base",
        ]:
            if key in clip_results:
                self.rolling_metrics[key].update(clip_results[key])

    def get_metrics(self) -> dict[str, Any]:
        """Get all current metrics."""
        metrics = {}

        # Loss metrics
        if self.rolling_metrics["loss"].count > 0:
            metrics["loss"] = self.rolling_metrics["loss"].avg

        # Learning rate
        if self.rolling_metrics["lr"].count > 0:
            metrics["lr"] = self.rolling_metrics["lr"].avg

        # Gradient metrics
        if self.rolling_metrics["grad_norm"].count > 0:
            metrics["grad_norm"] = self.rolling_metrics["grad_norm"].avg

        # Gradient statistics
        for stat_name in ["grad_zero_pct", "grad_nan_pct", "grad_inf_pct"]:
            if self.rolling_metrics[stat_name].count > 0:
                metrics[stat_name] = self.rolling_metrics[stat_name].avg

        # Aesthetic metrics
        if self.rolling_metrics["aesthetic_score"].count > 0:
            metrics["aesthetic_score"] = self.rolling_metrics["aesthetic_score"].avg

        # Color statistics
        for color_stat in ["color_saturation", "color_brightness", "color_contrast"]:
            if self.rolling_metrics[color_stat].count > 0:
                metrics[color_stat] = self.rolling_metrics[color_stat].avg

        # Timing metrics - now tracking both iteration and optimizer step timing
        if self.step_times:
            avg_iteration_time = sum(self.step_times) / len(self.step_times)
            metrics["iteration_time"] = avg_iteration_time
            metrics["iterations_per_sec"] = (
                1.0 / avg_iteration_time if avg_iteration_time > 0 else 0.0
            )

        if self.optimizer_step_times:
            avg_optimizer_step_time = sum(self.optimizer_step_times) / len(
                self.optimizer_step_times
            )
            metrics["optimizer_step_time"] = avg_optimizer_step_time
            metrics["optimizer_steps_per_sec"] = (
                1.0 / avg_optimizer_step_time if avg_optimizer_step_time > 0 else 0.0
            )
            # Legacy compatibility
            metrics["steps_per_sec"] = metrics["optimizer_steps_per_sec"]
            metrics["step_time"] = avg_optimizer_step_time

        # Throughput metrics - both raw and effective
        total_time = time.time() - self.start_time
        if total_time > 0:
            # Raw throughput (all samples processed)
            metrics["samples_per_sec_raw"] = self.total_samples_processed / total_time
            # Effective throughput (samples per optimizer step)
            metrics["samples_per_sec_effective"] = self.total_samples_effective / total_time
            # Legacy compatibility
            metrics["samples_per_sec"] = metrics["samples_per_sec_effective"]

        # Training progress - comprehensive tracking
        metrics["total_iterations"] = self.total_iterations
        metrics["total_optimizer_steps"] = self.total_optimizer_steps
        metrics["total_samples_processed"] = self.total_samples_processed
        metrics["total_samples_effective"] = self.total_samples_effective
        # Legacy compatibility
        metrics["total_steps"] = self.total_optimizer_steps
        metrics["total_samples"] = self.total_samples_effective

        # CLIP evaluation metrics
        for clip_metric in [
            "clip_score_current",
            "clip_score_base",
            "clip_score_delta",
            "win_rate_vs_base",
            "clip_eval_time",
        ]:
            if self.rolling_metrics[clip_metric].count > 0:
                metrics[clip_metric] = self.rolling_metrics[clip_metric].avg

        return metrics

    def get_console_summary(self) -> str:
        """Get a formatted string for console output."""
        metrics = self.get_metrics()

        parts = []
        if "loss" in metrics:
            parts.append(f"Loss: {metrics['loss']:.4f}")
        if "lr" in metrics:
            parts.append(f"LR: {metrics['lr']:.2e}")
        if "grad_norm" in metrics:
            parts.append(f"GradNorm: {metrics['grad_norm']:.3f}")
        if "grad_zero_pct" in metrics:
            parts.append(f"ZeroGrad%: {metrics['grad_zero_pct']:.1f}")
        if "grad_nan_pct" in metrics and metrics["grad_nan_pct"] > 0:
            parts.append(f"NaN%: {metrics['grad_nan_pct']:.3f}")
        if "aesthetic_score" in metrics:
            parts.append(f"Aesthetic: {metrics['aesthetic_score']:.3f}")
        if "clip_score_current" in metrics:
            parts.append(f"CLIP: {metrics['clip_score_current']:.3f}")
        if "win_rate_vs_base" in metrics:
            parts.append(f"WinRate: {metrics['win_rate_vs_base']:.3f}")

        # Show optimizer steps/sec (the important metric)
        if "optimizer_steps_per_sec" in metrics:
            parts.append(f"OptSteps/s: {metrics['optimizer_steps_per_sec']:.2f}")
        elif "steps_per_sec" in metrics:  # Fallback for compatibility
            parts.append(f"Steps/s: {metrics['steps_per_sec']:.2f}")

        # Show effective samples/sec (accounting for gradient accumulation)
        if "samples_per_sec_effective" in metrics:
            parts.append(f"Samples/s: {metrics['samples_per_sec_effective']:.1f}")
        elif "samples_per_sec" in metrics:  # Fallback for compatibility
            parts.append(f"Samples/s: {metrics['samples_per_sec']:.1f}")

        # Add next milestone info if CLIP evaluation is enabled
        if self.enable_clip_eval and hasattr(self, "eval_milestones"):
            next_milestone = self.get_next_milestone(self.total_steps)
            if next_milestone:
                parts.append(f"Next eval: {next_milestone}")

        return " | ".join(parts)

    def reset(self) -> None:
        """Reset all metrics."""
        self.rolling_metrics.clear()
        self.step_times.clear()
        self.optimizer_step_times.clear()
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.last_optimizer_step_time = time.time()
        self.total_iterations = 0
        self.total_optimizer_steps = 0
        self.total_samples_processed = 0
        self.total_samples_effective = 0
        # Compatibility aliases
        self.total_steps = 0
        self.total_samples = 0


def calculate_model_size(model: th.nn.Module) -> dict[str, int | float]:
    """Calculate model parameter statistics."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_percent": float((trainable_params / total_params * 100) if total_params > 0 else 0),
    }


def format_number(num: int | float) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(int(num)) if isinstance(num, float) and num.is_integer() else f"{num:.1f}"


def print_model_info(model: th.nn.Module, model_name: str = "Model") -> None:
    """Print detailed model information."""
    stats = calculate_model_size(model)

    logger.info(f"\n📊 {model_name} Statistics:")
    logger.info(f"  Total parameters: {format_number(stats['total_params'])}")
    logger.info(
        f"  Trainable parameters: {format_number(stats['trainable_params'])} ({stats['trainable_percent']:.1f}%)"
    )
    logger.info(f"  Frozen parameters: {format_number(stats['frozen_params'])}")
