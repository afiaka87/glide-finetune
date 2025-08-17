#!/usr/bin/env python3
"""
Training Monitor Dashboard for FP16 GLIDE Training
Comprehensive WandB integration with advanced monitoring and visualization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, List, Optional, Any, Tuple
import time
import psutil
import GPUtil
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging


@dataclass
class TrainingMetrics:
    """Container for comprehensive training metrics."""
    # Basic metrics
    step: int
    epoch: int
    loss: float
    learning_rate: float
    
    # FP16 specific metrics
    loss_scale: float
    grad_norm: float
    overflow_occurred: bool
    nan_detected: bool
    
    # Performance metrics
    samples_per_second: float
    step_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float
    
    # Memory metrics
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    gpu_utilization: float
    
    # Model statistics
    weight_norm: float
    weight_std: float
    gradient_std: float
    
    # Optional metrics
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    clip_score: Optional[float] = None
    fid_score: Optional[float] = None


class TrainingMonitorDashboard:
    """
    Comprehensive training monitor with WandB integration.
    
    Features:
    - Real-time FP16 training metrics
    - GPU/System monitoring
    - Loss scale tracking
    - Gradient analysis
    - Model weight visualization
    - Sample quality tracking
    - Custom alerts and notifications
    """
    
    def __init__(self, 
                 project_name: str = "glide_fp16_training",
                 entity: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None):
        """
        Initialize training monitor.
        
        Args:
            project_name: WandB project name
            entity: WandB entity/team name
            config: Training configuration
            tags: Tags for the run
            notes: Notes about the run
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize WandB
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config=config or {},
            tags=tags or ["fp16", "glide", "diffusion"],
            notes=notes,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )
        
        # Define custom metrics with appropriate summaries
        self._define_metrics()
        
        # Initialize tracking
        self.metrics_history = []
        self.alert_thresholds = self._setup_alert_thresholds()
        self.has_gpu = torch.cuda.is_available()
        
        # Setup custom charts
        self._setup_custom_charts()
    
    def _define_metrics(self):
        """Define custom metrics and their summaries."""
        # Loss metrics with min/max tracking
        self.run.define_metric("loss", summary="min")
        self.run.define_metric("loss", summary="last")
        self.run.define_metric("validation_loss", summary="min")
        
        # FP16 specific metrics
        self.run.define_metric("loss_scale", summary="last")
        self.run.define_metric("grad_norm", summary="max")
        self.run.define_metric("overflow_rate", summary="mean")
        
        # Performance metrics
        self.run.define_metric("samples_per_second", summary="max")
        self.run.define_metric("gpu_utilization", summary="mean")
        self.run.define_metric("gpu_memory_allocated", summary="max")
        
        # Model quality metrics
        self.run.define_metric("clip_score", summary="max")
        self.run.define_metric("fid_score", summary="min")
        
        # Set step as x-axis for all metrics
        self.run.define_metric("*", step_metric="step")
    
    def _setup_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup thresholds for automatic alerts."""
        return {
            "loss_scale": {"min": 1.0, "max": 65536.0},
            "grad_norm": {"max": 100.0},
            "overflow_rate": {"max": 0.1},  # 10% overflow rate
            "gpu_memory_allocated": {"max": 0.95},  # 95% memory usage
            "nan_rate": {"max": 0.01},  # 1% NaN rate
        }
    
    def _setup_custom_charts(self):
        """Setup custom chart configurations."""
        # These will be logged as custom plots
        self.custom_charts = {
            "loss_scale_history": {
                "title": "Loss Scale Evolution",
                "x": "step",
                "y": "loss_scale",
                "log_y": True
            },
            "gradient_distribution": {
                "title": "Gradient Norm Distribution",
                "type": "histogram"
            },
            "fp16_efficiency": {
                "title": "FP16 Training Efficiency",
                "metrics": ["samples_per_second", "gpu_utilization", "overflow_rate"]
            }
        }
    
    def watch_model(self, model: nn.Module, log_freq: int = 100, log: str = "all"):
        """
        Watch model for gradient and parameter logging.
        
        Args:
            model: PyTorch model to watch
            log_freq: Frequency of gradient logging
            log: What to log ("gradients", "parameters", "all")
        """
        self.run.watch(model, log=log, log_freq=log_freq)
        self.logger.info(f"Watching model with log_freq={log_freq}, log={log}")
    
    def log_training_step(self, 
                          metrics: TrainingMetrics,
                          additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Log comprehensive training metrics.
        
        Args:
            metrics: Training metrics dataclass
            additional_metrics: Any additional metrics to log
        """
        # Convert dataclass to dict
        log_dict = asdict(metrics)
        
        # Remove None values
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        
        # Add FP16-specific computed metrics
        if metrics.loss_scale > 0:
            log_dict["loss_scale_log10"] = np.log10(metrics.loss_scale)
        
        # Add efficiency metrics
        log_dict["fp16_efficiency"] = (
            metrics.samples_per_second * (1 - int(metrics.overflow_occurred))
        )
        
        # Add memory pressure indicator
        if metrics.gpu_memory_allocated > 0:
            log_dict["memory_pressure"] = metrics.gpu_memory_allocated / metrics.gpu_memory_reserved
        
        # Add additional metrics if provided
        if additional_metrics:
            log_dict.update(additional_metrics)
        
        # Log to WandB
        self.run.log(log_dict, step=metrics.step)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: TrainingMetrics):
        """Check metrics against alert thresholds."""
        alerts = []
        
        # Check loss scale
        if metrics.loss_scale < self.alert_thresholds["loss_scale"]["min"]:
            alerts.append(f"⚠️ Loss scale too low: {metrics.loss_scale}")
        elif metrics.loss_scale > self.alert_thresholds["loss_scale"]["max"]:
            alerts.append(f"⚠️ Loss scale too high: {metrics.loss_scale}")
        
        # Check gradient norm
        if metrics.grad_norm > self.alert_thresholds["grad_norm"]["max"]:
            alerts.append(f"⚠️ Gradient explosion: norm={metrics.grad_norm:.2f}")
        
        # Check memory
        if metrics.gpu_memory_allocated > self.alert_thresholds["gpu_memory_allocated"]["max"]:
            alerts.append(f"⚠️ High GPU memory usage: {metrics.gpu_memory_allocated*100:.1f}%")
        
        # Check for NaN
        if metrics.nan_detected:
            alerts.append("🚨 NaN detected in training!")
        
        # Log alerts
        if alerts:
            for alert in alerts:
                self.logger.warning(alert)
                self.run.alert(title="Training Alert", text=alert, level="WARNING")
    
    def log_images(self, 
                  images: List[np.ndarray],
                  captions: Optional[List[str]] = None,
                  step: Optional[int] = None):
        """
        Log generated images.
        
        Args:
            images: List of images as numpy arrays
            captions: Optional captions for images
            step: Training step
        """
        wandb_images = []
        
        for i, img in enumerate(images):
            caption = captions[i] if captions else f"Sample {i}"
            wandb_images.append(wandb.Image(img, caption=caption))
        
        log_dict = {"generated_samples": wandb_images}
        
        if step is not None:
            self.run.log(log_dict, step=step)
        else:
            self.run.log(log_dict)
    
    def log_gradient_histogram(self, 
                               model: nn.Module,
                               step: int):
        """
        Log gradient histogram for model parameters.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        gradients = []
        gradient_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data.cpu().numpy().flatten()
                gradients.extend(grad_data.tolist())
                gradient_norms.append(np.linalg.norm(grad_data))
        
        if gradients:
            # Create histogram
            data = [[g] for g in gradients[:10000]]  # Limit to 10k values
            table = wandb.Table(data=data, columns=["gradient_values"])
            
            self.run.log({
                "gradient_histogram": wandb.plot.histogram(
                    table, "gradient_values", 
                    title="Gradient Distribution"
                ),
                "gradient_norm_mean": np.mean(gradient_norms),
                "gradient_norm_std": np.std(gradient_norms),
            }, step=step)
    
    def log_weight_statistics(self, 
                             model: nn.Module,
                             step: int):
        """
        Log detailed weight statistics.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        weight_stats = {
            "fp16_layers": 0,
            "fp32_layers": 0,
            "total_parameters": 0,
            "weight_norms": [],
            "weight_means": [],
            "weight_stds": [],
        }
        
        for name, param in model.named_parameters():
            weight_data = param.data.cpu().float().numpy()
            
            # Count precision
            if param.dtype == torch.float16:
                weight_stats["fp16_layers"] += 1
            else:
                weight_stats["fp32_layers"] += 1
            
            weight_stats["total_parameters"] += param.numel()
            weight_stats["weight_norms"].append(np.linalg.norm(weight_data))
            weight_stats["weight_means"].append(np.mean(weight_data))
            weight_stats["weight_stds"].append(np.std(weight_data))
        
        # Log aggregated statistics
        self.run.log({
            "model/fp16_layer_ratio": weight_stats["fp16_layers"] / 
                                     (weight_stats["fp16_layers"] + weight_stats["fp32_layers"]),
            "model/total_parameters": weight_stats["total_parameters"],
            "model/avg_weight_norm": np.mean(weight_stats["weight_norms"]),
            "model/avg_weight_std": np.mean(weight_stats["weight_stds"]),
            "model/weight_norm_variance": np.var(weight_stats["weight_norms"]),
        }, step=step)
    
    def log_loss_scale_chart(self, step: int, window_size: int = 100):
        """
        Log custom loss scale evolution chart.
        
        Args:
            step: Current training step
            window_size: Window size for chart
        """
        if len(self.metrics_history) < 2:
            return
        
        # Get recent loss scale history
        recent_metrics = self.metrics_history[-window_size:]
        steps = [m.step for m in recent_metrics]
        loss_scales = [m.loss_scale for m in recent_metrics]
        
        # Create line plot
        data = [[s, ls] for s, ls in zip(steps, loss_scales)]
        table = wandb.Table(data=data, columns=["step", "loss_scale"])
        
        self.run.log({
            "loss_scale_evolution": wandb.plot.line(
                table, "step", "loss_scale",
                title="Loss Scale Evolution (Log Scale)"
            )
        }, step=step)
    
    def log_fp16_efficiency_metrics(self, step: int):
        """
        Log FP16 training efficiency metrics.
        
        Args:
            step: Current training step
        """
        if len(self.metrics_history) < 10:
            return
        
        # Calculate efficiency metrics over recent window
        recent = self.metrics_history[-100:]
        
        overflow_rate = sum(1 for m in recent if m.overflow_occurred) / len(recent)
        nan_rate = sum(1 for m in recent if m.nan_detected) / len(recent)
        avg_throughput = np.mean([m.samples_per_second for m in recent])
        avg_gpu_util = np.mean([m.gpu_utilization for m in recent])
        
        # Calculate speedup estimate (compare with theoretical FP32)
        # Assume FP32 would be ~50% slower
        estimated_fp32_throughput = avg_throughput / 1.5
        speedup = avg_throughput / estimated_fp32_throughput
        
        self.run.log({
            "fp16/overflow_rate": overflow_rate,
            "fp16/nan_rate": nan_rate,
            "fp16/avg_throughput": avg_throughput,
            "fp16/gpu_efficiency": avg_gpu_util,
            "fp16/estimated_speedup": speedup,
            "fp16/stability_score": (1 - overflow_rate) * (1 - nan_rate),
        }, step=step)
    
    def log_system_metrics(self, step: int):
        """
        Log detailed system metrics.
        
        Args:
            step: Current training step
        """
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            "system/cpu_percent": cpu_percent,
            "system/memory_percent": memory.percent,
            "system/memory_used_gb": memory.used / 1e9,
        }
        
        # GPU metrics if available
        if self.has_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.update({
                        "system/gpu_utilization": gpu.load * 100,
                        "system/gpu_memory_used": gpu.memoryUsed,
                        "system/gpu_memory_total": gpu.memoryTotal,
                        "system/gpu_temperature": gpu.temperature,
                    })
                
                # PyTorch GPU metrics
                metrics.update({
                    "system/cuda_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "system/cuda_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                })
            except Exception as e:
                self.logger.debug(f"Failed to get GPU metrics: {e}")
        
        self.run.log(metrics, step=step)
    
    def log_validation_metrics(self,
                              validation_loss: float,
                              validation_metrics: Dict[str, float],
                              generated_samples: Optional[List[np.ndarray]] = None,
                              step: int = 0):
        """
        Log validation metrics and samples.
        
        Args:
            validation_loss: Validation loss
            validation_metrics: Dictionary of validation metrics
            generated_samples: Optional generated samples
            step: Training step
        """
        log_dict = {
            "validation/loss": validation_loss,
        }
        
        # Add all validation metrics with prefix
        for key, value in validation_metrics.items():
            log_dict[f"validation/{key}"] = value
        
        # Log generated samples if provided
        if generated_samples:
            wandb_images = [wandb.Image(img) for img in generated_samples[:8]]  # Limit to 8
            log_dict["validation/samples"] = wandb_images
        
        self.run.log(log_dict, step=step)
    
    def create_summary_plots(self):
        """Create and log summary plots at the end of training."""
        if len(self.metrics_history) < 10:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curve
        ax = axes[0, 0]
        steps = [m.step for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        ax.plot(steps, losses)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        
        # Loss scale evolution
        ax = axes[0, 1]
        loss_scales = [m.loss_scale for m in self.metrics_history]
        ax.semilogy(steps, loss_scales)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss Scale (log)")
        ax.set_title("Loss Scale Evolution")
        ax.grid(True, alpha=0.3)
        
        # Gradient norms
        ax = axes[0, 2]
        grad_norms = [m.grad_norm for m in self.metrics_history]
        ax.plot(steps, grad_norms)
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm History")
        ax.grid(True, alpha=0.3)
        
        # Throughput
        ax = axes[1, 0]
        throughputs = [m.samples_per_second for m in self.metrics_history]
        ax.plot(steps, throughputs)
        ax.set_xlabel("Step")
        ax.set_ylabel("Samples/sec")
        ax.set_title("Training Throughput")
        ax.grid(True, alpha=0.3)
        
        # GPU utilization
        ax = axes[1, 1]
        gpu_utils = [m.gpu_utilization for m in self.metrics_history]
        ax.plot(steps, gpu_utils)
        ax.set_xlabel("Step")
        ax.set_ylabel("GPU Utilization (%)")
        ax.set_title("GPU Utilization")
        ax.grid(True, alpha=0.3)
        
        # Overflow rate over time (sliding window)
        ax = axes[1, 2]
        window = 100
        overflow_rates = []
        for i in range(window, len(self.metrics_history)):
            window_metrics = self.metrics_history[i-window:i]
            rate = sum(1 for m in window_metrics if m.overflow_occurred) / window
            overflow_rates.append(rate * 100)
        
        if overflow_rates:
            ax.plot(steps[window:], overflow_rates)
            ax.set_xlabel("Step")
            ax.set_ylabel("Overflow Rate (%)")
            ax.set_title(f"Overflow Rate ({window}-step window)")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to wandb
        self.run.log({"summary_plots": wandb.Image(fig)})
        plt.close()
    
    def log_final_summary(self):
        """Log final training summary."""
        if not self.metrics_history:
            return
        
        # Calculate summary statistics
        losses = [m.loss for m in self.metrics_history if not m.nan_detected]
        grad_norms = [m.grad_norm for m in self.metrics_history]
        throughputs = [m.samples_per_second for m in self.metrics_history]
        
        overflow_count = sum(1 for m in self.metrics_history if m.overflow_occurred)
        nan_count = sum(1 for m in self.metrics_history if m.nan_detected)
        
        summary = {
            "summary/final_loss": losses[-1] if losses else float('nan'),
            "summary/best_loss": min(losses) if losses else float('nan'),
            "summary/avg_loss": np.mean(losses) if losses else float('nan'),
            "summary/avg_grad_norm": np.mean(grad_norms),
            "summary/max_grad_norm": max(grad_norms),
            "summary/avg_throughput": np.mean(throughputs),
            "summary/total_steps": len(self.metrics_history),
            "summary/overflow_steps": overflow_count,
            "summary/nan_steps": nan_count,
            "summary/success_rate": (len(self.metrics_history) - overflow_count - nan_count) / len(self.metrics_history),
        }
        
        # Update wandb run summary
        for key, value in summary.items():
            self.run.summary[key] = value
        
        # Create final plots
        self.create_summary_plots()
        
        self.logger.info("Training summary logged to WandB")
    
    def finish(self):
        """Finish the WandB run and cleanup."""
        self.log_final_summary()
        self.run.finish()


def example_usage():
    """Example of how to use the training monitor."""
    import torch
    import random
    
    # Initialize monitor
    config = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "use_fp16": True,
        "gradient_clip": 1.0,
    }
    
    monitor = TrainingMonitorDashboard(
        project_name="glide_fp16_test",
        config=config,
        tags=["test", "fp16"],
        notes="Testing FP16 training monitor"
    )
    
    # Simulate training loop
    for step in range(100):
        # Create fake metrics
        metrics = TrainingMetrics(
            step=step,
            epoch=step // 10,
            loss=1.0 / (step + 1) + random.random() * 0.1,
            learning_rate=1e-4,
            loss_scale=2 ** (8 + step // 20),
            grad_norm=random.random() * 10,
            overflow_occurred=random.random() < 0.05,
            nan_detected=False,
            samples_per_second=32 + random.random() * 10,
            step_time=0.5 + random.random() * 0.2,
            forward_time=0.2,
            backward_time=0.2,
            optimizer_time=0.1,
            gpu_memory_allocated=0.6 + random.random() * 0.3,
            gpu_memory_reserved=0.8,
            gpu_utilization=70 + random.random() * 20,
            weight_norm=10.0 + random.random(),
            weight_std=0.1,
            gradient_std=0.01,
        )
        
        # Log metrics
        monitor.log_training_step(metrics)
        
        # Log system metrics every 10 steps
        if step % 10 == 0:
            monitor.log_system_metrics(step)
            monitor.log_fp16_efficiency_metrics(step)
        
        # Log validation every 20 steps
        if step % 20 == 0:
            monitor.log_validation_metrics(
                validation_loss=metrics.loss * 1.1,
                validation_metrics={
                    "accuracy": 0.8 + random.random() * 0.1,
                    "clip_score": 0.7 + random.random() * 0.2,
                },
                step=step
            )
        
        # Log charts every 50 steps
        if step % 50 == 0:
            monitor.log_loss_scale_chart(step)
    
    # Finish monitoring
    monitor.finish()
    print("✅ Training monitor example complete! Check WandB dashboard for results.")


if __name__ == "__main__":
    example_usage()