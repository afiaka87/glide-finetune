"""
Comprehensive metrics tracking for GLIDE training.
Inspired by VibeTune's metrics system with rolling averages and detailed logging.
"""

import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional
import torch as th
import numpy as np


class RollingAverage:
    """Rolling average calculator for smooth metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def update(self, value: float):
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
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rolling_metrics = defaultdict(lambda: RollingAverage(window_size))
        self.step_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_step_time = time.time()
        
        # Cumulative counters
        self.total_steps = 0
        self.total_samples = 0
        
    def update_loss(self, loss: float):
        """Update loss metrics."""
        self.rolling_metrics["loss"].update(loss)
        
    def update_lr(self, lr: float):
        """Update learning rate."""
        self.rolling_metrics["lr"].update(lr)
        
    def update_grad_norm(self, model: th.nn.Module):
        """Calculate and update gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.rolling_metrics["grad_norm"].update(total_norm)
        
    def update_timing(self):
        """Update timing metrics."""
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        self.total_steps += 1
        
    def update_batch_size(self, batch_size: int):
        """Update batch size for throughput calculation."""
        self.total_samples += batch_size
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        metrics = {}
        
        # Loss metrics
        if self.rolling_metrics["loss"].count > 0:
            metrics["loss"] = self.rolling_metrics["loss"].avg
            
        # Learning rate
        if self.rolling_metrics["lr"].count > 0:
            metrics["lr"] = self.rolling_metrics["lr"].avg
            
        # Gradient norm
        if self.rolling_metrics["grad_norm"].count > 0:
            metrics["grad_norm"] = self.rolling_metrics["grad_norm"].avg
            
        # Timing metrics
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            metrics["step_time"] = avg_step_time
            metrics["steps_per_sec"] = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
            
        # Throughput
        total_time = time.time() - self.start_time
        if total_time > 0:
            metrics["samples_per_sec"] = self.total_samples / total_time
            
        # Training progress
        metrics["total_steps"] = self.total_steps
        metrics["total_samples"] = self.total_samples
        
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
        if "steps_per_sec" in metrics:
            parts.append(f"Steps/s: {metrics['steps_per_sec']:.2f}")
        if "samples_per_sec" in metrics:
            parts.append(f"Samples/s: {metrics['samples_per_sec']:.1f}")
            
        return " | ".join(parts)
    
    def reset(self):
        """Reset all metrics."""
        self.rolling_metrics.clear()
        self.step_times.clear()
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.total_steps = 0
        self.total_samples = 0


def calculate_model_size(model: th.nn.Module) -> Dict[str, int]:
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
        "trainable_percent": (trainable_params / total_params * 100) if total_params > 0 else 0
    }


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def print_model_info(model: th.nn.Module, model_name: str = "Model"):
    """Print detailed model information."""
    stats = calculate_model_size(model)
    
    print(f"\n📊 {model_name} Statistics:")
    print(f"  Total parameters: {format_number(stats['total_params'])}")
    print(f"  Trainable parameters: {format_number(stats['trainable_params'])} ({stats['trainable_percent']:.1f}%)")
    print(f"  Frozen parameters: {format_number(stats['frozen_params'])}")