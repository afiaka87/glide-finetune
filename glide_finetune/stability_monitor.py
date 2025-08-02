"""
Stability monitoring utilities for CLIP adapter training.

This module provides tools to monitor the stability of CLIP adapter integration
during training, ensuring the pretrained GLIDE model isn't degraded.
"""

import torch
from typing import Dict, Optional, List, Tuple
import numpy as np
from collections import deque


class StabilityMonitor:
    """
    Monitor training stability and detect potential issues with CLIP adapter.
    
    This class tracks various metrics to ensure the CLIP adapter doesn't
    degrade the pretrained GLIDE model's performance.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        spike_threshold: float = 3.0,
        degradation_threshold: float = 0.1,
    ):
        """
        Args:
            window_size: Size of moving window for statistics
            spike_threshold: Number of std devs to consider a spike
            degradation_threshold: Relative increase in loss to trigger warning
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.degradation_threshold = degradation_threshold
        
        # Track loss history
        self.loss_history = deque(maxlen=window_size)
        self.baseline_loss_history = deque(maxlen=window_size)
        
        # Track KL divergence
        self.kl_history = deque(maxlen=window_size)
        
        # Track gate values
        self.gate_history = deque(maxlen=window_size)
        
        # Track gradient norms
        self.adapter_grad_history = deque(maxlen=window_size)
        self.main_grad_history = deque(maxlen=window_size)
        
        # Stability test results
        self.stability_test_results = []
        
        # Checkpoint for rollback
        self.best_checkpoint = None
        self.best_loss = float('inf')
    
    def update(
        self,
        loss: float,
        metrics: Dict[str, float],
        adapter_grad_norm: Optional[float] = None,
        main_grad_norm: Optional[float] = None,
    ):
        """Update monitor with latest training metrics."""
        self.loss_history.append(loss)
        
        # Track KL divergence if available
        if 'kl_divergence' in metrics:
            self.kl_history.append(metrics['kl_divergence'])
        
        # Track gate values
        if 'adapter_gate' in metrics:
            self.gate_history.append(metrics['adapter_gate'])
        
        # Track gradient norms
        if adapter_grad_norm is not None:
            self.adapter_grad_history.append(adapter_grad_norm)
        if main_grad_norm is not None:
            self.main_grad_history.append(main_grad_norm)
    
    def check_for_issues(self) -> Dict[str, any]:
        """
        Check for potential stability issues.
        
        Returns:
            Dict with detected issues and recommendations
        """
        issues = {
            'has_issues': False,
            'loss_spike': False,
            'loss_degradation': False,
            'kl_explosion': False,
            'gradient_explosion': False,
            'recommendations': []
        }
        
        if len(self.loss_history) < 10:
            return issues  # Not enough data
        
        # Check for loss spikes
        recent_losses = list(self.loss_history)[-10:]
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        
        if loss_std > 0 and abs(recent_losses[-1] - loss_mean) > self.spike_threshold * loss_std:
            issues['has_issues'] = True
            issues['loss_spike'] = True
            issues['recommendations'].append("Loss spike detected. Consider reducing learning rate.")
        
        # Check for overall degradation
        if len(self.loss_history) == self.window_size:
            early_mean = np.mean(list(self.loss_history)[:20])
            recent_mean = np.mean(list(self.loss_history)[-20:])
            
            if recent_mean > early_mean * (1 + self.degradation_threshold):
                issues['has_issues'] = True
                issues['loss_degradation'] = True
                issues['recommendations'].append("Loss increasing over time. Consider checkpoint rollback.")
        
        # Check KL divergence
        if len(self.kl_history) > 10:
            recent_kl = list(self.kl_history)[-10:]
            kl_mean = np.mean(recent_kl)
            
            if kl_mean > 1.0:  # High KL divergence
                issues['has_issues'] = True
                issues['kl_explosion'] = True
                issues['recommendations'].append("High KL divergence. Reduce adapter learning rate or gate warmup.")
        
        # Check gradient explosion
        if len(self.adapter_grad_history) > 5:
            recent_grads = list(self.adapter_grad_history)[-5:]
            if any(g > 10.0 for g in recent_grads):
                issues['has_issues'] = True
                issues['gradient_explosion'] = True
                issues['recommendations'].append("Gradient explosion in adapter. Reduce gradient clipping threshold.")
        
        return issues
    
    def run_stability_test(
        self,
        model,
        baseline_model,
        test_batch: Tuple[torch.Tensor, ...],
        device: str = 'cuda',
    ) -> Dict[str, float]:
        """
        Run a stability test comparing model with baseline.
        
        Args:
            model: CLIP-enabled model
            baseline_model: Original GLIDE model
            test_batch: (tokens, mask, images) tuple
            device: Device to run test on
            
        Returns:
            Dict with test results
        """
        tokens, mask, images = test_batch[:3]
        tokens = tokens.to(device)
        mask = mask.to(device)
        images = images.to(device)
        
        # Generate random timesteps
        batch_size = tokens.shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # Generate random noise
        x = torch.randn_like(images).to(device)
        
        model.eval()
        baseline_model.eval()
        
        with torch.no_grad():
            # Get outputs
            model_output = model(x, timesteps, tokens=tokens, mask=mask)
            baseline_output = baseline_model(x, timesteps, tokens=tokens, mask=mask)
            
            # Compute differences
            diff = torch.abs(model_output - baseline_output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # Check if outputs are identical
            identical = torch.allclose(model_output, baseline_output, rtol=1e-5, atol=1e-8)
        
        model.train()
        baseline_model.train()
        
        results = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'outputs_identical': identical,
            'timestamp': len(self.stability_test_results),
        }
        
        self.stability_test_results.append(results)
        
        return results
    
    def should_save_checkpoint(self, loss: float) -> bool:
        """Determine if current state should be saved as best checkpoint."""
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of monitoring statistics."""
        summary = {
            'num_updates': len(self.loss_history),
            'loss_stats': {},
            'kl_stats': {},
            'gate_stats': {},
            'grad_stats': {},
            'stability_tests': len(self.stability_test_results),
        }
        
        # Loss statistics
        if self.loss_history:
            losses = list(self.loss_history)
            summary['loss_stats'] = {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'min': np.min(losses),
                'max': np.max(losses),
                'recent_mean': np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses),
            }
        
        # KL statistics
        if self.kl_history:
            kls = list(self.kl_history)
            summary['kl_stats'] = {
                'mean': np.mean(kls),
                'std': np.std(kls),
                'max': np.max(kls),
                'recent_mean': np.mean(kls[-10:]) if len(kls) >= 10 else np.mean(kls),
            }
        
        # Gate statistics
        if self.gate_history:
            gates = list(self.gate_history)
            summary['gate_stats'] = {
                'current': gates[-1],
                'mean': np.mean(gates),
                'min': np.min(gates),
                'max': np.max(gates),
            }
        
        # Gradient statistics
        if self.adapter_grad_history:
            adapter_grads = list(self.adapter_grad_history)
            summary['grad_stats']['adapter'] = {
                'mean': np.mean(adapter_grads),
                'max': np.max(adapter_grads),
                'recent_mean': np.mean(adapter_grads[-10:]) if len(adapter_grads) >= 10 else np.mean(adapter_grads),
            }
        
        if self.main_grad_history:
            main_grads = list(self.main_grad_history)
            summary['grad_stats']['main'] = {
                'mean': np.mean(main_grads),
                'max': np.max(main_grads),
                'recent_mean': np.mean(main_grads[-10:]) if len(main_grads) >= 10 else np.mean(main_grads),
            }
        
        # Latest stability test
        if self.stability_test_results:
            summary['latest_stability_test'] = self.stability_test_results[-1]
        
        return summary
    
    def format_summary(self) -> str:
        """Format summary as a readable string."""
        summary = self.get_summary()
        
        lines = ["=== Stability Monitor Summary ==="]
        lines.append(f"Updates: {summary['num_updates']}")
        
        if summary['loss_stats']:
            lines.append(f"\nLoss: {summary['loss_stats']['recent_mean']:.4f} "
                        f"(std={summary['loss_stats']['std']:.4f})")
        
        if summary['kl_stats']:
            lines.append(f"KL Divergence: {summary['kl_stats']['recent_mean']:.4f} "
                        f"(max={summary['kl_stats']['max']:.4f})")
        
        if summary['gate_stats']:
            lines.append(f"Adapter Gate: {summary['gate_stats']['current']:.4f}")
        
        if 'adapter' in summary['grad_stats']:
            lines.append(f"Adapter Gradients: {summary['grad_stats']['adapter']['recent_mean']:.4f}")
        
        if summary.get('latest_stability_test'):
            test = summary['latest_stability_test']
            lines.append(f"\nLatest Stability Test:")
            lines.append(f"  Max Diff: {test['max_diff']:.2e}")
            lines.append(f"  Outputs Identical: {test['outputs_identical']}")
        
        return "\n".join(lines)