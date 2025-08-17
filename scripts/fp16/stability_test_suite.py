#!/usr/bin/env python3
"""
FP16 Training Stability Test Suite
Comprehensive tests to ensure FP16 training is stable under various conditions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
from dataclasses import dataclass
import time

from dynamic_loss_scaler import DynamicLossScaler, LossScalerConfig
from master_weight_manager import MasterWeightManager
from fp16_training_step import FP16TrainingStep, FP16TrainingConfig


@dataclass
class StabilityTestResult:
    """Result from a stability test."""
    test_name: str
    passed: bool
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class StabilityTestSuite:
    """Comprehensive stability tests for FP16 training."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize test suite."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.results: List[StabilityTestResult] = []
    
    def create_test_model(self, complexity: str = "simple") -> nn.Module:
        """Create test model of varying complexity."""
        if complexity == "simple":
            model = nn.Sequential(
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
            )
        elif complexity == "medium":
            model = nn.Sequential(
                nn.Linear(100, 500),
                nn.LayerNorm(500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(500, 100),
            )
        elif complexity == "complex":
            class ComplexModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(1000, 256)
                    self.layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(256, 8, 1024, dropout=0.1)
                        for _ in range(4)
                    ])
                    self.norm = nn.LayerNorm(256)
                    self.output = nn.Linear(256, 100)
                
                def forward(self, x):
                    if x.dim() == 2:
                        x = self.embed(x.long().clamp(0, 999))
                    for layer in self.layers:
                        x = layer(x)
                    x = self.norm(x)
                    return self.output(x.mean(dim=1) if x.dim() == 3 else x)
            
            model = ComplexModel()
        else:
            raise ValueError(f"Unknown complexity: {complexity}")
        
        return model.to(self.device)
    
    def test_gradient_overflow(self) -> StabilityTestResult:
        """Test handling of gradient overflow."""
        test_name = "gradient_overflow"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            model = self.create_test_model("simple").half()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            config = LossScalerConfig(init_scale=65536, scale_window=10)
            scaler = DynamicLossScaler(config)
            
            overflow_count = 0
            for step in range(50):
                x = torch.randn(32, 100).to(self.device).half()
                y = model(x)
                loss = y.mean()
                
                # Inject overflow every 10 steps
                if step % 10 == 0 and step > 0:
                    loss = loss * 1e10
                
                optimizer.zero_grad()
                scaler.backward(loss)
                
                if scaler.check_overflow(optimizer):
                    overflow_count += 1
                
                scaler.step(optimizer)
            
            metrics = {
                'overflow_count': overflow_count,
                'final_scale': scaler.scale,
                'skip_rate': scaler.stats['skipped_steps'] / scaler.stats['total_steps'],
            }
            
            passed = overflow_count > 0 and scaler.scale < 65536
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_gradient_underflow(self) -> StabilityTestResult:
        """Test handling of gradient underflow."""
        test_name = "gradient_underflow"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            model = self.create_test_model("medium").half()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)  # Very small LR
            
            config = LossScalerConfig(init_scale=1.0, scale_window=10)
            scaler = DynamicLossScaler(config)
            
            underflow_steps = 0
            for step in range(30):
                x = torch.randn(32, 100).to(self.device).half()
                y = model(x)
                loss = y.mean() * 1e-8  # Very small loss
                
                optimizer.zero_grad()
                scaler.backward(loss)
                
                # Check for very small gradients
                for param in model.parameters():
                    if param.grad is not None:
                        if param.grad.abs().max() < 1e-7:
                            underflow_steps += 1
                            break
                
                scaler.step(optimizer)
            
            metrics = {
                'underflow_steps': underflow_steps,
                'final_scale': scaler.scale,
            }
            
            passed = scaler.scale > 1.0  # Scale should increase to combat underflow
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_nan_recovery(self) -> StabilityTestResult:
        """Test NaN detection and recovery."""
        test_name = "nan_recovery"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            model = self.create_test_model("simple").half()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            config = FP16TrainingConfig(enable_nan_recovery=True, max_nan_recoveries=3)
            trainer = FP16TrainingStep(model, optimizer, config)
            
            nan_injected = 0
            nan_recovered = 0
            
            def compute_loss_with_nan():
                nonlocal nan_injected
                x = torch.randn(32, 100).to(self.device).half()
                y = model(x)
                loss = y.mean()
                # Inject NaN occasionally
                if nan_injected < 2:
                    loss = loss * float('nan')
                    nan_injected += 1
                return loss
            
            for step in range(20):
                result = trainer.training_step(compute_loss_with_nan)
                if result.get('recovered', False):
                    nan_recovered += 1
            
            metrics = {
                'nan_injected': nan_injected,
                'nan_recovered': trainer.stats['nan_recoveries'],
                'successful_steps': trainer.stats['successful_steps'],
            }
            
            passed = trainer.stats['nan_recoveries'] > 0 and trainer.stats['successful_steps'] > 10
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_mixed_precision_consistency(self) -> StabilityTestResult:
        """Test consistency between FP16 and FP32 training."""
        test_name = "mixed_precision_consistency"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            # Create two identical models
            torch.manual_seed(42)
            model_fp32 = self.create_test_model("medium")
            
            torch.manual_seed(42)
            model_fp16 = self.create_test_model("medium").half()
            
            # Ensure weights start identical
            with torch.no_grad():
                for (n1, p1), (n2, p2) in zip(model_fp32.named_parameters(), 
                                              model_fp16.named_parameters()):
                    p2.data = p1.data.half()
            
            # Create optimizers
            opt_fp32 = torch.optim.SGD(model_fp32.parameters(), lr=0.01)
            opt_fp16 = torch.optim.SGD(model_fp16.parameters(), lr=0.01)
            
            # Train both for a few steps
            torch.manual_seed(42)
            losses_fp32 = []
            losses_fp16 = []
            
            for step in range(10):
                # Same input for both
                x = torch.randn(32, 100).to(self.device)
                
                # FP32 step
                opt_fp32.zero_grad()
                y32 = model_fp32(x)
                loss32 = y32.mean()
                loss32.backward()
                opt_fp32.step()
                losses_fp32.append(loss32.item())
                
                # FP16 step
                opt_fp16.zero_grad()
                y16 = model_fp16(x.half())
                loss16 = y16.mean()
                loss16.backward()
                opt_fp16.step()
                losses_fp16.append(loss16.item())
            
            # Compare losses
            loss_diff = np.mean(np.abs(np.array(losses_fp32) - np.array(losses_fp16)))
            
            # Compare final weights
            weight_diffs = []
            for (n1, p1), (n2, p2) in zip(model_fp32.named_parameters(), 
                                          model_fp16.named_parameters()):
                diff = (p1.data.float() - p2.data.float()).abs().mean().item()
                weight_diffs.append(diff)
            
            metrics = {
                'avg_loss_diff': loss_diff,
                'max_weight_diff': max(weight_diffs),
                'avg_weight_diff': np.mean(weight_diffs),
            }
            
            # Allow some difference due to precision
            passed = loss_diff < 0.1 and max(weight_diffs) < 0.1
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_gradient_accumulation(self) -> StabilityTestResult:
        """Test gradient accumulation with FP16."""
        test_name = "gradient_accumulation"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            model = self.create_test_model("medium").half()
            
            # Test with accumulation
            config = FP16TrainingConfig(gradient_accumulation_steps=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = FP16TrainingStep(model, optimizer, config)
            
            accumulated_steps = 0
            optimizer_steps = 0
            
            for step in range(20):
                def compute_loss():
                    x = torch.randn(8, 100).to(self.device).half()  # Smaller batch
                    y = model(x)
                    return y.mean()
                
                result = trainer.training_step(compute_loss)
                accumulated_steps += 1
                
                if not result['skipped']:
                    optimizer_steps += 1
            
            metrics = {
                'accumulated_steps': accumulated_steps,
                'optimizer_steps': optimizer_steps,
                'expected_optimizer_steps': accumulated_steps // 4,
            }
            
            passed = abs(optimizer_steps - accumulated_steps // 4) <= 1
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_master_weight_sync(self) -> StabilityTestResult:
        """Test master weight synchronization."""
        test_name = "master_weight_sync"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            model = self.create_test_model("simple")
            model[0].half()  # Convert first layer to FP16
            model[2].half()  # Convert last layer to FP16
            
            manager = MasterWeightManager(model)
            optimizer = manager.create_optimizer(torch.optim.Adam, lr=1e-3)
            
            # Run training steps
            for step in range(10):
                x = torch.randn(32, 100).to(self.device)
                y = model(x)
                loss = y.mean()
                
                optimizer.zero_grad()
                manager.backward_sync(loss, loss_scale=256.0)
                manager.clip_grad_norm(max_norm=1.0)
                manager.step(optimizer)
            
            # Check synchronization
            sync_stats = manager.check_sync()
            
            metrics = {
                'max_sync_diff': sync_stats['max_diff'],
                'mean_sync_diff': sync_stats['mean_diff'],
                'mismatched_params': sync_stats['mismatched_params'],
                'master_params_created': manager.stats['master_params_created'],
            }
            
            passed = sync_stats['max_diff'] < 1e-3 and sync_stats['mismatched_params'] == 0
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_extreme_learning_rates(self) -> StabilityTestResult:
        """Test stability with extreme learning rates."""
        test_name = "extreme_learning_rates"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            results = {}
            
            for lr in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
                model = self.create_test_model("simple").half()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                config = FP16TrainingConfig(
                    gradient_clip_norm=1.0,
                    init_loss_scale=256.0 if lr < 1e-4 else 1.0
                )
                trainer = FP16TrainingStep(model, optimizer, config)
                
                success_count = 0
                for step in range(20):
                    def compute_loss():
                        x = torch.randn(32, 100).to(self.device).half()
                        y = model(x)
                        return y.mean()
                    
                    result = trainer.training_step(compute_loss)
                    if not result['skipped'] and not np.isnan(result['loss']):
                        success_count += 1
                
                results[f'lr_{lr}'] = success_count / 20
            
            metrics = results
            
            # Should handle at least moderate learning rates
            passed = results['lr_0.0001'] > 0.8 and results['lr_1e-06'] > 0.8
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def test_long_training_stability(self) -> StabilityTestResult:
        """Test stability over longer training runs."""
        test_name = "long_training_stability"
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            model = self.create_test_model("complex").half()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            
            config = FP16TrainingConfig(
                gradient_clip_norm=1.0,
                gradient_accumulation_steps=2,
                log_frequency=50
            )
            trainer = FP16TrainingStep(model, optimizer, config)
            
            losses = []
            grad_norms = []
            
            for step in range(100):
                def compute_loss():
                    x = torch.randn(16, 10, 100).to(self.device).half()
                    y = model(x)
                    return y.mean() + 0.01 * sum(p.norm() for p in model.parameters())
                
                result = trainer.training_step(compute_loss)
                
                if not np.isnan(result['loss']):
                    losses.append(result['loss'])
                    grad_norms.append(result['grad_norm'])
            
            # Check for training stability
            if len(losses) > 10:
                loss_std = np.std(losses[-10:])
                loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
                grad_norm_std = np.std(grad_norms[-10:])
            else:
                loss_std = float('inf')
                loss_trend = 0
                grad_norm_std = float('inf')
            
            metrics = {
                'total_steps': 100,
                'successful_steps': len(losses),
                'loss_std': loss_std,
                'loss_trend': loss_trend,
                'grad_norm_std': grad_norm_std,
                'skip_rate': trainer.stats['skipped_steps'] / trainer.stats['total_steps'],
            }
            
            # Training should be stable (low variance) and mostly successful
            passed = len(losses) > 80 and loss_std < 1.0 and grad_norm_std < 10.0
            
            return StabilityTestResult(
                test_name=test_name,
                passed=passed,
                metrics=metrics,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name=test_name,
                passed=False,
                metrics={},
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all stability tests."""
        self.logger.info("Starting FP16 Stability Test Suite")
        self.results = []
        
        # Run all tests
        tests = [
            self.test_gradient_overflow,
            self.test_gradient_underflow,
            self.test_nan_recovery,
            self.test_mixed_precision_consistency,
            self.test_gradient_accumulation,
            self.test_master_weight_sync,
            self.test_extreme_learning_rates,
            self.test_long_training_stability,
        ]
        
        for test in tests:
            result = test()
            self.results.append(result)
            
            if result.passed:
                self.logger.info(f"✅ {result.test_name}: PASSED ({result.duration_seconds:.2f}s)")
            else:
                self.logger.error(f"❌ {result.test_name}: FAILED ({result.duration_seconds:.2f}s)")
                if result.error_message:
                    self.logger.error(f"   Error: {result.error_message}")
        
        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        summary = {
            'total_tests': total_count,
            'passed': passed_count,
            'failed': total_count - passed_count,
            'success_rate': passed_count / total_count,
            'total_duration': sum(r.duration_seconds for r in self.results),
            'results': [
                {
                    'test': r.test_name,
                    'passed': r.passed,
                    'metrics': r.metrics,
                    'error': r.error_message,
                    'duration': r.duration_seconds,
                }
                for r in self.results
            ]
        }
        
        return summary
    
    def save_results(self, filepath: str = ".claude/scripts/stability_test_results.json"):
        """Save test results to file."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'results': [
                {
                    'test': r.test_name,
                    'passed': r.passed,
                    'metrics': r.metrics,
                    'error': r.error_message,
                    'duration': r.duration_seconds,
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")


def main():
    """Run stability test suite."""
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("FP16 TRAINING STABILITY TEST SUITE")
    print("="*80)
    
    suite = StabilityTestSuite(device="cuda" if torch.cuda.is_available() else "cpu")
    summary = suite.run_all_tests()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"Total Duration: {summary['total_duration']:.2f} seconds")
    
    if summary['success_rate'] == 1.0:
        print("\n🎉 ALL TESTS PASSED!")
    elif summary['success_rate'] >= 0.8:
        print("\n⚠️  Most tests passed, but some issues detected")
    else:
        print("\n❌ Multiple test failures - FP16 training may be unstable")
    
    suite.save_results()


if __name__ == "__main__":
    main()