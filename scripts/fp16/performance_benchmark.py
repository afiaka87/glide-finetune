#!/usr/bin/env python3
"""
FP16 Performance Benchmarking Tools
Measures and compares performance between FP32 and FP16 training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import gc

from fp16_converter import SelectiveFP16Converter
from fp16_training_step import FP16TrainingStep, FP16TrainingConfig


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    precision: str  # "fp32" or "fp16"
    batch_size: int
    sequence_length: int
    model_size: str
    
    # Timing metrics (in seconds)
    forward_time: float
    backward_time: float
    optimizer_time: float
    total_time: float
    throughput: float  # samples/second
    
    # Memory metrics (in MB)
    model_memory: float
    activation_memory: float
    gradient_memory: float
    peak_memory: float
    
    # GPU metrics
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    
    # Other metrics
    loss_scale: Optional[float] = None
    overflow_rate: float = 0.0


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for FP16 vs FP32."""
    
    def __init__(self, device: str = "cuda", warmup_steps: int = 10):
        """
        Initialize benchmark.
        
        Args:
            device: Device to run benchmarks on
            warmup_steps: Number of warmup steps before timing
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.warmup_steps = warmup_steps
        self.results: List[BenchmarkResult] = []
        
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    
    def create_model(self, size: str = "small") -> nn.Module:
        """Create model for benchmarking."""
        if size == "small":
            # ~10M parameters
            model = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )
        elif size == "medium":
            # ~50M parameters
            model = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 4096),
                nn.LayerNorm(4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.LayerNorm(2048),
                nn.Linear(2048, 1024),
            )
        elif size == "large":
            # ~100M parameters
            class LargeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=1024,
                            nhead=16,
                            dim_feedforward=4096,
                            dropout=0.1
                        )
                        for _ in range(12)
                    ])
                    self.output = nn.Linear(1024, 1024)
                
                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return self.output(x)
            
            model = LargeModel()
        else:
            raise ValueError(f"Unknown model size: {size}")
        
        return model.to(self.device)
    
    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        memory = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_used_gb': psutil.virtual_memory().used / 1e9,
        }
        
        if self.has_gpu:
            torch.cuda.synchronize()
            memory['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1e6
            memory['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1e6
            memory['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1e6
            
            if self.gpu:
                self.gpu = GPUtil.getGPUs()[0]
                memory['gpu_utilization'] = self.gpu.load * 100
                memory['gpu_memory_used_mb'] = self.gpu.memoryUsed
                memory['gpu_memory_total_mb'] = self.gpu.memoryTotal
        
        return memory
    
    def benchmark_training_step(self,
                               model: nn.Module,
                               optimizer: torch.optim.Optimizer,
                               batch_size: int,
                               sequence_length: int,
                               use_fp16: bool = False) -> BenchmarkResult:
        """
        Benchmark a single training configuration.
        
        Args:
            model: Model to benchmark
            optimizer: Optimizer to use
            batch_size: Batch size
            sequence_length: Sequence length (or feature dimension)
            use_fp16: Whether to use FP16 training
            
        Returns:
            Benchmark results
        """
        precision = "fp16" if use_fp16 else "fp32"
        
        # Setup FP16 if needed
        if use_fp16:
            converter = SelectiveFP16Converter(aggressive=True)
            model, _ = converter.convert_model_mixed_precision(model)
            
            config = FP16TrainingConfig(
                use_loss_scaling=True,
                init_loss_scale=256.0,
                gradient_clip_norm=1.0,
            )
            trainer = FP16TrainingStep(model, optimizer, config)
        
        # Clear cache and reset metrics
        if self.has_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Warmup
        for _ in range(self.warmup_steps):
            x = torch.randn(batch_size, sequence_length).to(self.device)
            if use_fp16:
                x = x.half()
            
            optimizer.zero_grad()
            y = model(x)
            loss = y.mean()
            loss.backward()
            optimizer.step()
        
        # Timed runs
        forward_times = []
        backward_times = []
        optimizer_times = []
        total_times = []
        overflow_count = 0
        
        num_iterations = 50
        
        for _ in range(num_iterations):
            x = torch.randn(batch_size, sequence_length).to(self.device)
            if use_fp16:
                x = x.half()
            
            # Forward pass
            torch.cuda.synchronize() if self.has_gpu else None
            t0 = time.perf_counter()
            
            y = model(x)
            loss = y.mean()
            
            torch.cuda.synchronize() if self.has_gpu else None
            t1 = time.perf_counter()
            forward_times.append(t1 - t0)
            
            # Backward pass
            optimizer.zero_grad()
            
            if use_fp16 and hasattr(trainer, 'loss_scaler'):
                loss = trainer.loss_scaler.scale_loss(loss)
            
            torch.cuda.synchronize() if self.has_gpu else None
            t0 = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize() if self.has_gpu else None
            t1 = time.perf_counter()
            backward_times.append(t1 - t0)
            
            # Optimizer step
            torch.cuda.synchronize() if self.has_gpu else None
            t0 = time.perf_counter()
            
            if use_fp16:
                result = trainer.training_step(lambda: model(x).mean())
                if result['skipped']:
                    overflow_count += 1
            else:
                optimizer.step()
            
            torch.cuda.synchronize() if self.has_gpu else None
            t1 = time.perf_counter()
            optimizer_times.append(t1 - t0)
            
            total_times.append(forward_times[-1] + backward_times[-1] + optimizer_times[-1])
        
        # Calculate metrics
        memory = self.measure_memory()
        
        # Model size in MB
        model_params = sum(p.numel() * p.element_size() for p in model.parameters())
        model_memory = model_params / 1e6
        
        result = BenchmarkResult(
            precision=precision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_size=self._get_model_size_label(model),
            
            forward_time=np.mean(forward_times),
            backward_time=np.mean(backward_times),
            optimizer_time=np.mean(optimizer_times),
            total_time=np.mean(total_times),
            throughput=batch_size / np.mean(total_times),
            
            model_memory=model_memory,
            activation_memory=memory.get('gpu_allocated_mb', 0) - model_memory,
            gradient_memory=model_memory,  # Approximate
            peak_memory=memory.get('gpu_max_allocated_mb', 0),
            
            gpu_utilization=memory.get('gpu_utilization', 0),
            gpu_memory_used=memory.get('gpu_memory_used_mb', 0),
            gpu_memory_total=memory.get('gpu_memory_total_mb', 0),
            
            loss_scale=trainer.loss_scaler.scale if use_fp16 else None,
            overflow_rate=overflow_count / num_iterations,
        )
        
        return result
    
    def _get_model_size_label(self, model: nn.Module) -> str:
        """Get human-readable model size label."""
        total_params = sum(p.numel() for p in model.parameters())
        
        if total_params < 20e6:
            return "small"
        elif total_params < 80e6:
            return "medium"
        else:
            return "large"
    
    def compare_precision(self,
                         model_sizes: List[str] = ["small", "medium"],
                         batch_sizes: List[int] = [1, 4, 16],
                         sequence_lengths: List[int] = [512, 1024]) -> Dict[str, List[BenchmarkResult]]:
        """
        Compare FP32 vs FP16 across different configurations.
        
        Args:
            model_sizes: List of model sizes to test
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary mapping configuration to results
        """
        results = {"fp32": [], "fp16": []}
        
        for model_size in model_sizes:
            for batch_size in batch_sizes:
                for seq_len in sequence_lengths:
                    print(f"\nBenchmarking {model_size} model, batch={batch_size}, seq={seq_len}")
                    
                    # Skip configurations that might OOM
                    if model_size == "large" and batch_size > 4:
                        print("  Skipping large model with large batch (OOM risk)")
                        continue
                    
                    # FP32 benchmark
                    print("  Testing FP32...")
                    model_fp32 = self.create_model(model_size)
                    opt_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=1e-4)
                    
                    try:
                        result_fp32 = self.benchmark_training_step(
                            model_fp32, opt_fp32, batch_size, seq_len, use_fp16=False
                        )
                        results["fp32"].append(result_fp32)
                        print(f"    FP32: {result_fp32.throughput:.1f} samples/sec")
                    except Exception as e:
                        print(f"    FP32 failed: {e}")
                    
                    # Clear memory
                    del model_fp32, opt_fp32
                    torch.cuda.empty_cache() if self.has_gpu else None
                    gc.collect()
                    
                    # FP16 benchmark
                    print("  Testing FP16...")
                    model_fp16 = self.create_model(model_size)
                    opt_fp16 = torch.optim.Adam(model_fp16.parameters(), lr=1e-4)
                    
                    try:
                        result_fp16 = self.benchmark_training_step(
                            model_fp16, opt_fp16, batch_size, seq_len, use_fp16=True
                        )
                        results["fp16"].append(result_fp16)
                        print(f"    FP16: {result_fp16.throughput:.1f} samples/sec")
                        
                        # Calculate speedup
                        if len(results["fp32"]) > 0:
                            speedup = result_fp16.throughput / results["fp32"][-1].throughput
                            print(f"    Speedup: {speedup:.2f}x")
                    except Exception as e:
                        print(f"    FP16 failed: {e}")
                    
                    # Clear memory
                    del model_fp16, opt_fp16
                    torch.cuda.empty_cache() if self.has_gpu else None
                    gc.collect()
        
        self.results = results["fp32"] + results["fp16"]
        return results
    
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Separate FP32 and FP16 results
        fp32_results = [r for r in self.results if r.precision == "fp32"]
        fp16_results = [r for r in self.results if r.precision == "fp16"]
        
        if not fp32_results or not fp16_results:
            return {"error": "Need both FP32 and FP16 results for comparison"}
        
        # Calculate averages
        def avg(results, attr):
            values = [getattr(r, attr) for r in results]
            return np.mean(values) if values else 0
        
        report = {
            "summary": {
                "num_configurations": len(fp32_results),
                "device": str(self.device),
                "has_gpu": self.has_gpu,
            },
            
            "performance": {
                "fp32": {
                    "avg_throughput": avg(fp32_results, "throughput"),
                    "avg_forward_time_ms": avg(fp32_results, "forward_time") * 1000,
                    "avg_backward_time_ms": avg(fp32_results, "backward_time") * 1000,
                    "avg_total_time_ms": avg(fp32_results, "total_time") * 1000,
                },
                "fp16": {
                    "avg_throughput": avg(fp16_results, "throughput"),
                    "avg_forward_time_ms": avg(fp16_results, "forward_time") * 1000,
                    "avg_backward_time_ms": avg(fp16_results, "backward_time") * 1000,
                    "avg_total_time_ms": avg(fp16_results, "total_time") * 1000,
                    "avg_overflow_rate": avg(fp16_results, "overflow_rate"),
                },
                "speedup": {
                    "throughput": avg(fp16_results, "throughput") / avg(fp32_results, "throughput"),
                    "forward": avg(fp32_results, "forward_time") / avg(fp16_results, "forward_time"),
                    "backward": avg(fp32_results, "backward_time") / avg(fp16_results, "backward_time"),
                    "total": avg(fp32_results, "total_time") / avg(fp16_results, "total_time"),
                },
            },
            
            "memory": {
                "fp32": {
                    "avg_model_memory_mb": avg(fp32_results, "model_memory"),
                    "avg_peak_memory_mb": avg(fp32_results, "peak_memory"),
                    "avg_gpu_memory_used_mb": avg(fp32_results, "gpu_memory_used"),
                },
                "fp16": {
                    "avg_model_memory_mb": avg(fp16_results, "model_memory"),
                    "avg_peak_memory_mb": avg(fp16_results, "peak_memory"),
                    "avg_gpu_memory_used_mb": avg(fp16_results, "gpu_memory_used"),
                },
                "reduction": {
                    "model_memory": 1 - avg(fp16_results, "model_memory") / avg(fp32_results, "model_memory"),
                    "peak_memory": 1 - avg(fp16_results, "peak_memory") / avg(fp32_results, "peak_memory"),
                },
            },
            
            "detailed_results": [asdict(r) for r in self.results],
        }
        
        return report
    
    def plot_results(self, save_path: str = ".claude/scripts/benchmark_plots.png"):
        """Generate visualization of benchmark results."""
        if not self.results:
            print("No results to plot")
            return
        
        fp32_results = [r for r in self.results if r.precision == "fp32"]
        fp16_results = [r for r in self.results if r.precision == "fp16"]
        
        if not fp32_results or not fp16_results:
            print("Need both FP32 and FP16 results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Throughput comparison
        ax = axes[0, 0]
        configs = [f"B{r.batch_size}_S{r.sequence_length}" for r in fp32_results]
        fp32_throughput = [r.throughput for r in fp32_results]
        fp16_throughput = [r.throughput for r in fp16_results[:len(fp32_results)]]
        
        x = np.arange(len(configs))
        width = 0.35
        ax.bar(x - width/2, fp32_throughput, width, label='FP32', color='blue')
        ax.bar(x + width/2, fp16_throughput, width, label='FP16', color='green')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Training Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Memory usage comparison
        ax = axes[0, 1]
        fp32_memory = [r.peak_memory for r in fp32_results]
        fp16_memory = [r.peak_memory for r in fp16_results[:len(fp32_results)]]
        
        ax.bar(x - width/2, fp32_memory, width, label='FP32', color='blue')
        ax.bar(x + width/2, fp16_memory, width, label='FP16', color='green')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Speedup by configuration
        ax = axes[1, 0]
        speedups = [fp16_throughput[i] / fp32_throughput[i] 
                   for i in range(min(len(fp32_throughput), len(fp16_throughput)))]
        
        ax.bar(configs, speedups, color='orange')
        ax.axhline(y=1.0, color='r', linestyle='--', label='No speedup')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('FP16 Speedup over FP32')
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Time breakdown
        ax = axes[1, 1]
        fp32_forward = np.mean([r.forward_time for r in fp32_results]) * 1000
        fp32_backward = np.mean([r.backward_time for r in fp32_results]) * 1000
        fp32_optimizer = np.mean([r.optimizer_time for r in fp32_results]) * 1000
        
        fp16_forward = np.mean([r.forward_time for r in fp16_results]) * 1000
        fp16_backward = np.mean([r.backward_time for r in fp16_results]) * 1000
        fp16_optimizer = np.mean([r.optimizer_time for r in fp16_results]) * 1000
        
        categories = ['Forward', 'Backward', 'Optimizer']
        fp32_times = [fp32_forward, fp32_backward, fp32_optimizer]
        fp16_times = [fp16_forward, fp16_backward, fp16_optimizer]
        
        x = np.arange(len(categories))
        ax.bar(x - width/2, fp32_times, width, label='FP32', color='blue')
        ax.bar(x + width/2, fp16_times, width, label='FP16', color='green')
        ax.set_xlabel('Operation')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Time Breakdown by Operation')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        print(f"Plots saved to {save_path}")
        plt.close()
    
    def save_results(self, filepath: str = ".claude/scripts/benchmark_results.json"):
        """Save benchmark results to file."""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Results saved to {filepath}")


def main():
    """Run performance benchmarks."""
    print("="*80)
    print("FP16 PERFORMANCE BENCHMARK")
    print("="*80)
    
    benchmark = PerformanceBenchmark(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Run comparisons
    results = benchmark.compare_precision(
        model_sizes=["small", "medium"],
        batch_sizes=[1, 4, 8],
        sequence_lengths=[512, 1024]
    )
    
    # Generate report
    report = benchmark.generate_report()
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    if "performance" in report:
        perf = report["performance"]
        print(f"\n📊 Performance:")
        print(f"  FP32 Throughput: {perf['fp32']['avg_throughput']:.1f} samples/sec")
        print(f"  FP16 Throughput: {perf['fp16']['avg_throughput']:.1f} samples/sec")
        print(f"  Speedup: {perf['speedup']['throughput']:.2f}x")
        
        print(f"\n⏱️  Timing:")
        print(f"  FP32 Total: {perf['fp32']['avg_total_time_ms']:.2f} ms/step")
        print(f"  FP16 Total: {perf['fp16']['avg_total_time_ms']:.2f} ms/step")
        print(f"  Time Reduction: {(1 - perf['fp16']['avg_total_time_ms'] / perf['fp32']['avg_total_time_ms']) * 100:.1f}%")
        
        if "memory" in report:
            mem = report["memory"]
            print(f"\n💾 Memory:")
            print(f"  FP32 Peak: {mem['fp32']['avg_peak_memory_mb']:.1f} MB")
            print(f"  FP16 Peak: {mem['fp16']['avg_peak_memory_mb']:.1f} MB")
            print(f"  Memory Reduction: {mem['reduction']['peak_memory'] * 100:.1f}%")
        
        if perf['fp16'].get('avg_overflow_rate', 0) > 0:
            print(f"\n⚠️  Overflow Rate: {perf['fp16']['avg_overflow_rate'] * 100:.1f}%")
    
    # Save results
    benchmark.save_results()
    benchmark.plot_results()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()