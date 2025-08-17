#!/usr/bin/env python3
"""
Inference Comparison Script for FP16 vs FP32 Models
Validates that FP16 model produces similar outputs to FP32.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'glide-text2im'))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


class InferenceComparison:
    """Compare inference between FP32 and FP16 models."""
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize comparison with checkpoint.
        
        Args:
            checkpoint_path: Path to base checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.results = {
            'outputs': {},
            'metrics': {},
            'timings': {},
            'memory': {},
        }
    
    def load_models(self) -> Tuple[nn.Module, nn.Module]:
        """
        Load FP32 and FP16 versions of the model.
        
        Returns:
            Tuple of (fp32_model, fp16_model)
        """
        print("Loading models...")
        
        # Load base FP32 model
        options = model_and_diffusion_defaults()
        options['use_fp16'] = False
        
        fp32_model, _ = create_model_and_diffusion(**options)
        fp32_model.eval()
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        fp32_model.load_state_dict(state_dict, strict=False)
        fp32_model = fp32_model.to(self.device)
        
        # Create FP16 version
        print("Converting to FP16...")
        fp16_model = self._convert_to_fp16(fp32_model)
        
        return fp32_model, fp16_model
    
    def _convert_to_fp16(self, model: nn.Module) -> nn.Module:
        """
        Convert model to mixed precision.
        
        Args:
            model: FP32 model
            
        Returns:
            Mixed precision model
        """
        # Import our converter
        from fp16_converter import SelectiveFP16Converter
        
        # Clone model to avoid modifying original
        import copy
        fp16_model = copy.deepcopy(model)
        
        # Convert to mixed precision
        converter = SelectiveFP16Converter(aggressive=True)
        fp16_model, stats = converter.convert_model_mixed_precision(fp16_model)
        
        print(f"  FP16 conversion: {stats['fp16_params']:,} FP16 params, "
              f"{stats['fp32_params']:,} FP32 params")
        
        return fp16_model
    
    def compare_outputs(self, 
                       fp32_model: nn.Module,
                       fp16_model: nn.Module,
                       inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compare outputs between FP32 and FP16 models.
        
        Args:
            fp32_model: FP32 model
            fp16_model: FP16 model
            inputs: Input tensors
            
        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}
        
        # Ensure models are in eval mode
        fp32_model.eval()
        fp16_model.eval()
        
        with torch.no_grad():
            # FP32 inference
            torch.cuda.synchronize()
            t0 = time.time()
            fp32_output = fp32_model(**inputs)
            torch.cuda.synchronize()
            fp32_time = time.time() - t0
            
            # FP16 inference
            torch.cuda.synchronize()
            t0 = time.time()
            fp16_output = fp16_model(**inputs)
            torch.cuda.synchronize()
            fp16_time = time.time() - t0
        
        # Handle different output types
        if isinstance(fp32_output, torch.Tensor):
            fp32_output = {'output': fp32_output}
            fp16_output = {'output': fp16_output}
        elif not isinstance(fp32_output, dict):
            fp32_output = {'output': fp32_output[0] if isinstance(fp32_output, tuple) else fp32_output}
            fp16_output = {'output': fp16_output[0] if isinstance(fp16_output, tuple) else fp16_output}
        
        # Compare each output tensor
        for key in fp32_output.keys():
            if not isinstance(fp32_output[key], torch.Tensor):
                continue
            
            fp32_tensor = fp32_output[key].float()
            fp16_tensor = fp16_output[key].float()
            
            # Compute metrics
            diff = (fp32_tensor - fp16_tensor).abs()
            
            metrics[f'{key}_max_diff'] = diff.max().item()
            metrics[f'{key}_mean_diff'] = diff.mean().item()
            metrics[f'{key}_relative_error'] = (diff / (fp32_tensor.abs() + 1e-8)).mean().item()
            
            # Cosine similarity
            fp32_flat = fp32_tensor.flatten()
            fp16_flat = fp16_tensor.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(
                fp32_flat.unsqueeze(0),
                fp16_flat.unsqueeze(0)
            ).item()
            metrics[f'{key}_cosine_similarity'] = cosine_sim
            
            # Statistical comparison
            metrics[f'{key}_fp32_mean'] = fp32_tensor.mean().item()
            metrics[f'{key}_fp16_mean'] = fp16_tensor.mean().item()
            metrics[f'{key}_fp32_std'] = fp32_tensor.std().item()
            metrics[f'{key}_fp16_std'] = fp16_tensor.std().item()
        
        # Timing comparison
        metrics['fp32_time_ms'] = fp32_time * 1000
        metrics['fp16_time_ms'] = fp16_time * 1000
        metrics['speedup'] = fp32_time / fp16_time if fp16_time > 0 else 0
        
        # Store outputs for visualization
        self.results['outputs']['fp32'] = fp32_output
        self.results['outputs']['fp16'] = fp16_output
        
        return metrics
    
    def compare_memory(self,
                      fp32_model: nn.Module,
                      fp16_model: nn.Module) -> Dict[str, float]:
        """
        Compare memory usage between models.
        
        Args:
            fp32_model: FP32 model
            fp16_model: FP16 model
            
        Returns:
            Dictionary of memory metrics
        """
        memory_metrics = {}
        
        # Calculate model sizes
        fp32_params = sum(p.numel() * p.element_size() for p in fp32_model.parameters())
        fp16_params = sum(p.numel() * p.element_size() for p in fp16_model.parameters())
        
        memory_metrics['fp32_model_size_mb'] = fp32_params / 1e6
        memory_metrics['fp16_model_size_mb'] = fp16_params / 1e6
        memory_metrics['model_size_reduction'] = 1 - (fp16_params / fp32_params)
        
        # Measure GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # FP32 memory
            torch.cuda.reset_peak_memory_stats()
            _ = fp32_model(torch.randn(1, 3, 64, 64).to(self.device))
            torch.cuda.synchronize()
            fp32_peak = torch.cuda.max_memory_allocated() / 1e6
            
            torch.cuda.empty_cache()
            
            # FP16 memory
            torch.cuda.reset_peak_memory_stats()
            _ = fp16_model(torch.randn(1, 3, 64, 64).to(self.device))
            torch.cuda.synchronize()
            fp16_peak = torch.cuda.max_memory_allocated() / 1e6
            
            memory_metrics['fp32_peak_memory_mb'] = fp32_peak
            memory_metrics['fp16_peak_memory_mb'] = fp16_peak
            memory_metrics['memory_reduction'] = 1 - (fp16_peak / fp32_peak) if fp32_peak > 0 else 0
        
        return memory_metrics
    
    def run_comprehensive_test(self,
                               num_samples: int = 10,
                               batch_size: int = 4) -> Dict[str, Any]:
        """
        Run comprehensive comparison test.
        
        Args:
            num_samples: Number of test samples
            batch_size: Batch size for testing
            
        Returns:
            Complete test results
        """
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE FP16 VS FP32 COMPARISON")
        print("="*80)
        
        # Load models
        fp32_model, fp16_model = self.load_models()
        
        # Memory comparison
        print("\nComparing memory usage...")
        memory_metrics = self.compare_memory(fp32_model, fp16_model)
        self.results['memory'] = memory_metrics
        
        print(f"  FP32 model size: {memory_metrics['fp32_model_size_mb']:.1f} MB")
        print(f"  FP16 model size: {memory_metrics['fp16_model_size_mb']:.1f} MB")
        print(f"  Size reduction: {memory_metrics['model_size_reduction']*100:.1f}%")
        
        # Run inference comparisons
        print(f"\nRunning {num_samples} inference comparisons...")
        all_metrics = []
        
        for i in range(num_samples):
            # Generate random inputs
            inputs = {
                'x': torch.randn(batch_size, 3, 64, 64).to(self.device),
                't': torch.randint(0, 1000, (batch_size,)).to(self.device),
            }
            
            # Compare outputs
            metrics = self.compare_outputs(fp32_model, fp16_model, inputs)
            all_metrics.append(metrics)
            
            if i == 0:
                print(f"\nSample {i} results:")
                print(f"  Max difference: {metrics.get('output_max_diff', 0):.6f}")
                print(f"  Mean difference: {metrics.get('output_mean_diff', 0):.6f}")
                print(f"  Relative error: {metrics.get('output_relative_error', 0):.4%}")
                print(f"  Cosine similarity: {metrics.get('output_cosine_similarity', 0):.6f}")
                print(f"  FP32 time: {metrics['fp32_time_ms']:.2f} ms")
                print(f"  FP16 time: {metrics['fp16_time_ms']:.2f} ms")
                print(f"  Speedup: {metrics['speedup']:.2f}x")
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        self.results['metrics'] = aggregated
        
        # Print summary
        self._print_summary(aggregated)
        
        return self.results
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across samples."""
        aggregated = {}
        
        # Collect all metric names
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Compute statistics for each metric
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                }
        
        return aggregated
    
    def _print_summary(self, aggregated: Dict[str, Dict[str, float]]) -> None:
        """Print summary of results."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Accuracy metrics
        print("\n📊 Accuracy Metrics (averaged):")
        if 'output_max_diff' in aggregated:
            print(f"  Max difference: {aggregated['output_max_diff']['mean']:.6f} "
                  f"(±{aggregated['output_max_diff']['std']:.6f})")
        if 'output_mean_diff' in aggregated:
            print(f"  Mean difference: {aggregated['output_mean_diff']['mean']:.6f} "
                  f"(±{aggregated['output_mean_diff']['std']:.6f})")
        if 'output_relative_error' in aggregated:
            print(f"  Relative error: {aggregated['output_relative_error']['mean']:.4%} "
                  f"(±{aggregated['output_relative_error']['std']:.4%})")
        if 'output_cosine_similarity' in aggregated:
            print(f"  Cosine similarity: {aggregated['output_cosine_similarity']['mean']:.6f} "
                  f"(±{aggregated['output_cosine_similarity']['std']:.6f})")
        
        # Performance metrics
        print("\n⚡ Performance Metrics:")
        if 'speedup' in aggregated:
            print(f"  Average speedup: {aggregated['speedup']['mean']:.2f}x "
                  f"(±{aggregated['speedup']['std']:.2f})")
        if 'fp32_time_ms' in aggregated:
            print(f"  FP32 inference: {aggregated['fp32_time_ms']['mean']:.2f} ms")
        if 'fp16_time_ms' in aggregated:
            print(f"  FP16 inference: {aggregated['fp16_time_ms']['mean']:.2f} ms")
        
        # Memory metrics
        if self.results['memory']:
            print(f"\n💾 Memory Metrics:")
            print(f"  Model size reduction: {self.results['memory']['model_size_reduction']*100:.1f}%")
            if 'memory_reduction' in self.results['memory']:
                print(f"  Runtime memory reduction: {self.results['memory']['memory_reduction']*100:.1f}%")
        
        # Verdict
        print("\n" + "="*80)
        print("VERDICT:")
        
        # Check if outputs are close enough
        max_diff = aggregated.get('output_max_diff', {}).get('mean', float('inf'))
        rel_error = aggregated.get('output_relative_error', {}).get('mean', float('inf'))
        cosine_sim = aggregated.get('output_cosine_similarity', {}).get('mean', 0)
        speedup = aggregated.get('speedup', {}).get('mean', 1)
        
        if max_diff < 0.01 and rel_error < 0.01 and cosine_sim > 0.99:
            print("✅ FP16 model produces nearly identical outputs to FP32!")
            print(f"✅ Performance improvement: {speedup:.2f}x faster")
            print(f"✅ Memory savings: {self.results['memory']['model_size_reduction']*100:.1f}%")
            print("\n🎉 FP16 conversion successful and ready for training!")
        elif max_diff < 0.1 and rel_error < 0.05 and cosine_sim > 0.95:
            print("⚠️  FP16 model shows small differences from FP32")
            print("   This is likely acceptable for training")
            print(f"   Performance gain: {speedup:.2f}x")
        else:
            print("❌ FP16 model outputs differ significantly from FP32")
            print("   Further investigation needed")
            print(f"   Max diff: {max_diff:.6f}")
            print(f"   Relative error: {rel_error:.4%}")
            print(f"   Cosine similarity: {cosine_sim:.6f}")
        
        print("="*80)
    
    def save_results(self, output_path: str = ".claude/scripts/inference_comparison.json") -> None:
        """Save results to file."""
        # Convert tensors to lists for JSON serialization
        serializable_results = {
            'metrics': self.results['metrics'],
            'memory': self.results['memory'],
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Run inference comparison test."""
    checkpoint_path = "glide_model_cache/glide50k.pt"
    
    if not os.path.exists(checkpoint_path):
        # Try FP16 converted version
        checkpoint_path = "glide_model_cache/glide50k_fp16.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("Please run fp16_converter.py first")
            sys.exit(1)
    
    comparison = InferenceComparison(checkpoint_path)
    results = comparison.run_comprehensive_test(num_samples=5, batch_size=2)
    comparison.save_results()


if __name__ == "__main__":
    main()