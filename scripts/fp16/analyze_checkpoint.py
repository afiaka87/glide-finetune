#!/usr/bin/env python3
"""
Checkpoint Weight Distribution Analysis for FP16 Conversion Safety
Analyzes glide50k.pt to identify potential FP16 conversion issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path
import json

# FP16 numerical limits
FP16_MAX = 65504.0
FP16_MIN = 6.103515625e-05  # Smallest normal positive number
FP16_SUBNORMAL_MIN = 5.960464477539063e-08  # Smallest subnormal

class CheckpointAnalyzer:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.stats = defaultdict(dict)
        self.problematic_layers = []
        self.layer_types = defaultdict(list)
        
    def load_checkpoint(self):
        """Load the checkpoint and extract state dict."""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            # Direct model state
            state_dict = checkpoint
            
        return state_dict
    
    def analyze_tensor(self, name, tensor):
        """Analyze a single tensor for FP16 safety."""
        if not isinstance(tensor, torch.Tensor):
            return None
            
        tensor_np = tensor.detach().cpu().numpy().flatten()
        
        stats = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': tensor.numel(),
            'mean': float(np.mean(tensor_np)),
            'std': float(np.std(tensor_np)),
            'min': float(np.min(tensor_np)),
            'max': float(np.max(tensor_np)),
            'abs_min': float(np.min(np.abs(tensor_np[tensor_np != 0]))) if np.any(tensor_np != 0) else 0,
            'abs_max': float(np.max(np.abs(tensor_np))),
            'zeros_pct': float(np.mean(tensor_np == 0) * 100),
            'nan_count': int(np.sum(np.isnan(tensor_np))),
            'inf_count': int(np.sum(np.isinf(tensor_np))),
        }
        
        # Check FP16 safety
        stats['fp16_overflow_risk'] = stats['abs_max'] > FP16_MAX
        stats['fp16_underflow_risk'] = (stats['abs_min'] < FP16_MIN and stats['abs_min'] > 0)
        stats['fp16_subnormal_risk'] = (stats['abs_min'] < FP16_SUBNORMAL_MIN and stats['abs_min'] > 0)
        
        # Count values outside FP16 range
        overflow_count = np.sum(np.abs(tensor_np) > FP16_MAX)
        underflow_count = np.sum((np.abs(tensor_np) < FP16_MIN) & (tensor_np != 0))
        
        stats['overflow_values'] = int(overflow_count)
        stats['underflow_values'] = int(underflow_count)
        stats['overflow_pct'] = float(overflow_count / tensor.numel() * 100)
        stats['underflow_pct'] = float(underflow_count / tensor.numel() * 100)
        
        # Quantiles for distribution understanding
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        stats['quantiles'] = {f'q{int(q*100)}': float(np.quantile(np.abs(tensor_np), q)) 
                              for q in quantiles}
        
        return stats
    
    def categorize_layer(self, name):
        """Categorize layer type from name."""
        if 'norm' in name.lower() or 'ln' in name.lower():
            return 'normalization'
        elif 'embed' in name.lower() or 'positional' in name.lower():
            return 'embedding'
        elif 'conv' in name.lower():
            return 'convolution'
        elif 'linear' in name.lower() or 'fc' in name.lower() or 'proj' in name.lower():
            return 'linear'
        elif 'attention' in name.lower() or 'attn' in name.lower():
            return 'attention'
        elif 'qkv' in name.lower():
            return 'qkv_projection'
        elif 'out' in name.lower() and 'output' not in name.lower():
            return 'output_projection'
        elif 'transformer' in name.lower():
            return 'transformer'
        else:
            return 'other'
    
    def analyze(self):
        """Perform complete analysis of the checkpoint."""
        state_dict = self.load_checkpoint()
        
        print(f"\nAnalyzing {len(state_dict)} tensors...")
        print("="*80)
        
        for name, tensor in state_dict.items():
            stats = self.analyze_tensor(name, tensor)
            if stats:
                self.stats[name] = stats
                layer_type = self.categorize_layer(name)
                self.layer_types[layer_type].append(name)
                
                # Track problematic layers
                if stats['fp16_overflow_risk'] or stats['fp16_underflow_risk']:
                    self.problematic_layers.append({
                        'name': name,
                        'type': layer_type,
                        'issue': 'overflow' if stats['fp16_overflow_risk'] else 'underflow',
                        'severity': stats['overflow_pct'] + stats['underflow_pct']
                    })
        
        self._print_summary()
        self._generate_recommendations()
        return self.stats
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*80)
        print("CHECKPOINT ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_params = sum(s['numel'] for s in self.stats.values())
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Total Layers: {len(self.stats)}")
        
        # Layer type breakdown
        print("\nLayer Type Distribution:")
        for layer_type, names in self.layer_types.items():
            count = len(names)
            params = sum(self.stats[n]['numel'] for n in names)
            print(f"  {layer_type:20s}: {count:4d} layers, {params:12,} params")
        
        # FP16 safety analysis
        print("\nFP16 Conversion Safety:")
        overflow_layers = sum(1 for s in self.stats.values() if s['fp16_overflow_risk'])
        underflow_layers = sum(1 for s in self.stats.values() if s['fp16_underflow_risk'])
        
        print(f"  Layers with overflow risk:  {overflow_layers}")
        print(f"  Layers with underflow risk: {underflow_layers}")
        
        if self.problematic_layers:
            print("\nMost Problematic Layers (Top 10):")
            sorted_problems = sorted(self.problematic_layers, 
                                    key=lambda x: x['severity'], reverse=True)[:10]
            for prob in sorted_problems:
                print(f"  - {prob['name'][:50]:50s} ({prob['type']:15s}): "
                      f"{prob['issue']:10s} severity={prob['severity']:.2f}%")
        
        # Value distribution
        all_max = max(s['abs_max'] for s in self.stats.values())
        all_min = min(s['abs_min'] for s in self.stats.values() if s['abs_min'] > 0)
        
        print(f"\nGlobal Value Range:")
        print(f"  Maximum absolute value: {all_max:.6e}")
        print(f"  Minimum absolute value: {all_min:.6e}")
        print(f"  FP16 max: {FP16_MAX:.6e}")
        print(f"  FP16 min normal: {FP16_MIN:.6e}")
        
    def _generate_recommendations(self):
        """Generate FP16 conversion recommendations."""
        print("\n" + "="*80)
        print("FP16 CONVERSION RECOMMENDATIONS")
        print("="*80)
        
        # Layers to keep in FP32
        fp32_layers = set()
        
        # Always keep normalization layers in FP32
        for name in self.layer_types.get('normalization', []):
            fp32_layers.add(name)
        
        # Keep embeddings in FP32 for stability
        for name in self.layer_types.get('embedding', []):
            fp32_layers.add(name)
        
        # Keep problematic layers in FP32
        for prob in self.problematic_layers:
            if prob['severity'] > 0.1:  # More than 0.1% of values problematic
                fp32_layers.add(prob['name'])
        
        print(f"\n1. Keep {len(fp32_layers)} layers in FP32 for stability:")
        print(f"   - All normalization layers ({len(self.layer_types.get('normalization', []))})")
        print(f"   - All embedding layers ({len(self.layer_types.get('embedding', []))})")
        print(f"   - Layers with >0.1% values outside FP16 range")
        
        # Estimate memory savings
        fp32_params = sum(self.stats[n]['numel'] for n in fp32_layers)
        fp16_params = sum(s['numel'] for n, s in self.stats.items() if n not in fp32_layers)
        total_params = fp32_params + fp16_params
        
        memory_before = total_params * 4  # All FP32
        memory_after = fp32_params * 4 + fp16_params * 2  # Mixed precision
        savings_pct = (1 - memory_after / memory_before) * 100
        
        print(f"\n2. Memory Savings Estimate:")
        print(f"   - FP32 params: {fp32_params:,} ({fp32_params/total_params*100:.1f}%)")
        print(f"   - FP16 params: {fp16_params:,} ({fp16_params/total_params*100:.1f}%)")
        print(f"   - Memory reduction: {savings_pct:.1f}%")
        
        print(f"\n3. Loss Scaling Recommendations:")
        if any(s['fp16_underflow_risk'] for s in self.stats.values()):
            print(f"   - Start with loss scale: 256 (2^8)")
            print(f"   - Use dynamic scaling with backoff")
        else:
            print(f"   - Model appears safe for FP16")
            print(f"   - Start with loss scale: 128")
        
        print(f"\n4. Gradient Clipping:")
        max_grad_estimate = max(s['std'] for s in self.stats.values()) * 10
        print(f"   - Recommended max grad norm: {min(1.0, max_grad_estimate):.2f}")
        
    def save_report(self, output_dir="."):
        """Save detailed analysis report."""
        output_path = Path(output_dir) / "checkpoint_analysis.json"
        
        report = {
            'checkpoint': str(self.checkpoint_path),
            'summary': {
                'total_layers': len(self.stats),
                'total_params': sum(s['numel'] for s in self.stats.values()),
                'problematic_layers': len(self.problematic_layers),
                'layer_types': {k: len(v) for k, v in self.layer_types.items()}
            },
            'problematic_layers': self.problematic_layers,
            'detailed_stats': self.stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_path}")
        

def main():
    checkpoint_path = "glide_model_cache/glide50k.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    analyzer = CheckpointAnalyzer(checkpoint_path)
    analyzer.analyze()
    analyzer.save_report(".claude/scripts")
    

if __name__ == "__main__":
    main()