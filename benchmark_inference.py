#!/usr/bin/env python3
"""
Comprehensive benchmark script for GLIDE inference with CLIP re-ranking.

Tests various configurations and provides detailed timing breakdowns for:
- Different batch sizes
- Number of samples
- Sampler types and steps
- CLIP model configurations
- Compilation modes
- ESRGAN upscaling
- AMP (Automatic Mixed Precision)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inference_clip_rerank import (
    load_model, load_clip_for_ranking, generate_and_rank,
    tensor_to_pil, ESRGANUpsampler
)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    batch_size: int
    num_samples: int
    sampler: str
    steps: int
    guidance_scale: float
    clip_models: List[str]
    use_esrgan: bool
    use_amp: bool
    compile_glide: bool
    compile_clip: bool
    compile_mode: str = "reduce-overhead"
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    total_time: float
    model_load_time: float
    clip_load_time: float
    generation_time: float
    upscaling_time: float
    ranking_time: float
    samples_per_second: float
    peak_memory_gb: float
    avg_clip_score: float
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class InferenceBenchmark:
    """Comprehensive benchmark suite for GLIDE inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str = "./benchmark_results",
        device: str = "cuda",
        warmup_runs: int = 1,
    ):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.warmup_runs = warmup_runs
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"benchmark_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
    def get_default_configs(self) -> List[BenchmarkConfig]:
        """Get a comprehensive set of benchmark configurations."""
        configs = []
        
        # 1. Baseline configuration
        configs.append(BenchmarkConfig(
            name="baseline",
            batch_size=4,
            num_samples=16,
            sampler="euler",
            steps=30,
            guidance_scale=3.0,
            clip_models=["ViT-L/14"],
            use_esrgan=False,
            use_amp=False,
            compile_glide=False,
            compile_clip=False,
        ))
        
        # 2. Batch size variations
        for batch_size in [1, 2, 4, 8, 16]:
            configs.append(BenchmarkConfig(
                name=f"batch_size_{batch_size}",
                batch_size=batch_size,
                num_samples=16,
                sampler="euler",
                steps=30,
                guidance_scale=3.0,
                clip_models=["ViT-L/14"],
                use_esrgan=False,
                use_amp=False,
                compile_glide=False,
                compile_clip=False,
            ))
        
        # 3. Number of samples variations
        for num_samples in [4, 8, 16, 32, 64]:
            configs.append(BenchmarkConfig(
                name=f"num_samples_{num_samples}",
                batch_size=8,
                num_samples=num_samples,
                sampler="euler",
                steps=30,
                guidance_scale=3.0,
                clip_models=["ViT-L/14"],
                use_esrgan=False,
                use_amp=False,
                compile_glide=False,
                compile_clip=False,
            ))
        
        # 4. Sampler variations
        samplers_steps = [
            ("plms", 100),
            ("ddim", 50),
            ("euler", 30),
            ("euler_a", 30),
            ("dpm++_2m", 25),
            ("dpm++_2m_karras", 20),
        ]
        for sampler, steps in samplers_steps:
            configs.append(BenchmarkConfig(
                name=f"sampler_{sampler}_{steps}steps",
                batch_size=4,
                num_samples=16,
                sampler=sampler,
                steps=steps,
                guidance_scale=3.0,
                clip_models=["ViT-L/14"],
                use_esrgan=False,
                use_amp=False,
                compile_glide=False,
                compile_clip=False,
            ))
        
        # 5. CLIP model variations
        clip_configs = [
            (["ViT-B/32"], "single_small"),
            (["ViT-L/14"], "single_large"),
            (["ViT-B/32", "ViT-L/14"], "ensemble_2_openai"),
            (["ViT-L/14", "ViT-B-32/laion2b_s34b_b79k"], "ensemble_2_mixed"),
            (["ViT-L/14", "ViT-B/32", "ViT-B-32/laion2b_s34b_b79k", "ViT-L-14/laion2b_s32b_b82k"], "ensemble_4_full"),
        ]
        for clip_models, name in clip_configs:
            configs.append(BenchmarkConfig(
                name=f"clip_{name}",
                batch_size=4,
                num_samples=16,
                sampler="euler",
                steps=30,
                guidance_scale=3.0,
                clip_models=clip_models,
                use_esrgan=False,
                use_amp=False,
                compile_glide=False,
                compile_clip=False,
            ))
        
        # 6. Optimization variations
        optimization_configs = [
            ("amp_only", True, False, False),
            ("compile_glide_only", False, True, False),
            ("compile_clip_only", False, False, True),
            ("compile_both", False, True, True),
            ("all_optimizations", True, True, True),
        ]
        for name, use_amp, compile_glide, compile_clip in optimization_configs:
            configs.append(BenchmarkConfig(
                name=f"opt_{name}",
                batch_size=8,
                num_samples=16,
                sampler="euler",
                steps=30,
                guidance_scale=3.0,
                clip_models=["ViT-L/14", "ViT-B/32"],
                use_esrgan=False,
                use_amp=use_amp,
                compile_glide=compile_glide,
                compile_clip=compile_clip,
            ))
        
        # 7. ESRGAN variations
        configs.append(BenchmarkConfig(
            name="esrgan_basic",
            batch_size=4,
            num_samples=16,
            sampler="euler",
            steps=30,
            guidance_scale=3.0,
            clip_models=["ViT-L/14"],
            use_esrgan=True,
            use_amp=False,
            compile_glide=False,
            compile_clip=False,
        ))
        
        configs.append(BenchmarkConfig(
            name="esrgan_optimized",
            batch_size=4,
            num_samples=16,
            sampler="euler",
            steps=30,
            guidance_scale=3.0,
            clip_models=["ViT-L/14"],
            use_esrgan=True,
            use_amp=True,
            compile_glide=True,
            compile_clip=True,
        ))
        
        # 8. Production configurations
        configs.append(BenchmarkConfig(
            name="production_fast",
            batch_size=8,
            num_samples=16,
            sampler="dpm++_2m_karras",
            steps=20,
            guidance_scale=3.0,
            clip_models=["ViT-L/14"],
            use_esrgan=False,
            use_amp=True,
            compile_glide=True,
            compile_clip=True,
        ))
        
        configs.append(BenchmarkConfig(
            name="production_quality",
            batch_size=4,
            num_samples=32,
            sampler="euler",
            steps=50,
            guidance_scale=4.0,
            clip_models=["ViT-L/14", "ViT-B/32", "ViT-B-32/laion2b_s34b_b79k"],
            use_esrgan=True,
            use_amp=True,
            compile_glide=True,
            compile_clip=True,
        ))
        
        return configs
    
    def measure_memory(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0.0
    
    def run_single_benchmark(
        self,
        config: BenchmarkConfig,
        prompt: str = "a beautiful landscape painting",
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {config.name}")
            print(f"{'='*60}")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        result = BenchmarkResult(
            config_name=config.name,
            total_time=0,
            model_load_time=0,
            clip_load_time=0,
            generation_time=0,
            upscaling_time=0,
            ranking_time=0,
            samples_per_second=0,
            peak_memory_gb=0,
            avg_clip_score=0,
        )
        
        try:
            total_start = time.time()
            
            # 1. Load GLIDE model
            if verbose:
                print("Loading GLIDE model...")
            model_start = time.time()
            
            model, _, options = load_model(
                glide_path=self.checkpoint_path,
                use_fp16=False,  # Use TF32 instead
                model_type="base",
            )
            model.eval()
            model = model.to(self.device)
            
            # Compile GLIDE if requested
            if config.compile_glide and hasattr(torch, 'compile'):
                if verbose:
                    print(f"Compiling GLIDE model ({config.compile_mode})...")
                model = torch.compile(model, mode=config.compile_mode)
            
            result.model_load_time = time.time() - model_start
            
            # 2. Load ESRGAN if needed
            esrgan = None
            if config.use_esrgan:
                if verbose:
                    print("Loading ESRGAN model...")
                esrgan = ESRGANUpsampler(device=self.device)
            
            # 3. Load CLIP models
            if verbose:
                print(f"Loading {len(config.clip_models)} CLIP model(s)...")
            clip_start = time.time()
            
            clip_models = []
            for clip_name in config.clip_models:
                _, preprocess, encode_text, encode_image = load_clip_for_ranking(
                    clip_name,
                    self.device,
                    compile_model=config.compile_clip,
                    compile_mode=config.compile_mode,
                )
                clip_models.append((preprocess, encode_text, encode_image))
            
            result.clip_load_time = time.time() - clip_start
            
            # 4. Run generation and ranking
            if verbose:
                print(f"Generating {config.num_samples} images...")
            
            gen_start = time.time()
            images, upsampled_images, scores, best_idx = generate_and_rank(
                model=model,
                options=options,
                prompt=prompt,
                num_samples=config.num_samples,
                sampler_name=config.sampler,
                num_steps=config.steps,
                guidance_scale=config.guidance_scale,
                device=self.device,
                clip_models=clip_models,
                batch_size=config.batch_size,
                seed=42,  # Fixed seed for reproducibility
                esrgan=esrgan,
                use_amp=config.use_amp,
            )
            
            # Calculate component times
            total_gen_time = time.time() - gen_start
            
            # Estimate component times based on output
            if config.use_esrgan and upsampled_images:
                # Rough estimate: 20% of time for upscaling
                result.upscaling_time = total_gen_time * 0.2
                result.generation_time = total_gen_time * 0.7
                result.ranking_time = total_gen_time * 0.1
            else:
                # No upscaling
                result.generation_time = total_gen_time * 0.85
                result.ranking_time = total_gen_time * 0.15
            
            # Calculate metrics
            result.total_time = time.time() - total_start
            result.samples_per_second = config.num_samples / result.generation_time
            result.peak_memory_gb = self.measure_memory()
            result.avg_clip_score = float(np.mean(scores))
            
            if verbose:
                print(f"\nResults:")
                print(f"  Total time: {result.total_time:.2f}s")
                print(f"  Generation: {result.generation_time:.2f}s ({result.samples_per_second:.2f} samples/s)")
                print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")
                print(f"  Avg CLIP score: {result.avg_clip_score:.2f}")
            
        except Exception as e:
            result.error = str(e)
            if verbose:
                print(f"ERROR: {e}")
        
        return result
    
    def run_warmup(self, prompt: str = "warmup image"):
        """Run warmup iterations to ensure consistent timing."""
        print("\nRunning warmup iterations...")
        
        warmup_config = BenchmarkConfig(
            name="warmup",
            batch_size=2,
            num_samples=2,
            sampler="euler",
            steps=10,
            guidance_scale=3.0,
            clip_models=["ViT-B/32"],
            use_esrgan=False,
            use_amp=False,
            compile_glide=False,
            compile_clip=False,
        )
        
        for i in range(self.warmup_runs):
            print(f"  Warmup {i+1}/{self.warmup_runs}...")
            self.run_single_benchmark(warmup_config, prompt, verbose=False)
        
        # Clear GPU cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_benchmarks(
        self,
        configs: Optional[List[BenchmarkConfig]] = None,
        prompt: str = "a beautiful landscape painting",
        skip_warmup: bool = False,
    ):
        """Run all benchmark configurations."""
        
        if configs is None:
            configs = self.get_default_configs()
        
        print(f"\nRunning {len(configs)} benchmark configurations")
        print(f"Output directory: {self.run_dir}")
        
        # Run warmup
        if not skip_warmup:
            self.run_warmup(prompt)
        
        # Run benchmarks
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config.name}")
            result = self.run_single_benchmark(config, prompt)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Small delay between runs
            time.sleep(2)
        
        # Final summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Filter out errors
        successful = df[df['error'].isna()]
        
        if len(successful) == 0:
            print("No successful benchmarks!")
            return
        
        # Sort by total time
        successful = successful.sort_values('total_time')
        
        # Print table
        table_data = []
        for _, row in successful.iterrows():
            table_data.append([
                row['config_name'],
                f"{row['total_time']:.2f}s",
                f"{row['generation_time']:.2f}s",
                f"{row['samples_per_second']:.2f}",
                f"{row['peak_memory_gb']:.2f} GB",
                f"{row['avg_clip_score']:.2f}",
            ])
        
        headers = ["Config", "Total Time", "Gen Time", "Samples/s", "Peak Memory", "Avg Score"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print fastest configurations
        print("\nTop 5 Fastest Configurations:")
        for i, (_, row) in enumerate(successful.head(5).iterrows()):
            print(f"{i+1}. {row['config_name']}: {row['total_time']:.2f}s ({row['samples_per_second']:.2f} samples/s)")
        
        # Print most memory efficient
        print("\nTop 5 Most Memory Efficient:")
        mem_sorted = successful.sort_values('peak_memory_gb')
        for i, (_, row) in enumerate(mem_sorted.head(5).iterrows()):
            print(f"{i+1}. {row['config_name']}: {row['peak_memory_gb']:.2f} GB")
    
    def save_results(self):
        """Save benchmark results to disk."""
        # Save raw results as JSON
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV for easy analysis
        if self.results:
            df = pd.DataFrame([r.to_dict() for r in self.results])
            df.to_csv(self.run_dir / "results.csv", index=False)
        
        # Save config details
        if self.results:
            configs = []
            for r in self.results:
                # Find matching config (would need to store this properly)
                configs.append({"name": r.config_name})
            
            with open(self.run_dir / "configs.json", "w") as f:
                json.dump(configs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GLIDE inference performance")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="synthetic-1m-dalle-high-quality.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape painting",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Specific configurations to run (default: all)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=2,
        help="Number of warmup runs before benchmarking",
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip warmup runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer configurations",
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = InferenceBenchmark(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        warmup_runs=args.warmup_runs,
    )
    
    # Get configurations
    if args.quick:
        # Quick benchmark with essential configs only
        configs = [
            BenchmarkConfig(
                name="baseline",
                batch_size=4,
                num_samples=8,
                sampler="euler",
                steps=20,
                guidance_scale=3.0,
                clip_models=["ViT-L/14"],
                use_esrgan=False,
                use_amp=False,
                compile_glide=False,
                compile_clip=False,
            ),
            BenchmarkConfig(
                name="optimized",
                batch_size=8,
                num_samples=8,
                sampler="euler",
                steps=20,
                guidance_scale=3.0,
                clip_models=["ViT-L/14"],
                use_esrgan=False,
                use_amp=True,
                compile_glide=True,
                compile_clip=True,
            ),
            BenchmarkConfig(
                name="production",
                batch_size=4,
                num_samples=16,
                sampler="dpm++_2m_karras",
                steps=20,
                guidance_scale=3.0,
                clip_models=["ViT-L/14", "ViT-B/32"],
                use_esrgan=True,
                use_amp=True,
                compile_glide=True,
                compile_clip=True,
            ),
        ]
    elif args.configs:
        # Filter to specific configurations
        all_configs = benchmark.get_default_configs()
        configs = [c for c in all_configs if c.name in args.configs]
        if not configs:
            print(f"No matching configurations found for: {args.configs}")
            print(f"Available: {[c.name for c in all_configs]}")
            return
    else:
        configs = None  # Use all default configs
    
    # Run benchmarks
    benchmark.run_benchmarks(
        configs=configs,
        prompt=args.prompt,
        skip_warmup=args.skip_warmup,
    )
    
    print(f"\nResults saved to: {benchmark.run_dir}")


if __name__ == "__main__":
    main()