#!/usr/bin/env python3
"""
Test script for verifying all sampling methods work correctly.

This script loads a GLIDE model and generates samples using each available
sampler, allowing for visual comparison and performance benchmarking.
"""

import argparse
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from glide_finetune.glide_util import load_model, sample


def test_sampler(
    model,
    options,
    sampler_name: str,
    prompt: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
    guidance_scale: float = 3.0,
    steps: str = "50",
    eta: float = 0.0,
    dpm_order: int = 2,
):
    """Test a single sampler and return the generated image and timing."""
    print(f"\nTesting {sampler_name} sampler...")
    
    start_time = time.time()
    
    samples = sample(
        glide_model=model,
        glide_options=options,
        side_x=64,
        side_y=64,
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        device=device,
        prediction_respacing=steps,
        sampler=sampler_name,
        sampler_eta=eta,
        dpm_order=dpm_order,
    )
    
    elapsed_time = time.time() - start_time
    print(f"{sampler_name} completed in {elapsed_time:.2f} seconds")
    
    # Convert to numpy for visualization
    samples_np = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    samples_np = samples_np.permute(0, 2, 3, 1).cpu().numpy()
    
    return samples_np, elapsed_time


def main():
    parser = argparse.ArgumentParser(description="Test all GLIDE samplers")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a painting of a cat",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (empty for default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="50",
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampler_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Eta parameter for stochastic samplers",
    )
    parser.add_argument(
        "--dpm-order",
        type=int,
        default=2,
        help="Order for DPM++ solver (1 or 2)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model...")
    model, diffusion, options = load_model(
        glide_path=args.checkpoint,
        use_fp16=False,
        model_type="base",
    )
    model.to(args.device)
    model.eval()
    
    # Test each sampler
    samplers = ["plms", "ddim", "euler", "euler_a", "dpm++"]
    results = {}
    
    for sampler in samplers:
        # Use appropriate eta for each sampler
        if sampler == "euler_a":
            eta = args.eta  # Stochastic
        elif sampler == "ddim":
            eta = args.eta  # Can be stochastic
        else:
            eta = 0.0  # Deterministic
            
        samples, timing = test_sampler(
            model=model,
            options=options,
            sampler_name=sampler,
            prompt=args.prompt,
            device=args.device,
            batch_size=args.batch_size,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            eta=eta,
            dpm_order=args.dpm_order if sampler == "dpm++" else 2,
        )
        
        results[sampler] = {
            "samples": samples,
            "time": timing,
        }
        
        # Save individual results
        for i, sample in enumerate(samples):
            plt.imsave(
                output_dir / f"{sampler}_sample_{i}.png",
                sample,
            )
    
    # Create comparison figure
    fig, axes = plt.subplots(1, len(samplers), figsize=(20, 4))
    fig.suptitle(f'Sampler Comparison: "{args.prompt}"', fontsize=16)
    
    for idx, (sampler, ax) in enumerate(zip(samplers, axes)):
        ax.imshow(results[sampler]["samples"][0])
        ax.set_title(f"{sampler}\n{results[sampler]['time']:.1f}s")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "sampler_comparison.png", dpi=150)
    print(f"\nResults saved to {output_dir}")
    
    # Print timing summary
    print("\nTiming Summary:")
    print("-" * 40)
    for sampler in samplers:
        print(f"{sampler:10s}: {results[sampler]['time']:6.2f} seconds")
    
    # Calculate relative speeds
    plms_time = results["plms"]["time"]
    print("\nRelative Speed (vs PLMS):")
    print("-" * 40)
    for sampler in samplers:
        speedup = plms_time / results[sampler]["time"]
        print(f"{sampler:10s}: {speedup:5.2f}x")


if __name__ == "__main__":
    main()