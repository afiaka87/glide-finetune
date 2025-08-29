#!/usr/bin/env python3
"""
Test script to verify multi-GPU training works with all training modes.

This script tests:
1. Standard GLIDE finetuning
2. CLIP adapter-only training  
3. Frozen transformer (train UNet only)
4. Frozen diffusion (train transformer only)
5. Randomized transformer
6. Randomized UNet/diffusion

Run with: uv run python test_multi_gpu_all_modes.py
"""

import subprocess
import tempfile
import sys
from pathlib import Path
import time
import json
import numpy as np
from PIL import Image


def create_test_dataset(data_dir: Path, num_samples: int = 8):
    """Create a minimal test dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create random image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(data_dir / f"img_{i}.jpg")
        
        # Create caption file
        with open(data_dir / f"img_{i}.txt", "w") as f:
            f.write(f"Test image number {i} for multi-GPU training")
    
    print(f"Created {num_samples} test samples in {data_dir}")


def run_training_test(
    mode_name: str,
    extra_args: list[str],
    data_dir: Path,
    output_dir: Path,
    num_gpus: int = 2,
    num_steps: int = 10,
) -> bool:
    """Run a training test with specified configuration.
    
    Args:
        mode_name: Name of the training mode for logging
        extra_args: Additional command-line arguments
        data_dir: Path to test dataset
        output_dir: Path for outputs
        num_gpus: Number of GPUs to use
        num_steps: Number of training steps
        
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing: {mode_name}")
    print(f"{'='*60}")
    
    # Build the command
    cmd = [
        "uv", "run", "accelerate", "launch",
        "--num_processes", str(num_gpus),
        "--mixed_precision", "no",
        "train.py",
        "--data_dir", str(data_dir),
        "--save_directory", str(output_dir / mode_name),
        "--batch_size", "2",
        "--learning_rate", "1e-4",
        "--num_epochs", "1",
        "--log_frequency", "2",
        "--save_frequency", "999999",  # Don't save during test
        "--sample_frequency", "999999",  # Don't sample during test
        "--seed", "42",
        "--max_steps", str(num_steps),  # Limit steps for quick test
    ]
    
    # Add mode-specific arguments
    cmd.extend(extra_args)
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Extra args: {extra_args}")
    
    try:
        # Run the training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            check=False,
        )
        
        # Check for success
        if result.returncode != 0:
            print(f"✗ {mode_name} failed with exit code {result.returncode}")
            if "CUDA out of memory" in result.stderr:
                print("  Note: Out of memory - try reducing batch size")
            else:
                print(f"  Error output (last 500 chars):\n{result.stderr[-500:]}")
            return False
        
        # Check for key indicators of success in output
        output = result.stdout + result.stderr
        
        # Check for successful training indicators
        success_indicators = [
            "step=",  # Training steps happening
            "loss=",  # Loss being computed
            "lr=",    # Learning rate being reported
        ]
        
        indicators_found = sum(1 for indicator in success_indicators if indicator in output)
        
        if indicators_found >= 2:
            print(f"✓ {mode_name} passed! Found {indicators_found}/3 success indicators")
            
            # Check for mode-specific success
            if "freeze_transformer" in extra_args:
                if "Freezing Transformer Components" in output or "Froze" in output:
                    print("  ✓ Transformer freezing confirmed")
            elif "freeze_diffusion" in extra_args:
                if "Freezing Diffusion Components" in output or "Froze" in output:
                    print("  ✓ Diffusion freezing confirmed")
            elif "randomize_transformer" in extra_args:
                if "Randomizing transformer" in output or "Randomized" in output:
                    print("  ✓ Transformer randomization confirmed")
            elif "randomize_diffusion" in extra_args:
                if "Randomizing diffusion" in output or "Randomized" in output:
                    print("  ✓ Diffusion randomization confirmed")
            elif "clip_adapter_only" in extra_args:
                if "Freezing base model" in output or "adapter" in output.lower():
                    print("  ✓ CLIP adapter mode confirmed")
            
            return True
        else:
            print(f"✗ {mode_name} unclear - only found {indicators_found}/3 success indicators")
            print(f"  Output sample (last 500 chars):\n{output[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {mode_name} timed out after 3 minutes")
        return False
    except Exception as e:
        print(f"✗ {mode_name} failed with exception: {e}")
        return False


def main():
    """Run all multi-GPU training mode tests."""
    print("="*60)
    print("Multi-GPU Training Mode Compatibility Test")
    print("="*60)
    
    # Check for GPUs
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
        )
        num_gpus = len(result.stdout.strip().split('\n'))
        print(f"Found {num_gpus} GPUs")
        
        if num_gpus < 2:
            print("WARNING: Less than 2 GPUs available. Tests may fail.")
            print("Continuing with available GPUs...")
            num_gpus = max(1, num_gpus)
    except Exception:
        print("WARNING: Could not detect GPUs. Assuming 2 for testing.")
        num_gpus = 2
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        data_dir = tmpdir_path / "data"
        output_dir = tmpdir_path / "outputs"
        
        # Create test dataset
        create_test_dataset(data_dir, num_samples=8)
        
        # Define test configurations
        test_configs = [
            # 1. Standard GLIDE finetuning
            ("standard_glide", ["--use_captions"]),
            
            # 2. CLIP adapter-only training
            ("clip_adapter_only", [
                "--use_clip_adapter",
                "--clip_adapter_only",
                "--use_captions",
            ]),
            
            # 3. Frozen transformer (train UNet only)
            ("frozen_transformer", [
                "--freeze_transformer",
                "--use_captions",
            ]),
            
            # 4. Frozen diffusion (train transformer only)  
            ("frozen_diffusion", [
                "--freeze_diffusion",
                "--use_captions",
            ]),
            
            # 5. Randomized transformer
            ("randomized_transformer", [
                "--randomize_transformer",
                "--randomize_init_std", "0.02",
                "--use_captions",
            ]),
            
            # 6. Randomized diffusion/UNet
            ("randomized_diffusion", [
                "--randomize_diffusion", 
                "--randomize_init_std", "0.02",
                "--use_captions",
            ]),
        ]
        
        # Run tests
        results = {}
        for mode_name, extra_args in test_configs:
            success = run_training_test(
                mode_name=mode_name,
                extra_args=extra_args,
                data_dir=data_dir,
                output_dir=output_dir,
                num_gpus=num_gpus,
                num_steps=10,  # Quick test with just 10 steps
            )
            results[mode_name] = success
            
            # Small delay between tests
            time.sleep(2)
        
        # Print summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for mode, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{mode:30s}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All multi-GPU training modes work correctly!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {total - passed} test(s) failed. Please review the errors above.")
            sys.exit(1)


if __name__ == "__main__":
    main()