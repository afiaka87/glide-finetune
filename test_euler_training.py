#!/usr/bin/env python3
"""
Test script to verify Euler sampling works during training.
"""

import torch
from glide_finetune.glide_util import load_model, sample
import time


def test_euler_training_sampling():
    """Test that Euler sampler works with training defaults."""
    
    print("Testing Euler sampler with training defaults (30 steps)...")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, diffusion, options = load_model(model_type="base")
    model.to(device)
    model.eval()
    
    # Test with default training parameters
    prompt = "a test image during training"
    
    # Time the sampling
    start_time = time.time()
    
    # Sample using Euler with 30 steps (new training defaults)
    samples = sample(
        glide_model=model,
        glide_options=options,
        side_x=64,
        side_y=64,
        prompt=prompt,
        batch_size=1,
        guidance_scale=4.0,  # Default training guidance scale
        device=device,
        prediction_respacing="30",  # New default
        sampler="euler",  # New default
    )
    
    elapsed = time.time() - start_time
    
    print(f"✓ Euler sampling completed successfully in {elapsed:.2f} seconds")
    print(f"  Shape: {samples.shape}")
    print(f"  Device: {samples.device}")
    print(f"  Range: [{samples.min():.2f}, {samples.max():.2f}]")
    
    # Compare with old default (PLMS with 100 steps)
    print("\nComparing with old default (PLMS, 100 steps)...")
    start_time = time.time()
    
    samples_plms = sample(
        glide_model=model,
        glide_options=options,
        side_x=64,
        side_y=64,
        prompt=prompt,
        batch_size=1,
        guidance_scale=4.0,
        device=device,
        prediction_respacing="100",  # Old default
        sampler="plms",  # Old default
    )
    
    elapsed_plms = time.time() - start_time
    
    print(f"✓ PLMS sampling completed in {elapsed_plms:.2f} seconds")
    
    # Calculate speedup
    speedup = elapsed_plms / elapsed
    print(f"\n📊 Performance Comparison:")
    print(f"  Euler (30 steps): {elapsed:.2f}s")
    print(f"  PLMS (100 steps): {elapsed_plms:.2f}s")
    print(f"  Speedup: {speedup:.1f}x faster")
    
    return True


if __name__ == "__main__":
    success = test_euler_training_sampling()
    if success:
        print("\n✅ Euler sampler is ready for training!")
    else:
        print("\n❌ Test failed")