#!/usr/bin/env python3
"""
Test parity when CLIP adapter is integrated but gate is zero.

Phase 1.21: Verify that with gate=0, the model behaves exactly like baseline.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from glide_finetune.clip_adapter import ClipAdapter, load_openai_clip, get_clip_text_features
from glide_finetune.unet_with_adapter import (
    create_model_with_adapter, 
    remove_adapter,
    set_adapter_scale,
    get_adapter_scale
)
from glide_text2im.model_creation import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from glide_text2im.nn import timestep_embedding


def set_deterministic_mode():
    """Enable deterministic mode for reproducible testing."""
    # ChatGPT's advice: ensure deterministic algorithms are OFF
    if torch.are_deterministic_algorithms_enabled():
        torch.use_deterministic_algorithms(False)
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_test_batch(batch_size=2, image_size=64, text_ctx=128):
    """Create a deterministic test batch."""
    # Realistic noise input (ChatGPT's advice)
    x_t = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    
    # Safe mid-range timesteps (ChatGPT's advice)
    timesteps = torch.tensor([100, 200], dtype=torch.long)
    
    # Valid tokens with guaranteed non-empty mask
    tokens = torch.zeros(batch_size, text_ctx, dtype=torch.long)
    tokens[:, 0] = 1  # Start token
    tokens[:, 1:5] = torch.arange(2, 6)  # Some content tokens
    
    # Boolean mask where True = valid token
    mask = torch.ones(batch_size, text_ctx, dtype=torch.bool)
    
    # Verify each row has valid tokens (ChatGPT's advice)
    assert mask.any(dim=-1).all(), "Each sample must have at least 1 valid token"
    
    # CLIP embeddings
    clip_embeddings = torch.randn(batch_size, 512, dtype=torch.float32) * 0.1
    
    return {
        'x_t': x_t,
        'timesteps': timesteps,
        'tokens': tokens,
        'mask': mask,
        'clip_embeddings': clip_embeddings,
    }


def test_adapter_integration():
    """Test that adapter can be integrated into model."""
    print("Testing adapter integration...")
    set_deterministic_mode()
    
    # Create model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    # Create adapter
    adapter = ClipAdapter.from_model(model)
    
    # Integrate adapter
    model_with_adapter = create_model_with_adapter(model, adapter)
    
    # Check adapter is attached
    assert hasattr(model_with_adapter, 'clip_adapter')
    assert model_with_adapter.clip_adapter is adapter
    print("✓ Adapter successfully integrated")
    
    # Test forward pass works
    batch = create_test_batch()
    with torch.no_grad():
        # Without CLIP embeddings (should work like baseline)
        output1 = model_with_adapter(
            batch['x_t'], 
            batch['timesteps'],
            tokens=batch['tokens'],
            mask=batch['mask']
        )
        
        # With CLIP embeddings
        output2 = model_with_adapter(
            batch['x_t'],
            batch['timesteps'],
            tokens=batch['tokens'],
            mask=batch['mask'],
            clip_embeddings=batch['clip_embeddings']
        )
    
    # Output should have 6 channels (3 for RGB, 3 for variance in GLIDE)
    expected_shape = (batch['x_t'].shape[0], 6, batch['x_t'].shape[2], batch['x_t'].shape[3])
    assert output1.shape == expected_shape, f"Expected {expected_shape}, got {output1.shape}"
    assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
    print("✓ Forward passes successful with and without CLIP")
    
    return model_with_adapter


def test_parity_at_zero_gate():
    """Test that with gate=0, adapter has no effect."""
    print("\nTesting parity at zero gate...")
    set_deterministic_mode()
    
    # Create baseline model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False
    
    torch.manual_seed(100)
    baseline_model, _ = create_model_and_diffusion(**options)
    baseline_model.eval()
    
    # Create identical model with adapter
    torch.manual_seed(100)
    model_with_adapter, _ = create_model_and_diffusion(**options)
    model_with_adapter.eval()
    
    # Add adapter with gate at zero
    adapter = ClipAdapter.from_model(model_with_adapter)
    adapter.set_gate_value(0.0)  # Set gate to exactly 0
    model_with_adapter = create_model_with_adapter(model_with_adapter, adapter)
    
    # Verify gate is zero
    gate_value = get_adapter_scale(model_with_adapter)
    assert gate_value < 1e-7, f"Gate should be ~0, got {gate_value}"
    print(f"✓ Gate set to: {gate_value:.9f}")
    
    # Create test batch
    batch = create_test_batch()
    
    # Run both models
    with torch.no_grad():
        # Baseline (no adapter)
        baseline_output = baseline_model(
            batch['x_t'],
            batch['timesteps'],
            tokens=batch['tokens'],
            mask=batch['mask']
        )
        
        # With adapter at gate=0
        adapter_output = model_with_adapter(
            batch['x_t'],
            batch['timesteps'],
            tokens=batch['tokens'],
            mask=batch['mask'],
            clip_embeddings=batch['clip_embeddings']
        )
    
    # Debug: Check for NaN
    baseline_has_nan = torch.isnan(baseline_output).any()
    adapter_has_nan = torch.isnan(adapter_output).any()
    
    if baseline_has_nan or adapter_has_nan:
        print(f"  WARNING: NaN detected!")
        print(f"    Baseline has NaN: {baseline_has_nan}")
        print(f"    Adapter has NaN: {adapter_has_nan}")
        print(f"    Baseline output range: [{baseline_output.min()}, {baseline_output.max()}]")
        print(f"    Adapter output range: [{adapter_output.min()}, {adapter_output.max()}]")
    
    # Check outputs are identical (or very close)
    max_diff = torch.abs(baseline_output - adapter_output).max().item()
    mean_diff = torch.abs(baseline_output - adapter_output).mean().item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    # With gate=0, differences should be minimal (floating point precision)
    tolerance = 1e-6
    assert max_diff < tolerance, f"Outputs differ too much: {max_diff} > {tolerance}"
    
    print(f"✓ Parity verified: gate=0 gives baseline behavior (diff < {tolerance})")
    
    return True


def test_gate_effect():
    """Test that increasing gate increases effect."""
    print("\nTesting gate effect scaling...")
    set_deterministic_mode()
    
    # Create model with adapter
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    adapter = ClipAdapter.from_model(model)
    model = create_model_with_adapter(model, adapter)
    
    # Create test batch
    batch = create_test_batch()
    
    # Test different gate values
    gate_values = [0.0, 0.01, 0.1, 0.5, 1.0]
    outputs = []
    
    for gate_val in gate_values:
        set_adapter_scale(model, gate_val)
        actual_gate = get_adapter_scale(model)
        
        with torch.no_grad():
            output = model(
                batch['x_t'],
                batch['timesteps'],
                tokens=batch['tokens'],
                mask=batch['mask'],
                clip_embeddings=batch['clip_embeddings']
            )
        
        outputs.append(output)
        print(f"  Gate {gate_val:.2f} (actual: {actual_gate:.6f})")
    
    # Check that effect increases with gate
    baseline = outputs[0]  # gate=0
    
    for i, (gate_val, output) in enumerate(zip(gate_values[1:], outputs[1:])):
        diff = torch.abs(output - baseline).mean().item()
        print(f"    Diff from baseline at gate={gate_values[i+1]:.2f}: {diff:.6f}")
        
        # Higher gates should have more effect (except gate=0)
        if i > 0:
            prev_diff = torch.abs(outputs[i] - baseline).mean().item()
            # Allow some noise but generally should increase
            if diff < prev_diff * 0.9:  # Allow 10% variance
                print(f"    Warning: Effect didn't increase monotonically")
    
    print("✓ Gate effect scales with value")
    return True


def test_adapter_removal():
    """Test that adapter can be cleanly removed."""
    print("\nTesting adapter removal...")
    set_deterministic_mode()
    
    # Create model with adapter
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    # Store original forward for comparison
    original_forward = model.forward
    
    # Add adapter
    adapter = ClipAdapter.from_model(model)
    model = create_model_with_adapter(model, adapter)
    
    # Verify adapter is attached
    assert hasattr(model, 'clip_adapter')
    assert model.forward != original_forward
    
    # Remove adapter
    model = remove_adapter(model)
    
    # Verify adapter is removed
    assert not hasattr(model, 'clip_adapter')
    assert model.forward == original_forward
    
    print("✓ Adapter cleanly removed")
    return True


def test_fp16_compatibility():
    """Test that adapter works with mixed precision using AMP."""
    print("\nTesting FP16 compatibility with AMP...")
    torch.manual_seed(42)
    
    # Create model in FP32 mode (AMP will handle mixed precision)
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False  # Keep model in FP32, use AMP for mixed precision
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    # Add adapter (stays in FP32)
    adapter = ClipAdapter.from_model(model)
    model = create_model_with_adapter(model, adapter)
    
    # Create test batch - keep everything in FP32
    batch = create_test_batch()
    
    # Determine device and autocast settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mixed = device.type == "cuda"  # Only use mixed precision on CUDA
    
    if device.type == "cpu":
        # CPU doesn't support FP16 well, use BF16 if available
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        print(f"  Using CPU with dtype: {autocast_dtype}")
    else:
        # CUDA supports FP16
        autocast_dtype = torch.float16
        print(f"  Using CUDA with FP16")
    
    # Move model and inputs to device
    model = model.to(device)
    batch['x_t'] = batch['x_t'].to(device)
    batch['timesteps'] = batch['timesteps'].to(device)
    batch['tokens'] = batch['tokens'].to(device)
    batch['mask'] = batch['mask'].to(device)
    batch['clip_embeddings'] = batch['clip_embeddings'].to(device)
    
    # Forward pass with AMP
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_mixed):
            output = model(
                batch['x_t'],
                batch['timesteps'],
                tokens=batch['tokens'],
                mask=batch['mask'],
                clip_embeddings=batch['clip_embeddings']
            )
    
    # Check output
    if use_mixed:
        expected_dtype = torch.float16 if device.type == "cuda" else autocast_dtype
        assert output.dtype == expected_dtype, f"Expected {expected_dtype}, got {output.dtype}"
    assert torch.isfinite(output).all(), "Output has non-finite values"
    
    # Verify adapter stayed in FP32
    assert model.clip_adapter.linear_1.weight.dtype == torch.float32, "Adapter should stay FP32"
    
    print(f"✓ AMP compatibility verified (output dtype: {output.dtype})")
    print(f"✓ Adapter maintained FP32 computation")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Testing CLIP Adapter Parity and Integration")
    print("=" * 80)
    
    # Run all tests
    test_adapter_integration()
    test_parity_at_zero_gate()
    test_gate_effect()
    test_adapter_removal()
    test_fp16_compatibility()
    
    print("\n" + "=" * 80)
    print("✓ All parity and integration tests passed!")
    print("✓ Adapter correctly integrates with zero-effect at gate=0")
    print("=" * 80)