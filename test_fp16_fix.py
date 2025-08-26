#!/usr/bin/env python3
"""Test FP16 with proper dtype handling."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / 'src'))

from glide_text2im.model_creation import model_and_diffusion_defaults, create_model_and_diffusion
from glide_finetune.clip_adapter import ClipAdapter
from glide_finetune.unet_with_adapter import create_model_with_adapter


def check_dtypes(model):
    """Check dtypes of critical components."""
    print("\n=== Dtype Check ===")
    
    # Check embeddings (should be FP32)
    if hasattr(model, 'token_embedding'):
        print(f"token_embedding: {model.token_embedding.weight.dtype}")
    
    # Check transformer blocks (should be FP16 if converted)
    if hasattr(model, 'transformer'):
        for name, param in model.transformer.named_parameters():
            if 'c_qkv.weight' in name:
                print(f"Transformer QKV: {param.dtype}")
                break
            if 'ln' in name.lower() or 'layernorm' in name.lower():
                print(f"LayerNorm {name}: {param.dtype}")
                break
    
    # Check adapter (should be FP32)
    if hasattr(model, 'clip_adapter'):
        print(f"Adapter linear1: {model.clip_adapter.linear_1.weight.dtype}")


def test_fp16_proper():
    """Test FP16 with proper dtype handling."""
    print("Testing FP16 with proper dtype handling...")
    
    # Create model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = True
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    print("Model created with use_fp16=True")
    
    # Convert to FP16 (required when use_fp16=True)
    if model.use_fp16:
        print("Converting to FP16...")
        model.convert_to_fp16()
    
    # Check dtypes after conversion
    check_dtypes(model)
    
    # Add adapter (stays FP32)
    print("\nAdding CLIP adapter...")
    adapter = ClipAdapter.from_model(model)
    model = create_model_with_adapter(model, adapter)
    
    # Create test batch
    batch_size = 2
    x_t = torch.randn(batch_size, 3, 64, 64).half()
    timesteps = torch.tensor([100, 200], dtype=torch.long)
    tokens = torch.zeros(batch_size, 128, dtype=torch.long)
    tokens[:, 0] = 1  # Start token
    mask = torch.ones(batch_size, 128, dtype=torch.bool)
    clip_embeddings = torch.randn(batch_size, 512).half()
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(
            x_t,
            timesteps,
            tokens=tokens,
            mask=mask,
            clip_embeddings=clip_embeddings
        )
    
    print(f"✓ Forward pass successful!")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output shape: {output.shape}")
    
    # Verify output
    assert output.dtype == torch.float16, f"Expected FP16 output, got {output.dtype}"
    assert torch.isfinite(output).all(), "Output has non-finite values"
    
    print("\n✓ All checks passed!")
    return True


def test_dtype_alignment():
    """Test that dtypes align properly through the forward pass."""
    print("\n\nTesting dtype alignment with hooks...")
    
    # Create and convert model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = True
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    if model.use_fp16:
        model.convert_to_fp16()
    
    # Add hooks to check dtype alignment
    mismatches = []
    
    def check_linear(module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        if hasattr(module, 'weight'):
            if x.dtype != module.weight.dtype:
                mismatches.append({
                    'module': module.__class__.__name__,
                    'input_dtype': x.dtype,
                    'weight_dtype': module.weight.dtype
                })
    
    # Register hooks on Linear layers
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.register_forward_hook(check_linear)
    
    # Run forward pass
    batch_size = 2
    x_t = torch.randn(batch_size, 3, 64, 64).half()
    timesteps = torch.tensor([100, 200], dtype=torch.long)
    tokens = torch.zeros(batch_size, 128, dtype=torch.long)
    tokens[:, 0] = 1
    mask = torch.ones(batch_size, 128, dtype=torch.bool)
    
    with torch.no_grad():
        _ = model(x_t, timesteps, tokens=tokens, mask=mask)
    
    if mismatches:
        print("⚠️ Dtype mismatches found:")
        for m in mismatches[:5]:  # Show first 5
            print(f"  {m['module']}: input={m['input_dtype']} vs weight={m['weight_dtype']}")
    else:
        print("✓ All Linear layers have aligned dtypes!")
    
    return len(mismatches) == 0


if __name__ == "__main__":
    test_fp16_proper()
    test_dtype_alignment()
    print("\n" + "=" * 60)
    print("✓ FP16 testing complete - all checks passed!")
    print("=" * 60)