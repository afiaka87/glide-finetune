#!/usr/bin/env python
"""
Simple test to verify BF16 works with the model.
"""

import torch as th
from glide_finetune.glide_util import load_model

def test_bf16():
    print("Testing BF16 support...")
    
    # Check hardware support
    if th.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"✓ BF16 supported: {th.cuda.is_bf16_supported()}")
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Load model with BF16
    print("\nLoading model with BF16...")
    model, diffusion, options = load_model(precision="bf16")
    model = model.to(device)
    
    # Check dtypes
    bf16_params = sum(1 for p in model.parameters() if p.dtype == th.bfloat16)
    fp32_params = sum(1 for p in model.parameters() if p.dtype == th.float32)
    
    print(f"✓ Model loaded: {bf16_params} BF16 params, {fp32_params} FP32 params")
    
    # Test forward pass with autocast
    print("\nTesting forward pass with autocast...")
    batch_size = 2
    
    with th.cuda.amp.autocast(dtype=th.bfloat16):
        # Create dummy inputs
        x = th.randn(batch_size, 3, 64, 64, device=device)
        timesteps = th.randint(0, 1000, (batch_size,), device=device)
        tokens = th.randint(0, 1000, (batch_size, 128), device=device)
        mask = th.ones((batch_size, 128), dtype=th.bool, device=device)
        
        # Forward pass
        try:
            output = model(x, timesteps, tokens=tokens, mask=mask)
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
    
    # Test backward pass
    print("\nTesting backward pass...")
    optimizer = th.optim.Adam(model.parameters(), lr=1e-5)
    
    with th.cuda.amp.autocast(dtype=th.bfloat16):
        x = th.randn(batch_size, 3, 64, 64, device=device)
        timesteps = th.randint(0, 1000, (batch_size,), device=device)
        tokens = th.randint(0, 1000, (batch_size, 128), device=device)
        mask = th.ones((batch_size, 128), dtype=th.bool, device=device)
        
        output = model(x, timesteps, tokens=tokens, mask=mask)
        loss = output.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Backward pass successful")
    print(f"  Loss: {loss.item():.6f}")
    
    print("\n" + "="*50)
    print("✅ BF16 test completed successfully!")
    print("BF16 training is supported and working.")
    print("="*50)
    return True

if __name__ == "__main__":
    test_bf16()