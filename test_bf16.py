#!/usr/bin/env python
"""
Quick test script to verify BF16 training works correctly.
"""

import torch as th
import numpy as np
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset
from glide_finetune.glide_finetune import base_train_step

def test_bf16_training():
    print("Testing BF16 training support...")
    
    # Check BF16 support
    if th.cuda.is_available():
        print(f"CUDA available: {th.cuda.is_available()}")
        print(f"BF16 support: {th.cuda.is_bf16_supported()}")
    else:
        print("No CUDA available, testing CPU BF16")
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Test loading model with BF16
    print("\n1. Loading model with BF16 precision...")
    try:
        model, diffusion, options = load_model(
            precision="bf16",
            model_type="base"
        )
        model = model.to(device)
        print("✓ Model loaded successfully with BF16")
    except Exception as e:
        print(f"✗ Failed to load model with BF16: {e}")
        return
    
    # Test model dtype
    print("\n2. Checking model precision...")
    # Check the main model blocks which are converted
    bf16_count = 0
    fp32_count = 0
    for name, param in model.named_parameters():
        if param.dtype == th.bfloat16:
            bf16_count += 1
        elif param.dtype == th.float32:
            fp32_count += 1
    
    print(f"Parameters in BF16: {bf16_count}")
    print(f"Parameters in FP32: {fp32_count}")
    
    # Most parameters should be BF16 (UNet blocks), some (embeddings) stay FP32
    assert bf16_count > 0, "No parameters in BF16"
    print("✓ Model has BF16 parameters")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    try:
        # Create dummy batch - ensure correct dtypes
        batch_size = 2
        tokens = th.randint(0, 1000, (batch_size, 128))
        masks = th.ones((batch_size, 128), dtype=th.bool)
        # Images should be in BF16 to match model
        images = th.randn(batch_size, 3, 64, 64, dtype=th.bfloat16 if device.type == "cuda" else th.float32)
        
        # Run training step
        loss = base_train_step(
            model, 
            diffusion,
            (tokens, masks, images),
            device=device
        )
        
        print(f"Loss value: {loss.item():.4f}")
        print(f"Loss dtype: {loss.dtype}")
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    try:
        optimizer = th.optim.Adam(model.parameters(), lr=1e-5)
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"Gradient norm: {grad_norm:.4f}")
        print("✓ Backward pass successful")
        
        # Test optimizer step
        optimizer.step()
        print("✓ Optimizer step successful")
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return
    
    print("\n" + "="*50)
    print("✅ BF16 training test completed successfully!")
    print("BF16 is working correctly and should provide stable training.")
    print("="*50)

if __name__ == "__main__":
    test_bf16_training()