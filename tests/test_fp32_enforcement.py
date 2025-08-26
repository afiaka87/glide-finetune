#!/usr/bin/env python3
"""
Test that ClipAdapter enforces FP32 computation throughout.

Phase 1.17: Verify adapter runs in FP32, only output casts to match input dtype.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from glide_finetune.clip_adapter import ClipAdapter, load_openai_clip, get_clip_text_features
from glide_text2im.model_creation import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


def test_fp32_enforcement_in_fp16_model():
    """Test that adapter stays in FP32 even when model is FP16."""
    print("Testing FP32 enforcement in mixed precision context...")
    
    # Create model and convert to FP16
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False  # Create in FP32 first
    model, _ = create_model_and_diffusion(**options)
    model.eval()
    
    # Convert model to FP16
    model.convert_to_fp16()
    
    # Verify model main blocks are FP16 (time_embed stays FP32 by design)
    # Check input blocks instead
    input_block_dtype = next(model.input_blocks.parameters()).dtype
    assert input_block_dtype == torch.float16, f"Model blocks should be FP16, got {input_block_dtype}"
    print(f"✓ Model main blocks are in FP16: {input_block_dtype}")
    print(f"  (Note: time_embed stays FP32 by design)")
    
    # Create adapter
    adapter = ClipAdapter.from_model(
        model=model,
        clip_model=None,
        first_linear_std=0.02,
        layer_norm_gamma=0.001,
        gate_init=-5.0,
    )
    
    # Check adapter internal layers are FP32
    assert adapter.layer_norm.weight.dtype == torch.float32, "LayerNorm should be FP32"
    assert adapter.linear_1.weight.dtype == torch.float32, "Linear1 should be FP32"
    assert adapter.linear_2.weight.dtype == torch.float32, "Linear2 should be FP32"
    assert adapter.gate.dtype == torch.float32, "Gate should be FP32"
    
    print("✓ All adapter layers are FP32:")
    print(f"  - LayerNorm: {adapter.layer_norm.weight.dtype}")
    print(f"  - Linear1: {adapter.linear_1.weight.dtype}")
    print(f"  - Linear2: {adapter.linear_2.weight.dtype}")
    print(f"  - Gate: {adapter.gate.dtype}")
    
    return adapter, model


def test_fp32_computation_with_fp16_input():
    """Test that adapter handles FP16 input correctly."""
    print("\nTesting FP32 computation with FP16 input...")
    
    adapter, model = test_fp32_enforcement_in_fp16_model()
    
    # Create FP16 input (as would come from CLIP in mixed precision)
    batch_size = 2
    clip_emb_fp16 = torch.randn(batch_size, 512, dtype=torch.float16)
    time_emb_fp16 = torch.randn(batch_size, 768, dtype=torch.float16)
    
    print(f"Input dtypes: clip={clip_emb_fp16.dtype}, time={time_emb_fp16.dtype}")
    
    # Track computation dtype through forward pass
    class DtypeTracker(nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.adapter = adapter
            self.computation_dtypes = []
            
        def forward(self, clip_emb, time_emb=None):
            # Hook to track dtype through computation
            orig_forward = self.adapter.linear_1.forward
            
            def tracked_forward(x):
                self.computation_dtypes.append(x.dtype)
                return orig_forward(x)
            
            self.adapter.linear_1.forward = tracked_forward
            output = self.adapter(clip_emb, time_emb)
            self.adapter.linear_1.forward = orig_forward
            
            return output
    
    tracker = DtypeTracker(adapter)
    output = tracker(clip_emb_fp16, time_emb_fp16)
    
    # Check that computation happened in FP32
    assert all(dtype == torch.float32 for dtype in tracker.computation_dtypes), \
        f"Computation should be FP32, got {tracker.computation_dtypes}"
    print(f"✓ Internal computation dtypes: {tracker.computation_dtypes}")
    
    # Output should match input dtype (FP16)
    assert output.dtype == torch.float16, f"Output should match input dtype, got {output.dtype}"
    print(f"✓ Output casted to match input: {output.dtype}")
    
    return True


def test_gradient_flow_in_fp32():
    """Test that gradients flow correctly in FP32."""
    print("\nTesting gradient flow in FP32...")
    
    # Create adapter
    adapter = ClipAdapter(
        time_embed_dim=768,
        clip_embed_dim=512,
        first_linear_std=0.02,
        layer_norm_gamma=0.001,
        gate_init=-5.0,
    )
    
    # Enable gradients
    for param in adapter.parameters():
        param.requires_grad = True
    
    # Forward pass with FP16 input
    clip_emb = torch.randn(2, 512, dtype=torch.float16, requires_grad=True)
    output = adapter(clip_emb)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients are computed
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.dtype == torch.float32, f"Gradient should be FP32 for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        print(f"  ✓ {name}: grad shape={param.grad.shape}, dtype={param.grad.dtype}")
    
    print("✓ All gradients flow correctly in FP32")
    return True


def test_edge_cast_behavior():
    """Test that casting only happens at input/output edges."""
    print("\nTesting edge-cast behavior...")
    
    adapter = ClipAdapter(
        time_embed_dim=768,
        clip_embed_dim=512,
    )
    
    # Test with various input dtypes
    test_cases = [
        (torch.float16, "FP16"),
        (torch.float32, "FP32"),
        (torch.bfloat16, "BF16") if torch.cuda.is_available() else None,
    ]
    
    for case in test_cases:
        if case is None:
            continue
            
        dtype, name = case
        clip_emb = torch.randn(2, 512, dtype=dtype)
        
        # Forward pass
        output = adapter(clip_emb)
        
        # Output should match input dtype
        assert output.dtype == dtype, f"{name}: Output dtype mismatch"
        print(f"✓ {name} input -> {name} output (internal FP32)")
    
    return True


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\nTesting numerical stability with extreme values...")
    
    adapter = ClipAdapter(
        time_embed_dim=768,
        clip_embed_dim=512,
    )
    
    # Test with various input scales
    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    
    for scale in scales:
        clip_emb = torch.randn(2, 512) * scale
        output = adapter(clip_emb)
        
        # Check output is finite
        assert torch.isfinite(output).all(), f"Non-finite output at scale {scale}"
        
        # Check output scale is reasonable (due to L2 norm + gate)
        output_scale = output.abs().max().item()
        assert output_scale < 100, f"Output too large at scale {scale}: {output_scale}"
        
        print(f"✓ Scale {scale:g}: max output = {output_scale:.6f}")
    
    print("✓ Numerically stable across input scales")
    return True


def test_with_actual_clip_features():
    """Test with real CLIP features to ensure compatibility."""
    print("\nTesting with actual CLIP features...")
    
    # Load CLIP model
    clip_model, _ = load_openai_clip("ViT-B/32", device="cpu")
    
    # Create adapter with explicit dimensions since we don't have a model
    adapter = ClipAdapter(
        time_embed_dim=768,  # Standard GLIDE dimension
        clip_embed_dim=512,  # ViT-B/32 dimension
    )
    
    # Get real CLIP features
    texts = ["a beautiful sunset", "a cute cat", "abstract art"]
    clip_features = get_clip_text_features(clip_model, texts, device="cpu")
    
    print(f"CLIP features: shape={clip_features.shape}, dtype={clip_features.dtype}")
    
    # Test with FP16 conversion (as would happen in mixed precision)
    clip_features_fp16 = clip_features.to(torch.float16)
    
    # Forward through adapter
    output = adapter(clip_features_fp16)
    
    # Check output
    assert output.dtype == torch.float16, "Output should match input dtype"
    assert torch.isfinite(output).all(), "Output should be finite"
    assert output.shape == (3, 768), f"Wrong output shape: {output.shape}"
    
    print(f"✓ Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"✓ Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Testing FP32 Enforcement in ClipAdapter")
    print("=" * 80)
    
    # Run all tests
    test_fp32_enforcement_in_fp16_model()
    test_fp32_computation_with_fp16_input()
    test_gradient_flow_in_fp32()
    test_edge_cast_behavior()
    test_numerical_stability()
    test_with_actual_clip_features()
    
    print("\n" + "=" * 80)
    print("✓ All FP32 enforcement tests passed!")
    print("✓ Adapter correctly maintains FP32 computation with edge casting")
    print("=" * 80)