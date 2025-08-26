#!/usr/bin/env python3
"""
Test that we're using standard OpenAI CLIP, NOT noise-aware CLIP.

This is critical: we must use the standard OpenAI CLIP ViT-B/32 model,
not the noise-aware CLIP from the glide_text2im codebase.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from glide_finetune.clip_adapter import (
    ClipAdapter,
    load_openai_clip,
    get_clip_text_features,
)
from glide_text2im.model_creation import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


def test_openai_clip_loading():
    """Test that we load standard OpenAI CLIP correctly."""
    print("Testing OpenAI CLIP loading...")
    
    # Load OpenAI CLIP
    clip_model, tokenize = load_openai_clip(model_name="ViT-B/32", device="cpu")
    
    # Verify it's the correct model
    assert hasattr(clip_model, 'text_projection'), "Should have text_projection"
    assert hasattr(clip_model, 'visual'), "Should have visual encoder"
    assert hasattr(clip_model, 'transformer'), "Should have transformer"
    
    # Check text projection dimensions
    assert isinstance(clip_model.text_projection, torch.nn.Parameter)
    text_proj_shape = clip_model.text_projection.shape
    assert text_proj_shape == (512, 512), f"Expected (512, 512), got {text_proj_shape}"
    
    # Verify model is frozen
    for param in clip_model.parameters():
        assert not param.requires_grad, "CLIP should be frozen"
    
    print(f"✓ Loaded OpenAI CLIP ViT-B/32 with text projection shape {text_proj_shape}")
    return clip_model, tokenize


def test_no_noise_aware_clip():
    """Ensure we're NOT using noise-aware CLIP from glide_text2im."""
    print("\nVerifying we're NOT using noise-aware CLIP...")
    
    # The noise-aware CLIP would have different signatures
    clip_model, _ = load_openai_clip("ViT-B/32", device="cpu")
    
    # OpenAI CLIP encode_text should NOT accept timestep parameter
    import inspect
    encode_text_sig = inspect.signature(clip_model.encode_text)
    params = list(encode_text_sig.parameters.keys())
    
    # Should only have 'text' parameter (or 'self' and 'text' if unbound)
    assert 't' not in params, "encode_text should NOT have timestep parameter"
    assert 'timestep' not in params, "encode_text should NOT have timestep parameter"
    
    print(f"✓ encode_text parameters: {params} (no timestep - correct!)")
    
    # Test encoding
    test_text = "a beautiful sunset"
    tokens = torch.zeros(1, 77, dtype=torch.long)  # CLIP uses 77 token context
    tokens[0, 0] = 49406  # Start token
    
    # This should work without timestep
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
    
    assert features.shape == (1, 512), f"Expected (1, 512), got {features.shape}"
    print(f"✓ Text encoding produces shape {features.shape} without timestep")
    
    return True


def test_clip_adapter_with_openai_clip():
    """Test ClipAdapter with actual OpenAI CLIP model."""
    print("\nTesting ClipAdapter with OpenAI CLIP...")
    
    # Load models
    clip_model, tokenize = load_openai_clip("ViT-B/32", device="cpu")
    
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False
    glide_model, _ = create_model_and_diffusion(**options)
    
    # Create adapter with OpenAI CLIP
    adapter = ClipAdapter.from_model(
        model=glide_model,
        clip_model=clip_model,
        first_linear_std=0.02,
        layer_norm_gamma=0.001,
        gate_init=-5.0,
    )
    
    # Verify dimensions
    assert adapter.clip_embed_dim == 512, f"Expected clip_embed_dim=512, got {adapter.clip_embed_dim}"
    assert adapter.time_embed_dim == 768, f"Expected time_embed_dim=768, got {adapter.time_embed_dim}"
    
    print(f"✓ Adapter created with OpenAI CLIP dimensions: {adapter.clip_embed_dim} -> {adapter.time_embed_dim}")
    
    # Test with actual CLIP features
    test_texts = ["a cat", "a dog"]
    clip_features = get_clip_text_features(clip_model, test_texts, device="cpu")
    
    assert clip_features.shape == (2, 512), f"Expected (2, 512), got {clip_features.shape}"
    
    # Forward through adapter
    adapter_output = adapter(clip_features)
    assert adapter_output.shape == (2, 768), f"Expected (2, 768), got {adapter_output.shape}"
    
    print(f"✓ Full pipeline working: text -> CLIP ({clip_features.shape}) -> adapter -> ({adapter_output.shape})")
    
    return adapter


def test_clip_text_encoding():
    """Test that text encoding with OpenAI CLIP works correctly."""
    print("\nTesting text encoding...")
    
    clip_model, tokenize = load_openai_clip("ViT-B/32", device="cpu")
    
    # Test single text
    single_features = get_clip_text_features(clip_model, "a beautiful painting", device="cpu")
    assert single_features.shape == (1, 512)
    print(f"✓ Single text encoded to shape {single_features.shape}")
    
    # Test batch of texts
    batch_texts = [
        "a cat sitting on a mat",
        "a dog playing in the park", 
        "a beautiful sunset over mountains"
    ]
    batch_features = get_clip_text_features(clip_model, batch_texts, device="cpu")
    assert batch_features.shape == (3, 512)
    print(f"✓ Batch of {len(batch_texts)} texts encoded to shape {batch_features.shape}")
    
    # Features should be normalized
    norms = torch.norm(batch_features, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"✓ Features are L2-normalized (norms: {norms.tolist()})")
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Testing OpenAI CLIP Integration (NOT noise-aware CLIP)")
    print("=" * 80)
    
    # Run tests
    clip_model, tokenize = test_openai_clip_loading()
    test_no_noise_aware_clip()
    adapter = test_clip_adapter_with_openai_clip()
    test_clip_text_encoding()
    
    print("\n" + "=" * 80)
    print("✓ All OpenAI CLIP integration tests passed!")
    print("✓ Confirmed: Using standard OpenAI CLIP ViT-B/32, NOT noise-aware CLIP")
    print("=" * 80)