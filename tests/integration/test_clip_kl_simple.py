#!/usr/bin/env python3
"""
Simplified test for CLIP KL divergence to debug the zero output issue.
"""

import torch
import torch.nn as nn
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

from glide_finetune.adapters.glide_clip_integration import (
    create_clip_model_from_options,
)


def test_simple_clip_difference():
    """Test that CLIP pathway actually changes outputs."""
    device = 'cpu'
    
    # Use configuration that matches pretrained base.pt model
    options = model_and_diffusion_defaults()
    # Override only what's necessary for testing
    options['use_fp16'] = False  # CPU testing
    options['timestep_respacing'] = '10'  # Faster testing
    
    # Create CLIP-enabled model
    model = create_clip_model_from_options(
        options,
        clip_model_name="ViT-B/32",
        use_clip=True,
        clip_gate_init=0.5,  # Start with 0.5 for stronger effect
        device=device,
    )
    model = model.to(device)
    model.eval()
    
    # Load pretrained weights to avoid NaN issues
    import os
    pretrained_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "glide_model_cache", "base.pt"
    )
    
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        
        # Load into CLIP model (will have missing keys for CLIP components)
        model.load_state_dict(pretrained_state, strict=False)
        print("Successfully loaded pretrained weights")
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        print("Test may produce NaN values without proper initialization")
    
    # Create test inputs
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.tensor([50, 50]).to(device)
    
    # Create tokens (just random for now)
    tokens = torch.randint(0, 1000, (batch_size, 128)).to(device)
    mask = torch.ones_like(tokens).bool()
    
    # Get CLIP embeddings
    clip_prompts = ["a red car", "a blue sky"]
    clip_embeddings = model.get_clip_text_emb(clip_prompts)
    
    print("Testing model outputs with and without CLIP...")
    
    with torch.no_grad():
        # Test with CLIP
        output_with_clip = model(
            x, timesteps, tokens=tokens, mask=mask,
            clip_embeddings=clip_embeddings,
            use_clip_override=True,
        )
        
        # Test without CLIP
        output_without_clip = model(
            x, timesteps, tokens=tokens, mask=mask,
            use_clip_override=False,
        )
    
    # Check statistics
    print(f"\nWith CLIP:")
    print(f"  Mean: {output_with_clip.mean().item():.6f}")
    print(f"  Std: {output_with_clip.std().item():.6f}")
    print(f"  Abs max: {output_with_clip.abs().max().item():.6f}")
    
    print(f"\nWithout CLIP:")
    print(f"  Mean: {output_without_clip.mean().item():.6f}")
    print(f"  Std: {output_without_clip.std().item():.6f}")
    print(f"  Abs max: {output_without_clip.abs().max().item():.6f}")
    
    # Check difference
    diff = (output_with_clip - output_without_clip).abs()
    print(f"\nDifference:")
    print(f"  Mean: {diff.mean().item():.6f}")
    print(f"  Max: {diff.max().item():.6f}")
    print(f"  Outputs identical: {torch.allclose(output_with_clip, output_without_clip)}")
    
    # Also check intermediate activations by enabling debug mode
    print("\n\nChecking with debug mode...")
    model._debug_clip = True
    
    with torch.no_grad():
        print("\nWith CLIP (debug):")
        _ = model(
            x, timesteps, tokens=tokens, mask=mask,
            clip_embeddings=clip_embeddings,
            use_clip_override=True,
        )
        
        print("\nWithout CLIP (debug):")
        _ = model(
            x, timesteps, tokens=tokens, mask=mask,
            use_clip_override=False,
        )


if __name__ == "__main__":
    test_simple_clip_difference()