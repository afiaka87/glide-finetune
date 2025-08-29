#!/usr/bin/env python3
"""Quick test to verify CLIP adapter is loaded and produces different outputs."""

import torch
import numpy as np
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from glide_finetune.clip_adapter import integrate_clip_adapter_to_model
from glide_finetune.clip_compute import CLIPFeatureComputer

# Test prompt
test_prompt = "A beautiful sunset over mountains"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your checkpoint with adapter
checkpoint_path = "/mnt/usb_nvme_512gb/Checkpoints/glide-finetune-ckpt/clip_adapter_20250828_210125/glide-ft-0x14000.pt"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Check if it has adapter weights
has_adapter = any(key.startswith("clip_adapter.") for key in checkpoint.keys())
print(f"Checkpoint has CLIP adapter weights: {has_adapter}")

if has_adapter:
    print("\nCreating model with CLIP adapter architecture...")
    
    # Create base model
    options = model_and_diffusion_defaults()
    options["use_fp16"] = False
    model, diffusion = create_model_and_diffusion(**options)
    
    # Load OpenAI base weights
    model.load_state_dict(load_checkpoint("base", "cpu"))
    model = model.to(device)
    
    # Add CLIP adapter
    model = integrate_clip_adapter_to_model(
        model,
        clip_model_name="ViT-B/32",
        device=device,
    )
    print("Added CLIP adapter to model")
    
    # Load full checkpoint with adapter weights
    model.load_state_dict(checkpoint)
    print("Loaded checkpoint weights including adapter")
    
    # Verify adapter exists
    if hasattr(model, 'clip_adapter') and model.clip_adapter is not None:
        print("\n✓ CLIP adapter successfully integrated!")
        print(f"Adapter gate value: {torch.sigmoid(model.clip_adapter.gate).item():.6f}")
        
        # Test with CLIP embeddings
        print("\nTesting CLIP conditioning...")
        clip_computer = CLIPFeatureComputer("ViT-B/32", device=device)
        clip_embeddings = clip_computer.compute_text_features([test_prompt])
        
        # Create dummy inputs
        batch_size = 1
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        t = torch.tensor([100]).to(device)
        
        # Get text tokens
        tokens = model.tokenizer.encode(test_prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
        
        model_kwargs = {
            "tokens": torch.tensor([tokens], device=device),
            "mask": torch.tensor([mask], dtype=torch.bool, device=device),
        }
        
        # Test without CLIP
        print("Output without CLIP embeddings...")
        with torch.no_grad():
            out_no_clip = model(x, t, **model_kwargs)
        
        # Test with CLIP
        print("Output with CLIP embeddings...")
        model_kwargs["clip_embeddings"] = clip_embeddings
        with torch.no_grad():
            out_with_clip = model(x, t, **model_kwargs)
        
        # Compare outputs
        diff = torch.abs(out_with_clip - out_no_clip).mean().item()
        max_diff = torch.abs(out_with_clip - out_no_clip).max().item()
        
        print(f"\nMean absolute difference: {diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        
        if diff > 1e-6:
            print("✓ CLIP adapter is producing different outputs - working correctly!")
        else:
            print("✗ Warning: Outputs are identical - adapter may not be working")
            
    else:
        print("\n✗ Error: CLIP adapter not found on model!")
else:
    print("\n✗ This checkpoint doesn't contain CLIP adapter weights")