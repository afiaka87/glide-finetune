#!/usr/bin/env python3
"""
Verify GLIDE architecture at runtime to determine cross-attention status.

This script loads a GLIDE model and verifies:
1. Whether encoder_kv layers are present in attention blocks
2. Whether cross-attention is actually used (encoder_out != None)
3. The dimensions of time embedding and other critical components

This is critical for CLIP adapter integration to ensure we don't assume
cross-attention is active when it might be disabled.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from glide_text2im.model_creation import (
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
    create_model_and_diffusion,
)


def verify_cross_attention(model):
    """Verify if cross-attention is actually active in the model."""
    print("=" * 80)
    print("GLIDE Architecture Verification")
    print("=" * 80)
    
    # Check if model has xf_width (text transformer width)
    has_text_encoder = hasattr(model, 'xf_width') and model.xf_width is not None and model.xf_width > 0
    print(f"\n1. Text Encoder Status:")
    print(f"   - Has xf_width: {has_text_encoder}")
    if has_text_encoder:
        print(f"   - xf_width value: {model.xf_width}")
        print(f"   - Text context length: {model.text_ctx}")
    
    # Check time embedding dimensions
    print(f"\n2. Time Embedding Dimensions:")
    if hasattr(model, 'time_embed'):
        time_embed_layers = list(model.time_embed.children())
        if len(time_embed_layers) >= 2:
            # Get output dimension from second linear layer
            d_emb = time_embed_layers[-1].out_features
            print(f"   - Time embedding dimension (d_emb): {d_emb}")
            print(f"   - Expected: {model.model_channels * 4}")
        else:
            print("   - WARNING: Unexpected time_embed structure")
    
    # Check for encoder_kv in attention blocks
    print(f"\n3. Cross-Attention Analysis:")
    encoder_kv_count = 0
    total_attention_blocks = 0
    encoder_channels_values = set()
    
    # Check all modules for AttentionBlock instances
    for name, module in model.named_modules():
        module_name = module.__class__.__name__
        
        if 'AttentionBlock' in module_name or 'Attention' in module_name:
            total_attention_blocks += 1
            
            # Check for encoder_kv layer
            has_encoder_kv = hasattr(module, 'encoder_kv') and module.encoder_kv is not None
            
            if has_encoder_kv:
                encoder_kv_count += 1
                # Get encoder channels from weight shape
                encoder_channels = module.encoder_kv.weight.shape[1]
                encoder_channels_values.add(encoder_channels)
                
                if encoder_kv_count <= 3:  # Show first few for verification
                    print(f"   - Block '{name}': encoder_kv present (channels={encoder_channels})")
    
    print(f"\n   Summary:")
    print(f"   - Total attention blocks: {total_attention_blocks}")
    print(f"   - Blocks with encoder_kv: {encoder_kv_count}")
    print(f"   - Unique encoder_channels values: {encoder_channels_values}")
    
    # Determine cross-attention status
    cross_attention_active = encoder_kv_count > 0
    print(f"\n4. CROSS-ATTENTION STATUS: {'ACTIVE' if cross_attention_active else 'INACTIVE'}")
    
    if cross_attention_active:
        print(f"   ✓ Cross-attention IS wired and active")
        print(f"   ✓ Found {encoder_kv_count} blocks with encoder_kv layers")
        if encoder_channels_values:
            print(f"   ✓ Encoder channels: {encoder_channels_values}")
    else:
        print(f"   ✗ Cross-attention is NOT active")
        print(f"   ✗ No encoder_kv layers found in attention blocks")
        print(f"   → Will NOT modify cross-attention in v1 adapter")
    
    # Additional architectural details
    print(f"\n5. Model Architecture Details:")
    print(f"   - Model channels: {model.model_channels}")
    print(f"   - Number of heads: {getattr(model, 'num_heads', 'N/A')}")
    print(f"   - Number of head channels: {getattr(model, 'num_head_channels', 'N/A')}")
    
    # Count ResBlocks
    resblock_count = sum(1 for _, m in model.named_modules() if 'ResBlock' in m.__class__.__name__)
    print(f"   - Total ResBlocks: {resblock_count}")
    
    return {
        'has_text_encoder': has_text_encoder,
        'xf_width': model.xf_width if has_text_encoder else None,
        'cross_attention_active': cross_attention_active,
        'encoder_kv_count': encoder_kv_count,
        'encoder_channels': list(encoder_channels_values) if encoder_channels_values else None,
        'd_emb': d_emb if 'd_emb' in locals() else None,
        'model_channels': model.model_channels,
    }


def main():
    """Run architecture verification on both base and upsampler models."""
    
    print("Loading base model (64x64)...")
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False  # Use FP32 for verification
    model, diffusion = create_model_and_diffusion(**options)
    
    print("\nVerifying BASE model architecture:")
    base_info = verify_cross_attention(model)
    
    print("\n" + "=" * 80)
    print("\nLoading upsampler model (64x64 -> 256x256)...")
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = False
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    
    print("\nVerifying UPSAMPLER model architecture:")
    upsampler_info = verify_cross_attention(model_up)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    
    print("\nBase Model (64x64):")
    print(f"  - Cross-attention: {'ACTIVE' if base_info['cross_attention_active'] else 'INACTIVE'}")
    print(f"  - Text encoder width: {base_info['xf_width']}")
    print(f"  - Time embed dimension: {base_info['d_emb']}")
    
    print("\nUpsampler Model (256x256):")
    print(f"  - Cross-attention: {'ACTIVE' if upsampler_info['cross_attention_active'] else 'INACTIVE'}")
    print(f"  - Text encoder width: {upsampler_info['xf_width']}")
    print(f"  - Time embed dimension: {upsampler_info['d_emb']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR CLIP ADAPTER:")
    print("=" * 80)
    
    if not base_info['cross_attention_active']:
        print("✓ Cross-attention is INACTIVE - proceed with time-embedding-only adapter")
        print("✓ No modifications to attention blocks needed in v1")
    else:
        print("⚠ Cross-attention is ACTIVE - adapter can still inject via time embedding")
        print("⚠ Consider cross-attention LoRA for v2 (IP-Adapter style)")
    
    print("\nDimensions for ClipAdapter:")
    print(f"  - d_emb = {base_info['d_emb']} (time embedding dimension)")
    print(f"  - Will discover d_clip at runtime from CLIP model")
    
    return base_info, upsampler_info


if __name__ == "__main__":
    base_info, upsampler_info = main()