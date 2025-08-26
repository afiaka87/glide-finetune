"""
UNet with CLIP adapter integration.

This module extends the GLIDE Text2ImUNet to support CLIP adapter injection
into the time embedding pathway.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from glide_text2im.nn import timestep_embedding


def create_model_with_adapter(base_model: nn.Module, clip_adapter: Optional[nn.Module] = None) -> nn.Module:
    """
    Wrap a GLIDE model with CLIP adapter support.
    
    Args:
        base_model: GLIDE Text2ImUNet model
        clip_adapter: ClipAdapter instance (optional)
    
    Returns:
        Model with integrated CLIP adapter
    """
    # Add adapter as an attribute
    base_model.clip_adapter = clip_adapter
    
    # Store original forward
    base_model._original_forward = base_model.forward
    
    # Create new forward method
    def forward_with_adapter(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        clip_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with optional CLIP adapter injection.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            timesteps: Timestep tensor [batch_size]
            tokens: BPE tokens [batch_size, seq_len]
            mask: Token mask [batch_size, seq_len]
            clip_embeddings: CLIP text embeddings [batch_size, clip_dim]
            **kwargs: Additional arguments
        
        Returns:
            Model output [batch_size, channels, height, width]
        """
        # Compute time embedding
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Add text conditioning if available
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        
        # Add CLIP adapter conditioning if available
        if self.clip_adapter is not None and clip_embeddings is not None:
            # Pass CLIP embeddings through adapter
            # The adapter handles FP32 computation and edge casting internally
            adapter_output = self.clip_adapter(clip_embeddings, time_embedding=emb)
            
            # Add adapter output to time embedding
            emb = emb + adapter_output.to(emb.dtype)
        
        # Continue with standard forward pass
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h
    
    # Replace forward method
    base_model.forward = forward_with_adapter.__get__(base_model, base_model.__class__)
    
    return base_model


def remove_adapter(model: nn.Module) -> nn.Module:
    """
    Remove CLIP adapter from model and restore original forward.
    
    Args:
        model: Model with adapter
    
    Returns:
        Model without adapter
    """
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        del model._original_forward
    
    if hasattr(model, 'clip_adapter'):
        del model.clip_adapter
    
    return model


def set_adapter_scale(model: nn.Module, scale: float):
    """
    Set the gate scale for the CLIP adapter.
    
    Args:
        model: Model with adapter
        scale: Target gate value (0 to 1)
    """
    if hasattr(model, 'clip_adapter') and model.clip_adapter is not None:
        model.clip_adapter.set_gate_value(scale)
    else:
        raise ValueError("Model does not have a CLIP adapter")


def get_adapter_scale(model: nn.Module) -> float:
    """
    Get the current gate scale of the CLIP adapter.
    
    Args:
        model: Model with adapter
    
    Returns:
        Current gate value
    """
    if hasattr(model, 'clip_adapter') and model.clip_adapter is not None:
        return model.clip_adapter.get_gate_value()
    else:
        raise ValueError("Model does not have a CLIP adapter")