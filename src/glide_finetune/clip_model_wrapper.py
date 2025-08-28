"""Model wrapper to inject CLIP features through the adapter.

This wrapper intercepts the forward pass to inject CLIP features into the
time embeddings using the CLIP adapter.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.clip_model_wrapper")


class ClipModelWrapper(nn.Module):
    """Wraps a GLIDE model to inject CLIP features through an adapter.
    
    This wrapper intercepts the forward pass and:
    1. Extracts CLIP features from the batch
    2. Passes them through the CLIP adapter
    3. Adds the adapter output to the time embeddings
    4. Calls the original model forward with modified embeddings
    """
    
    def __init__(self, model: nn.Module):
        """Initialize the wrapper.
        
        Args:
            model: GLIDE model with clip_adapter attribute
        """
        super().__init__()
        self.model = model
        
        # Verify adapter is present
        if not hasattr(model, 'clip_adapter'):
            raise ValueError("Model must have clip_adapter attribute")
        
        # Store reference to adapter for easy access
        self.clip_adapter = model.clip_adapter
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        clip_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with CLIP feature injection.
        
        Args:
            x: Input tensor
            timesteps: Timestep tensor
            clip_embeddings: Optional CLIP embeddings to inject
            **kwargs: Additional arguments for the model
            
        Returns:
            Model output
        """
        # If no CLIP embeddings provided, use standard forward
        if clip_embeddings is None:
            return self.model(x, timesteps, **kwargs)
        
        # Store original forward method
        original_forward = self.model.forward
        
        # Create a modified forward that injects CLIP features
        def forward_with_clip(x_inner, timesteps_inner, **inner_kwargs):
            # Call original forward to get standard processing
            # But we need to intercept the time embedding computation
            
            # First, compute time embeddings the same way the model does
            from glide_text2im.nn import timestep_embedding
            time_emb = self.model.time_embed(
                timestep_embedding(timesteps_inner, self.model.model_channels)
            )
            
            # Pass CLIP embeddings through adapter
            clip_contribution = self.clip_adapter(
                clip_embeddings,
                time_embedding=time_emb
            )
            
            # Add CLIP contribution to time embeddings
            modified_time_emb = time_emb + clip_contribution
            
            # Now we need to run the model with modified embeddings
            # This requires manually running through the model layers
            hs = []
            
            # Handle text conditioning if this is a Text2ImUNet
            if hasattr(self.model, 'xf_width') and self.model.xf_width:
                tokens = inner_kwargs.get('tokens', None)
                mask = inner_kwargs.get('mask', None)
                if tokens is not None:
                    text_outputs = self.model.get_text_emb(tokens, mask)
                    xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
                    # Add text projection to the already-modified embeddings
                    emb = modified_time_emb + xf_proj.to(modified_time_emb)
                else:
                    xf_out = None
                    emb = modified_time_emb
            else:
                xf_out = None
                emb = modified_time_emb
            
            # Run through UNet blocks with modified embeddings
            h = x_inner.type(self.model.dtype)
            for module in self.model.input_blocks:
                h = module(h, emb, xf_out)
                hs.append(h)
            h = self.model.middle_block(h, emb, xf_out)
            for module in self.model.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, xf_out)
            h = h.type(x_inner.dtype)
            h = self.model.out(h)
            return h
        
        # Use the modified forward
        return forward_with_clip(x, timesteps, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        # First check if it's in our __dict__ (like 'model' or 'clip_adapter')
        if '_ClipModelWrapper__initialized' not in self.__dict__:
            # During __init__, just use normal attribute access
            return object.__getattribute__(self, name)
        
        # Check our own attributes first
        if name in self.__dict__:
            return self.__dict__[name]
        
        # Forward to wrapped model
        return getattr(self.model, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute setting."""
        if name in ['model', 'clip_adapter']:
            # These are our own attributes
            object.__setattr__(self, name, value)
        elif hasattr(self, 'model') and hasattr(self.model, name):
            # Forward to wrapped model if it has this attribute
            setattr(self.model, name, value)
        else:
            # Default behavior
            object.__setattr__(self, name, value)
        
        # Mark as initialized after __init__ completes
        if name == 'clip_adapter':
            object.__setattr__(self, '_ClipModelWrapper__initialized', True)


def wrap_model_with_clip_adapter(model: nn.Module) -> nn.Module:
    """Wrap a model with CLIP adapter injection.
    
    Args:
        model: Model with clip_adapter attribute
        
    Returns:
        Wrapped model that injects CLIP features
    """
    if not hasattr(model, 'clip_adapter'):
        raise ValueError("Model must have clip_adapter attribute before wrapping")
    
    logger.info("Wrapping model with CLIP adapter injection")
    return ClipModelWrapper(model)