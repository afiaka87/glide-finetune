"""
Dual attention module for integrating CLIP features into GLIDE's cross-attention.

Following IP-Adapter's decoupled cross-attention design, this module maintains
separate K/V projections for text tokens and CLIP embeddings while preserving
the original behavior when CLIP is disabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from glide_text2im.nn import conv_nd, zero_module
from glide_text2im.unet import AttentionBlock, QKVAttention


class DualAttentionBlock(AttentionBlock):
    """
    Extended AttentionBlock that supports dual conditioning on both
    text tokens (from GLIDE's encoder) and CLIP embeddings.
    
    This maintains backward compatibility - when clip_encoder_out is None,
    it behaves exactly like the original AttentionBlock.
    """
    
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        encoder_channels=None,
        clip_channels=None,
        clip_gate_init=0.0,
    ):
        """
        Args:
            channels: Number of channels in the input
            num_heads: Number of attention heads
            num_head_channels: Channels per head (if -1, uses num_heads)
            use_checkpoint: Whether to use gradient checkpointing
            encoder_channels: Channels for text encoder cross-attention
            clip_channels: Channels for CLIP encoder cross-attention
            clip_gate_init: Initial value for CLIP attention gate
        """
        # Initialize parent without encoder_channels first
        super().__init__(
            channels=channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_checkpoint=use_checkpoint,
            encoder_channels=encoder_channels,
        )
        
        # Add CLIP-specific components
        self.clip_channels = clip_channels
        if clip_channels is not None:
            # Separate K/V projection for CLIP features
            self.clip_kv = conv_nd(1, clip_channels, channels * 2, 1)
            
            # Learnable gate for blending attention outputs
            # Store in logit space so sigmoid(clip_gate) = clip_gate_init
            if clip_gate_init == 0.0:
                # Special case: sigmoid(-inf) = 0, but use large negative value
                gate_logit = -10.0
            elif clip_gate_init == 1.0:
                # Special case: sigmoid(inf) = 1, but use large positive value
                gate_logit = 10.0
            else:
                # Convert from probability to logit: logit = log(p / (1-p))
                gate_logit = torch.log(torch.tensor(clip_gate_init / (1 - clip_gate_init))).item()
            self.clip_gate = nn.Parameter(torch.tensor(gate_logit))
            
            # Optional: separate attention module for CLIP
            # This allows different attention patterns for text vs CLIP
            self.clip_attention = QKVAttention(self.num_heads)
    
    def forward(self, x, encoder_out=None, clip_encoder_out=None):
        """
        Apply dual attention to the input.
        
        Args:
            x: Input tensor [B, C, H, W]
            encoder_out: Text encoder output [B, text_channels, seq_len]
            clip_encoder_out: CLIP encoder output [B, clip_channels, 1]
            
        Returns:
            Output tensor with same shape as input
        """
        b, c, *spatial = x.shape
        
        # Self-attention path (always present)
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        
        # Determine if this is a cross-attention block
        is_cross_attention = hasattr(self, 'encoder_kv')
        
        if is_cross_attention and (encoder_out is not None or clip_encoder_out is not None):
            # Cross-attention block with text and/or CLIP features
            if encoder_out is not None and clip_encoder_out is not None:
                # Both text and CLIP features available
                h_text = self._apply_text_attention(qkv, encoder_out)
                h_clip = self._apply_clip_attention(qkv, clip_encoder_out)
                
                # Blend the two attention outputs
                gate = torch.sigmoid(self.clip_gate)
                h = (1 - gate) * h_text + gate * h_clip
                
                # Debug print
                if hasattr(self, '_debug') and self._debug:
                    print(f"  DualAttentionBlock cross-attention: gate={gate.item():.4f}, h_text norm={h_text.norm().item():.4f}, h_clip norm={h_clip.norm().item():.4f}")
                
            elif encoder_out is not None:
                # Text-only (original behavior)
                encoder_kv = self.encoder_kv(encoder_out)
                h = self.attention(qkv, encoder_kv)
                
            else:  # clip_encoder_out is not None
                # CLIP-only attention
                h = self._apply_clip_attention(qkv, clip_encoder_out)
        else:
            # Self-attention block
            if clip_encoder_out is not None and hasattr(self, 'clip_gate'):
                # Self-attention with CLIP modulation
                h_self = self.attention(qkv, None)
                h_clip = self._apply_clip_attention(qkv, clip_encoder_out)
                
                # Blend self-attention with CLIP-modulated attention
                gate = torch.sigmoid(self.clip_gate)
                h = (1 - gate) * h_self + gate * h_clip
                
                # Debug print
                if hasattr(self, '_debug') and self._debug:
                    print(f"  DualAttentionBlock self-attention: gate={gate.item():.4f}, h_self norm={h_self.norm().item():.4f}, h_clip norm={h_clip.norm().item():.4f}")
            else:
                # Pure self-attention
                h = self.attention(qkv, None)
        
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)
    
    def _apply_text_attention(self, qkv, encoder_out):
        """Apply cross-attention with text encoder output."""
        if hasattr(self, 'encoder_kv'):
            encoder_kv = self.encoder_kv(encoder_out)
            return self.attention(qkv, encoder_kv)
        else:
            # Self-attention case
            return self.attention(qkv, None)
    
    def _apply_clip_attention(self, qkv, clip_encoder_out):
        """Apply cross-attention with CLIP encoder output."""
        clip_kv = self.clip_kv(clip_encoder_out)
        return self.clip_attention(qkv, clip_kv)
    
    def get_clip_gate_value(self) -> float:
        """Get current CLIP attention gate value."""
        if hasattr(self, 'clip_gate'):
            return torch.sigmoid(self.clip_gate).item()
        return 0.0
    
    def set_clip_gate_value(self, value: float):
        """Set CLIP attention gate value (for schedules)."""
        if hasattr(self, 'clip_gate'):
            with torch.no_grad():
                # Convert from [0,1] to logit space
                logit = torch.log(torch.tensor(value) / (1 - value))
                self.clip_gate.fill_(logit)


class DualConditioningAdapter(nn.Module):
    """
    Adapter module that prepares and combines text and CLIP features
    for the dual attention mechanism.
    """
    
    def __init__(
        self,
        text_dim: int,
        clip_dim: int,
        output_dim: int,
        sequence_length: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            text_dim: Dimension of text encoder output
            clip_dim: Dimension of CLIP encoder output
            output_dim: Output dimension for attention
            sequence_length: Expected sequence length for text
            dropout: Dropout rate
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.clip_dim = clip_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # Transform CLIP features to sequence format
        # CLIP gives us a single pooled embedding, but attention expects a sequence
        self.clip_to_seq = nn.Sequential(
            nn.Linear(clip_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim * sequence_length),
        )
        
        # Optional: learnable positional embeddings for CLIP sequence
        self.clip_pos_embed = nn.Parameter(
            torch.randn(1, sequence_length, output_dim) * 0.02
        )
        
    def forward(
        self,
        text_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process text and CLIP features for dual attention.
        
        Args:
            text_features: Text encoder output [B, text_dim, seq_len]
            clip_features: CLIP embeddings [B, clip_dim]
            
        Returns:
            text_out: Processed text features
            clip_out: Processed CLIP features as sequence
        """
        batch_size = clip_features.shape[0]
        
        # Transform CLIP features to sequence
        clip_seq = self.clip_to_seq(clip_features)  # [B, output_dim * seq_len]
        clip_seq = clip_seq.view(batch_size, self.sequence_length, self.output_dim)
        
        # Add positional embeddings
        clip_seq = clip_seq + self.clip_pos_embed
        
        # Transpose to match expected format [B, C, L]
        clip_out = clip_seq.transpose(1, 2)
        
        return text_features, clip_out


def replace_attention_blocks(model, clip_channels: int, clip_gate_init: float = 0.0):
    """
    Replace AttentionBlock modules in a model with DualAttentionBlock.
    
    This function walks through the model and replaces attention blocks
    while preserving their weights, ensuring backward compatibility.
    
    Args:
        model: The model to modify
        clip_channels: Number of channels for CLIP features
        clip_gate_init: Initial gate value
        
    Returns:
        Number of blocks replaced
    """
    replaced = 0
    
    def replace_module(parent, name, module):
        if isinstance(module, AttentionBlock) and not isinstance(module, DualAttentionBlock):
            # Extract configuration from existing module
            dual_block = DualAttentionBlock(
                channels=module.channels,
                num_heads=module.num_heads,
                num_head_channels=getattr(module, 'num_head_channels', -1),
                use_checkpoint=module.use_checkpoint,
                encoder_channels=getattr(module, 'encoder_channels', None),
                clip_channels=clip_channels,
                clip_gate_init=clip_gate_init,
            )
            
            # Copy weights from original module
            with torch.no_grad():
                dual_block.norm.load_state_dict(module.norm.state_dict())
                dual_block.qkv.load_state_dict(module.qkv.state_dict())
                dual_block.attention = module.attention
                dual_block.proj_out.load_state_dict(module.proj_out.state_dict())
                
                # Only copy encoder_kv if this is a cross-attention block
                if hasattr(module, 'encoder_kv') and hasattr(dual_block, 'encoder_kv'):
                    dual_block.encoder_kv.load_state_dict(module.encoder_kv.state_dict())
            
            setattr(parent, name, dual_block)
            return 1
        return 0
    
    # Walk through all modules
    for name, child in model.named_children():
        if isinstance(child, AttentionBlock):
            replaced += replace_module(model, name, child)
        else:
            replaced += replace_attention_blocks(child, clip_channels, clip_gate_init)
    
    return replaced