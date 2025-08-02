"""
CLIP Adapter module for augmenting GLIDE's text conditioning with frozen CLIP features.

This implementation follows the approach from CLIP-Adapter and IP-Adapter papers,
carefully designed to preserve pretrained GLIDE behavior while enabling gradual
integration of CLIP features.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# CLIP model dimensions for common architectures
# Note: These are the actual output dimensions from OpenAI's CLIP models
CLIP_DIMENSIONS = {
    "ViT-B/32": 512,     # Corrected from 768
    "ViT-B/16": 512,     # Corrected from 768
    "ViT-L/14": 768,     # Corrected from 1024
    "ViT-L/14@336px": 768,  # Corrected from 1024
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
}


def load_clip_model(model_name: str = "ViT-L/14", device: str = "cuda"):
    """
    Load a frozen CLIP model for text encoding.
    
    Args:
        model_name: Name of CLIP model architecture (e.g., 'ViT-L/14', 'ViT-B/32')
        device: Device to load model on
        
    Returns:
        clip_model: The CLIP model
        clip_preprocess: The preprocessing function (for images if needed)
    """
    try:
        import clip
    except ImportError:
        raise ImportError("Please install OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git")
    
    clip_model, clip_preprocess = clip.load(model_name, device=device)
    clip_model.eval()
    
    # Freeze all CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False
        
    return clip_model, clip_preprocess


class ClipAdapter(nn.Module):
    """
    CLIP adapter with 2-layer MLP and learnable gates for stable integration
    with pretrained GLIDE models.
    
    Following CLIP-Adapter design:
    - 2-layer MLP with residual connection
    - LayerNorm for stability
    - Learnable scalar gate initialized to 0
    - Optional LoRA branch for parameter efficiency
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        use_lora: bool = False,
        lora_rank: int = 32,
        init_std: float = 0.02,
    ):
        """
        Args:
            input_dim: CLIP embedding dimension (e.g., 768 for ViT-B/32, 1024 for ViT-L/14)
            output_dim: GLIDE's xf_width dimension for compatibility
            hidden_dim: Hidden layer dimension (defaults to 2 * input_dim)
            dropout: Dropout rate for regularization
            gate_init: Initial value for learnable gate (0.0 for stability)
            use_lora: Whether to use LoRA instead of full MLP
            lora_rank: Rank for LoRA decomposition
            init_std: Standard deviation for weight initialization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or (input_dim * 2)
        self.use_lora = use_lora
        
        # LayerNorm for input stability
        self.ln_in = nn.LayerNorm(input_dim)
        
        if use_lora:
            # LoRA branch for parameter efficiency
            self.lora_down = nn.Linear(input_dim, lora_rank, bias=False)
            self.lora_up = nn.Linear(lora_rank, output_dim, bias=False)
            
            # Initialize LoRA weights
            nn.init.normal_(self.lora_down.weight, std=init_std)
            nn.init.zeros_(self.lora_up.weight)
        else:
            # Standard 2-layer MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, output_dim),
            )
            
            # Careful initialization for stability
            self._init_mlp_weights(init_std)
        
        # Projection layer if dimensions don't match
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
            nn.init.normal_(self.proj.weight, std=init_std)
            nn.init.zeros_(self.proj.bias)
        else:
            self.proj = nn.Identity()
        
        # Learnable gate for gradual integration
        self.gate = nn.Parameter(torch.tensor(gate_init))
        
    def _init_mlp_weights(self, std: float):
        """Initialize MLP weights carefully for stability with pretrained models."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Xavier/He initialization scaled down for stability
                fan_in = module.in_features
                scale = math.sqrt(2.0 / fan_in) * 0.1  # Scale down by 10x
                nn.init.normal_(module.weight, std=min(std, scale))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        clip_embeddings: torch.Tensor,
        gate_override: Optional[float] = None,
        return_pre_gate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the adapter.
        
        Args:
            clip_embeddings: CLIP text embeddings [batch_size, clip_dim]
            gate_override: Override the learnable gate value (for testing)
            return_pre_gate: Return both gated and ungated outputs
            
        Returns:
            adapted_embeddings: Adapted embeddings [batch_size, output_dim]
            (optional) pre_gate_embeddings: Embeddings before gating
        """
        # Apply LayerNorm for stability
        x = self.ln_in(clip_embeddings)
        
        # Apply adapter transformation
        if self.use_lora:
            # LoRA path
            adapted = self.lora_up(self.lora_down(x))
        else:
            # MLP path
            adapted = self.mlp(x)
        
        # Residual connection with projection
        residual = self.proj(clip_embeddings)
        adapted = adapted + residual
        
        # Store pre-gate output if requested
        if return_pre_gate:
            pre_gate = adapted.clone()
        
        # Apply gate for gradual integration
        gate_value = gate_override if gate_override is not None else self.gate
        adapted = gate_value * adapted + (1 - gate_value) * residual
        
        if return_pre_gate:
            return adapted, pre_gate
        return adapted
    
    def get_gate_value(self) -> float:
        """Get current gate value."""
        return self.gate.item()
    
    def set_gate_value(self, value: float):
        """Set gate value (useful for schedules)."""
        with torch.no_grad():
            self.gate.fill_(value)


class ClipTextEncoder(nn.Module):
    """
    Wrapper for CLIP text encoding with caching support.
    """
    
    def __init__(self, clip_model, tokenizer=None, device="cuda"):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = {}
        
    @torch.no_grad()
    def encode_text(self, text: Union[str, list], use_cache: bool = True) -> torch.Tensor:
        """
        Encode text using CLIP model.
        
        Args:
            text: Single string or list of strings
            use_cache: Whether to use cached embeddings
            
        Returns:
            embeddings: CLIP text embeddings [batch_size, embed_dim]
        """
        if isinstance(text, str):
            text = [text]
        
        # Check cache
        cache_key = tuple(text)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Tokenize and encode
        try:
            import clip
            # clip.tokenize returns CPU tensors, so we need to move to device
            tokens = clip.tokenize(text, truncate=True)
            tokens = tokens.to(self.device)
        except:
            # Fallback to simple tokenization if clip.tokenize not available
            raise NotImplementedError("CLIP tokenization not available")
        
        embeddings = self.clip_model.encode_text(tokens).float()
        
        # Cache results
        if use_cache:
            self.cache[cache_key] = embeddings
            
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()


def create_clip_adapter_config(
    clip_model_name: str = "ViT-L/14",
    glide_xf_width: int = 2048,
    use_lora: bool = False,
    **kwargs
) -> dict:
    """
    Create configuration for CLIP adapter based on model choices.
    
    Args:
        clip_model_name: Name of CLIP model
        glide_xf_width: GLIDE's transformer width
        use_lora: Whether to use LoRA
        **kwargs: Additional adapter arguments
        
    Returns:
        config: Configuration dictionary
    """
    clip_dim = CLIP_DIMENSIONS.get(clip_model_name)
    if clip_dim is None:
        raise ValueError(f"Unknown CLIP model: {clip_model_name}. Available: {list(CLIP_DIMENSIONS.keys())}")
    
    config = {
        "input_dim": clip_dim,
        "output_dim": glide_xf_width,
        "use_lora": use_lora,
    }
    config.update(kwargs)
    
    return config