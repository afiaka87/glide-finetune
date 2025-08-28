"""CLIP adapter module for GLIDE fine-tuning.

This module implements a zero-initialized residual adapter pattern (ControlNet-inspired)
that safely injects CLIP conditioning into the GLIDE time embedding pathway.

IMPORTANT: This uses the standard OpenAI CLIP ViT-B/32 model, NOT the noise-aware CLIP
from the GLIDE codebase. The noise-aware CLIP is not suitable for our adapter approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import clip  # OpenAI CLIP, not the glide_text2im.clip module


class ClipAdapter(nn.Module):
    """Zero-initialized residual adapter for CLIP conditioning.
    
    Architecture:
        CLIP embedding -> L2 norm -> LayerNorm -> Linear -> GELU -> Linear (zero-init) -> Gate -> Add to time_emb
    
    Key properties:
        - Zero-initialized output layer for perfect baseline preservation (gate=0)
        - Scalar gate with sigmoid activation starting at ~0.0067
        - FP32 computation throughout (even in mixed precision training)
        - Runtime dimension discovery from model architecture
    """
    
    def __init__(
        self,
        time_embed_dim: int,
        clip_embed_dim: int,
        hidden_dim: Optional[int] = None,
        first_linear_std: float = 0.02,
        layer_norm_gamma: float = 0.001,
        gate_init: float = -5.0,
        enable_rms_matching: bool = False,
    ):
        """Initialize CLIP adapter with zero-conv pattern.
        
        Args:
            time_embed_dim: Dimension of time embedding (d_emb), discovered at runtime
            clip_embed_dim: Dimension of CLIP embedding (d_clip), from CLIP model
            hidden_dim: Hidden layer dimension, defaults to time_embed_dim
            first_linear_std: Standard deviation for first linear layer init
            layer_norm_gamma: Initial gamma value for LayerNorm (small but non-zero)
            gate_init: Initial value for gate parameter (sigmoid(-5) ≈ 0.0067)
            enable_rms_matching: Whether to match RMS to xf_proj (optional)
        """
        super().__init__()
        
        # Store dimensions
        self.time_embed_dim = time_embed_dim
        self.clip_embed_dim = clip_embed_dim
        self.hidden_dim = hidden_dim or time_embed_dim
        
        # Store config
        self.first_linear_std = first_linear_std
        self.layer_norm_gamma = layer_norm_gamma
        self.gate_init = gate_init
        self.enable_rms_matching = enable_rms_matching
        
        # Build adapter layers
        self.layer_norm = nn.LayerNorm(clip_embed_dim, eps=1e-5)
        self.linear_1 = nn.Linear(clip_embed_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_dim, time_embed_dim)
        
        # Scalar gate parameter
        self.gate = nn.Parameter(torch.tensor(gate_init))
        
        # Initialize weights
        self._init_weights()
        
        # Always use FP32 for adapter computation
        self.to(dtype=torch.float32)
    
    def _init_weights(self):
        """Initialize weights with zero-conv pattern."""
        # LayerNorm: small gamma for gradient flow
        with torch.no_grad():
            self.layer_norm.weight.fill_(self.layer_norm_gamma)
            self.layer_norm.bias.zero_()
        
        # First linear: small random init
        nn.init.normal_(self.linear_1.weight, std=self.first_linear_std)
        nn.init.zeros_(self.linear_1.bias)
        
        # Second linear: zero-initialized (critical for baseline preservation)
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
    
    def forward(
        self,
        clip_embedding: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of CLIP adapter.
        
        Args:
            clip_embedding: CLIP text embeddings [batch_size, clip_embed_dim]
            time_embedding: Original time embeddings for RMS matching (optional)
        
        Returns:
            Adapted embedding to add to time embedding [batch_size, time_embed_dim]
        """
        # Store input dtype for final cast
        input_dtype = clip_embedding.dtype
        
        # Convert to FP32 for adapter computation
        clip_embedding = clip_embedding.to(dtype=torch.float32)
        
        # L2 normalize CLIP embeddings
        clip_embedding = F.normalize(clip_embedding, p=2, dim=-1)
        
        # Scale by sqrt(dim) to maintain magnitude
        clip_embedding = clip_embedding * (self.clip_embed_dim ** 0.5)
        
        # Forward through adapter
        hidden = self.layer_norm(clip_embedding)
        hidden = self.linear_1(hidden)
        hidden = self.activation(hidden)
        adapter_output = self.linear_2(hidden)
        
        # Optional RMS matching to time embedding
        if self.enable_rms_matching and time_embedding is not None:
            with torch.no_grad():
                # Compute RMS of time embedding
                time_rms = torch.sqrt(torch.mean(time_embedding ** 2, dim=-1, keepdim=True))
                # Compute RMS of adapter output
                adapter_rms = torch.sqrt(torch.mean(adapter_output ** 2, dim=-1, keepdim=True))
                # Scale factor to match RMS
                scale = (time_rms / (adapter_rms + 1e-8)).detach()
            adapter_output = adapter_output * scale
        
        # Apply scalar gate with sigmoid activation
        gate_value = torch.sigmoid(self.gate)
        adapter_output = adapter_output * gate_value
        
        # Cast back to input dtype if needed
        if adapter_output.dtype != input_dtype:
            adapter_output = adapter_output.to(dtype=input_dtype)
        
        return adapter_output
    
    def get_gate_value(self) -> float:
        """Get current gate value (after sigmoid)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate).item()
    
    def set_gate_value(self, value: float):
        """Set gate to achieve target value after sigmoid.
        
        Args:
            value: Target gate value after sigmoid (0 to 1)
        """
        with torch.no_grad():
            if value <= 0:
                # For zero or negative, set to a large negative number
                # sigmoid(-30) ≈ 9.4e-14 which is effectively 0
                self.gate.data.fill_(-30.0)
            elif value >= 1:
                # For 1 or greater, set to a large positive number  
                # sigmoid(30) ≈ 0.9999999999994
                self.gate.data.fill_(30.0)
            else:
                # Inverse sigmoid: logit = log(p / (1 - p))
                logit = torch.log(torch.tensor(value / (1 - value)))
                self.gate.data = logit
    
    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        clip_model: Optional[nn.Module] = None,
        clip_embed_dim: Optional[int] = None,
        **kwargs
    ) -> "ClipAdapter":
        """Create adapter with dimensions discovered from model architecture.
        
        Args:
            model: GLIDE UNet model to extract time_embed_dim from
            clip_model: CLIP model to extract clip_embed_dim from (optional)
            clip_embed_dim: Override for CLIP embedding dimension
            **kwargs: Additional arguments for ClipAdapter init
        
        Returns:
            ClipAdapter instance with discovered dimensions
        """
        # Discover time embedding dimension from model
        if hasattr(model, 'time_embed') and len(list(model.time_embed.children())) > 0:
            time_embed_layers = list(model.time_embed.children())
            # Last layer's output dimension is d_emb
            if hasattr(time_embed_layers[-1], 'out_features'):
                time_embed_dim = time_embed_layers[-1].out_features
            else:
                raise ValueError("Cannot determine time_embed_dim from model architecture")
        else:
            raise ValueError("Model does not have expected time_embed structure")
        
        # Discover CLIP embedding dimension
        if clip_embed_dim is None:
            if clip_model is not None:
                # For OpenAI CLIP model
                if hasattr(clip_model, 'text_projection'):
                    # OpenAI CLIP has text_projection as a parameter
                    if isinstance(clip_model.text_projection, nn.Parameter):
                        # text_projection shape is (transformer_width, embed_dim)
                        clip_embed_dim = clip_model.text_projection.shape[1]
                    elif hasattr(clip_model.text_projection, 'out_features'):
                        clip_embed_dim = clip_model.text_projection.out_features
                    else:
                        # Fallback: get from transformer width (before projection)
                        if hasattr(clip_model, 'transformer'):
                            # Note: This would be pre-projection dimension
                            # For ViT-B/32: transformer.width=512, projection output=512
                            clip_embed_dim = clip_model.transformer.width
                        else:
                            raise ValueError("Cannot determine CLIP embedding dimension from model")
                else:
                    raise ValueError("CLIP model does not have expected structure")
            else:
                # Default to standard OpenAI CLIP ViT-B/32 dimension
                clip_embed_dim = 512
        
        return cls(
            time_embed_dim=time_embed_dim,
            clip_embed_dim=clip_embed_dim,
            **kwargs
        )


def integrate_clip_adapter_to_model(
    model: nn.Module,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: Optional[int] = None,
    gate_init: float = -5.0,
    device: Union[str, torch.device] = "cuda",
) -> nn.Module:
    """Integrate CLIP adapter into GLIDE model.
    
    Args:
        model: GLIDE model to add adapter to
        clip_model_name: Name of CLIP model for dimension discovery
        hidden_dim: Hidden dimension for adapter (None for auto)
        gate_init: Initial gate value
        device: Device to load CLIP model on
        
    Returns:
        Model with integrated CLIP adapter
    """
    # Load CLIP model for dimension discovery
    clip_model, _ = load_openai_clip(clip_model_name, device=device)
    
    # Create adapter with runtime dimension discovery
    adapter = ClipAdapter.from_model(
        model, 
        clip_model=clip_model,
        hidden_dim=hidden_dim,
        gate_init=gate_init,
    )
    
    # Move adapter to same device as model
    adapter = adapter.to(device)
    
    # Add adapter to model
    model.clip_adapter = adapter
    
    # Clean up CLIP model (we only needed it for dimension discovery)
    del clip_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model


def load_openai_clip(
    model_name: str = "ViT-B/32",
    device: Union[str, torch.device] = "cpu"
) -> Tuple[nn.Module, callable]:
    """Load standard OpenAI CLIP model.
    
    IMPORTANT: This loads the standard OpenAI CLIP, NOT the noise-aware CLIP
    from glide_text2im. We use OpenAI's CLIP for cleaner conditioning.
    
    Args:
        model_name: CLIP model variant (default: "ViT-B/32")
        device: Device to load model on
    
    Returns:
        Tuple of (clip_model, tokenize_function)
    """
    clip_model, preprocess = clip.load(model_name, device=device)
    clip_model.eval()
    
    # Freeze CLIP model
    for param in clip_model.parameters():
        param.requires_grad = False
    
    return clip_model, clip.tokenize


def get_clip_text_features(
    clip_model: nn.Module,
    texts: Union[str, list[str]],
    device: Union[str, torch.device] = "cpu",
    normalize: bool = True
) -> torch.Tensor:
    """Extract text features from CLIP model.
    
    Args:
        clip_model: OpenAI CLIP model
        texts: Text or list of texts to encode
        device: Device for tokens
        normalize: Whether to L2-normalize features
    
    Returns:
        Text features [batch_size, 512] for ViT-B/32
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize
    tokens = clip.tokenize(texts, truncate=True).to(device)
    
    # Encode
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        
        # L2 normalize if requested (standard for CLIP similarity)
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
    
    return text_features


def integrate_clip_adapter(
    model: nn.Module,
    clip_adapter: ClipAdapter,
    replace: bool = False
) -> nn.Module:
    """Integrate CLIP adapter into UNet model.
    
    Args:
        model: GLIDE UNet model
        clip_adapter: ClipAdapter instance
        replace: Whether to replace existing adapter
    
    Returns:
        Model with integrated adapter
    """
    if hasattr(model, 'clip_adapter') and not replace:
        raise ValueError("Model already has clip_adapter. Set replace=True to override.")
    
    # Add adapter as model attribute
    model.clip_adapter = clip_adapter
    
    # Adapter will be used in forward pass modifications
    return model