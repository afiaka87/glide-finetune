"""
Runtime CLIP feature computation for training.

Computes CLIP features on-the-fly when precomputed features are not available.
"""

from typing import Any

import torch
import torch.nn as nn

from glide_finetune.clip_adapter import load_openai_clip, get_clip_text_features
from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.clip_compute")


class CLIPFeatureComputer:
    """Computes CLIP features on-the-fly during training.
    
    This is used as a fallback when precomputed CLIP features are not available.
    The CLIP model is loaded once and cached for the entire training session.
    """
    
    def __init__(
        self, 
        clip_model_name: str = "ViT-B/32",
        device: str | torch.device = "cuda",
        cache_features: bool = False,
    ):
        """Initialize CLIP feature computer.
        
        Args:
            clip_model_name: Name of CLIP model to load
            device: Device to run CLIP model on
            cache_features: Whether to cache computed features (not recommended for training)
        """
        self.clip_model_name = clip_model_name
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.cache_features = cache_features
        self.feature_cache = {} if cache_features else None
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model, self.clip_preprocess = load_openai_clip(
            clip_model_name, 
            device=str(self.device)
        )
        self.clip_model.eval()
        
        # Get expected feature dimension
        with torch.no_grad():
            dummy_text = ["test"]
            dummy_features = get_clip_text_features(
                self.clip_model, 
                dummy_text, 
                device=str(self.device)
            )
            self.clip_dim = dummy_features.shape[-1]
        
        logger.info(f"CLIP model loaded, feature dimension: {self.clip_dim}")
    
    def compute_text_features(
        self, 
        texts: list[str],
        return_on_device: bool = True,
    ) -> torch.Tensor:
        """Compute CLIP features for a batch of text.
        
        Args:
            texts: List of text strings to encode
            return_on_device: If True, return features on the model's device
            
        Returns:
            CLIP feature tensor of shape (batch_size, clip_dim)
        """
        if not texts:
            # Return zero features for empty input
            return torch.zeros(0, self.clip_dim, device=self.device if return_on_device else "cpu")
        
        # Check cache if enabled
        if self.cache_features:
            cache_key = tuple(texts)
            if cache_key in self.feature_cache:
                features = self.feature_cache[cache_key].clone()
                if return_on_device and features.device != self.device:
                    features = features.to(self.device)
                return features
        
        # Compute features
        with torch.no_grad():
            features = get_clip_text_features(
                self.clip_model,
                texts,
                device=str(self.device)
            )
        
        # Cache if enabled
        if self.cache_features:
            self.feature_cache[cache_key] = features.cpu().clone()
        
        if not return_on_device:
            features = features.cpu()
        
        return features
    
    def compute_from_tokens(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor | None = None,
        tokenizer: Any | None = None,
    ) -> torch.Tensor | None:
        """Compute CLIP features from BPE tokens.
        
        This requires decoding the tokens back to text, then encoding with CLIP.
        
        Args:
            tokens: BPE token tensor from GLIDE tokenizer
            masks: Mask tensor indicating valid tokens
            tokenizer: GLIDE tokenizer to decode tokens
            
        Returns:
            CLIP features or None if tokenizer not provided
        """
        if tokenizer is None:
            logger.warning("No tokenizer provided, cannot decode tokens to text")
            return None
        
        batch_size = tokens.shape[0]
        texts = []
        
        for i in range(batch_size):
            # Get valid tokens for this sample
            if masks is not None:
                valid_length = masks[i].sum().item()
                valid_tokens = tokens[i, :valid_length]
            else:
                # Assume all non-zero tokens are valid
                valid_tokens = tokens[i][tokens[i] != 0]
            
            # Decode to text
            try:
                text = tokenizer.decode(valid_tokens.cpu().numpy().tolist())
                # Clean up the text
                text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()
                texts.append(text)
            except Exception as e:
                logger.warning(f"Failed to decode tokens for sample {i}: {e}")
                texts.append("")  # Use empty string as fallback
        
        # Compute CLIP features
        return self.compute_text_features(texts)
    
    def __del__(self):
        """Clean up CLIP model from memory."""
        if hasattr(self, "clip_model"):
            del self.clip_model
        if hasattr(self, "feature_cache"):
            del self.feature_cache


# Global instance for reuse across training
_clip_computer_instance = None


def get_clip_computer(
    clip_model_name: str = "ViT-B/32",
    device: str | torch.device = "cuda",
    reset: bool = False,
) -> CLIPFeatureComputer:
    """Get or create a global CLIP feature computer instance.
    
    Args:
        clip_model_name: Name of CLIP model to use
        device: Device to run on
        reset: If True, create a new instance even if one exists
        
    Returns:
        CLIPFeatureComputer instance
    """
    global _clip_computer_instance
    
    if reset or _clip_computer_instance is None:
        if _clip_computer_instance is not None:
            del _clip_computer_instance
        _clip_computer_instance = CLIPFeatureComputer(
            clip_model_name=clip_model_name,
            device=device,
            cache_features=False,  # Don't cache during training
        )
    
    return _clip_computer_instance