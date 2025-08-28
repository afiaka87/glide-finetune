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
        accelerator: Any = None,
    ):
        """Initialize CLIP feature computer.
        
        Args:
            clip_model_name: Name of CLIP model to load
            device: Device to run CLIP model on
            cache_features: Whether to cache computed features (not recommended for distributed training)
            accelerator: Optional Accelerator instance for distributed sync
        """
        self.clip_model_name = clip_model_name
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.cache_features = cache_features
        self.feature_cache = {} if cache_features else None
        self.accelerator = accelerator
        
        # Load CLIP model once on initialization (with distributed sync if needed)
        if accelerator and accelerator.is_main_process:
            logger.info(f"Loading CLIP model: {clip_model_name} on {self.device}")
        
        self.clip_model, self.clip_preprocess = load_openai_clip(
            clip_model_name, 
            device=str(self.device),
            accelerator=accelerator
        )
        self.clip_model.eval()
        
        # Log feature dimension for verification
        logger.info(f"CLIP model loaded on {self.device}, feature dimension: {self.get_feature_dim()}")
        
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


class CLIPComputerManager:
    """Manages CLIP computer instances per device without globals.
    
    This class provides a functional approach to managing CLIP models,
    where each training strategy owns its manager instance.
    """
    
    def __init__(self):
        """Initialize the manager with an empty cache."""
        self._computers: dict[str, CLIPFeatureComputer] = {}
    
    def get_computer(
        self,
        clip_model_name: str = "ViT-B/32",
        device: str | torch.device = "cuda",
        reset: bool = False,
        accelerator: Any = None,
    ) -> CLIPFeatureComputer:
        """Get or create a CLIP computer for the specified device.
        
        Args:
            clip_model_name: Name of CLIP model to use
            device: Device to run on
            reset: If True, create a new instance even if one exists
            accelerator: Optional Accelerator instance for distributed sync
            
        Returns:
            CLIPFeatureComputer instance for this device
        """
        # Create unique key for this model/device combination
        device_str = str(device)
        key = f"{clip_model_name}_{device_str}"
        
        if reset and key in self._computers:
            # Clean up old instance
            old_instance = self._computers.pop(key)
            del old_instance
            if 'cuda' in device_str:
                torch.cuda.empty_cache()
        
        if key not in self._computers:
            # Create new instance for this device
            self._computers[key] = CLIPFeatureComputer(
                clip_model_name=clip_model_name,
                device=device,
                cache_features=False,  # Disable caching in distributed training
                accelerator=accelerator,
            )
            if not accelerator or accelerator.is_main_process:
                logger.info(f"Created CLIP computer for {key}")
        
        return self._computers[key]
    
    def cleanup(self):
        """Clean up all CLIP computer instances."""
        for computer in self._computers.values():
            del computer
        self._computers.clear()
        torch.cuda.empty_cache()


# Backward compatibility: create a default manager instance
_default_manager = CLIPComputerManager()

def get_clip_computer(
    clip_model_name: str = "ViT-B/32",
    device: str | torch.device = "cuda",
    reset: bool = False,
    accelerator: Any = None,
) -> CLIPFeatureComputer:
    """Backward-compatible function using default manager.
    
    For new code, prefer creating your own CLIPComputerManager instance.
    
    Args:
        clip_model_name: Name of CLIP model to use
        device: Device to run on
        reset: If True, create a new instance even if one exists
        accelerator: Optional Accelerator instance for distributed sync
    """
    return _default_manager.get_computer(clip_model_name, device, reset, accelerator)