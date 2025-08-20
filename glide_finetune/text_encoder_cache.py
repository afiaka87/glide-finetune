"""
Text encoder caching for frozen transformer mode.

When the transformer is frozen, text embeddings don't change during training.
This module provides caching to avoid redundant forward passes.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import hashlib
import logging


class TextEncoderCache:
    """
    Cache for text encoder outputs when transformer is frozen.
    
    This cache stores the outputs of the text encoder to avoid
    redundant forward passes when the same tokens/masks are seen again.
    Particularly useful for repeated validation prompts or common training captions.
    """
    
    def __init__(self, max_cache_size: int = 1000, device: str = "cuda"):
        """
        Initialize text encoder cache.
        
        Args:
            max_cache_size: Maximum number of cached embeddings
            device: Device to store cached tensors on
        """
        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.max_cache_size = max_cache_size
        self.device = device
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger(__name__)
        
    def _compute_cache_key(self, tokens: torch.Tensor, mask: torch.Tensor) -> str:
        """
        Compute a cache key from tokens and mask tensors.
        
        Args:
            tokens: Token tensor [batch_size, seq_len]
            mask: Mask tensor [batch_size, seq_len]
            
        Returns:
            Hash string for cache lookup
        """
        # Convert tensors to bytes and hash
        tokens_bytes = tokens.cpu().numpy().tobytes()
        mask_bytes = mask.cpu().numpy().tobytes()
        
        hasher = hashlib.sha256()
        hasher.update(tokens_bytes)
        hasher.update(mask_bytes)
        
        return hasher.hexdigest()
    
    def get(self, tokens: torch.Tensor, mask: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get cached text embeddings if available.
        
        Args:
            tokens: Token tensor
            mask: Mask tensor
            
        Returns:
            Cached embeddings dict or None if not found
        """
        # Handle batch dimension
        if tokens.dim() == 2 and tokens.shape[0] == 1:
            # Single sample - use caching
            key = self._compute_cache_key(tokens, mask)
            
            if key in self.cache:
                self.hits += 1
                # Return copies to avoid in-place modifications affecting cache
                return {
                    k: v.clone() if v is not None else None 
                    for k, v in self.cache[key].items()
                }
            else:
                self.misses += 1
                return None
        else:
            # Batch processing - check each sample
            batch_size = tokens.shape[0]
            batch_results = []
            
            for i in range(batch_size):
                token_slice = tokens[i:i+1]
                mask_slice = mask[i:i+1]
                key = self._compute_cache_key(token_slice, mask_slice)
                
                if key in self.cache:
                    self.hits += 1
                    batch_results.append(self.cache[key])
                else:
                    self.misses += 1
                    return None  # If any sample misses, compute all
            
            # Combine batch results
            if batch_results:
                combined = {}
                for key in batch_results[0].keys():
                    if batch_results[0][key] is not None:
                        combined[key] = torch.cat([
                            result[key] for result in batch_results
                        ], dim=0)
                    else:
                        combined[key] = None
                return combined
            
        return None
    
    def put(self, tokens: torch.Tensor, mask: torch.Tensor, 
            embeddings: Dict[str, torch.Tensor]) -> None:
        """
        Store text embeddings in cache.
        
        Args:
            tokens: Token tensor
            mask: Mask tensor
            embeddings: Dict of embedding tensors to cache
        """
        # Handle batch dimension
        if tokens.dim() == 2:
            batch_size = tokens.shape[0]
            
            for i in range(batch_size):
                token_slice = tokens[i:i+1]
                mask_slice = mask[i:i+1]
                key = self._compute_cache_key(token_slice, mask_slice)
                
                # Extract single sample from embeddings
                single_embeddings = {}
                for k, v in embeddings.items():
                    if v is not None:
                        single_embeddings[k] = v[i:i+1].clone().detach()
                    else:
                        single_embeddings[k] = None
                
                # Evict oldest if cache is full
                if len(self.cache) >= self.max_cache_size and key not in self.cache:
                    # Simple FIFO eviction
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                self.cache[key] = single_embeddings
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        self.logger.info(
            f"TextEncoderCache: size={stats['cache_size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.2%}"
        )


def cached_text_encoder_forward(
    model: nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    cache: Optional[TextEncoderCache] = None
) -> Dict[str, torch.Tensor]:
    """
    Forward pass through text encoder with optional caching.
    
    Args:
        model: The GLIDE model with text encoder
        tokens: Token tensor
        mask: Mask tensor
        cache: Optional cache instance
        
    Returns:
        Dict with text encoder outputs
    """
    # Check cache first if provided
    if cache is not None:
        cached_result = cache.get(tokens, mask)
        if cached_result is not None:
            return cached_result
    
    # Compute embeddings
    with torch.no_grad():
        text_outputs = model.get_text_emb(tokens, mask)
        
        # Detach outputs
        detached_outputs = {}
        for key, value in text_outputs.items():
            if value is not None:
                detached_outputs[key] = value.detach()
            else:
                detached_outputs[key] = None
    
    # Store in cache if provided
    if cache is not None:
        cache.put(tokens, mask, detached_outputs)
    
    return detached_outputs


class CachedTextEncoder(nn.Module):
    """
    Wrapper for GLIDE model that adds text encoder caching.
    """
    
    def __init__(self, model: nn.Module, cache_size: int = 1000):
        """
        Initialize cached text encoder wrapper.
        
        Args:
            model: GLIDE model to wrap
            cache_size: Maximum cache size
        """
        super().__init__()
        self.model = model
        self.cache = TextEncoderCache(max_cache_size=cache_size)
        
    def forward(self, x_t, timesteps, tokens=None, mask=None):
        """
        Forward pass with cached text encoding.
        """
        if tokens is not None and mask is not None:
            # Get cached or compute text embeddings
            text_outputs = cached_text_encoder_forward(
                self.model, tokens, mask, self.cache
            )
            
            # Use the cached embeddings in forward pass
            # This would need integration with the model's forward method
            # For now, just pass through
            return self.model(x_t, timesteps, tokens=tokens, mask=mask)
        else:
            return self.model(x_t, timesteps, tokens=tokens, mask=mask)
    
    def get_text_emb(self, tokens, mask):
        """Get text embeddings with caching."""
        return cached_text_encoder_forward(
            self.model, tokens, mask, self.cache
        )
    
    def clear_cache(self):
        """Clear the text encoder cache."""
        self.cache.clear()
    
    def log_cache_stats(self):
        """Log cache statistics."""
        self.cache.log_stats()


def test_text_encoder_cache():
    """Test text encoder cache functionality."""
    print("Testing Text Encoder Cache...")
    
    # Create dummy tokens and masks
    tokens1 = torch.randint(0, 1000, (1, 128))
    mask1 = torch.ones_like(tokens1, dtype=torch.bool)
    
    tokens2 = torch.randint(0, 1000, (1, 128))
    mask2 = torch.ones_like(tokens2, dtype=torch.bool)
    
    # Create cache
    cache = TextEncoderCache(max_cache_size=10)
    
    # Test cache miss
    result = cache.get(tokens1, mask1)
    assert result is None, "Should be cache miss"
    
    # Store in cache
    embeddings = {
        'xf_proj': torch.randn(1, 512),
        'xf_out': torch.randn(1, 128, 2048)
    }
    cache.put(tokens1, mask1, embeddings)
    
    # Test cache hit
    result = cache.get(tokens1, mask1)
    assert result is not None, "Should be cache hit"
    assert torch.allclose(result['xf_proj'], embeddings['xf_proj'])
    
    # Test different tokens
    result = cache.get(tokens2, mask2)
    assert result is None, "Should be cache miss for different tokens"
    
    # Test batch processing
    batch_tokens = torch.cat([tokens1, tokens2], dim=0)
    batch_mask = torch.cat([mask1, mask2], dim=0)
    
    # Store batch
    batch_embeddings = {
        'xf_proj': torch.randn(2, 512),
        'xf_out': torch.randn(2, 128, 2048)
    }
    cache.put(batch_tokens, batch_mask, batch_embeddings)
    
    # Check stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    print("✅ Text Encoder Cache test passed!")


if __name__ == "__main__":
    test_text_encoder_cache()