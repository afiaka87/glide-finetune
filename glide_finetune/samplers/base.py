"""Base sampler interface and registry for diffusion models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch as th


class Sampler(ABC):
    """Base class for all diffusion samplers."""

    def __init__(
        self,
        diffusion,
        model_fn: Callable,
        shape: Tuple[int, ...],
        device: str = "cpu",
        clip_denoised: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.diffusion = diffusion
        self.model_fn = model_fn
        self.shape = shape
        self.device = device
        self.clip_denoised = clip_denoised
        self.model_kwargs = model_kwargs or {}

    @abstractmethod
    def sample(self, num_steps: int, **kwargs) -> th.Tensor:
        """Generate samples using this sampler.

        Args:
            num_steps: Number of sampling steps
            **kwargs: Additional sampler-specific arguments

        Returns:
            Generated samples tensor
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this sampler."""
        pass

    def _wrap_model(self, model_fn: Callable) -> Callable:
        """Wrap model function to ensure consistent behavior."""

        def wrapped_fn(x, t, **kwargs):
            out = model_fn(x, t, **kwargs)
            # Ensure we handle both eps and v-prediction parameterizations
            if isinstance(out, tuple):
                return out
            return out, None

        return wrapped_fn


class SamplerRegistry:
    """Registry for managing available samplers."""

    _samplers: Dict[str, Type[Sampler]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a new sampler."""

        def decorator(sampler_cls: Type[Sampler]) -> Type[Sampler]:
            if name in cls._samplers:
                raise ValueError(f"Sampler '{name}' already registered")
            cls._samplers[name] = sampler_cls
            return sampler_cls

        return decorator

    @classmethod
    def get_sampler(cls, name: str) -> Type[Sampler]:
        """Get a sampler class by name."""
        if name not in cls._samplers:
            available = list(cls._samplers.keys())
            msg = f"Unknown sampler '{name}'. Available samplers: {available}"
            raise ValueError(msg)
        return cls._samplers[name]

    @classmethod
    def list_samplers(cls) -> list[str]:
        """List all available sampler names."""
        return list(cls._samplers.keys())
