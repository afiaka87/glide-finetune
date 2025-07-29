"""PLMS (Pseudo Linear Multi-Step) sampler implementation."""

from typing import Callable, Optional

import torch as th

from .base import Sampler, SamplerRegistry


@SamplerRegistry.register("plms")
class PLMSSampler(Sampler):
    """PLMS sampler - uses the existing guided-diffusion implementation."""

    @property
    def name(self) -> str:
        return "plms"

    def sample(
        self,
        num_steps: int,
        cond_fn: Optional[Callable] = None,
        progress: bool = True,
        **kwargs,
    ) -> th.Tensor:
        """Sample using PLMS method."""
        # Use existing plms_sample_loop from guided-diffusion
        result = self.diffusion.plms_sample_loop(
            self.model_fn,
            self.shape,
            clip_denoised=self.clip_denoised,
            model_kwargs=self.model_kwargs,
            device=self.device,
            progress=progress,
            cond_fn=cond_fn,
        )
        return th.tensor(result)
