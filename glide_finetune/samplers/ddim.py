"""DDIM (Denoising Diffusion Implicit Models) sampler implementation."""

from typing import Callable, Optional

import torch as th

from .base import Sampler, SamplerRegistry


@SamplerRegistry.register("ddim")
class DDIMSampler(Sampler):
    """DDIM sampler - deterministic sampling."""

    @property
    def name(self) -> str:
        return "ddim"

    def sample(
        self,
        num_steps: int,
        eta: float = 0.0,
        progress: bool = True,
        cond_fn: Optional[Callable] = None,
        **kwargs,
    ) -> th.Tensor:
        """Sample using DDIM method.

        Args:
            num_steps: Number of sampling steps
            eta: DDIM eta parameter (0.0 for deterministic)
            progress: Show progress bar
            cond_fn: Optional conditioning function
        """
        # Use existing ddim_sample_loop from guided-diffusion
        result = self.diffusion.ddim_sample_loop(
            self.model_fn,
            self.shape,
            clip_denoised=self.clip_denoised,
            model_kwargs=self.model_kwargs,
            device=self.device,
            progress=progress,
            eta=eta,
            cond_fn=cond_fn,
        )
        return th.tensor(result)
