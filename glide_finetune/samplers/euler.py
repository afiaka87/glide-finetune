"""Euler and Euler Ancestral sampler implementations using DDPM parameterization."""

from typing import List, Union

import numpy as np
import torch as th
from tqdm import tqdm

from .base import Sampler, SamplerRegistry


@SamplerRegistry.register("euler")
class EulerSampler(Sampler):
    """Euler sampler using DDPM parameterization compatible with GLIDE."""

    @property
    def name(self) -> str:
        return "euler"

    def sample(
        self, num_steps: int, eta: float = 0.0, progress: bool = True, **kwargs
    ) -> th.Tensor:
        """Sample using Euler method with DDPM parameterization.

        This uses the DDPM forward process parameterization:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        And solves the reverse process using Euler steps on the
        probability flow ODE.
        """
        # Get the respaced timesteps
        if hasattr(self.diffusion, "timestep_map"):
            # For SpacedDiffusion, map to original timesteps
            respaced_indices = np.linspace(
                self.diffusion.num_timesteps - 1, 0, num_steps, dtype=np.int64
            )
            timesteps = np.array(
                [self.diffusion.timestep_map[i] for i in respaced_indices]
            )
        else:
            # No respacing
            timesteps = np.linspace(
                self.diffusion.num_timesteps - 1, 0, num_steps, dtype=np.int64
            )

        # Get alpha values - need to handle respaced diffusion
        if hasattr(self.diffusion, "timestep_map"):
            # For respaced diffusion, we need the original alphas
            # We'll compute them from the cosine schedule
            from .util import get_glide_cosine_schedule

            original_steps = self.diffusion.timestep_map[-1] + 1
            _, alphas_cumprod, _ = get_glide_cosine_schedule(original_steps)
        else:
            # Use diffusion's alphas directly
            alphas_cumprod = self.diffusion.alphas_cumprod
            if isinstance(alphas_cumprod, th.Tensor):
                alphas_cumprod = alphas_cumprod.cpu().numpy()

        # Start from pure noise
        noisy_image = th.randn(self.shape, device=self.device)

        # Progress bar
        step_indices: Union[List[int], tqdm[int]] = list(range(len(timesteps)))
        if progress:
            step_indices = tqdm(step_indices)

        for step_idx in step_indices:
            current_timestep = timesteps[step_idx]
            timestep_batch = th.full(
                (self.shape[0],), current_timestep, device=self.device, dtype=th.long
            )

            # Get model prediction (predicted noise)
            with th.no_grad():
                model_output = self.model_fn(
                    noisy_image, timestep_batch, **self.model_kwargs
                )
                if isinstance(model_output, tuple):
                    predicted_noise = model_output[0][:, :3]
                else:
                    predicted_noise = model_output[:, :3]

            # Current noise level (cumulative product of alphas)
            current_noise_level = alphas_cumprod[current_timestep]

            # Predict the clean image from the noisy image and predicted noise
            # Formula: clean_image = (noisy_image - sqrt(1 - noise_level) *
            # predicted_noise) / sqrt(noise_level)
            predicted_clean_image = (
                noisy_image - np.sqrt(1 - current_noise_level) * predicted_noise
            ) / np.sqrt(current_noise_level)

            if self.clip_denoised:
                predicted_clean_image = predicted_clean_image.clamp(-1, 1)

            # Get next noise level (or 1.0 for final step meaning no noise)
            if step_idx < len(timesteps) - 1:
                next_timestep = timesteps[step_idx + 1]
                next_noise_level = alphas_cumprod[next_timestep]
            else:
                next_noise_level = 1.0

            # DDIM update (deterministic when eta=0)
            # This is essentially an Euler step on the probability flow ODE
            noise_amount = (
                eta
                * np.sqrt((1 - next_noise_level) / (1 - current_noise_level))
                * np.sqrt(1 - current_noise_level / next_noise_level)
            )

            # Compute the direction pointing from noisy to clean
            denoising_direction = (
                noisy_image - np.sqrt(current_noise_level) * predicted_clean_image
            ) / np.sqrt(1 - current_noise_level)

            # Take a step towards the clean image
            noisy_image = (
                np.sqrt(next_noise_level) * predicted_clean_image
                + np.sqrt(1 - next_noise_level - noise_amount**2) * denoising_direction
            )

            # Add noise if eta > 0 (stochastic sampling)
            if eta > 0 and step_idx < len(timesteps) - 1:
                random_noise = th.randn_like(noisy_image)
                noisy_image = noisy_image + noise_amount * random_noise

        return noisy_image


@SamplerRegistry.register("euler_a")
class EulerAncestralSampler(Sampler):
    """Euler Ancestral sampler using DDPM parameterization."""

    @property
    def name(self) -> str:
        return "euler_a"

    def sample(
        self, num_steps: int, eta: float = 1.0, progress: bool = True, **kwargs
    ) -> th.Tensor:
        """Sample using Euler Ancestral with DDPM parameterization.

        This is similar to Euler but always adds noise (eta=1.0 by default).
        """
        # Just use the Euler sampler with eta=1.0
        euler_sampler = EulerSampler(
            diffusion=self.diffusion,
            model_fn=self.model_fn,
            shape=self.shape,
            device=self.device,
            clip_denoised=self.clip_denoised,
            model_kwargs=self.model_kwargs,
        )
        return euler_sampler.sample(
            num_steps=num_steps, eta=eta, progress=progress, **kwargs
        )
