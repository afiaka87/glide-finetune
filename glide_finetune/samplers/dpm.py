"""DPM++ (Diffusion Probabilistic Model Plus Plus) sampler implementations using 
DDPM parameterization."""

import numpy as np
import torch as th
from tqdm import tqdm

from .base import Sampler, SamplerRegistry


@SamplerRegistry.register("dpm++_2m")
class DPMPlusPlusSampler(Sampler):
    """DPM++ 2M sampler using DDPM parameterization compatible with GLIDE."""

    @property
    def name(self) -> str:
        return "dpm++_2m"

    def sample(
        self, num_steps: int, eta: float = 0.0, progress: bool = True, **kwargs
    ) -> th.Tensor:
        """Sample using DPM++ 2M with DDPM parameterization.

        This uses the DDPM forward process parameterization:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        And implements a second-order multistep update.
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

        # DPM++ 2M specific - store previous clean image predictions for 
        # second-order updates
        previous_clean_image = None
        previous_timestep = None

        # Progress bar
        step_indices = list(range(len(timesteps)))
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

            if step_idx == 0:
                # First step - just store for multistep
                previous_clean_image = predicted_clean_image
                previous_timestep = current_timestep
            else:
                # Multistep update
                if step_idx < len(timesteps) - 1:
                    # Get next noise level
                    next_timestep = timesteps[step_idx + 1]
                    next_noise_level = alphas_cumprod[next_timestep]

                    # Compute timestep sizes for DPM++ formula
                    # We use log(noise_level) as our "time" variable for 
                    # numerical stability
                    log_snr_current = -0.5 * np.log(
                        current_noise_level
                    )  # log signal-to-noise ratio
                    log_snr_previous = -0.5 * np.log(alphas_cumprod[previous_timestep])
                    log_snr_next = -0.5 * np.log(next_noise_level)

                    step_size = log_snr_next - log_snr_current
                    previous_step_size = log_snr_current - log_snr_previous

                    if (
                        previous_clean_image is not None
                        and abs(previous_step_size) > 1e-8
                    ):
                        # Second-order update
                        # Compute the derivative estimate (rate of change of 
                        # clean image)
                        clean_image_derivative = (
                            predicted_clean_image - previous_clean_image
                        ) / previous_step_size

                        # Second-order DPM++ formula: extrapolate using derivative
                        extrapolated_clean_image = (
                            predicted_clean_image
                            + 0.5 * step_size * clean_image_derivative
                        )
                    else:
                        # First-order fallback
                        extrapolated_clean_image = predicted_clean_image

                    # DDIM-style update to next timestep
                    # Formula: next_image = sqrt(next_noise_level) * clean_image + 
                    # sqrt(1 - next_noise_level) * noise_direction
                    noise_amount = (
                        eta
                        * np.sqrt((1 - next_noise_level) / (1 - current_noise_level))
                        * np.sqrt(1 - current_noise_level / next_noise_level)
                    )

                    # Compute the direction pointing from noisy to clean
                    denoising_direction = (
                        noisy_image
                        - np.sqrt(current_noise_level) * predicted_clean_image
                    ) / np.sqrt(1 - current_noise_level)

                    # Take a step towards the extrapolated clean image
                    noisy_image = (
                        np.sqrt(next_noise_level) * extrapolated_clean_image
                        + np.sqrt(1 - next_noise_level - noise_amount**2)
                        * denoising_direction
                    )

                    # Add noise if eta > 0 (stochastic sampling)
                    if eta > 0:
                        random_noise = th.randn_like(noisy_image)
                        noisy_image = noisy_image + noise_amount * random_noise
                else:
                    # Final step - use the denoised prediction
                    noisy_image = predicted_clean_image

                # Update previous values for next iteration
                previous_clean_image = predicted_clean_image
                previous_timestep = current_timestep

        return noisy_image


@SamplerRegistry.register("dpm++_2m_karras")
class DPMPlusPlus2MKarrasSampler(DPMPlusPlusSampler):
    """DPM++ 2M with Karras schedule using DDPM parameterization.

    Note: For GLIDE's DDPM parameterization, the Karras schedule
    doesn't apply in the same way as k-diffusion, so this is
    equivalent to the base DPM++ 2M sampler.
    """

    @property
    def name(self) -> str:
        return "dpm++_2m_karras"
