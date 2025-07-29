"""Euler and Euler Ancestral sampler implementations."""

import torch as th

from .base import Sampler, SamplerRegistry
from .util import (
    get_glide_cosine_schedule,
    scale_model_input,
    sigma_to_timestep,
    predicted_noise_to_denoised,
    get_glide_schedule_timesteps,
)


@SamplerRegistry.register("euler")
class EulerSampler(Sampler):
    """Euler sampler - simple and fast first-order ODE solver."""
    
    @property
    def name(self) -> str:
        return "euler"
    
    def sample(
        self,
        num_steps: int,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        progress: bool = True,
        use_karras: bool = False,
        **kwargs
    ) -> th.Tensor:
        """Sample using Euler method with proper GLIDE schedule.
        
        Args:
            num_steps: Number of sampling steps
            s_churn: Amount of stochasticity (0 for deterministic)
            s_tmin: Minimum timestep for adding noise
            s_tmax: Maximum timestep for adding noise  
            s_noise: Noise scale factor
            progress: Show progress bar
            use_karras: Use Karras sigma schedule
        """
        import numpy as np
        from tqdm import tqdm
        
        # Get GLIDE's cosine schedule
        betas, alphas_cumprod, sigmas = get_glide_cosine_schedule(self.diffusion.num_timesteps)
        
        # Get timesteps and sigmas for sampling
        timesteps, sampling_sigmas = get_glide_schedule_timesteps(
            num_steps, self.diffusion.num_timesteps, sigmas, use_karras=use_karras
        )
        
        # Start from pure noise
        current_sample = th.randn(self.shape, device=self.device)
        
        # Progress bar setup
        iterator = tqdm(range(num_steps)) if progress else range(num_steps)
        
        for step_idx in iterator:
            timestep = timesteps[step_idx]
            sigma = sampling_sigmas[step_idx]
            
            # Scale model input
            scaled_sample = scale_model_input(current_sample, sigma)
            
            # Get timestep tensor
            batch_timesteps = th.full(
                (self.shape[0],), timestep, device=self.device, dtype=th.long
            )
            
            # Get model prediction
            with th.no_grad():
                model_output = self.model_fn(scaled_sample, batch_timesteps, **self.model_kwargs)
                if isinstance(model_output, tuple):
                    predicted_noise = model_output[0][:, :3]
                else:
                    predicted_noise = model_output[:, :3]
            
            # Convert to denoised prediction
            alpha_prod = alphas_cumprod[timestep]
            denoised = predicted_noise_to_denoised(
                scaled_sample, predicted_noise, sigma, alpha_prod, self.clip_denoised
            )
            
            # Get next sigma
            if step_idx < num_steps - 1:
                next_sigma = sampling_sigmas[step_idx + 1]
            else:
                next_sigma = 0.0
            
            # Add noise for stochasticity if needed
            if s_churn > 0 and s_tmin <= sigma <= s_tmax:
                gamma = min(s_churn / num_steps, np.sqrt(2.0) - 1)
                noise_factor = sigma * np.sqrt(1 + gamma**2)
                noise = th.randn_like(current_sample) * s_noise
                current_sample = current_sample + noise * (noise_factor - sigma)
                sigma = noise_factor
            
            # Euler step
            # d = (x - denoised) / sigma
            # x_next = x + d * (sigma_next - sigma)
            d = (current_sample - denoised) / sigma if sigma > 0 else 0
            dt = next_sigma - sigma
            current_sample = current_sample + d * dt
            
            # Free memory
            del scaled_sample, predicted_noise, denoised, d
            
        return current_sample


@SamplerRegistry.register("euler_a")
class EulerAncestralSampler(Sampler):
    """Euler Ancestral sampler - adds noise at each step for more variation."""
    
    @property 
    def name(self) -> str:
        return "euler_a"
    
    def sample(
        self,
        num_steps: int,
        eta: float = 1.0,
        progress: bool = True,
        use_karras: bool = False,
        **kwargs
    ) -> th.Tensor:
        """Sample using Euler Ancestral method with proper GLIDE schedule.
        
        Args:
            num_steps: Number of sampling steps
            eta: Amount of noise to add at each step (1.0 is default)
            progress: Show progress bar
            use_karras: Use Karras sigma schedule
        """
        import numpy as np
        from tqdm import tqdm
        
        # Get GLIDE's cosine schedule
        betas, alphas_cumprod, sigmas = get_glide_cosine_schedule(self.diffusion.num_timesteps)
        
        # Get timesteps and sigmas for sampling
        timesteps, sampling_sigmas = get_glide_schedule_timesteps(
            num_steps, self.diffusion.num_timesteps, sigmas, use_karras=use_karras
        )
        
        # Start from pure noise
        current_sample = th.randn(self.shape, device=self.device)
        
        # Progress bar setup
        iterator = tqdm(range(num_steps)) if progress else range(num_steps)
        
        for step_idx in iterator:
            timestep = timesteps[step_idx]
            sigma = sampling_sigmas[step_idx]
            
            # Scale model input
            scaled_sample = scale_model_input(current_sample, sigma)
            
            # Get timestep tensor
            batch_timesteps = th.full(
                (self.shape[0],), timestep, device=self.device, dtype=th.long
            )
            
            # Get model prediction
            with th.no_grad():
                model_output = self.model_fn(scaled_sample, batch_timesteps, **self.model_kwargs)
                if isinstance(model_output, tuple):
                    predicted_noise = model_output[0][:, :3]
                else:
                    predicted_noise = model_output[:, :3]
            
            # Convert to denoised prediction
            alpha_prod = alphas_cumprod[timestep]
            denoised = predicted_noise_to_denoised(
                scaled_sample, predicted_noise, sigma, alpha_prod, self.clip_denoised
            )
            
            # Ancestral sampling step
            if step_idx < num_steps - 1:
                # Get next sigma
                next_sigma = sampling_sigmas[step_idx + 1]
                
                # Calculate step sizes for ancestral sampling
                # sigma_up = sqrt(sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2)
                # sigma_down = sqrt(sigma_next^2 - sigma_up^2)
                if sigma > 0 and next_sigma < sigma:
                    sigma_up = np.sqrt(next_sigma**2 * (sigma**2 - next_sigma**2) / sigma**2)
                    sigma_down = np.sqrt(next_sigma**2 - sigma_up**2)
                else:
                    sigma_up = 0
                    sigma_down = next_sigma
                
                # Euler step with reduced noise
                d = (current_sample - denoised) / sigma if sigma > 0 else 0
                dt = sigma_down - sigma
                current_sample = current_sample + d * dt
                
                # Add noise back (ancestral part)
                if eta > 0 and sigma_up > 0:
                    noise = th.randn_like(current_sample)
                    current_sample = current_sample + noise * sigma_up * eta
            else:
                # Final step - deterministic
                current_sample = denoised
            
            # Free memory
            del scaled_sample, predicted_noise, denoised
            
        return current_sample