"""DPM++ (Diffusion Probabilistic Model Plus Plus) sampler implementations."""

import torch as th

from .base import Sampler, SamplerRegistry


@SamplerRegistry.register("dpm++_2m")
class DPMPlusPlusSampler(Sampler):
    """DPM++ 2M sampler - second-order multistep solver."""
    
    @property
    def name(self) -> str:
        return "dpm++_2m"
    
    def sample(
        self,
        num_steps: int,
        use_karras_sigmas: bool = True,
        progress: bool = True,
        **kwargs
    ) -> th.Tensor:
        """Sample using DPM++ 2M method.
        
        Args:
            num_steps: Number of sampling steps
            use_karras_sigmas: Use Karras sigma schedule for better quality
            progress: Show progress bar
        """
        # Pre-compute noise schedule on CPU to save memory
        import numpy as np
        betas_np = self.diffusion.betas
        alphas_np = 1.0 - betas_np
        alphas_cumprod_np = np.cumprod(alphas_np)
        
        # Calculate sigmas (noise levels) from alphas
        sigmas_np = np.sqrt((1 - alphas_cumprod_np) / alphas_cumprod_np)
        
        # Create timestep schedule
        if use_karras_sigmas:
            # Karras sigma schedule - better spacing for fewer steps
            sigma_min = float(sigmas_np[-1])
            sigma_max = float(sigmas_np[0])
            
            rho = 7.0  # Karras paper default
            ramp = np.linspace(0, 1, num_steps)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            karras_sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            
            # Map Karras sigmas to timesteps
            timesteps = []
            for karras_sigma in karras_sigmas:
                # Find closest timestep
                distances = np.abs(sigmas_np - karras_sigma)
                timestep = int(distances.argmin())
                timesteps.append(timestep)
            timesteps = th.tensor(timesteps, dtype=th.long)
        else:
            # Linear timestep schedule on CPU
            timesteps = th.linspace(
                self.diffusion.num_timesteps - 1, 0, num_steps, 
                dtype=th.long
            )
        
        # Start from pure noise
        current_sample = th.randn(self.shape, device=self.device)
        
        # DPM++ 2M specific - store previous denoised predictions
        old_denoised = None
        
        # Progress bar setup
        from tqdm import tqdm
        iterator = tqdm(enumerate(timesteps)) if progress else enumerate(timesteps)
        
        for step_idx, timestep in iterator:
            timestep_int = int(timestep.item())
            batch_timesteps = th.full(
                (self.shape[0],), timestep_int, device=self.device, dtype=th.long
            )
            
            # Get model prediction
            with th.no_grad():
                model_output = self.model_fn(current_sample, batch_timesteps, **self.model_kwargs)
                if isinstance(model_output, tuple):
                    predicted_noise = model_output[0][:, :3]
                else:
                    predicted_noise = model_output[:, :3]
            
            # Current values as scalars
            current_sigma = float(sigmas_np[timestep_int])
            current_alpha_prod = float(alphas_cumprod_np[timestep_int])
            sqrt_alpha_prod = np.sqrt(current_alpha_prod)
            
            # Convert to denoised prediction
            denoised_sample = (current_sample - current_sigma * predicted_noise) / sqrt_alpha_prod
            
            if self.clip_denoised:
                denoised_sample = denoised_sample.clamp(-1, 1)
            
            # DPM++ 2M update step
            if step_idx == 0:
                # First step - can't do multistep yet, just store
                old_denoised = denoised_sample
            else:
                # Multistep update using previous and current denoised predictions
                if step_idx < len(timesteps) - 1:
                    next_timestep_int = int(timesteps[step_idx + 1].item())
                    next_sigma = float(sigmas_np[next_timestep_int])
                    next_alpha_prod = float(alphas_cumprod_np[next_timestep_int])
                    sqrt_next_alpha_prod = np.sqrt(next_alpha_prod)
                    
                    # Second-order update
                    h = next_sigma - current_sigma  # Step size
                    
                    # Linear multistep coefficients
                    # These come from the DPM++ paper's second-order formula
                    if old_denoised is not None:
                        # Use both current and previous denoised estimates
                        denoised_prime = (3 * denoised_sample - old_denoised) / 2
                    else:
                        # Fall back to first-order
                        denoised_prime = denoised_sample
                    
                    # Update sample
                    current_sample = (
                        sqrt_next_alpha_prod * denoised_prime + 
                        next_sigma * predicted_noise
                    )
                    
                    # Free memory
                    del denoised_prime
                else:
                    # Final step
                    current_sample = denoised_sample
                
                # Store for next iteration and free old memory
                if old_denoised is not None:
                    del old_denoised
                old_denoised = denoised_sample
            
            # Free memory after each step
            del predicted_noise
        
        return current_sample


@SamplerRegistry.register("dpm++_2m_karras") 
class DPMPlusPlus2MKarrasSampler(DPMPlusPlusSampler):
    """DPM++ 2M with Karras sigma schedule - convenience alias."""
    
    @property
    def name(self) -> str:
        return "dpm++_2m_karras"
    
    def sample(self, num_steps: int, progress: bool = True, **kwargs) -> th.Tensor:
        """Sample using DPM++ 2M with Karras sigmas."""
        return super().sample(
            num_steps=num_steps,
            use_karras_sigmas=True,
            progress=progress,
            **kwargs
        )