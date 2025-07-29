"""DPM++ (Diffusion Probabilistic Model Plus Plus) sampler implementations."""

import torch as th

from .base import Sampler, SamplerRegistry
from .util import (
    get_glide_cosine_schedule,
    scale_model_input,
    sigma_to_timestep,
    predicted_noise_to_denoised,
    get_glide_schedule_timesteps,
    compute_lambda_min_clipped,
)


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
        """Sample using DPM++ 2M method with proper GLIDE schedule.
        
        Args:
            num_steps: Number of sampling steps
            use_karras_sigmas: Use Karras sigma schedule for better quality
            progress: Show progress bar
        """
        import numpy as np
        from tqdm import tqdm
        
        # Get GLIDE's cosine schedule
        # For SpacedDiffusion, we need to use the original number of timesteps
        if hasattr(self.diffusion, 'original_num_steps'):
            original_steps = self.diffusion.original_num_steps
        elif hasattr(self.diffusion, 'timestep_map'):
            # SpacedDiffusion stores the mapping but not original_num_steps
            # The last element of timestep_map + 1 gives us the original steps
            original_steps = self.diffusion.timestep_map[-1] + 1
        else:
            # Fall back to the diffusion's num_timesteps
            original_steps = self.diffusion.num_timesteps
            
        betas, alphas_cumprod, sigmas = get_glide_cosine_schedule(original_steps)
        
        # Get timesteps and sigmas for sampling
        # If we have a timestep map, we need to map our steps through it
        if hasattr(self.diffusion, 'timestep_map'):
            # For respaced diffusion, we sample from the respaced timesteps
            # and map them to the original timesteps
            respaced_timesteps = np.linspace(
                self.diffusion.num_timesteps - 1, 0, num_steps, dtype=np.int64
            )
            timesteps = np.array([self.diffusion.timestep_map[t] for t in respaced_timesteps])
            
            if use_karras_sigmas:
                # For Karras, we still use the Karras sigmas but map the timesteps
                sigma_min = float(sigmas[timesteps[-1]])
                sigma_max = float(sigmas[timesteps[0]])
                from .util import get_karras_sigmas
                karras_sigmas = get_karras_sigmas(num_steps, sigma_min, sigma_max)
                sampling_sigmas = karras_sigmas
            else:
                sampling_sigmas = sigmas[timesteps]
        else:
            # No respacing, use the normal schedule
            timesteps, sampling_sigmas = get_glide_schedule_timesteps(
                num_steps, original_steps, sigmas, use_karras=use_karras_sigmas
            )
        
        # Get lambda_min_clipped for cosine schedule stability
        lambda_min_clipped = compute_lambda_min_clipped()
        
        # Start from pure noise
        current_sample = th.randn(self.shape, device=self.device)
        
        # DPM++ 2M specific - store previous denoised and derivative
        old_denoised = None
        h_last = None
        
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
            
            # DPM++ 2M update
            if step_idx == 0:
                # First step - just store for multistep
                old_denoised = denoised
            else:
                # Multistep update
                if step_idx < num_steps - 1:
                    # Get next sigma
                    next_sigma = sampling_sigmas[step_idx + 1]
                    
                    # Compute lambdas for DPM++ formula
                    # lambda = log(sigma) - log(alpha)
                    lambda_s = np.log(sigma) - 0.5 * np.log(alpha_prod)
                    next_timestep = timesteps[step_idx + 1]
                    next_alpha_prod = alphas_cumprod[next_timestep]
                    lambda_next = np.log(next_sigma) - 0.5 * np.log(next_alpha_prod)
                    
                    # Clip lambda for stability with cosine schedule
                    lambda_s = max(lambda_s, lambda_min_clipped)
                    lambda_next = max(lambda_next, lambda_min_clipped)
                    
                    # Step size
                    h = lambda_next - lambda_s
                    
                    if old_denoised is not None and h_last is not None:
                        # Second-order update
                        # r = h / h_last
                        r = h / h_last if abs(h_last) > 1e-8 else 1.0
                        
                        # Compute D2 (second-order difference)
                        D2 = (denoised - old_denoised) / h_last if abs(h_last) > 1e-8 else 0
                        
                        # Second-order DPM++ formula
                        # x_next = alpha_next * (denoised + 0.5 * h * D2)
                        denoised_prime = denoised + 0.5 * h * D2
                    else:
                        # First-order fallback
                        denoised_prime = denoised
                    
                    # Update sample (convert back from denoised space)
                    # x_next = sqrt(alpha_next) * denoised_prime + sigma_next * noise
                    sqrt_next_alpha = np.sqrt(next_alpha_prod)
                    
                    # For DPM++, we need to compute the noise component
                    # noise = (x - sqrt(alpha) * denoised) / sigma
                    noise_component = (current_sample - np.sqrt(alpha_prod) * denoised) / sigma if sigma > 0 else predicted_noise
                    
                    current_sample = sqrt_next_alpha * denoised_prime + next_sigma * noise_component
                    
                    # Store for next iteration
                    h_last = h
                    del noise_component, denoised_prime
                else:
                    # Final step - deterministic
                    current_sample = denoised
                
                # Update old_denoised
                if old_denoised is not None:
                    del old_denoised
                old_denoised = denoised
            
            # Free memory
            del scaled_sample, predicted_noise
        
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