"""Euler and Euler Ancestral sampler implementations."""

import torch as th

from .base import Sampler, SamplerRegistry


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
        **kwargs
    ) -> th.Tensor:
        """Sample using Euler method.
        
        Args:
            num_steps: Number of sampling steps
            s_churn: Amount of stochasticity (0 for deterministic)
            s_tmin: Minimum timestep for adding noise
            s_tmax: Maximum timestep for adding noise  
            s_noise: Noise scale factor
            progress: Show progress bar
        """
        # Pre-compute alpha products only when needed to save memory
        import numpy as np
        betas_np = self.diffusion.betas
        alphas_np = 1.0 - betas_np
        alphas_cumprod_np = np.cumprod(alphas_np)
        
        # Create timestep schedule on CPU
        timesteps = th.linspace(
            self.diffusion.num_timesteps - 1, 0, num_steps, dtype=th.long
        )
        
        # Start from pure noise
        current_sample = th.randn(self.shape, device=self.device)
        
        # Progress bar setup
        from tqdm import tqdm
        iterator = tqdm(timesteps) if progress else timesteps
        
        for step_idx, timestep in enumerate(iterator):
            timestep_int = int(timestep.item())
            batch_timesteps = th.full(
                (self.shape[0],), timestep_int, device=self.device, dtype=th.long
            )
            
            # Get model prediction
            with th.no_grad():
                model_output = self.model_fn(current_sample, batch_timesteps, **self.model_kwargs)
                if isinstance(model_output, tuple):
                    predicted_noise = model_output[0][:, :3]  # Extract epsilon prediction
                else:
                    predicted_noise = model_output[:, :3]
            
            # Get alpha values for current timestep
            alpha_prod_t = float(alphas_cumprod_np[timestep_int])
            if step_idx < len(timesteps) - 1:
                next_timestep_int = int(timesteps[step_idx + 1].item())
                alpha_prod_next = float(alphas_cumprod_np[next_timestep_int])
            else:
                alpha_prod_next = 1.0
            
            # Add noise for stochasticity if needed (for Karras-style noise)
            if s_churn > 0 and s_tmin <= timestep_int <= s_tmax:
                gamma = min(s_churn / num_steps, np.sqrt(2.0) - 1)
                noise_level = np.sqrt((1 - alpha_prod_t) / alpha_prod_t)
                noise_level_higher = noise_level * np.sqrt(1 + gamma**2)
                noise = th.randn_like(current_sample) * s_noise
                current_sample = current_sample + noise * (noise_level_higher - noise_level)
                # Recompute alpha_prod_t
                alpha_prod_t = 1 / (1 + noise_level_higher**2)
            
            # Euler step - compute noise levels as scalars
            current_noise_level = np.sqrt((1 - alpha_prod_t) / alpha_prod_t)
            next_noise_level = np.sqrt((1 - alpha_prod_next) / alpha_prod_next)
            sqrt_alpha_prod_t = np.sqrt(alpha_prod_t)
            
            # Denoised estimate - in-place operations where possible
            denoised_sample = current_sample - current_noise_level * predicted_noise
            denoised_sample.div_(sqrt_alpha_prod_t)
            
            if self.clip_denoised:
                denoised_sample.clamp_(-1, 1)
            
            # Euler update - compute more efficiently
            # derivative = (x - sqrt_alpha * denoised) / sigma
            # new_x = x + derivative * (sigma_next - sigma)
            timestep_delta = next_noise_level - current_noise_level
            
            # Compute update more efficiently with fewer allocations
            update = (current_sample - sqrt_alpha_prod_t * denoised_sample) / current_noise_level
            current_sample = current_sample + update * timestep_delta
            
            # Free memory
            del denoised_sample, predicted_noise, update
            
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
        **kwargs
    ) -> th.Tensor:
        """Sample using Euler Ancestral method.
        
        Args:
            num_steps: Number of sampling steps
            eta: Amount of noise to add at each step (1.0 is default)
            progress: Show progress bar
        """
        # Pre-compute alpha products only when needed to save memory
        import numpy as np
        betas_np = self.diffusion.betas
        alphas_np = 1.0 - betas_np
        alphas_cumprod_np = np.cumprod(alphas_np)
        
        # Create timestep schedule on CPU
        timesteps = th.linspace(
            self.diffusion.num_timesteps - 1, 0, num_steps, dtype=th.long
        )
        
        # Start from pure noise
        current_sample = th.randn(self.shape, device=self.device)
        
        # Progress bar setup
        from tqdm import tqdm
        iterator = tqdm(timesteps) if progress else timesteps
        
        for step_idx, timestep in enumerate(iterator):
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
            
            # Current timestep values as scalars
            current_alpha_prod = float(alphas_cumprod_np[timestep_int])
            current_noise_level = np.sqrt((1 - current_alpha_prod) / current_alpha_prod)
            
            # Denoised estimate  
            sqrt_alpha_prod = np.sqrt(current_alpha_prod)
            denoised_sample = (current_sample - current_noise_level * predicted_noise) / sqrt_alpha_prod
            
            if self.clip_denoised:
                denoised_sample = denoised_sample.clamp(-1, 1)
            
            # Ancestral sampling step
            if step_idx < len(timesteps) - 1:
                # Next timestep values
                next_timestep_int = int(timesteps[step_idx + 1].item())
                next_alpha_prod = float(alphas_cumprod_np[next_timestep_int])
                next_noise_level = np.sqrt((1 - next_alpha_prod) / next_alpha_prod)
                
                # Calculate step sizes for ancestral sampling
                # This is where we "overshoot" and then add noise back
                noise_amount_up = np.sqrt(
                    next_noise_level**2 * (current_noise_level**2 - next_noise_level**2) / current_noise_level**2
                )
                noise_amount_down = np.sqrt(next_noise_level**2 - noise_amount_up**2)
                
                # Euler step with reduced noise
                derivative = (current_sample - sqrt_alpha_prod * denoised_sample) / current_noise_level
                timestep_delta = noise_amount_down - current_noise_level
                current_sample = current_sample + derivative * timestep_delta
                
                # Add noise back (ancestral part - this is what makes it non-convergent)
                if eta > 0:
                    random_noise = th.randn_like(current_sample)
                    current_sample = current_sample + random_noise * noise_amount_up * eta
                
                # Free memory
                del derivative, random_noise
            else:
                # Final step - no noise added
                final_alpha = float(alphas_cumprod_np[-1])
                current_sample = np.sqrt(final_alpha) * denoised_sample
            
            # Free memory
            del denoised_sample, predicted_noise
                
        return current_sample