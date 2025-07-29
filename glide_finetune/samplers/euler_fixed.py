"""Fixed Euler sampler that works with GLIDE's DDPM parameterization."""

import numpy as np
import torch as th
from tqdm import tqdm

from .base import Sampler, SamplerRegistry


@SamplerRegistry.register("euler_fixed")
class EulerFixedSampler(Sampler):
    """Euler sampler using DDPM parameterization compatible with GLIDE."""
    
    @property
    def name(self) -> str:
        return "euler_fixed"
    
    def sample(
        self,
        num_steps: int,
        eta: float = 0.0,
        progress: bool = True,
        **kwargs
    ) -> th.Tensor:
        """Sample using Euler method with DDPM parameterization.
        
        This uses the DDPM forward process parameterization:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        
        And solves the reverse process using Euler steps on the 
        probability flow ODE.
        """
        # Get the respaced timesteps
        if hasattr(self.diffusion, 'timestep_map'):
            # For SpacedDiffusion, map to original timesteps
            respaced_indices = np.linspace(
                self.diffusion.num_timesteps - 1, 0, num_steps, dtype=np.int64
            )
            timesteps = np.array([self.diffusion.timestep_map[i] for i in respaced_indices])
        else:
            # No respacing
            timesteps = np.linspace(
                self.diffusion.num_timesteps - 1, 0, num_steps, dtype=np.int64
            )
        
        # Get alpha values - need to handle respaced diffusion
        if hasattr(self.diffusion, 'timestep_map'):
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
        img = th.randn(self.shape, device=self.device)
        
        # Progress bar
        indices = list(range(len(timesteps)))
        if progress:
            indices = tqdm(indices)
        
        for i in indices:
            t = timesteps[i]
            t_batch = th.full((self.shape[0],), t, device=self.device, dtype=th.long)
            
            # Get model prediction (epsilon)
            with th.no_grad():
                out = self.model_fn(img, t_batch, **self.model_kwargs)
                if isinstance(out, tuple):
                    eps = out[0][:, :3]
                else:
                    eps = out[:, :3]
            
            # Current alpha values
            alpha_bar = alphas_cumprod[t]
            
            # Predict x_0
            # x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
            x_0 = (img - np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha_bar)
            
            if self.clip_denoised:
                x_0 = x_0.clamp(-1, 1)
            
            # Get next alpha (or 1.0 for final step)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = alphas_cumprod[t_next]
            else:
                alpha_bar_next = 1.0
            
            # DDIM update (deterministic when eta=0)
            # This is essentially an Euler step on the probability flow ODE
            sigma_t = eta * np.sqrt((1 - alpha_bar_next) / (1 - alpha_bar)) * np.sqrt(1 - alpha_bar / alpha_bar_next)
            
            # Compute the direction pointing to x_0
            pred_dir = (img - np.sqrt(alpha_bar) * x_0) / np.sqrt(1 - alpha_bar)
            
            # Take a step towards x_0
            img = np.sqrt(alpha_bar_next) * x_0 + np.sqrt(1 - alpha_bar_next - sigma_t**2) * pred_dir
            
            # Add noise if eta > 0
            if eta > 0 and i < len(timesteps) - 1:
                noise = th.randn_like(img)
                img = img + sigma_t * noise
        
        return img


@SamplerRegistry.register("euler_a_fixed") 
class EulerAncestralFixedSampler(Sampler):
    """Euler Ancestral sampler using DDPM parameterization."""
    
    @property
    def name(self) -> str:
        return "euler_a_fixed"
    
    def sample(
        self,
        num_steps: int,
        eta: float = 1.0,
        progress: bool = True,
        **kwargs
    ) -> th.Tensor:
        """Sample using Euler Ancestral with DDPM parameterization.
        
        This is similar to Euler but always adds noise (eta=1.0 by default).
        """
        # Just use the fixed Euler sampler with eta=1.0
        euler_sampler = EulerFixedSampler(
            diffusion=self.diffusion,
            model_fn=self.model_fn,
            shape=self.shape,
            device=self.device,
            clip_denoised=self.clip_denoised,
            model_kwargs=self.model_kwargs,
        )
        return euler_sampler.sample(num_steps=num_steps, eta=eta, progress=progress, **kwargs)