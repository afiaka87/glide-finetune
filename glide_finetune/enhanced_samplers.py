"""
Enhanced samplers for GLIDE that properly integrate with the GaussianDiffusion framework.
These samplers extend the built-in GLIDE diffusion methods to add Euler, Euler Ancestral, and DPM++ support.
"""

import torch as th
from typing import Optional, Tuple
import numpy as np
from tqdm.auto import tqdm


def add_enhanced_samplers_to_diffusion(diffusion_class):
    """Add enhanced sampling methods to a GaussianDiffusion class."""
    
    def euler_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        num_steps=50,
    ):
        """
        Generate samples from the model using Euler method.
        Same interface as ddim_sample_loop but with Euler integration.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        
        # CRITICAL: Use the actual timesteps from the diffusion schedule
        # For SpacedDiffusion, num_timesteps is already the respaced count
        indices = list(range(self.num_timesteps))[::-1]
        
        indices_iter = tqdm(indices, desc="Euler sampling") if progress else indices
        
        for i, timestep_idx in enumerate(indices_iter):
            t = th.tensor([timestep_idx] * shape[0], device=device)
            
            with th.no_grad():
                # Use GLIDE's p_mean_variance to get proper x_start prediction
                out = self.p_mean_variance(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs or {},
                )
                
                # Apply classifier guidance if specified
                if cond_fn is not None:
                    out = self.condition_score(cond_fn, out, img, t, model_kwargs=model_kwargs or {})
                
                # Get epsilon from x_start prediction (GLIDE's approach)
                epsilon = self._predict_eps_from_xstart(img, t, out["pred_xstart"])
                
                # Extract alpha values properly using GLIDE's method
                from glide_text2im.gaussian_diffusion import _extract_into_tensor
                alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, img.shape)
                alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, img.shape)
                
                # For Euler, we use eta=0 (deterministic)
                sigma = 0.0
                
                # Use GLIDE's DDIM mean prediction formula
                mean_pred = (
                    out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                    + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * epsilon
                )
                
                # No noise for deterministic Euler
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                img = mean_pred
                
                # Don't clamp intermediate values - only clamp pred_xstart via clip_denoised param
                if denoised_fn is not None:
                    img = denoised_fn(img)
        
        return img
    
    def euler_ancestral_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=1.0,
        num_steps=30,
        generator=None,
    ):
        """
        Generate samples from the model using Euler Ancestral method with stochasticity.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        
        # CRITICAL: Use the actual timesteps from the diffusion schedule
        # For SpacedDiffusion, num_timesteps is already the respaced count
        indices = list(range(self.num_timesteps))[::-1]
        
        indices_iter = tqdm(indices, desc="Euler Ancestral sampling") if progress else indices
        
        for i, timestep_idx in enumerate(indices_iter):
            t = th.tensor([timestep_idx] * shape[0], device=device)
            
            with th.no_grad():
                # Use GLIDE's p_mean_variance to get proper x_start prediction
                out = self.p_mean_variance(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs or {},
                )
                
                # Apply classifier guidance if specified
                if cond_fn is not None:
                    out = self.condition_score(cond_fn, out, img, t, model_kwargs=model_kwargs or {})
                
                # Get epsilon from x_start prediction (GLIDE's approach)
                epsilon = self._predict_eps_from_xstart(img, t, out["pred_xstart"])
                
                # Extract alpha values properly using GLIDE's method
                from glide_text2im.gaussian_diffusion import _extract_into_tensor
                alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, img.shape)
                alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, img.shape)
                
                # Calculate sigma for stochastic component (GLIDE's DDIM formula)
                sigma = (
                    eta
                    * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * th.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                
                # Use GLIDE's DDIM mean prediction formula
                mean_pred = (
                    out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                    + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * epsilon
                )
                
                # Add noise for ancestral sampling
                noise = th.randn_like(img)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                img = mean_pred + nonzero_mask * sigma * noise
                
                # Don't clamp intermediate values - only clamp pred_xstart via clip_denoised param
                if denoised_fn is not None:
                    img = denoised_fn(img)
        
        return img
    
    def dpm_solver_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        num_steps=20,
        order=2,
    ):
        """
        Generate samples from the model using DPM-Solver++ method.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        
        # CRITICAL: Use the actual timesteps from the diffusion schedule
        # For SpacedDiffusion, num_timesteps is already the respaced count
        indices = list(range(self.num_timesteps))[::-1]
        
        indices_iter = tqdm(indices, desc="DPM-Solver sampling") if progress else indices
        
        prev_epsilon = None
        
        for i, timestep_idx in enumerate(indices_iter):
            t = th.tensor([timestep_idx] * shape[0], device=device)
            
            with th.no_grad():
                # Use GLIDE's p_mean_variance to get proper x_start prediction
                out = self.p_mean_variance(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs or {},
                )
                
                # Apply classifier guidance if specified
                if cond_fn is not None:
                    out = self.condition_score(cond_fn, out, img, t, model_kwargs=model_kwargs or {})
                
                # Get epsilon from x_start prediction (GLIDE's approach)
                epsilon = self._predict_eps_from_xstart(img, t, out["pred_xstart"])
                
                # Extract alpha values properly using GLIDE's method
                from glide_text2im.gaussian_diffusion import _extract_into_tensor
                alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, img.shape)
                alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, img.shape)
                
                # DPM-Solver uses deterministic updates (eta=0)
                sigma = 0.0
                
                # DPM-Solver step with multi-step information
                if order == 1 or prev_epsilon is None:
                    # First order - same as DDIM
                    mean_pred = (
                        out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                        + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * epsilon
                    )
                elif order == 2:
                    # Second order DPM-Solver - use linear multistep method
                    # Combine current and previous epsilon predictions
                    combined_eps = 1.5 * epsilon - 0.5 * prev_epsilon
                    mean_pred = (
                        out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                        + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * combined_eps
                    )
                else:
                    raise NotImplementedError(f"DPM-Solver order {order} not implemented")
                
                prev_epsilon = epsilon
                img = mean_pred
                
                # Don't clamp intermediate values - only clamp pred_xstart via clip_denoised param
                if denoised_fn is not None:
                    img = denoised_fn(img)
        
        return img
    
    # Add the methods to the class
    diffusion_class.euler_sample_loop = euler_sample_loop
    diffusion_class.euler_ancestral_sample_loop = euler_ancestral_sample_loop
    diffusion_class.dpm_solver_sample_loop = dpm_solver_sample_loop
    
    return diffusion_class


def enhance_glide_diffusion(diffusion):
    """Add enhanced sampling methods to an existing GLIDE diffusion instance."""
    diffusion_class = type(diffusion)
    enhanced_class = add_enhanced_samplers_to_diffusion(diffusion_class)
    return diffusion