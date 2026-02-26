"""
Rectified Flow training and sampling for JiT models.

Implements the rectified flow framework from "Back to Basics: Let Denoising
Generative Models Denoise" (Li & He, 2025):
- Logit-normal time sampling
- x-prediction with v-loss
- Euler and Heun ODE solvers
- CFG with interval support
"""

import torch as th
import torch.nn.functional as F
from tqdm import tqdm


class RectifiedFlow:
    """Rectified flow training and sampling.

    Args:
        time_mu: Mean of the logit-normal distribution for time sampling.
        time_sigma: Std of the logit-normal distribution for time sampling.
        noise_scale: Scale factor for noise (1.0 for 64×64).
        cfg_drop_prob: Probability of dropping text conditioning during training (for CFG).
        cfg_interval: Time interval (low, high) in which CFG is applied during sampling.
    """

    def __init__(
        self,
        time_mu: float = -0.8,
        time_sigma: float = 0.8,
        noise_scale: float = 1.0,
        cfg_drop_prob: float = 0.1,
        cfg_interval: tuple = (0.1, 1.0),
    ):
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.noise_scale = noise_scale
        self.cfg_drop_prob = cfg_drop_prob
        self.cfg_interval = cfg_interval

    def sample_time(self, batch_size: int, device: th.device) -> th.Tensor:
        """Sample timesteps from logit-normal distribution in (0, 1).

        The logit-normal concentrates samples around sigmoid(mu) ≈ 0.31,
        spending more training compute on the noisier parts of the trajectory.
        """
        # Sample from normal, then apply sigmoid to get (0,1)
        u = th.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        t = th.sigmoid(u)
        return t

    def training_losses(
        self,
        model,
        x: th.Tensor,
        tokens: th.Tensor,
        masks: th.Tensor,
        device: th.device,
    ) -> th.Tensor:
        """Compute rectified flow training loss (v-loss with x-prediction).

        Args:
            model: JiTModel that predicts clean x from noisy z_t.
            x: Clean images [B, 3, 64, 64] in [-1, 1].
            tokens: Text token IDs [B, text_ctx].
            masks: Text attention masks [B, text_ctx].
            device: Device for computation.

        Returns:
            Scalar loss tensor with gradients.
        """
        B = x.shape[0]
        t = self.sample_time(B, device)  # [B]

        # Forward process: z_t = t * x + (1 - t) * eps
        eps = th.randn_like(x) * self.noise_scale
        t_expand = t[:, None, None, None]  # [B, 1, 1, 1]
        z_t = t_expand * x + (1 - t_expand) * eps

        # CFG dropout: replace tokens with empty (unconditional)
        drop_mask = th.rand(B, device=device) < self.cfg_drop_prob
        if drop_mask.any():
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
                [], model.text_ctx
            )
            uncond_tokens = th.tensor(uncond_tokens, device=device)
            uncond_mask = th.tensor(uncond_mask, dtype=th.bool, device=device)
            tokens = tokens.clone()
            masks = masks.clone()
            tokens[drop_mask] = uncond_tokens
            masks[drop_mask] = uncond_mask

        # x-prediction
        x_pred = model(z_t, t, tokens=tokens, mask=masks)

        # v-loss: weight by 1/(1-t) to emphasize cleaner timesteps
        # Clamp (1-t) to avoid division by zero near t=1
        weight = 1.0 / th.clamp(1 - t_expand, min=0.05)
        v_pred = (x_pred - z_t) * weight
        v_target = (x - z_t) * weight

        loss = F.mse_loss(v_pred, v_target)
        return loss

    def _cfg_predict(
        self,
        model,
        z: th.Tensor,
        t_scalar: float,
        guidance_scale: float,
        tokens_cond: th.Tensor,
        mask_cond: th.Tensor,
        tokens_uncond: th.Tensor,
        mask_uncond: th.Tensor,
    ) -> th.Tensor:
        """Shared CFG prediction logic for both solvers.

        Returns predicted clean x (with or without CFG depending on interval).
        """
        B = z.shape[0]
        t = th.full((B,), t_scalar, device=z.device, dtype=z.dtype)

        # Check if we should apply CFG at this timestep
        if (
            guidance_scale > 1.0
            and self.cfg_interval[0] <= t_scalar <= self.cfg_interval[1]
        ):
            # Double batch for CFG
            z_double = th.cat([z, z], dim=0)
            t_double = th.cat([t, t], dim=0)
            tokens_double = th.cat([tokens_cond, tokens_uncond], dim=0)
            mask_double = th.cat([mask_cond, mask_uncond], dim=0)

            x_pred_double = model(
                z_double, t_double, tokens=tokens_double, mask=mask_double
            )
            x_cond, x_uncond = x_pred_double.chunk(2, dim=0)
            x_pred = x_uncond + guidance_scale * (x_cond - x_uncond)
        else:
            # No CFG — just conditional prediction
            x_pred = model(z, t, tokens=tokens_cond, mask=mask_cond)

        return x_pred

    @th.inference_mode()
    def sample_euler(
        self,
        model,
        shape: tuple,
        num_steps: int,
        device: th.device,
        guidance_scale: float = 4.0,
        tokens_cond: th.Tensor = None,
        mask_cond: th.Tensor = None,
        tokens_uncond: th.Tensor = None,
        mask_uncond: th.Tensor = None,
        progress: bool = True,
    ) -> th.Tensor:
        """Euler ODE solver: 1 NFE per step.

        ODE: dz/dt = v(z, t) where v = (x_pred - z) / (1 - t)
        Integration from t=0 (noise) to t=1 (clean).
        """
        model.del_cache()
        z = th.randn(shape, device=device) * self.noise_scale

        dt = 1.0 / num_steps
        steps = range(num_steps)
        if progress:
            steps = tqdm(steps, desc="Euler sampling")

        for i in steps:
            t = i * dt  # current time

            x_pred = self._cfg_predict(
                model,
                z,
                t,
                guidance_scale,
                tokens_cond,
                mask_cond,
                tokens_uncond,
                mask_uncond,
            )

            # Velocity: v = (x_pred - z) / (1 - t)
            denom = max(1 - t, 1e-5)
            v = (x_pred - z) / denom
            z = z + dt * v

        model.del_cache()
        return z

    @th.inference_mode()
    def sample_heun(
        self,
        model,
        shape: tuple,
        num_steps: int,
        device: th.device,
        guidance_scale: float = 4.0,
        tokens_cond: th.Tensor = None,
        mask_cond: th.Tensor = None,
        tokens_uncond: th.Tensor = None,
        mask_uncond: th.Tensor = None,
        progress: bool = True,
    ) -> th.Tensor:
        """Heun ODE solver: 2 NFE per step (2nd-order).

        Uses trapezoidal rule for better accuracy at same number of steps.
        """
        model.del_cache()
        z = th.randn(shape, device=device) * self.noise_scale

        dt = 1.0 / num_steps
        steps = range(num_steps)
        if progress:
            steps = tqdm(steps, desc="Heun sampling")

        for i in steps:
            t = i * dt
            t_next = min((i + 1) * dt, 1.0)

            # First evaluation at t
            x_pred_1 = self._cfg_predict(
                model,
                z,
                t,
                guidance_scale,
                tokens_cond,
                mask_cond,
                tokens_uncond,
                mask_uncond,
            )
            denom_1 = max(1 - t, 1e-5)
            v_1 = (x_pred_1 - z) / denom_1

            # Euler step to get z_next estimate
            z_next_est = z + dt * v_1

            # Second evaluation at t_next (skip if last step)
            if i < num_steps - 1:
                x_pred_2 = self._cfg_predict(
                    model,
                    z_next_est,
                    t_next,
                    guidance_scale,
                    tokens_cond,
                    mask_cond,
                    tokens_uncond,
                    mask_uncond,
                )
                denom_2 = max(1 - t_next, 1e-5)
                v_2 = (x_pred_2 - z_next_est) / denom_2

                # Trapezoidal update
                z = z + dt * 0.5 * (v_1 + v_2)
            else:
                z = z_next_est

        model.del_cache()
        return z
