"""Unit tests for rectified flow training and sampling."""

import torch as th
from glide_finetune.rectified_flow import RectifiedFlow
from glide_finetune.jit_model import JiTModel


def _make_small_model():
    return JiTModel(
        image_size=64,
        patch_size=4,
        in_channels=3,
        hidden_dim=64,
        depth=2,
        heads=4,
        bottleneck_dim=32,
        text_ctx=128,
        xf_width=64,
        xf_layers=2,
        xf_heads=4,
    )


class TestTimeSampling:
    def test_values_in_range(self):
        flow = RectifiedFlow()
        t = flow.sample_time(10000, th.device("cpu"))
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_mean_near_expected(self):
        """Mean should be approximately sigmoid(-0.8) ≈ 0.31."""
        flow = RectifiedFlow(time_mu=-0.8, time_sigma=0.8)
        t = flow.sample_time(50000, th.device("cpu"))
        expected_mean = th.sigmoid(th.tensor(-0.8)).item()
        actual_mean = t.mean().item()
        assert abs(actual_mean - expected_mean) < 0.03, (
            f"Expected mean ~{expected_mean:.3f}, got {actual_mean:.3f}"
        )

    def test_shape(self):
        flow = RectifiedFlow()
        t = flow.sample_time(8, th.device("cpu"))
        assert t.shape == (8,)


class TestForwardProcess:
    def test_noise_at_t0(self):
        """At t=0, z_t should be pure noise (eps)."""
        x = th.randn(4, 3, 8, 8)
        eps = th.randn_like(x)
        t = th.zeros(4)
        t_expand = t[:, None, None, None]
        z_t = t_expand * x + (1 - t_expand) * eps
        assert th.allclose(z_t, eps)

    def test_clean_at_t1(self):
        """At t=1, z_t should be clean image (x)."""
        x = th.randn(4, 3, 8, 8)
        eps = th.randn_like(x)
        t = th.ones(4)
        t_expand = t[:, None, None, None]
        z_t = t_expand * x + (1 - t_expand) * eps
        assert th.allclose(z_t, x)

    def test_interpolation_midpoint(self):
        """At t=0.5, z_t should be average of x and eps."""
        x = th.ones(1, 3, 8, 8)
        eps = th.zeros(1, 3, 8, 8)
        t = th.tensor([0.5])
        t_expand = t[:, None, None, None]
        z_t = t_expand * x + (1 - t_expand) * eps
        assert th.allclose(z_t, th.ones_like(z_t) * 0.5)


class TestTrainingLosses:
    def test_loss_is_finite_scalar(self):
        model = _make_small_model()
        flow = RectifiedFlow()
        x = th.randn(2, 3, 64, 64)
        tokens = th.randint(0, 100, (2, 128))
        masks = th.ones(2, 128, dtype=th.bool)

        loss = flow.training_losses(model, x, tokens, masks, th.device("cpu"))
        assert loss.dim() == 0  # scalar
        assert th.isfinite(loss)

    def test_loss_has_gradients(self):
        model = _make_small_model()
        flow = RectifiedFlow()
        x = th.randn(2, 3, 64, 64)
        tokens = th.randint(0, 100, (2, 128))
        masks = th.ones(2, 128, dtype=th.bool)

        loss = flow.training_losses(model, x, tokens, masks, th.device("cpu"))
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "Expected gradients to flow through model"

    def test_cfg_dropout(self):
        """With cfg_drop_prob=1.0, all tokens should be replaced with unconditional."""
        model = _make_small_model()
        flow = RectifiedFlow(cfg_drop_prob=1.0)
        x = th.randn(2, 3, 64, 64)
        tokens = th.randint(0, 100, (2, 128))
        masks = th.ones(2, 128, dtype=th.bool)

        # Should not error even with 100% dropout
        loss = flow.training_losses(model, x, tokens, masks, th.device("cpu"))
        assert th.isfinite(loss)


class TestEulerSampling:
    def test_output_shape(self):
        model = _make_small_model()
        model.eval()
        flow = RectifiedFlow()

        tokens_cond = th.randint(0, 100, (1, 128))
        mask_cond = th.ones(1, 128, dtype=th.bool)
        tokens_uncond = th.zeros(1, 128, dtype=th.long)
        mask_uncond = th.zeros(1, 128, dtype=th.bool)

        samples = flow.sample_euler(
            model=model,
            shape=(1, 3, 64, 64),
            num_steps=5,
            device=th.device("cpu"),
            guidance_scale=1.0,
            tokens_cond=tokens_cond,
            mask_cond=mask_cond,
            tokens_uncond=tokens_uncond,
            mask_uncond=mask_uncond,
            progress=False,
        )
        assert samples.shape == (1, 3, 64, 64)

    def test_output_finite(self):
        model = _make_small_model()
        model.eval()
        flow = RectifiedFlow()

        tokens_cond = th.randint(0, 100, (1, 128))
        mask_cond = th.ones(1, 128, dtype=th.bool)
        tokens_uncond = th.zeros(1, 128, dtype=th.long)
        mask_uncond = th.zeros(1, 128, dtype=th.bool)

        samples = flow.sample_euler(
            model=model,
            shape=(1, 3, 64, 64),
            num_steps=5,
            device=th.device("cpu"),
            guidance_scale=1.0,
            tokens_cond=tokens_cond,
            mask_cond=mask_cond,
            tokens_uncond=tokens_uncond,
            mask_uncond=mask_uncond,
            progress=False,
        )
        assert th.isfinite(samples).all()


class TestHeunSampling:
    def test_output_shape(self):
        model = _make_small_model()
        model.eval()
        flow = RectifiedFlow()

        tokens_cond = th.randint(0, 100, (1, 128))
        mask_cond = th.ones(1, 128, dtype=th.bool)
        tokens_uncond = th.zeros(1, 128, dtype=th.long)
        mask_uncond = th.zeros(1, 128, dtype=th.bool)

        samples = flow.sample_heun(
            model=model,
            shape=(1, 3, 64, 64),
            num_steps=5,
            device=th.device("cpu"),
            guidance_scale=1.0,
            tokens_cond=tokens_cond,
            mask_cond=mask_cond,
            tokens_uncond=tokens_uncond,
            mask_uncond=mask_uncond,
            progress=False,
        )
        assert samples.shape == (1, 3, 64, 64)
        assert th.isfinite(samples).all()
