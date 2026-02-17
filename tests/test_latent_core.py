"""Core tests for latent diffusion architecture and diffusion math."""

import torch as th
from glide_text2im.gaussian_diffusion import get_named_beta_schedule
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_latent,
)


def _make_latent_model():
    opts = model_and_diffusion_defaults_latent()
    opts["use_fp16"] = False
    return create_model_and_diffusion(**opts), opts


class TestLatentModelConstruction:
    def test_output_shape(self):
        (model, _diffusion), _opts = _make_latent_model()
        x = th.randn(2, 4, 32, 32)
        t = th.tensor([10, 20])
        tokens = th.randint(0, 100, (2, 128))
        mask = th.ones(2, 128, dtype=th.bool)

        out = model(x, t, tokens=tokens, mask=mask)
        assert out.shape == (2, 8, 32, 32)

    def test_output_shape_with_clip_emb(self):
        (model, _diffusion), _opts = _make_latent_model()
        x = th.randn(2, 4, 32, 32)
        t = th.tensor([10, 20])
        tokens = th.randint(0, 100, (2, 128))
        mask = th.ones(2, 128, dtype=th.bool)
        clip_emb = th.randn(2, 768)

        out = model(x, t, tokens=tokens, mask=mask, clip_emb=clip_emb)
        assert out.shape == (2, 8, 32, 32)

    def test_clip_layers_exist(self):
        (model, _diffusion), _opts = _make_latent_model()
        assert hasattr(model, "clip_to_time")
        assert hasattr(model, "clip_to_xf")
        # clip_to_time: Linear -> SiLU -> Linear
        assert len(model.clip_to_time) == 3
        # clip_to_xf input is 768
        assert model.clip_to_xf.in_features == 768


class TestLinearLatentSchedule:
    def test_length(self):
        betas = get_named_beta_schedule("linear_latent", 1000)
        assert len(betas) == 1000

    def test_endpoints(self):
        betas = get_named_beta_schedule("linear_latent", 1000)
        assert abs(betas[0] - 0.00085) < 1e-4
        assert abs(betas[-1] - 0.012) < 1e-4

    def test_monotonically_increasing(self):
        betas = get_named_beta_schedule("linear_latent", 1000)
        assert all(betas[i] <= betas[i + 1] for i in range(len(betas) - 1))


class TestDiffusionOnLatents:
    def test_q_sample_preserves_shape(self):
        (_model, diffusion), _opts = _make_latent_model()
        latents = th.randn(2, 4, 32, 32)
        noise = th.randn_like(latents)
        timesteps = th.tensor([100, 500])

        x_t = diffusion.q_sample(latents, timesteps, noise=noise)
        assert x_t.shape == (2, 4, 32, 32)

    def test_q_sample_at_t0_close_to_input(self):
        (_model, diffusion), _opts = _make_latent_model()
        latents = th.randn(1, 4, 32, 32)
        noise = th.randn_like(latents)
        timesteps = th.tensor([0])

        x_t = diffusion.q_sample(latents, timesteps, noise=noise)
        # At t=0, x_t should be very close to the original latents
        # (beta_0=0.00085, so noise contribution ~0.03 * noise)
        assert th.allclose(x_t, latents, atol=0.15)


class TestPixelBackwardCompatibility:
    def test_pixel_model_unchanged(self):
        pixel_opts = model_and_diffusion_defaults()
        pixel_opts["use_fp16"] = False
        pixel_model, _diffusion = create_model_and_diffusion(**pixel_opts)

        x = th.randn(1, 3, 64, 64)
        t = th.tensor([10])
        tokens = th.randint(0, 100, (1, 128))
        mask = th.ones(1, 128, dtype=th.bool)

        out = pixel_model(x, t, tokens=tokens, mask=mask)
        assert out.shape == (1, 6, 64, 64)

    def test_default_options_differ(self):
        pixel_opts = model_and_diffusion_defaults()
        latent_opts = model_and_diffusion_defaults_latent()

        assert pixel_opts["image_size"] == 64
        assert latent_opts["image_size"] == 32
        assert pixel_opts["noise_schedule"] != latent_opts["noise_schedule"]
        assert latent_opts["latent_mode"] is True
        assert "latent_mode" not in pixel_opts or pixel_opts.get("latent_mode") is False
