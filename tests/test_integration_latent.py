"""Integration tests for the latent diffusion pipeline.

Tests marked @pytest.mark.slow require downloading external models
(VAE ~300MB, CLIP ~900MB) and may take several minutes on first run.
Run with:  uv run pytest tests/test_integration_latent.py -v
Slow only: uv run pytest tests/test_integration_latent.py -v -m slow
Skip slow: uv run pytest tests/test_integration_latent.py -v -m "not slow"
"""

import subprocess

import pytest
import torch as th
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_latent,
)


def _make_latent_model():
    opts = model_and_diffusion_defaults_latent()
    opts["use_fp16"] = False
    model, diffusion = create_model_and_diffusion(**opts)
    return model, diffusion, opts


# ---------------------------------------------------------------------------
# Mock helpers for tests that don't need real VAE/CLIP downloads
# ---------------------------------------------------------------------------


class _MockVAE:
    """Returns correctly-shaped random latents without a real model."""

    @th.no_grad()
    def encode(self, images: th.Tensor) -> th.Tensor:
        B = images.shape[0]
        return th.randn(B, 4, 32, 32)

    @th.no_grad()
    def decode(self, latents: th.Tensor) -> th.Tensor:
        B = latents.shape[0]
        return th.randn(B, 3, 256, 256).clamp(-1, 1)


class _MockCLIP:
    """Returns correctly-shaped random embeddings without a real model."""

    @th.no_grad()
    def encode_text_batch(self, texts: list[str]) -> th.Tensor:
        return th.randn(len(texts), 768)


# ===================================================================
# VAE roundtrip (requires downloading stabilityai/sd-vae-ft-mse)
# ===================================================================


class TestVAERoundtrip:
    @pytest.fixture(scope="class")
    def vae(self):
        pytest.importorskip("diffusers")
        from glide_finetune.latent_util import LatentVAE

        return LatentVAE(device="cpu", dtype=th.float32)

    @pytest.mark.slow
    def test_encode_shape(self, vae):
        images = th.randn(1, 3, 256, 256).clamp(-1, 1)
        latents = vae.encode(images)
        assert latents.shape == (1, 4, 32, 32)

    @pytest.mark.slow
    def test_decode_shape(self, vae):
        latents = th.randn(1, 4, 32, 32) * 0.18215
        decoded = vae.decode(latents)
        assert decoded.shape == (1, 3, 256, 256)

    @pytest.mark.slow
    def test_roundtrip_output_range(self, vae):
        images = th.randn(1, 3, 256, 256).clamp(-1, 1)
        latents = vae.encode(images)
        decoded = vae.decode(latents)
        assert decoded.min() >= -1.0
        assert decoded.max() <= 1.0


# ===================================================================
# Training smoke test
# ===================================================================


class TestLatentTrainStep:
    def _make_batch(self, batch_size: int = 2):
        return (
            th.randint(0, 100, (batch_size, 128)),  # tokens
            th.ones(batch_size, 128, dtype=th.bool),  # masks
            th.randn(batch_size, 3, 256, 256),  # images_256
            ["a photo of a cat"] * batch_size,  # captions
        )

    def test_returns_positive_scalar_loss(self):
        from glide_finetune.glide_finetune import latent_train_step

        model, diffusion, _opts = _make_latent_model()
        loss = latent_train_step(
            glide_model=model,
            glide_diffusion=diffusion,
            batch=self._make_batch(),
            device="cpu",
            vae=_MockVAE(),
            clip_encoder=_MockCLIP(),
        )
        assert loss.dim() == 0, "loss should be a scalar"
        assert loss.item() > 0, "MSE loss should be positive"
        assert not th.isnan(loss)
        assert not th.isinf(loss)

    def test_loss_is_differentiable(self):
        from glide_finetune.glide_finetune import latent_train_step

        model, diffusion, _opts = _make_latent_model()
        loss = latent_train_step(
            glide_model=model,
            glide_diffusion=diffusion,
            batch=self._make_batch(batch_size=1),
            device="cpu",
            vae=_MockVAE(),
            clip_encoder=_MockCLIP(),
        )
        assert loss.requires_grad
        loss.backward()

        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(grads) > 0, "backward pass should produce gradients"

    def test_optimizer_step_reduces_loss(self):
        """Three optimizer steps on same data should reduce loss."""
        from glide_finetune.glide_finetune import latent_train_step

        # Use generator-based seeding for deterministic model init independent
        # of global RNG state (which CUDA compile tests may have altered).
        gen = th.Generator().manual_seed(123)
        model, diffusion, _opts = _make_latent_model()
        # Re-init parameters with our generator for reproducibility
        for p in model.parameters():
            if p.data.is_floating_point():
                p.data = th.randn_like(p.data, generator=gen) * 0.02
        optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

        batch = self._make_batch(batch_size=2)
        mock_vae = _MockVAE()
        mock_clip = _MockCLIP()

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss = latent_train_step(
                glide_model=model,
                glide_diffusion=diffusion,
                batch=batch,
                device="cpu",
                vae=mock_vae,
                clip_encoder=mock_clip,
            )
            assert not th.isnan(loss), f"NaN loss encountered, losses so far: {losses}"
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease over 3 steps on same data
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ===================================================================
# Weight transfer from pixel checkpoint
# ===================================================================


class TestInitLatentFromPixel:
    def test_input_conv_transfer(self):
        """First 3 channels copied from pixel, 4th zero-initialized."""
        from glide_finetune.latent_util import init_latent_from_pixel

        pixel_opts = model_and_diffusion_defaults()
        pixel_opts["use_fp16"] = False
        pixel_model, _ = create_model_and_diffusion(**pixel_opts)
        pixel_sd = pixel_model.state_dict()

        latent_model, _, _ = _make_latent_model()
        init_latent_from_pixel(latent_model, pixel_sd)
        latent_sd = latent_model.state_dict()

        w = latent_sd["input_blocks.0.0.weight"]
        assert w.shape == (192, 4, 3, 3)
        assert th.allclose(w[:, :3], pixel_sd["input_blocks.0.0.weight"])
        assert th.allclose(w[:, 3], th.zeros_like(w[:, 3]))

    def test_output_conv_transfer(self):
        """Epsilon and variance channels mapped correctly with new channels zero-initialized."""
        from glide_finetune.latent_util import init_latent_from_pixel

        pixel_opts = model_and_diffusion_defaults()
        pixel_opts["use_fp16"] = False
        pixel_model, _ = create_model_and_diffusion(**pixel_opts)
        pixel_sd = pixel_model.state_dict()

        latent_model, _, _ = _make_latent_model()
        init_latent_from_pixel(latent_model, pixel_sd)
        latent_sd = latent_model.state_dict()

        # Weight: pixel [6, in, kH, kW] → latent [8, in, kH, kW]
        w = latent_sd["out.2.weight"]
        psw = pixel_sd["out.2.weight"]
        assert w.shape[0] == 8
        assert th.allclose(w[:3], psw[:3])  # epsilon channels copied
        assert th.allclose(w[3], th.zeros_like(w[3]))  # new epsilon channel zero
        assert th.allclose(w[4:7], psw[3:])  # variance channels copied
        assert th.allclose(w[7], th.zeros_like(w[7]))  # new variance channel zero

        # Bias: pixel [6] → latent [8]
        b = latent_sd["out.2.bias"]
        psb = pixel_sd["out.2.bias"]
        assert b.shape[0] == 8
        assert th.allclose(b[:3], psb[:3])  # epsilon channels copied
        assert b[3] == 0.0  # new epsilon channel zero
        assert th.allclose(b[4:7], psb[3:])  # variance channels copied
        assert b[7] == 0.0  # new variance channel zero

    def test_text_transformer_transferred(self):
        """Text transformer weights should be copied exactly."""
        from glide_finetune.latent_util import init_latent_from_pixel

        pixel_opts = model_and_diffusion_defaults()
        pixel_opts["use_fp16"] = False
        pixel_model, _ = create_model_and_diffusion(**pixel_opts)
        pixel_sd = pixel_model.state_dict()

        latent_model, _, _ = _make_latent_model()
        init_latent_from_pixel(latent_model, pixel_sd)
        latent_sd = latent_model.state_dict()

        # token_embedding and transformer weights should match exactly
        assert th.allclose(
            latent_sd["token_embedding.weight"],
            pixel_sd["token_embedding.weight"],
        )
        assert th.allclose(
            latent_sd["transformer_proj.weight"],
            pixel_sd["transformer_proj.weight"],
        )

    def test_clip_layers_remain_random(self):
        """CLIP projection layers should NOT be overwritten (they're new)."""
        from glide_finetune.latent_util import init_latent_from_pixel

        latent_model, _, _ = _make_latent_model()

        # Snapshot CLIP layer weights before transfer
        pre_clip_time = latent_model.clip_to_time[0].weight.clone()
        pre_clip_xf = latent_model.clip_to_xf.weight.clone()

        pixel_opts = model_and_diffusion_defaults()
        pixel_opts["use_fp16"] = False
        pixel_model, _ = create_model_and_diffusion(**pixel_opts)
        init_latent_from_pixel(latent_model, pixel_model.state_dict())

        # CLIP layers should be unchanged (pixel model has no clip_to_*)
        assert th.allclose(latent_model.clip_to_time[0].weight, pre_clip_time)
        assert th.allclose(latent_model.clip_to_xf.weight, pre_clip_xf)


# ===================================================================
# Collate function
# ===================================================================


class TestLatentCollateFn:
    def test_stacks_tensors_collects_strings(self):
        from glide_finetune.wds_loader import latent_collate_fn

        batch = [
            (th.ones(128), th.ones(128, dtype=th.bool), th.randn(3, 256, 256), "cat"),
            (th.ones(128), th.ones(128, dtype=th.bool), th.randn(3, 256, 256), "dog"),
        ]
        tokens, masks, images, captions = latent_collate_fn(batch)

        assert tokens.shape == (2, 128)
        assert masks.shape == (2, 128)
        assert images.shape == (2, 3, 256, 256)
        assert captions == ["cat", "dog"]


# ===================================================================
# torch.compile + bf16 backward pass
# ===================================================================


class TestTorchCompileBf16:
    """Reproduce dtype errors that occur with torch.compile + bf16 + CLIP conditioning."""

    def _make_bf16_model(self, device="cpu"):
        model, diffusion, opts = _make_latent_model()
        model.convert_to_bf16()
        model = model.to(device)
        return model, diffusion, opts

    def test_eager_bf16_backward(self):
        """bf16 latent model backward WITHOUT torch.compile should work."""
        model, _diffusion, _opts = self._make_bf16_model()
        x = th.randn(1, 4, 32, 32)
        t = th.tensor([100])
        tokens = th.randint(0, 100, (1, 128))
        mask = th.ones(1, 128, dtype=th.bool)
        clip_emb = th.randn(1, 768)

        out = model(x, t, tokens=tokens, mask=mask, clip_emb=clip_emb)
        loss = out.float().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_eager_bf16_backward_no_clip(self):
        """bf16 latent model backward without CLIP (clip_emb=None)."""
        model, _diffusion, _opts = self._make_bf16_model()
        x = th.randn(1, 4, 32, 32)
        t = th.tensor([100])
        tokens = th.randint(0, 100, (1, 128))
        mask = th.ones(1, 128, dtype=th.bool)

        out = model(x, t, tokens=tokens, mask=mask, clip_emb=None)
        loss = out.float().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    @pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA")
    def test_compile_bf16_backward_no_clip(self):
        """torch.compile + bf16 backward WITHOUT CLIP conditioning."""
        model, _diffusion, _opts = self._make_bf16_model(device="cuda")
        compiled = th.compile(model)
        x = th.randn(1, 4, 32, 32, device="cuda")
        t = th.tensor([100], device="cuda")
        tokens = th.randint(0, 100, (1, 128), device="cuda")
        mask = th.ones(1, 128, dtype=th.bool, device="cuda")

        out = compiled(x, t, tokens=tokens, mask=mask, clip_emb=None)
        loss = out.float().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "backward should produce gradients"

    @pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA")
    def test_compile_bf16_backward_with_clip(self):
        """torch.compile + bf16 backward WITH CLIP conditioning."""
        model, _diffusion, _opts = self._make_bf16_model(device="cuda")
        compiled = th.compile(model)
        x = th.randn(1, 4, 32, 32, device="cuda")
        t = th.tensor([100], device="cuda")
        tokens = th.randint(0, 100, (1, 128), device="cuda")
        mask = th.ones(1, 128, dtype=th.bool, device="cuda")
        clip_emb = th.randn(1, 768, device="cuda")

        out = compiled(x, t, tokens=tokens, mask=mask, clip_emb=clip_emb)
        loss = out.float().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "backward should produce gradients"

    @pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA")
    def test_compile_bf16_channels_last_backward_with_clip(self):
        """torch.compile + bf16 + channels_last + CLIP (matches training loop)."""
        model, _diffusion, _opts = self._make_bf16_model(device="cuda")
        model.to(memory_format=th.channels_last)
        th.set_float32_matmul_precision("high")
        compiled = th.compile(model)
        x = th.randn(1, 4, 32, 32, device="cuda")
        t = th.tensor([100], device="cuda")
        tokens = th.randint(0, 100, (1, 128), device="cuda")
        mask = th.ones(1, 128, dtype=th.bool, device="cuda")
        clip_emb = th.randn(1, 768, device="cuda")

        out = compiled(x, t, tokens=tokens, mask=mask, clip_emb=clip_emb)
        loss = out.float().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "backward should produce gradients"

    @pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA")
    def test_compile_bf16_backward_with_bf16_input(self):
        """torch.compile + bf16 backward when input tensor is bf16 (like VAE output).

        This was the root cause: bf16 VAE returns bf16 latents, so x_t is bf16,
        model output is bf16, loss is bf16. Backward from bf16 loss through
        mixed-dtype compiled model fails. Fix: cast latents to float32.
        """
        model, diffusion, _opts = self._make_bf16_model(device="cuda")
        model.to(memory_format=th.channels_last)
        compiled = th.compile(model)

        # Simulate bf16 VAE output → bf16 x_t (the failing path before fix)
        latents_bf16 = th.randn(1, 4, 32, 32, device="cuda", dtype=th.bfloat16)
        noise_bf16 = th.randn_like(latents_bf16)
        t = th.tensor([100], device="cuda")
        x_t_bf16 = diffusion.q_sample(latents_bf16, t, noise=noise_bf16)
        tokens = th.randint(0, 100, (1, 128), device="cuda")
        mask = th.ones(1, 128, dtype=th.bool, device="cuda")
        clip_emb = th.randn(1, 768, device="cuda")

        # The fix: cast to float32 before model forward
        x_t = x_t_bf16.float()
        noise = noise_bf16.float()

        out = compiled(x_t, t, tokens=tokens, mask=mask, clip_emb=clip_emb)
        C = x_t.shape[1]
        epsilon, _ = th.split(out, C, dim=1)
        loss = th.nn.functional.mse_loss(epsilon, noise.detach())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "backward should produce gradients"

    @pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA")
    def test_compile_bf16_with_pixel_weight_transfer(self):
        """Full training path: pixel weight transfer + bf16 + channels_last + compile."""
        from glide_finetune.latent_util import init_latent_from_pixel

        # Create pixel model and extract weights
        pixel_opts = model_and_diffusion_defaults()
        pixel_opts["use_fp16"] = False
        pixel_model, _ = create_model_and_diffusion(**pixel_opts)
        pixel_sd = pixel_model.state_dict()
        del pixel_model

        # Create latent model and transfer weights (matches load_latent_model path)
        model, diffusion, _opts = _make_latent_model()
        init_latent_from_pixel(model, pixel_sd)
        model.convert_to_bf16()
        model = model.to("cuda")
        model.to(memory_format=th.channels_last)
        model.train()
        th.set_float32_matmul_precision("high")
        compiled = th.compile(model)

        # Replicate latent_train_step
        tokens = th.randint(0, 100, (2, 128), device="cuda")
        masks = th.ones(2, 128, dtype=th.bool, device="cuda")
        clip_emb = th.randn(2, 768, device="cuda")
        latents = th.randn(2, 4, 32, 32, device="cuda")
        timesteps = th.randint(0, len(diffusion.betas) - 1, (2,), device="cuda")
        noise = th.randn_like(latents)
        x_t = diffusion.q_sample(latents, timesteps, noise=noise)

        model_output = compiled(
            x_t, timesteps, tokens=tokens, mask=masks, clip_emb=clip_emb
        )
        C = x_t.shape[1]
        epsilon, _ = th.split(model_output, C, dim=1)
        loss = th.nn.functional.mse_loss(epsilon, noise.detach())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "backward should produce gradients"


# ===================================================================
# Lint / format checks on latent code
# ===================================================================


class TestCodeQuality:
    _LATENT_FILES = [
        "glide_finetune/latent_util.py",
        "glide_finetune/wds_loader.py",
        "glide_finetune/loader.py",
        "tests/",
    ]

    def test_ruff_check_latent_util(self):
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "glide_finetune/latent_util.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ruff check failed:\n{result.stdout}"

    def test_ruff_format(self):
        result = subprocess.run(
            ["uv", "run", "ruff", "format", "--check", *self._LATENT_FILES],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Formatting issues:\n{result.stderr}"
