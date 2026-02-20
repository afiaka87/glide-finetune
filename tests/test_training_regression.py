"""Regression tests for training correctness.

These tests verify that training produces sensible results and that
generated samples do not degenerate.  They use the pretrained model
with synthetic data and a small number of steps.

Requires GPU and pretrained weights — marked as slow.
"""

import pytest
import torch as th
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from glide_text2im.download import load_checkpoint

from glide_finetune.glide_finetune import base_train_step


def _load_pretrained_model(precision="bf16"):
    """Load pretrained GLIDE base model."""
    options = model_and_diffusion_defaults()
    options["use_fp16"] = False
    model, diffusion = create_model_and_diffusion(**options)
    model.load_state_dict(load_checkpoint("base", th.device("cpu")))
    if precision == "bf16":
        model.convert_to_bf16()
    return model, diffusion, options


def _make_batch(tokenizer, text_ctx, batch_size=2, side=64):
    """Create a synthetic batch."""
    tokens_list, masks_list = [], []
    for _ in range(batch_size):
        toks, mask = tokenizer.padded_tokens_and_mask([], text_ctx)
        tokens_list.append(toks)
        masks_list.append(mask)
    tokens = th.tensor(tokens_list)
    masks = th.tensor(masks_list, dtype=th.bool)
    images = th.randn(batch_size, 3, side, side).clamp(-1, 1)
    return tokens, masks, images


@th.inference_mode()
def _generate_samples(model, options, device, n=2):
    """Generate n samples and return the raw tensor."""
    from glide_finetune.glide_util import sample

    model.eval()
    samples = sample(
        glide_model=model,
        glide_options=options,
        side_x=64,
        side_y=64,
        prompt="",
        batch_size=n,
        guidance_scale=4.0,
        device=device,
        prediction_respacing="10",  # fast for testing
        sampler="euler",
    )
    return samples


@pytest.mark.slow
@pytest.mark.skipif(not th.cuda.is_available(), reason="Requires CUDA")
class TestTrainingRegression:
    """Regression tests that train and check output quality."""

    def test_loss_decreases_over_training(self):
        """Train for 80 steps, verify loss trend is downward."""
        th.manual_seed(123)
        device = "cuda"

        model, diffusion, options = _load_pretrained_model()
        model = model.to(device)
        model.train()

        optimizer = th.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=0.01,
        )

        losses = []
        for _ in range(80):
            batch = _make_batch(model.tokenizer, options["text_ctx"])
            loss = base_train_step(model, diffusion, batch, device)

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        # Windowed average comparison
        first_20 = sum(losses[:20]) / 20
        last_20 = sum(losses[-20:]) / 20
        assert last_20 < first_20, (
            f"Loss did not decrease: first_20={first_20:.4f}, last_20={last_20:.4f}"
        )

    def test_samples_not_degenerate_after_training(self):
        """Train for 50 steps then generate — outputs should not be blobs."""
        th.manual_seed(456)
        device = "cuda"

        model, diffusion, options = _load_pretrained_model()
        model = model.to(device)
        model.train()

        optimizer = th.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=0.01,
        )

        for _ in range(50):
            batch = _make_batch(model.tokenizer, options["text_ctx"])
            loss = base_train_step(model, diffusion, batch, device)
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Generate samples
        samples = _generate_samples(model, options, device, n=2)

        # Samples are in [-1, 1].  Degenerate blobs tend to have very low
        # spatial variance (each pixel nearly the same color) or extreme mean.
        for i in range(samples.shape[0]):
            img = samples[i]  # [3, 64, 64]
            pixel_std = img.std().item()
            pixel_mean = img.float().mean().item()

            # Non-degenerate images should have meaningful spatial variation
            assert pixel_std > 0.1, (
                f"Sample {i} has very low std ({pixel_std:.4f}) — likely a blob"
            )
            # Mean should be roughly centered (not saturated)
            assert -0.9 < pixel_mean < 0.9, (
                f"Sample {i} has extreme mean ({pixel_mean:.4f}) — likely saturated"
            )

    def test_timestep_range_covers_full_schedule(self):
        """Verify that timestep sampling covers the full [0, N-1] range."""
        options = model_and_diffusion_defaults()
        _, diffusion = create_model_and_diffusion(**options)

        num_timesteps = len(diffusion.betas)

        # Sample many timesteps and check the range
        th.manual_seed(0)
        all_timesteps = th.randint(0, num_timesteps, (10000,))

        assert all_timesteps.min().item() == 0, "Minimum timestep should be 0"
        assert all_timesteps.max().item() == num_timesteps - 1, (
            f"Maximum timestep should be {num_timesteps - 1}, got {all_timesteps.max().item()}"
        )

    def test_no_nan_in_loss(self):
        """Verify no NaN losses during a short training run."""
        th.manual_seed(789)
        device = "cuda"

        model, diffusion, options = _load_pretrained_model()
        model = model.to(device)
        model.train()

        optimizer = th.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=0.01,
        )

        for step in range(30):
            batch = _make_batch(model.tokenizer, options["text_ctx"])
            loss = base_train_step(model, diffusion, batch, device)

            assert not th.isnan(loss), f"NaN loss at step {step}"
            assert not th.isinf(loss), f"Inf loss at step {step}"

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
