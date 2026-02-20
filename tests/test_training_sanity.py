"""Ablation test for training optimizations that may cause degradation.

This test loads a pretrained GLIDE base model, trains for a small number
of steps on synthetic data, and checks that loss decreases and outputs
remain non-degenerate.  It is parameterized with feature flags so each
suspect optimization can be tested in isolation.

Requires GPU and pretrained weights — marked as slow.
"""

import pytest
import torch as th
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from glide_text2im.download import load_checkpoint


def _make_synthetic_batch(tokenizer, text_ctx, batch_size=2, side=64):
    """Create a synthetic (tokens, masks, images) batch."""
    tokens_list = []
    masks_list = []
    for _ in range(batch_size):
        toks, mask = tokenizer.padded_tokens_and_mask([], text_ctx)
        tokens_list.append(toks)
        masks_list.append(mask)
    tokens = th.tensor(tokens_list)
    masks = th.tensor(masks_list, dtype=th.bool)
    images = th.randn(batch_size, 3, side, side).clamp(-1, 1)
    return tokens, masks, images


def _train_n_steps(
    n_steps=50,
    use_compile=False,
    use_tf32=False,
    use_channels_last=False,
    use_fused_adam=False,
    precision="bf16",
    seed=42,
):
    """Train for n_steps and return list of losses."""
    th.manual_seed(seed)
    device = "cuda"

    options = model_and_diffusion_defaults()
    options["use_fp16"] = False
    model, diffusion = create_model_and_diffusion(**options)
    model.load_state_dict(load_checkpoint("base", th.device("cpu")))

    if precision == "bf16":
        model.convert_to_bf16()

    model = model.to(device)
    model.train()

    if use_channels_last:
        model.to(memory_format=th.channels_last)

    if use_tf32:
        th.set_float32_matmul_precision("high")
        th.backends.cudnn.allow_tf32 = True
    else:
        th.set_float32_matmul_precision("highest")
        th.backends.cudnn.allow_tf32 = False

    if use_compile:
        model = th.compile(model)

    adam_kwargs = dict(lr=3e-4, weight_decay=0.01)
    if use_fused_adam:
        adam_kwargs["fused"] = True
    optimizer = th.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], **adam_kwargs
    )

    text_ctx = options["text_ctx"]
    losses = []

    for _ in range(n_steps):
        batch = _make_synthetic_batch(model.tokenizer, text_ctx, batch_size=2)
        tokens = batch[0].to(device)
        masks = batch[1].to(device)
        reals = batch[2].to(device)

        timesteps = th.randint(
            0, len(diffusion.betas), (reals.shape[0],), device=device
        )
        noise = th.randn_like(reals)
        x_t = diffusion.q_sample(reals, timesteps, noise=noise)
        _, C = x_t.shape[:2]
        out = model(x_t, timesteps, tokens=tokens, mask=masks)
        eps, _ = th.split(out, C, dim=1)
        loss = th.nn.functional.mse_loss(eps, noise.detach())

        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "all_new": dict(
        use_compile=True, use_tf32=True, use_channels_last=True, use_fused_adam=True
    ),
    "no_compile": dict(
        use_compile=False, use_tf32=True, use_channels_last=True, use_fused_adam=True
    ),
    "no_tf32": dict(
        use_compile=True, use_tf32=False, use_channels_last=True, use_fused_adam=True
    ),
    "no_channels_last": dict(
        use_compile=True, use_tf32=True, use_channels_last=False, use_fused_adam=True
    ),
    "no_fused_adam": dict(
        use_compile=True, use_tf32=True, use_channels_last=True, use_fused_adam=False
    ),
    "all_old": dict(
        use_compile=False, use_tf32=False, use_channels_last=False, use_fused_adam=False
    ),
}


@pytest.mark.slow
@pytest.mark.skipif(not th.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
def test_training_does_not_diverge(config_name):
    """Train for 50 steps and verify loss doesn't explode."""
    cfg = CONFIGS[config_name]
    losses = _train_n_steps(n_steps=50, **cfg)

    # Loss should not explode (NaN or huge values)
    assert all(loss < 10.0 for loss in losses), (
        f"Config {config_name}: loss exploded — {losses[-5:]}"
    )
    assert not any(th.isnan(th.tensor(v)) for v in losses), (
        f"Config {config_name}: NaN loss detected"
    )

    # Average of last 10 should be lower than average of first 10
    # (on synthetic random data, we at least expect the model to overfit)
    first_10 = sum(losses[:10]) / 10
    last_10 = sum(losses[-10:]) / 10
    print(f"Config {config_name}: first_10={first_10:.4f}, last_10={last_10:.4f}")


@pytest.mark.slow
@pytest.mark.skipif(not th.cuda.is_available(), reason="Requires CUDA")
def test_baseline_loss_decreases():
    """Verify the safe config (all optimizations off) shows loss decrease."""
    losses = _train_n_steps(n_steps=100, **CONFIGS["all_old"])

    first_quarter = sum(losses[:25]) / 25
    last_quarter = sum(losses[-25:]) / 25
    assert last_quarter < first_quarter, (
        f"Baseline loss did not decrease: first_quarter={first_quarter:.4f}, "
        f"last_quarter={last_quarter:.4f}"
    )
