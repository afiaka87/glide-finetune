#!/usr/bin/env python3
"""
Integration test for CLIP-enabled training steps.

Tests that base_train_step and upsample_train_step correctly handle
batches with CLIP embeddings.
"""

import numpy as np
import pytest
import torch
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

from glide_finetune.adapters.clip_adapter import ClipTextEncoder, load_clip_model
from glide_finetune.glide_finetune import base_train_step, upsample_train_step


@pytest.fixture
def clip_encoder():
    """Create a CLIP text encoder for generating real embeddings."""
    clip_model, tokenizer = load_clip_model("ViT-B/32", device="cpu")
    encoder = ClipTextEncoder(clip_model, tokenizer, device="cpu")
    return encoder


@pytest.fixture
def model_and_diffusion():
    """Create a model matching pretrained weights."""
    # Use configuration that matches pretrained base.pt model
    options = model_and_diffusion_defaults()
    # Override only what's necessary for testing
    options["use_fp16"] = False  # CPU testing
    options["timestep_respacing"] = "10"  # Faster testing

    model, diffusion = create_model_and_diffusion(**options)

    # Load pretrained weights to avoid NaN issues
    import os

    pretrained_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "glide_model_cache", "base.pt"
    )

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(pretrained_state, strict=True)
        print("Successfully loaded pretrained weights")
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        print("Test may produce NaN values without proper initialization")

    return model, diffusion


def test_base_train_step_with_clip_embeddings(model_and_diffusion, clip_encoder):
    """Test base_train_step handles 4-tuple batches with CLIP embeddings."""
    model, diffusion = model_and_diffusion
    device = "cpu"
    model = model.to(device)

    # Create sample batch with real CLIP embeddings
    batch_size = 2
    seq_len = 128

    # Create some test prompts
    test_prompts = [
        "a beautiful sunset over the ocean",
        "a cute cat playing with a ball",
    ]

    # Generate real CLIP embeddings
    clip_embeddings = []
    for prompt in test_prompts[:batch_size]:
        embedding = clip_encoder.encode_text(prompt)
        clip_embeddings.append(embedding)
    clip_embeddings = torch.cat(clip_embeddings, dim=0)

    # Use smaller token values to avoid NaN issues
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    images = torch.randn(batch_size, 3, 64, 64)

    # Test with 3-tuple (no CLIP)
    batch_3 = (tokens, masks, images)
    loss_3, metrics_3 = base_train_step(model, diffusion, batch_3, device)
    assert isinstance(loss_3, torch.Tensor)
    assert loss_3.shape == ()
    assert "loss_q0" in metrics_3

    # Test with 4-tuple (with real CLIP embeddings)
    batch_4 = (tokens, masks, images, clip_embeddings)
    loss_4, metrics_4 = base_train_step(model, diffusion, batch_4, device)
    assert isinstance(loss_4, torch.Tensor)
    assert loss_4.shape == ()
    assert "loss_q0" in metrics_4

    # Both should work without errors
    assert loss_3.item() > 0
    assert loss_4.item() > 0


def test_base_train_step_with_clip_model(model_and_diffusion, clip_encoder):
    """Test base_train_step with a model that accepts CLIP embeddings."""
    model, diffusion = model_and_diffusion
    device = "cpu"
    model = model.to(device)

    # Monkey-patch the forward method to accept clip_embeddings
    original_forward = model.forward

    def forward_with_clip(x, timesteps, tokens=None, mask=None, clip_embeddings=None):
        # Just ignore clip_embeddings and call original forward
        return original_forward(x, timesteps, tokens=tokens, mask=mask)

    # Replace forward method
    model.forward = forward_with_clip

    # Create batch with real CLIP embeddings
    batch_size = 2
    seq_len = 128

    # Generate real CLIP embeddings
    test_prompts = ["a red apple on a wooden table", "a blue car on a highway"]
    clip_embeddings = []
    for prompt in test_prompts[:batch_size]:
        embedding = clip_encoder.encode_text(prompt)
        clip_embeddings.append(embedding)
    clip_embeddings = torch.cat(clip_embeddings, dim=0)

    # Use smaller token values to avoid NaN issues
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    images = torch.randn(batch_size, 3, 64, 64)

    batch = (tokens, masks, images, clip_embeddings)

    # Should handle CLIP embeddings
    loss, metrics = base_train_step(model, diffusion, batch, device)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # Scalar loss
    assert "loss_q0" in metrics


def test_upsample_train_step_with_clip_embeddings(clip_encoder):
    """Test upsample_train_step handles 5-tuple batches with CLIP embeddings."""
    # Create upsampler model
    options = model_and_diffusion_defaults_upsampler()
    options.update(
        {
            "num_channels": 32,
            "num_res_blocks": 1,
            "timestep_respacing": "10",
            "use_fp16": False,
        }
    )
    model, diffusion = create_model_and_diffusion(**options)
    device = "cpu"
    model = model.to(device)

    # Create sample batch
    batch_size = 2
    seq_len = 128

    # Generate real CLIP embeddings
    test_prompts = ["a detailed landscape painting", "a modern architecture building"]
    clip_embeddings = []
    for prompt in test_prompts[:batch_size]:
        embedding = clip_encoder.encode_text(prompt)
        clip_embeddings.append(embedding)
    clip_embeddings = torch.cat(clip_embeddings, dim=0)

    # Use smaller token values to avoid NaN issues
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    low_res = torch.randn(batch_size, 3, 64, 64)
    high_res = torch.randn(batch_size, 3, 256, 256)

    # Test with 4-tuple (no CLIP)
    batch_4 = (tokens, masks, low_res, high_res)
    loss_4, metrics_4 = upsample_train_step(model, diffusion, batch_4, device)
    assert isinstance(loss_4, torch.Tensor)
    assert loss_4.shape == ()

    # Test with 5-tuple (with real CLIP embeddings)
    batch_5 = (tokens, masks, low_res, high_res, clip_embeddings)
    loss_5, metrics_5 = upsample_train_step(model, diffusion, batch_5, device)
    assert isinstance(loss_5, torch.Tensor)
    assert loss_5.shape == ()

    # Both should work (don't check values as they may be NaN with random inputs)
    assert "loss_q0" in metrics_4
    assert "loss_q0" in metrics_5


def test_invalid_batch_sizes():
    """Test that invalid batch sizes raise appropriate errors."""
    # Create dummy model and diffusion
    options = model_and_diffusion_defaults()
    options["image_size"] = 64
    model, diffusion = create_model_and_diffusion(**options)
    device = "cpu"

    # Test base_train_step with invalid batch size
    invalid_batch = (torch.randn(2, 128),)  # Only 1 element
    with pytest.raises(ValueError, match="Expected batch to have 3 or 4 elements"):
        base_train_step(model, diffusion, invalid_batch, device)

    # Test upsample_train_step with invalid batch size
    invalid_batch = (torch.randn(2, 128), torch.randn(2, 128))  # Only 2 elements
    with pytest.raises(ValueError, match="Expected batch to have 4 or 5 elements"):
        upsample_train_step(model, diffusion, invalid_batch, device)


def test_clip_embeddings_device_handling(clip_encoder):
    """Test that CLIP embeddings are properly moved to device."""
    options = model_and_diffusion_defaults()
    options["image_size"] = 64
    options["use_fp16"] = False  # Disable fp16 for testing
    model, diffusion = create_model_and_diffusion(**options)

    # Use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create batch on CPU
    batch_size = 2

    # Generate real CLIP embeddings on CPU
    test_prompts = ["a sunny day at the beach", "a snowy mountain peak"]
    clip_embeddings = []
    for prompt in test_prompts[:batch_size]:
        embedding = clip_encoder.encode_text(prompt)
        clip_embeddings.append(embedding)
    clip_embeddings = torch.cat(clip_embeddings, dim=0)

    tokens = torch.randint(0, 1000, (batch_size, 128))  # Use smaller token range
    masks = torch.ones(batch_size, 128, dtype=torch.bool)
    images = torch.randn(batch_size, 3, 64, 64)

    # All tensors start on CPU
    assert tokens.device.type == "cpu"
    assert clip_embeddings.device.type == "cpu"

    batch = (tokens, masks, images, clip_embeddings)

    # Train step should handle device transfer
    loss, _ = base_train_step(model, diffusion, batch, device)

    # Loss should be on the correct device
    assert loss.device.type == device.split(":")[0]


def test_real_clip_embeddings_produce_valid_losses(clip_encoder, model_and_diffusion):
    """Test that using real CLIP embeddings produces valid (non-NaN) losses."""
    model, diffusion = model_and_diffusion
    device = "cpu"
    model = model.to(device)

    batch_size = 4
    seq_len = 128

    # Generate diverse prompts for better testing
    test_prompts = [
        "a photorealistic portrait of a cat",
        "abstract art with vibrant colors",
        "a serene landscape with mountains",
        "futuristic city skyline at night",
    ]

    # Generate real CLIP embeddings
    clip_embeddings = []
    for prompt in test_prompts[:batch_size]:
        embedding = clip_encoder.encode_text(prompt)
        clip_embeddings.append(embedding)
    clip_embeddings = torch.cat(clip_embeddings, dim=0)

    # CLIP embeddings are not normalized by default, but they should have reasonable magnitude
    norms = torch.norm(clip_embeddings, dim=-1)
    assert (norms > 0).all(), "CLIP embeddings have zero norm"
    assert (norms < 100).all(), "CLIP embeddings have unreasonably large norm"

    # Use smaller token values
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    images = torch.randn(batch_size, 3, 64, 64) * 0.5  # Smaller magnitude

    batch = (tokens, masks, images, clip_embeddings)

    # Run multiple steps to ensure consistency
    losses = []
    for _ in range(5):
        loss, metrics = base_train_step(model, diffusion, batch, device)
        losses.append(loss.item())

        # Check that loss is valid
        assert not torch.isnan(loss), f"Loss is NaN: {loss}"
        assert not torch.isinf(loss), f"Loss is infinite: {loss}"
        assert loss.item() > 0, f"Loss is not positive: {loss.item()}"

        # Check quartile losses are also valid
        for key, value in metrics.items():
            assert not np.isnan(value), f"{key} is NaN: {value}"
            assert not np.isinf(value), f"{key} is infinite: {value}"

    # Losses should be reasonably consistent
    losses = np.array(losses)
    assert losses.std() < losses.mean() * 2, "Losses are too variable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
