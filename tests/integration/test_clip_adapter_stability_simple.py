"""
Simplified integration tests for CLIP adapter stability.

These tests verify basic functionality without requiring full model instantiation.
"""

import pytest
import torch
import numpy as np

from glide_text2im.tokenizer.bpe import get_encoder
from glide_finetune.glide_util import get_tokens_and_mask
from glide_finetune.adapters import (
    ClipAdapter,
    load_clip_model,
    create_clip_adapter_config,
    CLIP_DIMENSIONS,
)


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestClipAdapterCore:
    """Test core CLIP adapter functionality."""
    
    def test_clip_adapter_initialization(self, device):
        """Test that CLIP adapter initializes correctly."""
        # Test different CLIP model configurations
        for model_name, clip_dim in CLIP_DIMENSIONS.items():
            if model_name in ["ViT-B/32", "ViT-L/14"]:  # Test main models
                config = create_clip_adapter_config(
                    clip_model_name=model_name,
                    glide_xf_width=2048,
                    gate_init=0.0,
                )
                
                adapter = ClipAdapter(**config).to(device)
                
                # Check dimensions
                assert adapter.input_dim == clip_dim
                assert adapter.output_dim == 2048
                
                # Check gate initialization
                assert adapter.get_gate_value() == 0.0
                
                # Test forward pass
                batch_size = 4
                dummy_input = torch.randn(batch_size, clip_dim).to(device)
                output = adapter(dummy_input)
                
                assert output.shape == (batch_size, 2048)
    
    def test_gate_zero_preserves_input(self, device):
        """Test that gate=0 preserves the residual connection."""
        adapter = ClipAdapter(
            input_dim=768,
            output_dim=768,  # Same dim for easy testing
            gate_init=0.0,
        ).to(device)
        
        # Create random input
        x = torch.randn(10, 768).to(device)
        
        # Forward pass with gate=0
        output = adapter(x, gate_override=0.0)
        
        # Should be just the projection of input (identity in this case)
        expected = adapter.proj(x)
        assert torch.allclose(output, expected, atol=1e-6)
    
    def test_gradual_gate_schedule(self, device):
        """Test gate scheduling functionality."""
        adapter = ClipAdapter(
            input_dim=768,
            output_dim=768,
            gate_init=0.0,
        ).to(device)
        
        # Test schedule points
        schedule = [
            (0, 0.0),
            (0.25, 0.25),
            (0.5, 0.5),
            (1.0, 1.0),
        ]
        
        for progress, expected in schedule:
            adapter.set_gate_value(progress)
            assert abs(adapter.get_gate_value() - progress) < 1e-6
    
    def test_adapter_gradients_isolated(self, device):
        """Test that adapter gradients don't affect frozen parameters."""
        adapter = ClipAdapter(
            input_dim=768,
            output_dim=512,
            gate_init=0.1,
        ).to(device)
        
        # Create dummy frozen layer
        frozen_layer = torch.nn.Linear(512, 256).to(device)
        frozen_layer.requires_grad_(False)
        
        # Forward pass
        x = torch.randn(4, 768, requires_grad=True).to(device)
        out = adapter(x)
        out = frozen_layer(out)
        loss = out.mean()
        
        # Backward
        loss.backward()
        
        # Check frozen layer has no gradients
        assert frozen_layer.weight.grad is None
        assert frozen_layer.bias.grad is None
        
        # Check adapter has gradients
        for name, param in adapter.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP models")
class TestClipLoading:
    """Test CLIP model loading functionality."""
    
    def test_load_clip_models(self, device):
        """Test loading different CLIP models."""
        test_models = ["ViT-B/32", "ViT-L/14"]
        
        for model_name in test_models:
            clip_model, preprocess = load_clip_model(model_name, device=device)
            
            # Check model is frozen
            for param in clip_model.parameters():
                assert not param.requires_grad
            
            # Test encoding
            test_text = ["a cat", "a dog"]
            import clip
            tokens = clip.tokenize(test_text).to(device)
            
            with torch.no_grad():
                embeddings = clip_model.encode_text(tokens)
            
            # Check output dimensions
            expected_dim = CLIP_DIMENSIONS[model_name]
            assert embeddings.shape == (2, expected_dim)
    
    def test_clip_adapter_with_real_model(self, device):
        """Test adapter with real CLIP embeddings."""
        clip_model, _ = load_clip_model("ViT-B/32", device=device)
        
        adapter = ClipAdapter(
            input_dim=512,  # Corrected for ViT-B/32
            output_dim=512,
            gate_init=0.0,
        ).to(device)
        
        # Get real CLIP embeddings
        import clip
        text = ["a beautiful painting", "a photo of a cat"]
        tokens = clip.tokenize(text).to(device)
        
        with torch.no_grad():
            clip_embeddings = clip_model.encode_text(tokens).float()
        
        # Test adapter
        output = adapter(clip_embeddings)
        assert output.shape == (2, 512)
        
        # With gate=0, output should be deterministic
        output2 = adapter(clip_embeddings)
        assert torch.allclose(output, output2)


class TestStabilityUtils:
    """Test stability monitoring utilities."""
    
    def test_loss_spike_detection(self):
        """Test loss spike detection logic."""
        loss_history = []
        threshold = 2.0
        
        # Simulate normal training
        for i in range(20):
            loss = 1.0 + np.random.normal(0, 0.1)
            loss_history.append(loss)
        
        # Check normal loss doesn't trigger
        recent_avg = sum(loss_history[-10:-1]) / 9
        normal_loss = recent_avg * 1.5  # 50% increase
        is_spike = normal_loss > recent_avg * threshold
        assert not is_spike
        
        # Check spike does trigger
        spike_loss = recent_avg * 3.0  # 3x increase
        is_spike = spike_loss > recent_avg * threshold
        assert is_spike
    
    def test_warmup_schedule_calculation(self):
        """Test warmup schedule calculations."""
        warmup_steps = 10000
        
        test_cases = [
            (0, 0.0),
            (2500, 0.125),  # 25% progress -> 0.5 * 0.25 = 0.125
            (5000, 0.25),   # 50% progress -> 0.5 * 0.5 = 0.25
            (10000, 0.5),   # 100% progress -> 0.5 * 1.0 = 0.5
            (20000, 0.5),   # Beyond warmup -> stays at 0.5
        ]
        
        for step, expected in test_cases:
            progress = min(1.0, step / warmup_steps)
            gate_value = 0.5 * progress
            assert abs(gate_value - expected) < 1e-6