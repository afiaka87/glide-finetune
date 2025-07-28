"""Integration tests for freezing strategies."""

import pytest
import torch as th
import torch.nn.functional as F
from pathlib import Path

from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset


@pytest.mark.gpu
class TestFreezing:
    """Test freezing strategies with actual training."""
    
    def setup_method(self):
        """Set up test data."""
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.batch_size = 2
        self.steps = 100
        
    def create_dummy_batch(self, model):
        """Create a dummy batch for training."""
        tokens = th.randint(0, 1000, (self.batch_size, 128), device=self.device)
        mask = th.ones_like(tokens, dtype=th.bool)
        timesteps = th.randint(0, 1000, (self.batch_size,), device=self.device)
        images = th.randn(self.batch_size, 3, 64, 64, device=self.device)
        noise = th.randn_like(images)
        
        return {
            "tokens": tokens,
            "mask": mask,
            "images": images,
            "timesteps": timesteps,
            "noise": noise,
        }
    
    def verify_parameter_counts(self, model, expected_trainable_ratio):
        """Verify parameter counts match expectations."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        actual_ratio = trainable_params / total_params
        
        print(f"\nParameter summary:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({actual_ratio*100:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        
        # Allow 1% tolerance for ratio
        assert abs(actual_ratio - expected_trainable_ratio) < 0.01, \
            f"Expected {expected_trainable_ratio*100:.1f}% trainable, got {actual_ratio*100:.1f}%"
        
        return trainable_params, frozen_params
    
    def train_steps(self, model, diffusion, optimizer):
        """Train for a fixed number of steps."""
        model.train()
        losses = []
        
        for step in range(self.steps):
            batch = self.create_dummy_batch(model)
            
            # Add noise to images
            t = batch["timesteps"]
            noise = batch["noise"]
            images = batch["images"]
            
            # Forward diffusion process
            noised_images = diffusion.q_sample(images, t, noise=noise)
            
            # Model prediction
            model_output = model(
                noised_images,
                t,
                tokens=batch["tokens"],
                mask=batch["mask"]
            )
            
            # Compute loss (simplified - just MSE on noise prediction)
            pred_noise = model_output[:, :3]  # First 3 channels are noise prediction
            loss = F.mse_loss(pred_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            if step == 0:
                self.verify_gradients(model)
            
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 20 == 0:
                print(f"  Step {step}: loss = {loss.item():.4f}")
        
        return losses
    
    def verify_gradients(self, model):
        """Verify gradients are computed only for trainable parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for trainable parameter: {name}"
            else:
                assert param.grad is None, f"Gradient computed for frozen parameter: {name}"
    
    def test_freeze_transformer(self):
        """Test training with frozen transformer."""
        print("\n=== Testing freeze_transformer ===")
        
        # Load model with frozen transformer
        model, diffusion, _ = load_model(freeze_transformer=True)
        model = model.to(self.device)
        
        # Verify parameter counts (expect ~80.1% trainable)
        self.verify_parameter_counts(model, expected_trainable_ratio=0.801)
        
        # Create optimizer only for trainable parameters
        optimizer = th.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5
        )
        
        # Train for 100 steps
        losses = self.train_steps(model, diffusion, optimizer)
        
        # Verify training is working (loss should change)
        assert losses[0] != losses[-1], "Loss didn't change during training"
        print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    
    def test_freeze_diffusion(self):
        """Test training with frozen diffusion."""
        print("\n=== Testing freeze_diffusion ===")
        
        # Load model with frozen diffusion
        model, diffusion, _ = load_model(freeze_diffusion=True)
        model = model.to(self.device)
        
        # Verify parameter counts (expect ~20.1% trainable)
        self.verify_parameter_counts(model, expected_trainable_ratio=0.201)
        
        # Create optimizer only for trainable parameters
        optimizer = th.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5
        )
        
        # Train for 100 steps
        losses = self.train_steps(model, diffusion, optimizer)
        
        # Verify training is working (loss should change)
        assert losses[0] != losses[-1], "Loss didn't change during training"
        print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    
    def test_freeze_both(self):
        """Test training with both components frozen."""
        print("\n=== Testing freeze_transformer + freeze_diffusion ===")
        
        # Load model with both frozen
        model, diffusion, _ = load_model(freeze_transformer=True, freeze_diffusion=True)
        model = model.to(self.device)
        
        # Verify parameter counts (expect ~0.2% trainable)
        self.verify_parameter_counts(model, expected_trainable_ratio=0.002)
        
        # Create optimizer only for trainable parameters
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(trainable_params) > 0, "No trainable parameters found"
        
        optimizer = th.optim.AdamW(trainable_params, lr=1e-5)
        
        # Train for 100 steps
        losses = self.train_steps(model, diffusion, optimizer)
        
        # Loss might not change much with so few parameters
        print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
        
        # Verify only time_embed has gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "time_embed" in name, f"Unexpected trainable parameter: {name}"
    
    def test_no_freezing(self):
        """Test training without freezing (baseline)."""
        print("\n=== Testing no freezing (baseline) ===")
        
        # Load model without freezing
        model, diffusion, _ = load_model()
        model = model.to(self.device)
        
        # Verify parameter counts (expect 100% trainable)
        self.verify_parameter_counts(model, expected_trainable_ratio=1.0)
        
        # Create optimizer for all parameters
        optimizer = th.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Train for 100 steps
        losses = self.train_steps(model, diffusion, optimizer)
        
        # Verify training is working (loss should change)
        assert losses[0] != losses[-1], "Loss didn't change during training"
        print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    # Run tests directly
    test = TestFreezing()
    test.setup_method()
    
    test.test_no_freezing()
    test.test_freeze_transformer()
    test.test_freeze_diffusion()
    test.test_freeze_both()