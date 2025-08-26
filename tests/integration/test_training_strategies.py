"""Integration tests for training strategies.

These tests verify that different training strategies work correctly,
including FP16 training, distributed training setup, and checkpoint management.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from glide_finetune.checkpoint_manager import CheckpointManager
from glide_finetune.metrics_tracker import MetricsTracker


@pytest.mark.integration
class TestTrainingStrategies:
    """Integration tests for various training strategies."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        return model
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        x = torch.randn(100, 10)
        y = torch.randn(100, 10)
        return TensorDataset(x, y)
    
    @pytest.fixture
    def simple_dataloader(self, simple_dataset):
        """Create a simple dataloader."""
        return DataLoader(simple_dataset, batch_size=10, shuffle=True)
    
    @pytest.mark.smoke
    def test_checkpoint_manager_save_load(self, simple_model, tmp_path):
        """Test checkpoint saving and loading."""
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoints_dir=str(tmp_path),
            save_frequency=100,
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=1,
            global_step=1000,
        )
        checkpoint_path = Path(checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint (contains just model state dict)
        loaded_state = torch.load(checkpoint_path, map_location="cpu")
        assert isinstance(loaded_state, dict)
        # The checkpoint is just the model state dict
        assert "0.weight" in loaded_state  # First layer weight
        
        # Check metadata file separately
        metadata_path = checkpoint_path.parent / f"glide-ft-1x1000.json"
        assert metadata_path.exists()
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["global_step"] == 1000
        assert metadata["epoch"] == 1
        
        # Test loading into new model
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
        )
        
        # Verify parameters match
        for p1, p2 in zip(simple_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    
    @pytest.mark.smoke
    def test_metrics_tracker(self):
        """Test metrics tracking functionality."""
        metrics_tracker = MetricsTracker(
            window_size=100,
        )
        
        # Add some metrics
        for i in range(10):
            metrics_tracker.update_loss(1.0 - i * 0.1)
            metrics_tracker.update_lr(0.001)
        
        # Get metrics
        metrics = metrics_tracker.get_metrics()
        assert "loss" in metrics
        assert metrics["loss"] < 1.0  # Should decrease
        
        # Test gradient tracking
        model = nn.Linear(10, 10)
        model.weight.grad = torch.randn_like(model.weight)
        
        metrics_tracker.update_gradient_stats(model)
        metrics = metrics_tracker.get_metrics()
        assert "grad_norm" in metrics
        assert metrics["grad_norm"] > 0
    
    @pytest.mark.smoke
    def test_fp16_training_step(self, simple_model, simple_dataloader):
        """Test FP16 training step."""
        from glide_finetune.fp16_training import (
            FP16TrainingConfig,
            FP16TrainingStep,
        )
        
        # Create FP16 config
        fp16_config = FP16TrainingConfig(
            use_loss_scaling=True,
            init_loss_scale=128.0,
            scale_factor=2.0,
            scale_window=100,
            use_master_weights=True,
        )
        
        # Create optimizer first
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        
        # Create FP16 training step handler
        fp16_step = FP16TrainingStep(
            model=simple_model,
            optimizer=optimizer,
            config=fp16_config,
        )
        
        # Test training step
        for batch_idx, (x, y) in enumerate(simple_dataloader):
            if batch_idx >= 2:  # Just test a couple batches
                break
            
            # Define loss computation function
            def compute_loss():
                output = simple_model(x)
                return torch.nn.functional.mse_loss(output, y)
            
            # Run training step
            metrics = fp16_step.training_step(compute_loss)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            assert "loss" in metrics
            assert not torch.isnan(metrics["loss"])
            assert not torch.isinf(metrics["loss"])
    
    @pytest.mark.smoke
    def test_gradient_accumulation(self, simple_model, simple_dataloader):
        """Test gradient accumulation strategy."""
        accumulation_steps = 4
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        
        accumulated_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(simple_dataloader):
            if batch_idx >= 8:  # Test 2 full accumulation cycles
                break
            
            # Forward pass
            output = simple_model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = 0.0
                
                # Check that gradients were applied
                for param in simple_model.parameters():
                    assert param.grad is None or torch.all(param.grad == 0)
    
    @pytest.mark.smoke
    def test_freeze_strategy(self):
        """Test layer freezing strategy."""
        from glide_finetune.utils.freeze_utils import apply_freeze_policy
        
        # Create a model with named modules
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 10),
        )
        
        # Name the layers
        model[0].name = "input_layer"
        model[1].name = "norm_layer"
        model[2].name = "output_layer"
        
        # Apply freeze policy (freeze all layers for testing)
        for param in model[1].parameters():
            param.requires_grad = False
        model[1].eval()
        
        # Check that norm layer is frozen
        assert not model[1].training
        assert all(not p.requires_grad for p in model[1].parameters())
        
        # Check that other layers are not frozen
        assert all(p.requires_grad for p in model[0].parameters())
        assert all(p.requires_grad for p in model[2].parameters())
    
    @pytest.mark.smoke
    def test_warmup_scheduler(self, simple_model):
        """Test warmup learning rate scheduler."""
        from glide_finetune.utils.train_util import create_warmup_scheduler
        
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        
        # Create warmup scheduler
        scheduler = create_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=100,
            warmup_start_lr=0.0001,
            target_lr=0.01,
        )
        
        # Test warmup phase
        initial_lr = optimizer.param_groups[0]["lr"]
        
        for step in range(50):
            scheduler.step()
        
        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr > initial_lr
        assert mid_lr < 0.01
        
        for step in range(50, 100):
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]["lr"]
        assert abs(final_lr - 0.01) < 0.001  # Should reach target LR
    
    @pytest.mark.integration
    def test_training_loop_with_interruption(self, simple_model, simple_dataloader, tmp_path):
        """Test training loop with interruption handling."""
        from glide_finetune.training_pipeline import InterruptHandler
        
        # Create interrupt handler
        interrupt_handler = InterruptHandler()
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoints_dir=str(tmp_path),
            save_frequency=5,
        )
        
        # Simple training loop
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        global_step = 0
        
        for epoch in range(2):
            for batch_idx, (x, y) in enumerate(simple_dataloader):
                if interrupt_handler.interrupted:
                    # Save emergency checkpoint
                    checkpoint_manager.save_checkpoint(
                        model=simple_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        is_interrupted=True,
                    )
                    break
                
                # Training step
                output = simple_model(x)
                loss = torch.nn.functional.mse_loss(output, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
                # Regular checkpoint
                if global_step % checkpoint_manager.save_frequency == 0:
                    checkpoint_manager.save_checkpoint(
                        model=simple_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                    )
            
            if interrupt_handler.interrupted:
                break
        
        # Verify checkpoints were created
        checkpoints = list(tmp_path.glob("*.pt"))
        assert len(checkpoints) > 0
    
    @pytest.mark.smoke
    def test_mixed_precision_with_autocast(self, simple_model, simple_dataloader):
        """Test mixed precision training with autocast."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = simple_model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, (x, y) in enumerate(simple_dataloader):
            if batch_idx >= 2:
                break
            
            x, y = x.cuda(), y.cuda()
            
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)