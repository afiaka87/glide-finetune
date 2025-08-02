#!/usr/bin/env python3
"""
Unit tests for early stopping functionality in ClipAdapterTrainer.

Tests that the trainer properly monitors pretrained model performance
and stops training if degradation is detected.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

from glide_finetune.adapters.glide_clip_integration import ClipAdapterTrainer


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock ClipText2ImUNet model."""
        model = Mock()
        model.use_clip = True
        model.parameters = Mock(return_value=[])
        model.get_adapter_params = Mock(return_value=[])
        model.get_stability_metrics = Mock(return_value={
            'adapter_gate': 0.0,
            'attention_gate_mean': 0.0
        })
        model.set_adapter_gate_schedule = Mock()
        return model
    
    @pytest.fixture
    def mock_diffusion(self):
        """Create a mock diffusion object."""
        diffusion = Mock()
        
        # Mock q_sample to return noisy version
        def mock_q_sample(x0, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x0)
            return x0 + 0.1 * noise
        
        diffusion.q_sample = mock_q_sample
        
        # Mock training_losses to return consistent loss
        def mock_training_losses(model, x_t, t, model_kwargs=None, noise=None):
            batch_size = x_t.shape[0]
            # Return slightly different losses based on use_clip setting
            base_loss = 0.5 if model.use_clip else 0.4
            loss_tensor = torch.full((batch_size,), base_loss)
            return {'loss': loss_tensor}
        
        diffusion.training_losses = mock_training_losses
        
        return diffusion
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = Mock()
        optimizer.step = Mock()
        optimizer.zero_grad = Mock()
        optimizer.state_dict = Mock(return_value={})
        optimizer.load_state_dict = Mock()
        return optimizer
    
    @pytest.fixture
    def trainer(self, mock_model, mock_diffusion, mock_optimizer):
        """Create a ClipAdapterTrainer instance."""
        return ClipAdapterTrainer(
            model=mock_model,
            diffusion=mock_diffusion,
            optimizer=mock_optimizer,
            warmup_steps=1000,
            stability_threshold=10.0,
            checkpoint_dir="./test_checkpoints",
            adapter_grad_clip=1.0,
            main_grad_clip=1.0,
            early_stop_threshold=0.1,  # 10% degradation threshold
            early_stop_patience=50,    # Small patience for testing
            baseline_eval_interval=10,  # Check frequently for testing
        )
    
    def test_baseline_performance_evaluation(self, trainer):
        """Test that baseline performance is correctly evaluated."""
        # Create test batch
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
            'tokens': torch.randint(0, 1000, (4, 128)),
            'mask': torch.ones(4, 128, dtype=torch.bool),
        }
        
        # Evaluate baseline
        baseline_loss = trainer.evaluate_baseline_performance(batch, num_samples=2)
        
        # Should return the baseline loss (0.4 without CLIP)
        assert abs(baseline_loss - 0.4) < 1e-6
        
        # Model's use_clip should be restored
        assert trainer.model.use_clip == True
    
    def test_early_stopping_initialization(self, trainer):
        """Test that early stopping state is properly initialized."""
        assert trainer.baseline_performance is None
        assert trainer.degradation_detected_step is None
        assert trainer.should_stop is False
        assert trainer.early_stop_threshold == 0.1
        assert trainer.early_stop_patience == 50
        assert trainer.baseline_eval_interval == 10
    
    def test_early_stopping_not_triggered_below_threshold(self, trainer):
        """Test that early stopping is not triggered when performance is stable."""
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
            'tokens': torch.randint(0, 1000, (4, 128)),
            'mask': torch.ones(4, 128, dtype=torch.bool),
        }
        
        # Set baseline performance
        trainer.baseline_performance = 0.4
        trainer.step = 10  # At eval interval
        
        # Mock evaluate_baseline_performance to return slight improvement
        trainer.evaluate_baseline_performance = Mock(return_value=0.39)
        
        # Check early stopping
        should_stop = trainer.check_early_stopping(batch)
        
        assert should_stop is False
        assert trainer.baseline_performance == 0.39  # Updated to better value
        assert trainer.degradation_detected_step is None
    
    def test_early_stopping_triggered_after_patience(self, trainer):
        """Test that early stopping is triggered after patience is exceeded."""
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
            'tokens': torch.randint(0, 1000, (4, 128)),
            'mask': torch.ones(4, 128, dtype=torch.bool),
        }
        
        # Set baseline performance
        trainer.baseline_performance = 0.4
        
        # First detection of degradation
        trainer.step = 10
        trainer.evaluate_baseline_performance = Mock(return_value=0.45)  # 12.5% worse
        should_stop = trainer.check_early_stopping(batch)
        
        assert should_stop is False
        assert trainer.degradation_detected_step == 10
        
        # Still within patience period
        trainer.step = 50
        should_stop = trainer.check_early_stopping(batch)
        assert should_stop is False
        
        # Patience exceeded
        trainer.step = 60
        should_stop = trainer.check_early_stopping(batch)
        assert should_stop is True
        assert trainer.should_stop is True
    
    def test_early_stopping_recovery(self, trainer):
        """Test that early stopping resets if performance recovers."""
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
            'tokens': torch.randint(0, 1000, (4, 128)),
            'mask': torch.ones(4, 128, dtype=torch.bool),
        }
        
        # Set baseline and detect degradation
        trainer.baseline_performance = 0.4
        trainer.step = 10
        trainer.evaluate_baseline_performance = Mock(return_value=0.45)  # Degraded
        trainer.check_early_stopping(batch)
        
        assert trainer.degradation_detected_step == 10
        
        # Performance recovers
        trainer.step = 20
        trainer.evaluate_baseline_performance = Mock(return_value=0.41)  # Within threshold
        should_stop = trainer.check_early_stopping(batch)
        
        assert should_stop is False
        assert trainer.degradation_detected_step is None  # Reset
    
    def test_training_step_with_early_stopping(self, trainer):
        """Test that training_step integrates early stopping checks."""
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
            'tokens': torch.randint(0, 1000, (4, 128)),
            'mask': torch.ones(4, 128, dtype=torch.bool),
        }
        
        # Mock compute_loss_fn
        mock_loss = torch.tensor(0.5, requires_grad=True)
        compute_loss_fn = Mock(return_value=mock_loss)
        
        # Mock methods
        trainer.update_gates = Mock()
        trainer.clip_gradients = Mock(return_value={'grad_norm_adapter_pre_clip': 0.5})
        trainer.check_stability = Mock(return_value=True)
        trainer.check_early_stopping = Mock(return_value=False)
        trainer.get_metrics = Mock(return_value={'adapter_gate': 0.1})
        
        # Run training step
        metrics = trainer.training_step(batch, compute_loss_fn)
        
        # Verify calls
        trainer.update_gates.assert_called_once()
        trainer.check_early_stopping.assert_called_once_with(batch)
        
        # Check metrics
        assert metrics['loss'] == 0.5
        assert metrics['step'] == 1
        assert metrics['should_stop'] is False
        assert 'grad_norm_adapter_pre_clip' in metrics
    
    def test_skip_check_outside_interval(self, trainer):
        """Test that early stopping check is skipped outside evaluation interval."""
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
        }
        
        # Not at evaluation interval
        trainer.step = 5  # baseline_eval_interval is 10
        trainer.evaluate_baseline_performance = Mock()
        
        should_stop = trainer.check_early_stopping(batch)
        
        # Should not evaluate
        trainer.evaluate_baseline_performance.assert_not_called()
        assert should_stop is False
    
    def test_early_stopping_metrics_in_training_step(self, trainer):
        """Test that early stopping metrics are included in training step output."""
        batch = {
            'x': torch.randn(4, 3, 64, 64),
            'timesteps': torch.randint(0, 100, (4,)),
            'tokens': torch.randint(0, 1000, (4, 128)),
            'mask': torch.ones(4, 128, dtype=torch.bool),
        }
        
        # Set up state
        trainer.baseline_performance = 0.4
        trainer.degradation_detected_step = 10
        trainer.step = 20
        
        # Mock methods
        mock_loss = torch.tensor(0.5, requires_grad=True)
        compute_loss_fn = Mock(return_value=mock_loss)
        trainer.update_gates = Mock()
        trainer.clip_gradients = Mock(return_value={})
        trainer.check_stability = Mock(return_value=True)
        trainer.check_early_stopping = Mock(return_value=False)
        trainer.get_metrics = Mock(return_value={})
        
        # Run training step
        metrics = trainer.training_step(batch, compute_loss_fn)
        
        # Check early stopping metrics
        assert 'baseline_performance' in metrics
        assert metrics['baseline_performance'] == 0.4
        assert 'steps_since_degradation' in metrics
        assert metrics['steps_since_degradation'] == 11  # 21 - 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])