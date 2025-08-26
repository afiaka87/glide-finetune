"""Unit tests for configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from glide_finetune.settings import (
    CheckpointSettings,
    DatasetSettings,
    FP16Mode,
    FP16Settings,
    ModelSettings,
    SamplerType,
    SamplingSettings,
    Settings,
    SystemSettings,
    TrainingMode,
    TrainingSettings,
)


@pytest.mark.unit
class TestDatasetSettings:
    """Test dataset configuration validation."""
    
    def test_valid_dataset_config(self, temp_dir: Path):
        """Test valid dataset configuration."""
        config = DatasetSettings(
            data_dir=str(temp_dir),
            batch_size=4,
            side_x=64,
            side_y=64,
        )
        
        assert config.batch_size == 4
        assert config.side_x == 64
        assert config.side_y == 64
    
    def test_invalid_image_dimensions(self):
        """Test that non-divisible-by-8 dimensions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetSettings(
                data_dir="/tmp",
                side_x=63,  # Not divisible by 8
                side_y=64,
            )
        
        assert "divisible by 8" in str(exc_info.value)
    
    def test_webdataset_pattern(self):
        """Test WebDataset pattern validation."""
        config = DatasetSettings(
            data_dir="/path/to/data-{00000..00099}.tar",
            use_webdataset=True,
        )
        
        assert config.use_webdataset
        assert "{" in config.data_dir and "}" in config.data_dir
    
    def test_optimized_loader_validation(self):
        """Test optimized loader configuration validation."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetSettings(
                data_dir="/tmp",
                use_webdataset=False,
                use_optimized_loader=True,  # Requires WebDataset
            )
        
        assert "requires WebDataset" in str(exc_info.value)
    
    def test_negative_workers(self):
        """Test that negative workers are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetSettings(
                data_dir="/tmp",
                num_workers=-1,
            )
        
        assert "greater than or equal to 0" in str(exc_info.value).lower()


@pytest.mark.unit
class TestModelSettings:
    """Test model configuration validation."""
    
    def test_valid_model_config(self):
        """Test valid model configuration."""
        config = ModelSettings(
            train_upsample=False,
            freeze_transformer=False,
            freeze_diffusion=False,
        )
        
        assert not config.train_upsample
        assert not config.freeze_transformer
        assert not config.freeze_diffusion
    
    def test_freeze_both_error(self):
        """Test that freezing both transformer and diffusion is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSettings(
                freeze_transformer=True,
                freeze_diffusion=True,
            )
        
        assert "Cannot freeze both" in str(exc_info.value)
    
    def test_randomize_both_error(self):
        """Test that randomizing both components is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSettings(
                randomize_transformer=True,
                randomize_diffusion=True,
            )
        
        assert "Cannot randomize both" in str(exc_info.value)
    
    def test_freeze_and_randomize_error(self):
        """Test that freezing and randomizing same component is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSettings(
                freeze_transformer=True,
                randomize_transformer=True,
            )
        
        assert "simultaneously" in str(exc_info.value)
    
    def test_randomize_init_std_default(self):
        """Test that randomize_init_std gets default value when randomization enabled."""
        config = ModelSettings(
            randomize_transformer=True,
        )
        
        assert config.randomize_init_std == 0.02  # Default value


@pytest.mark.unit
class TestTrainingSettings:
    """Test training configuration validation."""
    
    def test_valid_training_config(self):
        """Test valid training configuration."""
        config = TrainingSettings(
            learning_rate=1e-4,
            batch_size=4,
            num_epochs=10,
        )
        
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.num_epochs == 10
    
    def test_invalid_learning_rate(self):
        """Test that non-positive learning rate is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingSettings(learning_rate=0.0)
        
        assert "greater than 0" in str(exc_info.value).lower()
    
    def test_invalid_batch_size(self):
        """Test that invalid batch size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingSettings(batch_size=0)
        
        assert "greater than or equal to 1" in str(exc_info.value).lower()
    
    def test_microbatch_exceeds_batch(self):
        """Test that microbatch > batch is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingSettings(
                batch_size=4,
                microbatch_size=8,
            )
        
        assert "cannot exceed batch size" in str(exc_info.value).lower()
    
    def test_warmup_lr_validation(self):
        """Test warmup LR validation."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingSettings(
                learning_rate=1e-4,
                warmup_steps=100,
                warmup_start_lr=1e-3,  # Higher than target LR
            )
        
        assert "less than target" in str(exc_info.value).lower()
    
    def test_adam_betas_validation(self):
        """Test Adam beta parameter validation."""
        config = TrainingSettings(
            adam_beta1=0.9,
            adam_beta2=0.999,
        )
        
        assert 0 < config.adam_beta1 < 1
        assert 0 < config.adam_beta2 < 1


@pytest.mark.unit
class TestFP16Settings:
    """Test FP16 configuration validation."""
    
    def test_valid_fp16_config(self):
        """Test valid FP16 configuration."""
        config = FP16Settings(
            use_fp16=True,
            fp16_loss_scale=256.0,
        )
        
        assert config.use_fp16
        assert config.fp16_loss_scale == 256.0
    
    def test_fp16_mode_enum(self):
        """Test FP16 mode enumeration."""
        config = FP16Settings(
            use_fp16=True,
            fp16_mode=FP16Mode.AGGRESSIVE,
        )
        
        assert config.fp16_mode == FP16Mode.AGGRESSIVE
    
    def test_loss_scale_range_validation(self):
        """Test loss scale range validation."""
        with pytest.raises(ValidationError) as exc_info:
            FP16Settings(
                use_fp16=True,
                fp16_min_loss_scale=1000.0,
                fp16_max_loss_scale=100.0,  # Max < Min
            )
        
        assert "less than max" in str(exc_info.value).lower()
    
    def test_loss_scale_auto_clamp(self):
        """Test that loss scale is clamped to valid range."""
        config = FP16Settings(
            use_fp16=True,
            fp16_loss_scale=1e30,  # Too large
            fp16_max_loss_scale=1e20,
        )
        
        # Should be clamped to max
        assert config.fp16_loss_scale == 1e20


@pytest.mark.unit
class TestSamplingSettings:
    """Test sampling configuration validation."""
    
    def test_valid_sampling_config(self):
        """Test valid sampling configuration."""
        config = SamplingSettings(
            timestep_respacing="50",
            guidance_scale=3.0,
            num_steps=50,
        )
        
        assert config.timestep_respacing == "50"
        assert config.guidance_scale == 3.0
        assert config.num_steps == 50
    
    def test_sampler_type_enum(self):
        """Test sampler type enumeration."""
        config = SamplingSettings(
            sampler=SamplerType.DPM_PLUS_PLUS,
        )
        
        assert config.sampler == SamplerType.DPM_PLUS_PLUS
    
    def test_timestep_respacing_validation(self):
        """Test timestep respacing format validation."""
        # Valid formats
        config1 = SamplingSettings(timestep_respacing="50")
        config2 = SamplingSettings(timestep_respacing="ddim50")
        
        assert config1.timestep_respacing == "50"
        assert config2.timestep_respacing == "ddim50"
        
        # Invalid format
        with pytest.raises(ValidationError) as exc_info:
            SamplingSettings(timestep_respacing="invalid")
        
        assert "Invalid timestep respacing" in str(exc_info.value)
    
    def test_prompt_file_validation(self, temp_dir: Path):
        """Test prompt file path validation."""
        # Non-existent file
        with pytest.raises(ValidationError) as exc_info:
            SamplingSettings(
                eval_prompt_file=temp_dir / "nonexistent.txt"
            )
        
        assert "does not exist" in str(exc_info.value)
        
        # Valid file
        prompt_file = temp_dir / "prompts.txt"
        prompt_file.write_text("test prompt")
        
        config = SamplingSettings(eval_prompt_file=prompt_file)
        assert config.eval_prompt_file == prompt_file


@pytest.mark.unit
class TestCheckpointSettings:
    """Test checkpoint configuration validation."""
    
    def test_valid_checkpoint_config(self, temp_dir: Path):
        """Test valid checkpoint configuration."""
        config = CheckpointSettings(
            save_directory=temp_dir / "checkpoints",
            checkpoint_frequency=1000,
            max_checkpoints=5,
        )
        
        assert config.save_directory.exists()  # Should be created
        assert config.checkpoint_frequency == 1000
        assert config.max_checkpoints == 5
    
    def test_directory_creation(self, temp_dir: Path):
        """Test that checkpoint directory is created if missing."""
        checkpoint_dir = temp_dir / "new_checkpoints"
        assert not checkpoint_dir.exists()
        
        config = CheckpointSettings(save_directory=checkpoint_dir)
        assert checkpoint_dir.exists()
    
    def test_frequency_validation(self):
        """Test checkpoint frequency validation."""
        with pytest.raises(ValidationError) as exc_info:
            CheckpointSettings(
                checkpoint_frequency=50,  # Too low (min is 100)
            )
        
        assert "greater than or equal to 100" in str(exc_info.value).lower()


@pytest.mark.unit
class TestSystemSettings:
    """Test system configuration validation."""
    
    def test_valid_system_config(self):
        """Test valid system configuration."""
        config = SystemSettings(
            seed=42,
            device="cuda",
            enable_tf32=True,
        )
        
        assert config.seed == 42
        assert config.device == "cuda"
        assert config.enable_tf32
    
    def test_seed_range_validation(self):
        """Test seed range validation."""
        # Valid seed
        config = SystemSettings(seed=12345)
        assert config.seed == 12345
        
        # Negative seed
        with pytest.raises(ValidationError) as exc_info:
            SystemSettings(seed=-1)
        
        assert "greater than or equal to 0" in str(exc_info.value).lower()
        
        # Too large seed
        with pytest.raises(ValidationError) as exc_info:
            SystemSettings(seed=2**32)
        
        assert "less than" in str(exc_info.value).lower()
    
    def test_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            config = SystemSettings(log_level=level)
            assert config.log_level == level


@pytest.mark.unit
class TestCompleteSettings:
    """Test complete Settings model."""
    
    def test_complete_settings_creation(self, temp_dir: Path):
        """Test creating complete settings."""
        settings = Settings(
            dataset=DatasetSettings(data_dir=str(temp_dir)),
            model=ModelSettings(),
            training=TrainingSettings(),
            fp16=FP16Settings(),
            sampling=SamplingSettings(),
            checkpoint=CheckpointSettings(save_directory=temp_dir / "ckpt"),
            system=SystemSettings(),
        )
        
        assert settings.dataset is not None
        assert settings.model is not None
        assert settings.training is not None
        assert settings.fp16 is not None
        assert settings.sampling is not None
        assert settings.checkpoint is not None
        assert settings.system is not None
    
    def test_settings_to_dict(self, temp_dir: Path):
        """Test converting settings to dictionary."""
        settings = Settings(
            dataset=DatasetSettings(data_dir=str(temp_dir)),
            checkpoint=CheckpointSettings(save_directory=temp_dir / "ckpt"),
        )
        
        config_dict = settings.model_dump()
        
        assert "dataset" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert config_dict["dataset"]["data_dir"] == str(temp_dir)
    
    def test_settings_from_dict(self, temp_dir: Path):
        """Test creating settings from dictionary."""
        config_dict = {
            "dataset": {
                "data_dir": str(temp_dir),
                "batch_size": 8,
            },
            "training": {
                "learning_rate": 5e-5,
                "num_epochs": 20,
            },
            "checkpoint": {
                "save_directory": str(temp_dir / "ckpt"),
            },
        }
        
        settings = Settings(**config_dict)
        
        assert settings.dataset.batch_size == 8
        assert settings.training.learning_rate == 5e-5
        assert settings.training.num_epochs == 20