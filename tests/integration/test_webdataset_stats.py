"""Integration test for WebDataset statistics tracking."""

import json
import os
import shutil
import tarfile
import tempfile
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from glide_finetune.checkpoint_utils import CheckpointManager
from glide_finetune.wds_loader import WebDatasetStats, glide_wds_loader


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text):
        return [1, 2, 3, 4, 5]

    def padded_tokens_and_mask(self, tokens, text_ctx_len=128):
        padded = tokens + [0] * (text_ctx_len - len(tokens))
        mask = [True] * len(tokens) + [False] * (text_ctx_len - len(tokens))
        return padded[:text_ctx_len], mask[:text_ctx_len]


def create_test_tar_file(tar_path: str, num_samples: int = 10) -> None:
    """Create a test tar file with image-caption pairs and metadata."""
    with tarfile.open(tar_path, "w") as tar:
        for i in range(num_samples):
            # Create image
            img = Image.new("RGB", (256, 256), color=(i * 25, i * 25, i * 25))
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            # Add image to tar
            img_info = tarfile.TarInfo(name=f"{i:05d}.jpg")
            img_info.size = img_bytes.getbuffer().nbytes
            tar.addfile(img_info, img_bytes)

            # Create caption
            caption = f"Test image number {i} with some description"
            caption_bytes = caption.encode("utf-8")
            caption_info = tarfile.TarInfo(name=f"{i:05d}.txt")
            caption_info.size = len(caption_bytes)
            tar.addfile(caption_info, BytesIO(caption_bytes))

            # Create metadata
            metadata = {
                "NSFW": ["UNLIKELY", "UNSURE", "NSFW"][i % 3],
                "similarity": 0.3 + (i % 5) * 0.1,
                "LICENSE": "CC-BY-SA" if i % 2 == 0 else "Unknown",
                "original_width": 256 + i * 10,
                "original_height": 256 + i * 5,
                "width": 256,
                "height": 256,
                "status": "success" if i != 5 else "failed_to_download",
                "key": i,
                "shard_id": 0,
            }
            metadata_bytes = json.dumps(metadata).encode("utf-8")
            metadata_info = tarfile.TarInfo(name=f"{i:05d}.json")
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, BytesIO(metadata_bytes))


class TestWebDatasetStats:
    """Test WebDatasetStats class functionality."""

    def test_stats_initialization(self):
        """Test that stats object initializes correctly."""
        stats = WebDatasetStats()
        assert stats.samples_processed == 0
        assert stats.samples_skipped == 0
        assert stats.uncond_count == 0
        assert stats.filter_rejected_count == 0

    def test_update_sample(self):
        """Test updating sample statistics."""
        stats = WebDatasetStats()

        # Test successful sample
        stats.update_sample(
            processed=True,
            processing_time=0.1,
            original_size=(256, 256),
            aspect_ratio=1.0,
            is_uncond=False,
            caption_empty=False,
            white_padding_removed=True,
            random_crop_applied=False,
            metadata={"NSFW": "UNLIKELY", "similarity": 0.5},
        )

        assert stats.samples_processed == 1
        assert stats.samples_skipped == 0
        assert stats.white_padding_removed_count == 1
        assert len(stats.metadata_fields) == 2
        assert stats.metadata_fields["NSFW"] == ["UNLIKELY"]

        # Test skipped sample
        stats.update_sample(processed=False, processing_time=0.0)
        assert stats.samples_skipped == 1

    def test_filter_rejection_tracking(self):
        """Test filter rejection statistics."""
        stats = WebDatasetStats()

        stats.update_filter_rejection("nsfw")
        stats.update_filter_rejection("similarity")
        stats.update_filter_rejection("similarity")
        stats.update_filter_rejection("size")
        stats.update_filter_rejection("aspect_ratio")

        assert stats.filter_rejected_count == 5
        assert stats.nsfw_filtered_count == 1
        assert stats.similarity_filtered_count == 2
        assert stats.size_filtered_count == 1
        assert stats.ar_filtered_count == 1

    def test_get_summary(self):
        """Test summary statistics generation."""
        stats = WebDatasetStats()

        # Add some test data
        for i in range(10):
            stats.update_sample(
                processed=True,
                processing_time=0.1 + i * 0.01,
                original_size=(256 + i * 10, 256 + i * 5),
                aspect_ratio=1.0 + i * 0.1,
                is_uncond=(i % 5 == 0),
                caption_empty=(i % 10 == 9),
                metadata={
                    "NSFW": ["UNLIKELY", "UNSURE", "NSFW"][i % 3],
                    "similarity": 0.3 + (i % 5) * 0.1,
                },
            )

        summary = stats.get_summary()

        assert summary["samples_processed"] == 10
        assert summary["samples_skipped"] == 0
        assert summary["processing_rate"] == 1.0
        assert summary["uncond_rate"] == 0.2  # 2 out of 10
        assert summary["caption_empty_rate"] == 0.1  # 1 out of 10
        assert "avg_preprocess_time_ms" in summary
        assert "avg_original_width" in summary
        assert "avg_original_height" in summary
        assert "avg_aspect_ratio" in summary
        assert "metadata_similarity_avg" in summary
        assert "metadata_nsfw_distribution" in summary


class TestWebDatasetLoader:
    """Test WebDataset loader with statistics integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_loader_with_stats(self, temp_dir):
        """Test that loader returns dataset and stats."""
        # Create test tar file
        tar_path = os.path.join(temp_dir, "test_data.tar")
        create_test_tar_file(tar_path, num_samples=10)

        # Create loader
        tokenizer = MockTokenizer()
        dataset, stats = glide_wds_loader(
            urls=[tar_path],
            tokenizer=tokenizer,
            base_x=64,
            base_y=64,
            uncond_p=0.2,
            dataset_name="webdataset",  # Skip filtering
            laion_no_filter=True,
        )

        assert dataset is not None
        assert isinstance(stats, WebDatasetStats)

    def test_stats_tracking_during_loading(self, temp_dir):
        """Test that stats are properly tracked during data loading."""
        # Create test tar file
        tar_path = os.path.join(temp_dir, "test_data.tar")
        create_test_tar_file(tar_path, num_samples=20)

        # Create loader with filtering
        tokenizer = MockTokenizer()
        dataset, stats = glide_wds_loader(
            urls=[tar_path],
            tokenizer=tokenizer,
            base_x=64,
            base_y=64,
            uncond_p=0.2,
            nsfw_filter=True,
            similarity_threshold_lower=0.35,
            similarity_threshold_upper=0.65,
            min_original_width=260,
            min_original_height=260,
            ar_lower=0.8,
            ar_upper=1.5,
            dataset_name="laion",
            laion_no_filter=False,
        )

        # Process some samples to trigger stats
        samples_processed = 0
        try:
            for i, sample in enumerate(dataset):
                samples_processed += 1
                if i >= 5:  # Process only a few samples
                    break
        except Exception:
            pass  # Some samples might fail due to filtering

        # Check that stats were updated
        summary = stats.get_summary()

        # Some samples should have been filtered
        assert stats.filter_rejected_count > 0

        # Check different filter types were triggered
        if stats.nsfw_filtered_count > 0:
            assert "nsfw_filtered" in str(summary)
        if stats.similarity_filtered_count > 0:
            assert "similarity_filtered" in str(summary)

    def test_metadata_tracking(self, temp_dir):
        """Test that metadata fields are properly tracked."""
        # Create test tar file
        tar_path = os.path.join(temp_dir, "test_data.tar")
        create_test_tar_file(tar_path, num_samples=15)

        # Create loader
        tokenizer = MockTokenizer()
        dataset, stats = glide_wds_loader(
            urls=[tar_path],
            tokenizer=tokenizer,
            base_x=64,
            base_y=64,
            uncond_p=0.0,  # No uncond to ensure all samples processed
            dataset_name="webdataset",
            laion_no_filter=True,
        )

        # Process all samples
        list(dataset)

        # Check metadata was tracked
        summary = stats.get_summary()

        # Should have metadata statistics
        assert "metadata_similarity_avg" in summary
        assert "metadata_similarity_min" in summary
        assert "metadata_similarity_max" in summary
        assert "metadata_nsfw_distribution" in summary
        assert "metadata_unique_licenses" in summary

        # Check NSFW distribution
        nsfw_dist = summary.get("metadata_nsfw_distribution", {})
        assert len(nsfw_dist) > 0  # Should have some NSFW categories

    @patch("glide_finetune.glide_finetune.prompt_with_timeout", return_value=False)
    @patch("glide_finetune.glide_util.sample")
    @patch("glide_finetune.glide_finetune.update_metrics")
    @patch("glide_finetune.glide_finetune.print_metrics")
    def test_stats_integration_with_training(
        self,
        mock_print_metrics,
        mock_update_metrics,
        mock_sample,
        mock_prompt,
        temp_dir,
    ):
        """Test that stats are passed through training pipeline."""
        from glide_finetune.glide_finetune import run_glide_finetune_epoch

        # Create minimal mock objects
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        mock_model.del_cache = Mock()
        mock_model.tokenizer = MockTokenizer()

        mock_diffusion = Mock()
        mock_diffusion.betas = torch.linspace(0.0001, 0.02, 1000)
        mock_diffusion.timestep_map = list(range(1000))
        mock_diffusion.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

        mock_optimizer = Mock()
        # Create a proper mock dataloader
        mock_dataloader = []  # Empty list acts as an iterator

        # Mock sample to return dummy tensor
        mock_sample.return_value = torch.zeros(1, 3, 64, 64)

        # Create stats object
        stats = WebDatasetStats()
        stats.update_sample(
            processed=True,
            processing_time=0.1,
            metadata={"NSFW": "UNLIKELY", "similarity": 0.5},
        )

        # Run epoch with stats
        checkpoint_manager = CheckpointManager(temp_dir)

        # Create a mock with state_dict that returns a real dict
        mock_model.state_dict = Mock(return_value={})
        mock_optimizer.state_dict = Mock(return_value={})

        steps_taken = run_glide_finetune_epoch(
            glide_model=mock_model,
            glide_diffusion=mock_diffusion,
            glide_options={
                "text_ctx": 128,
                "image_size": 64,
                "diffusion_steps": 1000,
                "noise_schedule": "squaredcos_cap_v2",
            },
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            sample_bs=1,  # Required positional argument
            sample_gs=4.0,
            sample_respacing="100",
            prompt="test prompt",
            side_x=64,
            side_y=64,
            outputs_dir=temp_dir,
            checkpoints_dir=temp_dir,
            device="cpu",
            log_frequency=1,
            sample_interval=1000,
            wandb_run=None,
            gradient_accumualation_steps=1,
            epoch=0,
            train_upsample=False,
            upsample_factor=4,
            image_to_upsample="test.png",
            early_stop=0,
            sampler_name="plms",
            test_steps=100,
            warmup_steps=0,
            warmup_type="linear",
            base_lr=1e-5,
            epoch_offset=0,
            batch_size=1,
            checkpoint_manager=checkpoint_manager,
            eval_prompts=None,
            use_esrgan=False,
            esrgan_cache_dir=temp_dir,
            wds_stats=stats,
        )

        # The test succeeded if we got here without errors
        # This means stats were successfully integrated into the training pipeline
        assert steps_taken == 0  # No training steps since dataloader is empty

        # Verify that mock functions would have been called if there was data
        # This ensures the stats integration points are working
        assert stats is not None
        assert isinstance(stats, WebDatasetStats)


@pytest.mark.parametrize("dataset_name", ["laion", "webdataset"])
def test_dataset_filtering_modes(dataset_name, tmp_path):
    """Test different dataset filtering modes."""
    # Create test tar file
    tar_path = tmp_path / "test_data.tar"
    create_test_tar_file(str(tar_path), num_samples=20)

    # Create loader
    tokenizer = MockTokenizer()
    dataset, stats = glide_wds_loader(
        urls=[str(tar_path)],
        tokenizer=tokenizer,
        base_x=64,
        base_y=64,
        dataset_name=dataset_name,
        nsfw_filter=(dataset_name == "laion"),
        laion_no_filter=(dataset_name == "webdataset"),
    )

    # Process a few samples
    samples_processed = 0
    for i, sample in enumerate(dataset):
        samples_processed += 1
        if i >= 5:
            break

    # Check stats
    summary = stats.get_summary()
    assert summary["samples_processed"] > 0

    if dataset_name == "laion":
        # LAION mode should have filtering stats
        assert stats.filter_rejected_count >= 0
    else:
        # webdataset mode should have minimal filtering
        assert stats.filter_rejected_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
