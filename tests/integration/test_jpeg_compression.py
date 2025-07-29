"""Test JPEG compression implementation for image saving."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from glide_finetune.train_util import pred_to_pil, save_image_compressed


class TestJPEGCompression:
    """Test cases for JPEG compression functionality."""

    def test_jpeg_compression_saves_space(self):
        """Test that JPEG compression produces smaller files than PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test tensor (64x64 RGB image)
            test_tensor = torch.randn(1, 3, 64, 64)

            # Convert to PIL image
            pil_image = pred_to_pil(test_tensor)

            # Save as PNG for comparison
            png_path = Path(tmpdir) / "test.png"
            pil_image.save(png_path)
            png_size = os.path.getsize(png_path)

            # Save as JPEG using our function
            jpg_path = Path(tmpdir) / "test.png"  # Extension will be changed to .jpg
            actual_path = save_image_compressed(pil_image, jpg_path)
            jpg_size = os.path.getsize(actual_path)

            # Verify the file has .jpg extension
            assert actual_path.endswith(".jpg"), (
                f"Expected .jpg extension, got {actual_path}"
            )

            # Verify the JPEG is smaller (should be at least 30% smaller)
            compression_ratio = 1 - (jpg_size / png_size)
            assert compression_ratio > 0.3, (
                f"Expected at least 30% compression, got {compression_ratio:.1%} "
                f"(PNG: {png_size}, JPEG: {jpg_size})"
            )

            # Verify we can load the JPEG
            loaded_img = Image.open(actual_path)
            assert loaded_img.size == (64, 64), (
                f"Expected size (64, 64), got {loaded_img.size}"
            )
            assert loaded_img.mode == "RGB", f"Expected RGB mode, got {loaded_img.mode}"

    def test_rgba_to_rgb_conversion(self):
        """Test that RGBA images are properly converted to RGB with white background."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an RGBA image with transparency
            rgba_image = Image.new("RGBA", (64, 64), (255, 0, 0, 128))

            # Save using our function
            jpg_path = Path(tmpdir) / "rgba_test.png"
            actual_path = save_image_compressed(rgba_image, jpg_path)

            # Load and verify
            loaded_img = Image.open(actual_path)
            assert loaded_img.mode == "RGB", f"Expected RGB mode, got {loaded_img.mode}"
            assert loaded_img.size == (64, 64), (
                f"Expected size (64, 64), got {loaded_img.size}"
            )

    def test_different_image_modes(self):
        """Test conversion of various image modes to RGB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modes_to_test = [
                ("L", (64, 64), 128),  # Grayscale
                ("P", (64, 64), 0),  # Palette
                ("LA", (64, 64), (128, 255)),  # Grayscale with alpha
            ]

            for mode, size, color in modes_to_test:
                # Create test image
                img = Image.new(mode, size, color)

                # Save using our function
                jpg_path = Path(tmpdir) / f"{mode}_test.png"
                actual_path = save_image_compressed(img, jpg_path)

                # Verify conversion to RGB
                loaded_img = Image.open(actual_path)
                assert loaded_img.mode == "RGB", (
                    f"Expected RGB mode for {mode}, got {loaded_img.mode}"
                )

    def test_path_extension_handling(self):
        """Test that various input paths are correctly converted to .jpg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Image.new("RGB", (32, 32), (100, 100, 100))

            test_cases = [
                ("test.png", "test.jpg"),
                ("test.PNG", "test.jpg"),
                ("test", "test.jpg"),
                ("test.txt", "test.txt.jpg"),
                ("test.jpg", "test.jpg"),
                ("test.jpeg", "test.jpeg"),
            ]

            for input_name, expected_suffix in test_cases:
                input_path = Path(tmpdir) / input_name
                actual_path = save_image_compressed(test_image, input_path)
                assert actual_path.endswith(expected_suffix), (
                    f"For input {input_name}, expected to end with {expected_suffix}, "
                    f"got {actual_path}"
                )

    def test_quality_parameter(self):
        """Test that quality parameter affects file size appropriately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a detailed test image (more compressible)
            test_tensor = torch.randn(1, 3, 256, 256)
            pil_image = pred_to_pil(test_tensor)

            sizes = {}
            qualities = [50, 75, 95]

            for quality in qualities:
                jpg_path = Path(tmpdir) / f"test_q{quality}.png"
                actual_path = save_image_compressed(
                    pil_image, jpg_path, quality=quality
                )
                sizes[quality] = os.path.getsize(actual_path)

            # Verify that higher quality produces larger files
            assert sizes[75] > sizes[50], (
                f"Quality 75 ({sizes[75]}) should be larger than 50 ({sizes[50]})"
            )
            assert sizes[95] > sizes[75], (
                f"Quality 95 ({sizes[95]}) should be larger than 75 ({sizes[75]})"
            )

            # Verify reasonable compression even at high quality
            # Create reference PNG
            png_path = Path(tmpdir) / "reference.png"
            pil_image.save(png_path)
            png_size = os.path.getsize(png_path)

            # Even at quality 95, JPEG should be smaller than PNG
            assert sizes[95] < png_size, (
                f"JPEG at quality 95 ({sizes[95]}) should be smaller "
                f"than PNG ({png_size})"
            )

    def test_optimize_parameter(self):
        """Test that optimize parameter is working."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Image.new("RGB", (128, 128), (100, 150, 200))

            # Save with optimize=False
            path_no_opt = Path(tmpdir) / "no_optimize.jpg"
            actual_no_opt = save_image_compressed(
                test_image, path_no_opt, optimize=False
            )
            size_no_opt = os.path.getsize(actual_no_opt)

            # Save with optimize=True (default)
            path_opt = Path(tmpdir) / "optimize.jpg"
            actual_opt = save_image_compressed(test_image, path_opt, optimize=True)
            size_opt = os.path.getsize(actual_opt)

            # Optimized should be same size or smaller
            assert size_opt <= size_no_opt, (
                f"Optimized ({size_opt}) should be <= non-optimized ({size_no_opt})"
            )

    @pytest.mark.parametrize("size", [(64, 64), (256, 256), (512, 512)])
    def test_various_image_sizes(self, size):
        """Test compression works for various image sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            test_tensor = torch.randn(1, 3, size[0], size[1])
            pil_image = pred_to_pil(test_tensor)

            # Save as JPEG
            jpg_path = Path(tmpdir) / f"test_{size[0]}x{size[1]}.png"
            actual_path = save_image_compressed(pil_image, jpg_path)

            # Verify it saved correctly
            assert os.path.exists(actual_path), f"File not created: {actual_path}"

            # Verify dimensions preserved
            loaded_img = Image.open(actual_path)
            assert loaded_img.size == size, (
                f"Expected size {size}, got {loaded_img.size}"
            )
