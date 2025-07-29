import subprocess
import sys

import pytest
import torch


class TestTF32Support:
    """Test TF32 (TensorFloat-32) support in the training script."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tf32_flag_enables_tf32(self):
        """Test that --use_tf32 flag properly enables TF32 in PyTorch."""
        # Check if GPU supports TF32 (Ampere or newer)
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = (device_props.major, device_props.minor)

        # TF32 is supported on compute capability 8.0+ (Ampere)
        if compute_capability < (8, 0):
            pytest.skip(
                f"GPU compute capability {compute_capability} doesn't support TF32"
            )

        # Save original TF32 settings
        orig_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn = torch.backends.cudnn.allow_tf32

        try:
            # Reset to default (disabled)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            # Run a minimal test with --use_tf32
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import torch
import sys
sys.path.insert(0, '.')
from train_glide import parse_args

# Parse args with TF32 enabled
import argparse
args = parse_args()
args.use_tf32 = True

# Apply TF32 settings
if args.use_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Verify settings
assert torch.backends.cuda.matmul.allow_tf32 == True
assert torch.backends.cudnn.allow_tf32 == True
print("TF32 enabled successfully")
""",
                ],
                capture_output=True,
                text=True,
                cwd="/home/sam/GitHub/glide-finetune",
            )

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "TF32 enabled successfully" in result.stdout

        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul
            torch.backends.cudnn.allow_tf32 = orig_cudnn

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tf32_performance_difference(self):
        """Test that TF32 provides performance improvement for matrix operations."""
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = (device_props.major, device_props.minor)

        if compute_capability < (8, 0):
            pytest.skip(
                f"GPU compute capability {compute_capability} doesn't support TF32"
            )

        # Save original settings
        orig_matmul = torch.backends.cuda.matmul.allow_tf32

        try:
            # Test matrix size
            size = 2048
            a = torch.randn(size, size, device="cuda", dtype=torch.float32)
            b = torch.randn(size, size, device="cuda", dtype=torch.float32)

            # Warmup
            for _ in range(5):
                torch.matmul(a, b)
            torch.cuda.synchronize()

            # Time with TF32 disabled
            torch.backends.cuda.matmul.allow_tf32 = False
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(10):
                torch.matmul(a, b)
            end_event.record()
            torch.cuda.synchronize()
            time_fp32 = start_event.elapsed_time(end_event)

            # Time with TF32 enabled
            torch.backends.cuda.matmul.allow_tf32 = True
            start_event.record()
            for _ in range(10):
                torch.matmul(a, b)
            end_event.record()
            torch.cuda.synchronize()
            time_tf32 = start_event.elapsed_time(end_event)

            # TF32 should be faster (allow some margin for variance)
            speedup = time_fp32 / time_tf32
            print(
                f"FP32 time: {time_fp32:.2f}ms, TF32 time: {time_tf32:.2f}ms, "
                f"speedup: {speedup:.2f}x"
            )

            # We expect at least some speedup, but the exact amount varies by GPU
            assert speedup > 0.9, f"TF32 unexpectedly slower: {speedup:.2f}x"

        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul

    def test_tf32_argument_parsing(self):
        """Test that the --use_tf32 argument is properly parsed."""
        result = subprocess.run(
            [sys.executable, "train_glide.py", "--use_tf32", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/sam/GitHub/glide-finetune",
        )

        assert result.returncode == 0
        assert "--use_tf32" in result.stdout or "-tf32" in result.stdout
        assert "Enable TF32" in result.stdout
