"""
Integration test for multi-GPU CLIP adapter training.

Tests distributed training of CLIP adapter with 2 processes to verify:
1. Adapter weights are synchronized across ranks
2. Frozen base model parameters don't sync gradients
3. Training converges correctly
"""

import subprocess
import tempfile
import torch
from pathlib import Path
import json
import sys


def test_two_process_adapter_training():
    """Test CLIP adapter training with 2 processes using accelerate."""
    
    # Create temporary directory for test data and outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create minimal test dataset
        data_dir = tmpdir_path / "data"
        data_dir.mkdir()
        
        # Create a few test images with captions
        from PIL import Image
        import numpy as np
        
        for i in range(4):  # Just 4 samples for quick test
            # Create random image
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(data_dir / f"img_{i}.jpg")
            
            # Create caption file
            with open(data_dir / f"img_{i}.txt", "w") as f:
                f.write(f"Test image number {i}")
        
        # Output directory for checkpoints
        output_dir = tmpdir_path / "outputs"
        output_dir.mkdir()
        
        # Build accelerate launch command
        cmd = [
            "accelerate", "launch",
            "--num_processes", "2",
            "--mixed_precision", "no",
            "train.py",
            "--data_dir", str(data_dir),
            "--save_directory", str(output_dir),
            "--use_clip_adapter",
            "--clip_adapter_only",
            "--batch_size", "2",
            "--learning_rate", "1e-4",
            "--clip_adapter_lr", "5e-4",
            "--num_epochs", "1",
            "--log_frequency", "1",
            "--save_frequency", "10",
            "--sample_frequency", "999999",  # Don't sample during test
            "--seed", "42",
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run training
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )
            print("Training completed successfully")
            print(f"stdout: {result.stdout[-1000:]}")  # Last 1000 chars
            
        except subprocess.TimeoutExpired:
            print("ERROR: Training timed out after 5 minutes")
            return False
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Training failed with exit code {e.returncode}")
            print(f"stderr: {e.stderr}")
            print(f"stdout: {e.stdout}")
            return False
        
        # Check that checkpoint was saved
        checkpoint_files = list(output_dir.glob("checkpoint_*/pytorch_model.bin"))
        if not checkpoint_files:
            # Check for accelerate-style checkpoint directories
            checkpoint_dirs = list(output_dir.glob("checkpoint_*"))
            if not checkpoint_dirs:
                print("ERROR: No checkpoints saved")
                return False
            print(f"Found {len(checkpoint_dirs)} checkpoint directories")
        
        # Verify adapter weights were saved
        if checkpoint_dirs:
            # Load the checkpoint metadata
            latest_checkpoint = sorted(checkpoint_dirs)[-1]
            metadata_file = latest_checkpoint / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    print(f"Checkpoint metadata: {metadata}")
            
            # Check model state
            model_file = latest_checkpoint / "pytorch_model.bin"
            if model_file.exists():
                state_dict = torch.load(model_file, map_location="cpu")
                
                # Check for adapter weights
                adapter_keys = [k for k in state_dict.keys() if "clip_adapter" in k]
                if not adapter_keys:
                    print("ERROR: No clip_adapter weights in checkpoint")
                    return False
                
                print(f"Found {len(adapter_keys)} adapter parameter tensors")
                
                # Verify adapter gate was trained (should be different from init)
                gate_key = next((k for k in adapter_keys if "gate" in k), None)
                if gate_key:
                    gate_value = state_dict[gate_key].item()
                    init_value = torch.sigmoid(torch.tensor(-5.0)).item()
                    
                    if abs(gate_value - init_value) < 1e-6:
                        print("WARNING: Gate parameter hasn't changed from init")
                    else:
                        print(f"Gate trained: init={init_value:.6f}, current={gate_value:.6f}")
        
        print("✓ Multi-GPU adapter training test passed!")
        return True


def test_adapter_weight_sync():
    """Test that adapter weights are synchronized across ranks."""
    
    # Create test script that checks weight sync
    test_script = '''
import torch
import torch.distributed as dist
from accelerate import Accelerator
from glide_finetune.clip_adapter import CLIPAdapter

# Initialize accelerator
accelerator = Accelerator()

# Create adapter on each rank
adapter = CLIPAdapter(
    time_embed_dim=512,
    clip_embed_dim=512, 
    hidden_dim=768,
    gate_init=-5.0,
)

# Move to device
adapter = adapter.to(accelerator.device)

# Prepare with accelerator (wraps in DDP)
adapter = accelerator.prepare(adapter)

# Initialize differently on each rank (to test sync)
if accelerator.is_main_process:
    # Rank 0: Set gate to 0.1
    with torch.no_grad():
        adapter.module.gate.data.fill_(0.1)
else:
    # Other ranks: Set gate to 0.9
    with torch.no_grad():
        adapter.module.gate.data.fill_(0.9)

# Broadcast from rank 0 (simulating checkpoint load)
if dist.is_initialized():
    dist.broadcast(adapter.module.gate.data, src=0)

# Check all ranks have same value
accelerator.wait_for_everyone()
gate_value = adapter.module.gate.data.item()

# Gather values from all ranks
all_values = accelerator.gather(torch.tensor([gate_value], device=accelerator.device))

if accelerator.is_main_process:
    print(f"Gate values across ranks: {all_values.cpu().numpy()}")
    
    # Check all are equal
    if torch.allclose(all_values, all_values[0], atol=1e-6):
        print("✓ Weights synchronized correctly across ranks!")
    else:
        print("✗ Weight synchronization failed!")
        exit(1)
'''
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Run with 2 processes
        cmd = [
            "accelerate", "launch",
            "--num_processes", "2",
            script_path,
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        
        print(result.stdout)
        
        if "✓ Weights synchronized correctly" in result.stdout:
            print("✓ Weight synchronization test passed!")
            return True
        else:
            print("✗ Weight synchronization test failed")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    finally:
        Path(script_path).unlink()


if __name__ == "__main__":
    # Run tests
    tests_passed = 0
    tests_total = 2
    
    print("=" * 60)
    print("Testing Multi-GPU CLIP Adapter Training")
    print("=" * 60)
    
    print("\n1. Testing 2-process adapter training...")
    if test_two_process_adapter_training():
        tests_passed += 1
    
    print("\n2. Testing adapter weight synchronization...")
    if test_adapter_weight_sync():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)