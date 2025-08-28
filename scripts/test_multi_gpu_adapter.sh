#!/bin/bash
# Test multi-GPU CLIP adapter training with minimal configuration

set -e  # Exit on error

echo "Testing Multi-GPU CLIP Adapter Training"
echo "========================================"

# Create temporary test data
TEST_DIR=$(mktemp -d)
echo "Creating test data in $TEST_DIR"

# Create minimal dataset (4 images)
mkdir -p "$TEST_DIR/data"
for i in {0..3}; do
    # Create 64x64 random images using ImageMagick or Python
    if command -v convert &> /dev/null; then
        convert -size 64x64 xc: +noise Random "$TEST_DIR/data/img_$i.jpg"
    else
        python -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
img.save('$TEST_DIR/data/img_$i.jpg')
"
    fi
    echo "Test caption for image $i" > "$TEST_DIR/data/img_$i.txt"
done

echo "Created 4 test images with captions"

# Test 1: Basic 2-process training
echo ""
echo "Test 1: Running 2-process adapter training..."
echo "----------------------------------------------"

accelerate launch --num_processes 2 train.py \
    --data_dir "$TEST_DIR/data" \
    --save_directory "$TEST_DIR/checkpoints" \
    --use_clip_adapter \
    --clip_adapter_only \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --clip_adapter_lr 5e-4 \
    --iterations 10 \
    --log_frequency 5 \
    --save_frequency 10 \
    --sample_frequency 999999 \
    --seed 42 \
    2>&1 | tail -n 20

# Check if checkpoint was created
if [ -d "$TEST_DIR/checkpoints/checkpoint_00000010" ]; then
    echo "✓ Checkpoint created successfully"
else
    echo "✗ No checkpoint found"
    exit 1
fi

# Test 2: Verify adapter weights in checkpoint
echo ""
echo "Test 2: Verifying adapter weights..."
echo "-------------------------------------"

python -c "
import torch
from pathlib import Path

checkpoint_dir = Path('$TEST_DIR/checkpoints/checkpoint_00000010')
if checkpoint_dir.exists():
    # Check for pytorch_model.bin or model.safetensors
    model_files = list(checkpoint_dir.glob('*.bin')) + list(checkpoint_dir.glob('*.safetensors'))
    if model_files:
        state_dict = torch.load(model_files[0], map_location='cpu')
        adapter_keys = [k for k in state_dict.keys() if 'clip_adapter' in k]
        print(f'Found {len(adapter_keys)} adapter parameters')
        if adapter_keys:
            print('✓ Adapter weights saved correctly')
        else:
            print('✗ No adapter weights found')
            exit(1)
    else:
        print('✗ No model file found in checkpoint')
        exit(1)
"

# Clean up
echo ""
echo "Cleaning up test directory..."
rm -rf "$TEST_DIR"

echo ""
echo "========================================"
echo "✓ All multi-GPU adapter tests passed!"
echo "========================================"