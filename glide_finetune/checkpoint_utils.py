"""Checkpoint saving and loading utilities with full training state support."""

import json
import os
import signal
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import torch as th


class CheckpointManager:
    """Manages checkpoint saving and loading with full training state."""
    
    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir
        self.sigint_handler_installed = False
        self.current_checkpoint_data = None
        
    def save_checkpoint(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        epoch: int,
        step: int,
        global_step: int,
        warmup_steps: int,
        warmup_type: str,
        base_lr: float,
        checkpoint_type: str = "regular",
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Save a complete checkpoint with all training state.
        
        Returns dict with paths to saved files.
        """
        
        # Determine base filename
        if checkpoint_type == "emergency":
            base_name = f"emergency_checkpoint_epoch{epoch}_step{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif checkpoint_type == "sigint":
            base_name = f"interrupted_checkpoint_epoch{epoch}_step{step}"
        else:
            base_name = f"glide-ft-{epoch}x{step}"
        
        # Save model weights (keep original format)
        model_path = os.path.join(self.checkpoints_dir, f"{base_name}.pt")
        th.save(model.state_dict(), model_path)
        
        # Save optimizer state separately
        optimizer_path = os.path.join(self.checkpoints_dir, f"{base_name}.optimizer.pt")
        optimizer_data = {
            "optimizer_state_dict": optimizer.state_dict(),
            # Save RNG states with optimizer for reproducibility
            "torch_rng_state": th.get_rng_state(),
            "cuda_rng_state": th.cuda.get_rng_state() if th.cuda.is_available() else None,
        }
        th.save(optimizer_data, optimizer_path)
        
        # Save training metadata as JSON
        metadata = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "warmup_steps": warmup_steps,
            "warmup_type": warmup_type,
            "base_lr": base_lr,
            "checkpoint_type": checkpoint_type,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add any additional state
        if additional_state:
            metadata.update(additional_state)
        
        metadata_path = os.path.join(self.checkpoints_dir, f"{base_name}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Saved {checkpoint_type} checkpoint:")
        print(f"  Model: {model_path}")
        print(f"  Optimizer: {optimizer_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Epoch: {epoch}, Step: {step}, Global Step: {global_step}")
        print(f"{'='*50}\n")
        
        return {
            "model": model_path,
            "optimizer": optimizer_path,
            "metadata": metadata_path,
        }
    
    def load_checkpoint(self, checkpoint_path: str, model: th.nn.Module, optimizer: Optional[th.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load a checkpoint and return the training state.
        
        checkpoint_path can be either:
        - Path to model .pt file (will look for associated .optimizer.pt and .json)
        - Path to metadata .json file (will look for associated .pt and .optimizer.pt)
        - Base path without extension (will look for all associated files)
        """
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Determine base path by removing known extensions
        if checkpoint_path.endswith('.pt'):
            # Handle both .pt and .optimizer.pt
            if checkpoint_path.endswith('.optimizer.pt'):
                base_path = checkpoint_path[:-13]  # Remove .optimizer.pt
            else:
                base_path = checkpoint_path[:-3]  # Remove .pt
        elif checkpoint_path.endswith('.json'):
            base_path = checkpoint_path[:-5]  # Remove .json
        else:
            base_path = checkpoint_path
        
        # Construct paths for associated files
        model_path = f"{base_path}.pt"
        optimizer_path = f"{base_path}.optimizer.pt"
        metadata_path = f"{base_path}.json"
        
        # Load model state (required)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        model_state = th.load(model_path, map_location="cpu")
        model.load_state_dict(model_state)
        print("✓ Loaded model state")
        
        # Initialize return state
        state = {
            "epoch": 0,
            "step": 0,
            "global_step": 0,
            "has_optimizer_state": False,
            "has_metadata": False,
        }
        
        # Try to load metadata (optional)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            state.update({
                "epoch": metadata.get("epoch", 0),
                "step": metadata.get("step", 0),
                "global_step": metadata.get("global_step", 0),
                "warmup_steps": metadata.get("warmup_steps", 0),
                "warmup_type": metadata.get("warmup_type", "linear"),
                "base_lr": metadata.get("base_lr", 1e-5),
                "has_metadata": True,
            })
            print("✓ Loaded training metadata")
        else:
            print("ℹ No metadata file found")
        
        # Try to load optimizer state (optional)
        if os.path.exists(optimizer_path) and optimizer is not None:
            optimizer_data = th.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(optimizer_data["optimizer_state_dict"])
            print("✓ Loaded optimizer state")
            
            state["has_optimizer_state"] = True
            
            # Restore RNG states if available
            if "torch_rng_state" in optimizer_data:
                th.set_rng_state(optimizer_data["torch_rng_state"])
                print("✓ Restored PyTorch RNG state")
            
            if "cuda_rng_state" in optimizer_data and th.cuda.is_available():
                th.cuda.set_rng_state(optimizer_data["cuda_rng_state"])
                print("✓ Restored CUDA RNG state")
        elif optimizer is not None:
            print("ℹ No optimizer state file found")
        
        # Determine resume strategy
        if state["has_metadata"] and state["has_optimizer_state"]:
            print(f"✓ Full checkpoint - Resuming from Epoch {state['epoch']}, Step {state['step']}, Global Step {state['global_step']}")
        elif state["has_metadata"]:
            print("ℹ Metadata found but no optimizer state - starting fresh optimizer")
        else:
            print("ℹ Model-only checkpoint - starting fresh training with loaded weights")
        
        print(f"{'='*50}\n")
        return state
    
    def setup_sigint_handler(self, get_current_state_fn):
        """Setup SIGINT (Ctrl+C) handler for graceful shutdown with checkpoint saving."""
        
        if self.sigint_handler_installed:
            return
            
        def sigint_handler(signum, frame):
            print("\n\n🛑 Training interrupted by user (Ctrl+C)")
            print("💾 Saving checkpoint before exit...")
            
            # Get current training state
            state = get_current_state_fn()
            
            if state:
                # Save checkpoint
                self.save_checkpoint(
                    model=state["model"],
                    optimizer=state["optimizer"],
                    epoch=state["epoch"],
                    step=state["step"],
                    global_step=state["global_step"],
                    warmup_steps=state["warmup_steps"],
                    warmup_type=state["warmup_type"],
                    base_lr=state["base_lr"],
                    checkpoint_type="sigint",
                )
            
            print("✅ Checkpoint saved. Exiting gracefully.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, sigint_handler)
        self.sigint_handler_installed = True
        print("✓ Installed SIGINT handler for graceful interruption")
    
    def save_model_only(self, model: th.nn.Module, epoch: int, step: int) -> str:
        """Save only model weights (backward compatibility)."""
        filename = f"glide-ft-{epoch}x{step}.pt"
        filepath = os.path.join(self.checkpoints_dir, filename)
        th.save(model.state_dict(), filepath)
        print(f"Saved model checkpoint to {filepath}")
        return filepath