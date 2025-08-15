"""Checkpoint management utilities for GLIDE fine-tuning."""

import os
import json
import signal
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch as th
from tqdm import tqdm


class CheckpointManager:
    """Manages checkpoint saving, loading, and interrupt handling."""
    
    def __init__(self, checkpoints_dir: str, save_frequency: int = 1000):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoints_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N steps
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.interrupted = False
        
        # Setup interrupt handlers
        signal.signal(signal.SIGINT, self._interrupt_handler)
        signal.signal(signal.SIGTERM, self._interrupt_handler)
    
    def _interrupt_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.interrupted = True
        print("\n⚠️  Interrupt received! Will save checkpoint at next opportunity...")
    
    def save_checkpoint(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        epoch: int,
        global_step: int,
        is_interrupted: bool = False
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch
            global_step: Current global step
            is_interrupted: Whether this is an interrupted checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if is_interrupted:
            # Save interrupted checkpoint with special naming
            checkpoint_path = self.checkpoints_dir / "interrupted_checkpoint.pt"
            optimizer_path = self.checkpoints_dir / "interrupted_optimizer.pt"
            state_path = self.checkpoints_dir / "interrupted_state.json"
            
            # Save model
            th.save(model.state_dict(), checkpoint_path)
            
            # Save optimizer
            th.save(optimizer.state_dict(), optimizer_path)
            
            # Save training state
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "interrupted": True
            }
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"\n💾 Saved interrupted checkpoint to {self.checkpoints_dir}")
            print(f"   - Model: interrupted_checkpoint.pt")
            print(f"   - Optimizer: interrupted_optimizer.pt")
            print(f"   - State: interrupted_state.json")
            
            return str(checkpoint_path)
        else:
            # Regular checkpoint
            checkpoint_path = self.checkpoints_dir / f"glide-ft-{epoch}x{global_step}.pt"
            th.save(model.state_dict(), checkpoint_path)
            
            # Also save metadata for regular checkpoints
            metadata_path = self.checkpoints_dir / f"glide-ft-{epoch}x{global_step}.json"
            metadata = {
                "epoch": epoch,
                "global_step": global_step,
                "interrupted": False
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            tqdm.write(f"💾 Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)
    
    def should_save(self, global_step: int) -> bool:
        """Check if we should save at this step."""
        return global_step > 0 and global_step % self.save_frequency == 0
    
    def load_checkpoint(
        self,
        resume_path: str,
        model: th.nn.Module,
        optimizer: Optional[th.optim.Optimizer] = None
    ) -> Tuple[int, int]:
        """
        Load checkpoint and optionally restore training state.
        
        Args:
            resume_path: Path to checkpoint file or directory
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            
        Returns:
            Tuple of (start_epoch, global_step)
        """
        resume_path = Path(resume_path)
        start_epoch = 0
        global_step = 0
        
        if not resume_path.exists():
            print(f"⚠️  Resume path {resume_path} does not exist, starting fresh")
            return start_epoch, global_step
        
        # Handle directory vs file
        if resume_path.is_dir():
            # Check for interrupted checkpoint first
            interrupted_ckpt = resume_path / "interrupted_checkpoint.pt"
            if interrupted_ckpt.exists():
                print(f"📂 Found interrupted checkpoint in {resume_path}")
                checkpoint_file = interrupted_ckpt
                
                # Load state
                state_file = resume_path / "interrupted_state.json"
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        start_epoch = state.get('epoch', 0)
                        global_step = state.get('global_step', 0)
                        print(f"   Resuming from epoch {start_epoch}, step {global_step}")
                
                # Load optimizer if available
                if optimizer:
                    optimizer_file = resume_path / "interrupted_optimizer.pt"
                    if optimizer_file.exists():
                        optimizer.load_state_dict(th.load(optimizer_file, map_location="cpu"))
                        print("   ✓ Restored optimizer state")
            else:
                # Look for latest regular checkpoint
                checkpoints = sorted(resume_path.glob("glide-ft-*.pt"))
                if checkpoints:
                    checkpoint_file = checkpoints[-1]
                    print(f"📂 Found checkpoint: {checkpoint_file}")
                    
                    # Try to load metadata
                    metadata_file = checkpoint_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            start_epoch = metadata.get('epoch', 0)
                            global_step = metadata.get('global_step', 0)
                            print(f"   Resuming from epoch {start_epoch}, step {global_step}")
                else:
                    print(f"⚠️  No checkpoints found in {resume_path}")
                    return start_epoch, global_step
        else:
            # Direct file path
            checkpoint_file = resume_path
            print(f"📄 Loading checkpoint from {checkpoint_file}")
            
            # Try to load metadata if it exists
            metadata_file = checkpoint_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    start_epoch = metadata.get('epoch', 0)
                    global_step = metadata.get('global_step', 0)
                    print(f"   Found metadata: epoch {start_epoch}, step {global_step}")
        
        # Load model weights
        if checkpoint_file and checkpoint_file.exists():
            state_dict = th.load(checkpoint_file, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"✓ Loaded model weights from {checkpoint_file}")
        
        return start_epoch, global_step
    
    def cleanup_interrupted_files(self):
        """Remove interrupted checkpoint files after successful resume."""
        files_to_remove = [
            self.checkpoints_dir / "interrupted_checkpoint.pt",
            self.checkpoints_dir / "interrupted_optimizer.pt",
            self.checkpoints_dir / "interrupted_state.json"
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                print(f"🧹 Cleaned up {file_path.name}")