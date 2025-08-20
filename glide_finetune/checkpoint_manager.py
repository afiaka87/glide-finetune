"""Checkpoint management utilities for GLIDE fine-tuning."""

import os
import json
import signal
from datetime import datetime
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
        self.interrupt_count = 0
        self.original_handlers = {}
        
        # Store original handlers and setup interrupt handlers
        self.original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._interrupt_handler)
        self.original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._interrupt_handler)
    
    def _interrupt_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.interrupt_count += 1
        
        if self.interrupt_count == 1:
            self.interrupted = True
            print("\n⚠️  Interrupt received! Will save checkpoint at next opportunity...")
            print("    Press Ctrl+C again to force exit without saving.")
        else:
            print("\n❌ Force exit requested. Exiting immediately...")
            # Restore original handler and re-raise
            if signum in self.original_handlers:
                signal.signal(signum, self.original_handlers[signum])
            os._exit(1)
    
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
            # Save interrupted checkpoint with timestamp in the name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoints_dir / f"interrupted_checkpoint_{timestamp}.pt"
            optimizer_path = self.checkpoints_dir / f"interrupted_optimizer_{timestamp}.pt"
            state_path = self.checkpoints_dir / f"interrupted_state_{timestamp}.json"
            
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
            print(f"   - Model: {checkpoint_path.name}")
            print(f"   - Optimizer: {optimizer_path.name}")
            print(f"   - State: {state_path.name}")
            
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
            # Check for interrupted checkpoints (both old and new format)
            # First try new timestamped format
            interrupted_ckpts = sorted(resume_path.glob("interrupted_checkpoint_*.pt"))
            
            # Fall back to old format if no timestamped ones found
            if not interrupted_ckpts:
                old_interrupted = resume_path / "interrupted_checkpoint.pt"
                if old_interrupted.exists():
                    interrupted_ckpts = [old_interrupted]
            
            if interrupted_ckpts:
                # Use the most recent interrupted checkpoint
                checkpoint_file = interrupted_ckpts[-1]
                print(f"📂 Found interrupted checkpoint: {checkpoint_file.name}")
                
                # Determine the corresponding state and optimizer files
                if checkpoint_file.name == "interrupted_checkpoint.pt":
                    # Old format
                    state_file = resume_path / "interrupted_state.json"
                    optimizer_file = resume_path / "interrupted_optimizer.pt"
                else:
                    # New timestamped format - extract timestamp
                    timestamp = checkpoint_file.stem.replace("interrupted_checkpoint_", "")
                    state_file = resume_path / f"interrupted_state_{timestamp}.json"
                    optimizer_file = resume_path / f"interrupted_optimizer_{timestamp}.pt"
                
                # Load state
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        start_epoch = state.get('epoch', 0)
                        global_step = state.get('global_step', 0)
                        print(f"   Resuming from epoch {start_epoch}, step {global_step}")
                
                # Load optimizer if available
                if optimizer and optimizer_file.exists():
                    optimizer.load_state_dict(th.load(optimizer_file, map_location="cpu", weights_only=False))
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
            state_dict = th.load(checkpoint_file, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded model weights from {checkpoint_file}")
        
        return start_epoch, global_step
    
    def cleanup_interrupted_files(self):
        """Remove interrupted checkpoint files after successful resume."""
        # Clean up old format files
        old_format_files = [
            self.checkpoints_dir / "interrupted_checkpoint.pt",
            self.checkpoints_dir / "interrupted_optimizer.pt",
            self.checkpoints_dir / "interrupted_state.json"
        ]
        
        for file_path in old_format_files:
            if file_path.exists():
                file_path.unlink()
                print(f"🧹 Cleaned up {file_path.name}")
        
        # Clean up new timestamped format files
        for pattern in ["interrupted_checkpoint_*.pt", "interrupted_optimizer_*.pt", "interrupted_state_*.json"]:
            for file_path in self.checkpoints_dir.glob(pattern):
                file_path.unlink()
                print(f"🧹 Cleaned up {file_path.name}")