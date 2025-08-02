"""
Integration module for using CLIP adapters with GLIDE models.

This module provides utilities for loading pretrained GLIDE models with
CLIP adapter support and managing the training process.
"""

import os
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.optim import AdamW

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

from .clip_text2im_model import ClipText2ImUNet
from .clip_adapter import CLIP_DIMENSIONS


def create_clip_model_from_options(
    model_options: Dict[str, Any],
    clip_model_name: str = "ViT-B/32",
    use_clip: bool = True,
    clip_gate_init: float = 0.0,
    freeze_glide_encoder: bool = True,
    device: str = "cuda",
) -> ClipText2ImUNet:
    """
    Create a ClipText2ImUNet from model options dict.
    
    This is a helper for testing that properly handles the model initialization.
    """
    from glide_text2im.tokenizer.bpe import get_encoder
    
    # Parse attention resolutions
    attention_ds = []
    if isinstance(model_options["attention_resolutions"], str):
        # Parse comma-separated string
        for res_str in model_options["attention_resolutions"].split(","):
            res = int(res_str.strip())
            attention_ds.append(model_options["image_size"] // res)
    else:
        # Already a list/tuple of integers
        for res in model_options["attention_resolutions"]:
            attention_ds.append(model_options["image_size"] // res)
    
    # Create the model with proper argument order
    model = ClipText2ImUNet(
        # Text2ImUNet positional args
        model_options["text_ctx"],
        model_options["xf_width"],
        model_options["xf_layers"],
        model_options["xf_heads"],
        model_options["xf_final_ln"],
        get_encoder(),
        # UNetModel keyword args
        in_channels=3,
        model_channels=model_options["num_channels"],
        out_channels=6,  # 3 * 2 for learned variance
        num_res_blocks=model_options["num_res_blocks"],
        attention_resolutions=tuple(attention_ds),
        dropout=model_options["dropout"],
        channel_mult=model_options["channel_mult"] if model_options["channel_mult"] else (1, 2, 3, 4),
        use_fp16=model_options.get("use_fp16", False),
        num_heads=model_options["num_heads"],
        num_head_channels=model_options["num_head_channels"],
        num_heads_upsample=model_options["num_heads_upsample"],
        use_scale_shift_norm=model_options["use_scale_shift_norm"],
        resblock_updown=model_options.get("resblock_updown", True),
        cache_text_emb=model_options.get("cache_text_emb", False),
        xf_padding=model_options["xf_padding"],
        # CLIP-specific args
        clip_model_name=clip_model_name,
        use_clip=use_clip,
        clip_gate_init=clip_gate_init,
        freeze_glide_encoder=freeze_glide_encoder,
    )
    
    return model.to(device)


def load_glide_model_with_clip(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = True,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
    clip_model_name: str = "ViT-L/14",
    use_clip: bool = True,
    clip_gate_init: float = 0.0,
    adapter_dropout: float = 0.1,
    use_lora: bool = False,
    lora_rank: int = 32,
):
    """
    Load a GLIDE model with optional CLIP adapter support.
    
    Args:
        glide_path: Path to GLIDE checkpoint
        use_fp16: Whether to use FP16
        freeze_transformer: Freeze GLIDE's text transformer
        freeze_diffusion: Freeze diffusion UNet
        activation_checkpointing: Use gradient checkpointing
        model_type: Type of model ('base', 'upsample', etc.)
        clip_model_name: CLIP model to use
        use_clip: Whether to enable CLIP
        clip_gate_init: Initial gate value (0.0 for stability)
        adapter_dropout: Dropout in adapter
        use_lora: Use LoRA adapter
        lora_rank: LoRA rank
        
    Returns:
        model: ClipText2ImUNet model
        diffusion: Diffusion object
        options: Model options dict
    """
    # Get model options
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if "inpaint" in model_type:
        options["inpaint"] = True
    
    options["use_fp16"] = use_fp16
    
    # Create base model with CLIP support
    model = ClipText2ImUNet(
        text_ctx=options["text_ctx"],
        xf_width=options["xf_width"],
        xf_layers=options["xf_layers"],
        xf_heads=options["xf_heads"],
        xf_final_ln=options["xf_final_ln"],
        tokenizer=options["tokenizer"],
        xf_padding=options["xf_padding"],
        image_size=options["image_size"],
        in_channels=options["in_channels"],
        model_channels=options["model_channels"],
        out_channels=options["out_channels"],
        num_res_blocks=options["num_res_blocks"],
        attention_resolutions=options["attention_resolutions"],
        dropout=options["dropout"],
        channel_mult=options["channel_mult"],
        num_heads=options["num_heads"],
        num_head_channels=options["num_head_channels"],
        num_heads_upsample=options["num_heads_upsample"],
        use_scale_shift_norm=options["use_scale_shift_norm"],
        resblock_updown=options["resblock_updown"],
        use_new_attention_order=options["use_new_attention_order"],
        # CLIP-specific arguments
        clip_model_name=clip_model_name,
        use_clip=use_clip,
        clip_gate_init=clip_gate_init,
        adapter_dropout=adapter_dropout,
        use_lora=use_lora,
        lora_rank=lora_rank,
        freeze_glide_encoder=freeze_transformer,
    )
    
    # Load pretrained weights if provided
    if glide_path:
        print(f"Loading GLIDE checkpoint from {glide_path}")
        state_dict = load_checkpoint(options["tokenizer"], glide_path)
        
        # Filter out any CLIP-related keys that might not exist in checkpoint
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not any(clip_key in k for clip_key in ['clip_', 'adapter', 'dual_']):
                filtered_state_dict[k] = v
        
        # Load with strict=False to allow new CLIP components
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded pretrained GLIDE weights (excluding CLIP components)")
    
    if activation_checkpointing:
        model.use_checkpoint = True
    
    # Handle freezing
    model.requires_grad_(True)
    
    if freeze_transformer:
        # This is handled in ClipText2ImUNet.__init__
        pass
    
    if freeze_diffusion:
        # Freeze all UNet components except CLIP-related ones
        for name, param in model.named_parameters():
            if not any(clip_key in name for clip_key in ['clip_', 'adapter', 'dual_']):
                if 'input_blocks' in name or 'middle_block' in name or 'output_blocks' in name:
                    param.requires_grad = False
    
    # Convert to FP16 if requested
    if use_fp16:
        model.convert_to_fp16()
    
    # Create diffusion
    diffusion = create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        learn_sigma=options["learn_sigma"],
        noise_schedule=options["noise_schedule"],
        use_kl=options["use_kl"],
        predict_xstart=options["predict_xstart"],
        rescale_timesteps=options["rescale_timesteps"],
        rescale_learned_sigmas=options["rescale_learned_sigmas"],
        timestep_respacing=options["timestep_respacing"],
    )
    
    return model, diffusion, options


def create_clip_adapter_optimizer(
    model: ClipText2ImUNet,
    adapter_lr: float = 1e-5,
    adapter_wd: float = 1e-2,
    adapter_beta2: float = 0.98,
    main_lr: Optional[float] = None,
    main_wd: float = 0.0,
    train_phases: str = "adapter_only",
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Create optimizers for CLIP adapter training with separate learning rates.
    
    Args:
        model: ClipText2ImUNet model
        adapter_lr: Learning rate for adapter components
        adapter_wd: Weight decay for adapter
        adapter_beta2: Beta2 for adapter optimizer
        main_lr: Learning rate for main model (if training)
        main_wd: Weight decay for main model
        train_phases: Training phase ('adapter_only', 'adapter_gates', 'full')
        
    Returns:
        optimizer: The optimizer
        optimizer_info: Information about parameter groups
    """
    # Get adapter parameters
    adapter_params = list(model.get_adapter_params())
    adapter_param_ids = {id(p) for p in adapter_params}
    
    # Separate parameters
    param_groups = []
    param_counts = {}
    
    if train_phases == "adapter_only":
        # Only train adapter
        param_groups.append({
            'params': adapter_params,
            'lr': adapter_lr,
            'weight_decay': adapter_wd,
            'betas': (0.9, adapter_beta2),
            'name': 'adapter',
        })
        param_counts['adapter'] = len(adapter_params)
        
    elif train_phases == "adapter_gates":
        # Train adapter + attention gates
        gate_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and id(param) not in adapter_param_ids:
                if 'clip_gate' in name or 'clip_kv' in name:
                    gate_params.append(param)
        
        param_groups.extend([
            {
                'params': adapter_params,
                'lr': adapter_lr,
                'weight_decay': adapter_wd,
                'betas': (0.9, adapter_beta2),
                'name': 'adapter',
            },
            {
                'params': gate_params,
                'lr': adapter_lr * 0.1,  # Lower LR for gates
                'weight_decay': 0.0,
                'betas': (0.9, 0.999),
                'name': 'gates',
            }
        ])
        param_counts['adapter'] = len(adapter_params)
        param_counts['gates'] = len(gate_params)
        
    elif train_phases == "full":
        # Train everything
        if main_lr is None:
            raise ValueError("main_lr must be specified for full training")
        
        main_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and id(param) not in adapter_param_ids:
                main_params.append(param)
        
        param_groups.extend([
            {
                'params': adapter_params,
                'lr': adapter_lr,
                'weight_decay': adapter_wd,
                'betas': (0.9, adapter_beta2),
                'name': 'adapter',
            },
            {
                'params': main_params,
                'lr': main_lr,
                'weight_decay': main_wd,
                'betas': (0.9, 0.999),
                'name': 'main',
            }
        ])
        param_counts['adapter'] = len(adapter_params)
        param_counts['main'] = len(main_params)
    
    else:
        raise ValueError(f"Unknown training phase: {train_phases}")
    
    # Create optimizer
    optimizer = AdamW(param_groups)
    
    # Create info dict
    optimizer_info = {
        'param_counts': param_counts,
        'train_phases': train_phases,
        'total_params': sum(param_counts.values()),
    }
    
    return optimizer, optimizer_info


class ClipAdapterTrainer:
    """
    Helper class for training CLIP adapters with stability monitoring.
    """
    
    def __init__(
        self,
        model: ClipText2ImUNet,
        diffusion,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10000,
        stability_threshold: float = 10.0,
        checkpoint_dir: str = "./checkpoints",
        adapter_grad_clip: float = 1.0,
        main_grad_clip: float = 1.0,
        early_stop_threshold: float = 0.1,
        early_stop_patience: int = 1000,
        baseline_eval_interval: int = 500,
    ):
        """
        Args:
            model: The model to train
            diffusion: Diffusion object
            optimizer: Optimizer
            warmup_steps: Steps for gate warmup
            stability_threshold: Loss spike threshold for rollback
            checkpoint_dir: Directory for checkpoints
            adapter_grad_clip: Max gradient norm for adapter parameters
            main_grad_clip: Max gradient norm for main model parameters
            early_stop_threshold: Max allowed degradation in pretrained performance (e.g., 0.1 = 10%)
            early_stop_patience: Steps to wait before early stopping after degradation detected
            baseline_eval_interval: How often to evaluate pretrained performance
        """
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stability_threshold = stability_threshold
        self.checkpoint_dir = checkpoint_dir
        self.adapter_grad_clip = adapter_grad_clip
        self.main_grad_clip = main_grad_clip
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.baseline_eval_interval = baseline_eval_interval
        
        self.step = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
        # Early stopping state
        self.baseline_performance = None
        self.degradation_detected_step = None
        self.should_stop = False
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def update_gates(self):
        """Update adapter gates based on warmup schedule."""
        self.model.set_adapter_gate_schedule(self.step, self.warmup_steps)
    
    def clip_gradients(self) -> Dict[str, float]:
        """
        Clip gradients separately for adapter and main model parameters.
        
        Returns:
            grad_norms: Dictionary with gradient norms before clipping
        """
        # Get adapter parameters
        adapter_params = list(self.model.get_adapter_params())
        adapter_param_set = set(adapter_params)
        
        # Separate parameters
        adapter_grads = []
        main_grads = []
        
        for p in self.model.parameters():
            if p.grad is not None:
                if p in adapter_param_set:
                    adapter_grads.append(p)
                else:
                    main_grads.append(p)
        
        # Calculate norms before clipping
        adapter_norm = 0.0
        if adapter_grads:
            adapter_norm = torch.nn.utils.clip_grad_norm_(
                adapter_grads, 
                self.adapter_grad_clip
            ).item()
        
        main_norm = 0.0
        if main_grads:
            main_norm = torch.nn.utils.clip_grad_norm_(
                main_grads,
                self.main_grad_clip
            ).item()
        
        return {
            'grad_norm_adapter_pre_clip': adapter_norm,
            'grad_norm_main_pre_clip': main_norm,
        }
    
    def check_stability(self, loss: float) -> bool:
        """
        Check if training is stable.
        
        Args:
            loss: Current loss value
            
        Returns:
            is_stable: Whether training is stable
        """
        self.loss_history.append(loss)
        
        # Keep only recent history
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        
        # Check for loss spike
        if len(self.loss_history) > 10:
            recent_avg = sum(self.loss_history[-10:-1]) / 9
            if loss > recent_avg * self.stability_threshold:
                return False
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
        
        return True
    
    def save_checkpoint(self, tag: str = ""):
        """Save a checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}_{self.step}.pt")
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.loss_history = checkpoint['loss_history']
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        metrics = {
            'step': self.step,
            'best_loss': self.best_loss,
        }
        
        # Add model stability metrics
        metrics.update(self.model.get_stability_metrics())
        
        # Add gradient norms (post-clipping)
        total_norm = 0.0
        adapter_norm = 0.0
        adapter_params = set(self.model.get_adapter_params())
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                if p in adapter_params:
                    adapter_norm += param_norm ** 2
        
        metrics['grad_norm_total'] = total_norm ** 0.5
        metrics['grad_norm_adapter'] = adapter_norm ** 0.5
        
        # Add gradient clipping thresholds
        metrics['adapter_grad_clip'] = self.adapter_grad_clip
        metrics['main_grad_clip'] = self.main_grad_clip
        
        return metrics
    
    def run_dry_run_test(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Run dry-run test on a batch to verify adapter doesn't affect outputs.
        
        Args:
            batch: Dictionary containing 'x', 'timesteps', 'tokens', 'mask', and optionally 'clip_embeddings'
            num_samples: Number of samples to test (default: 5, use -1 for all)
            
        Returns:
            Dictionary with test results including max/mean differences
        """
        # Limit number of samples for efficiency
        if num_samples > 0:
            batch_size = batch['x'].shape[0]
            num_samples = min(num_samples, batch_size)
            
            # Select subset of batch
            test_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    test_batch[key] = value[:num_samples]
                else:
                    test_batch[key] = value
        else:
            test_batch = batch
        
        # Run dry run test
        test_results = self.model.dry_run_test(
            x=test_batch.get('x'),
            timesteps=test_batch.get('timesteps'),
            tokens=test_batch.get('tokens'),
            mask=test_batch.get('mask'),
            clip_embeddings=test_batch.get('clip_embeddings'),
            return_metrics=True,
        )
        
        # Add training step info
        test_results['training_step'] = self.step
        test_results['num_samples_tested'] = num_samples if num_samples > 0 else batch['x'].shape[0]
        
        return test_results
    
    def log_dry_run_metrics(self, metrics: Dict[str, Any], logger=None):
        """
        Log dry-run test metrics.
        
        Args:
            metrics: Dictionary from run_dry_run_test
            logger: Optional logger (e.g., wandb) to use
        """
        log_dict = {
            'dry_run/output_diff_max': metrics.get('output_diff_max', 0.0),
            'dry_run/output_diff_mean': metrics.get('output_diff_mean', 0.0),
            'dry_run/outputs_identical': float(metrics.get('outputs_identical', False)),
            'dry_run/adapter_gate_value': metrics.get('adapter_gate_value', 0.0),
            'dry_run/step': metrics.get('training_step', 0),
        }
        
        if logger:
            logger.log(log_dict)
        else:
            # Print to console
            print(f"[Dry Run Test] Step {metrics.get('training_step', 0)}:")
            print(f"  Max output difference: {metrics.get('output_diff_max', 0.0):.6f}")
            print(f"  Mean output difference: {metrics.get('output_diff_mean', 0.0):.6f}")
            print(f"  Outputs identical: {metrics.get('outputs_identical', False)}")
            print(f"  Adapter gate value: {metrics.get('adapter_gate_value', 0.0):.4f}")
    
    def evaluate_baseline_performance(
        self,
        eval_batch: Dict[str, torch.Tensor],
        num_samples: int = 10,
    ) -> float:
        """
        Evaluate model performance without CLIP features to establish baseline.
        
        Args:
            eval_batch: Evaluation batch with 'x', 'timesteps', 'tokens', 'mask'
            num_samples: Number of samples to evaluate
            
        Returns:
            baseline_loss: Average loss without CLIP features
        """
        # Limit samples for efficiency
        if num_samples > 0:
            batch_size = eval_batch['x'].shape[0]
            num_samples = min(num_samples, batch_size)
            
            test_batch = {}
            for key, value in eval_batch.items():
                if isinstance(value, torch.Tensor):
                    test_batch[key] = value[:num_samples]
                else:
                    test_batch[key] = value
        else:
            test_batch = eval_batch
        
        # Store original use_clip setting
        original_use_clip = self.model.use_clip
        self.model.use_clip = False
        
        total_loss = 0.0
        with torch.no_grad():
            # Evaluate without CLIP features
            x = test_batch['x']
            timesteps = test_batch['timesteps']
            tokens = test_batch.get('tokens')
            mask = test_batch.get('mask')
            
            # Get model output
            model_output = self.model(x, timesteps, tokens=tokens, mask=mask)
            
            # Compute loss using diffusion
            t = timesteps
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x, t, noise=noise)
            
            # Compute diffusion loss
            terms = self.diffusion.training_losses(
                self.model,
                x_t,
                t,
                model_kwargs={'tokens': tokens, 'mask': mask},
                noise=noise,
            )
            
            loss = terms['loss'].mean()
            total_loss = loss.item()
        
        # Restore original setting
        self.model.use_clip = original_use_clip
        
        return total_loss
    
    def check_early_stopping(
        self,
        current_batch: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Check if training should stop due to pretrained performance degradation.
        
        Args:
            current_batch: Current training batch for evaluation
            
        Returns:
            should_stop: Whether to stop training
        """
        # Only check at intervals
        if self.step % self.baseline_eval_interval != 0:
            return self.should_stop
        
        # Evaluate current baseline performance
        current_performance = self.evaluate_baseline_performance(current_batch)
        
        # Initialize baseline on first evaluation
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            print(f"[Early Stopping] Initial baseline performance: {current_performance:.6f}")
            return False
        
        # Check for degradation
        degradation = (current_performance - self.baseline_performance) / self.baseline_performance
        
        if degradation > self.early_stop_threshold:
            if self.degradation_detected_step is None:
                self.degradation_detected_step = self.step
                print(f"[Early Stopping] Performance degradation detected at step {self.step}:")
                print(f"  Baseline: {self.baseline_performance:.6f}")
                print(f"  Current: {current_performance:.6f}")
                print(f"  Degradation: {degradation*100:.1f}%")
                print(f"  Waiting {self.early_stop_patience} steps before stopping...")
            
            # Check if patience exceeded
            if self.step - self.degradation_detected_step >= self.early_stop_patience:
                print(f"[Early Stopping] Patience exceeded. Stopping training.")
                self.should_stop = True
                return True
        else:
            # Reset if performance recovered
            if self.degradation_detected_step is not None:
                print(f"[Early Stopping] Performance recovered at step {self.step}")
                self.degradation_detected_step = None
            
            # Update baseline if performance improved
            if current_performance < self.baseline_performance:
                print(f"[Early Stopping] Baseline performance improved: {self.baseline_performance:.6f} → {current_performance:.6f}")
                self.baseline_performance = current_performance
        
        return False
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        compute_loss_fn,
    ) -> Dict[str, float]:
        """
        Execute a single training step with early stopping checks.
        
        Args:
            batch: Training batch
            compute_loss_fn: Function to compute loss
            
        Returns:
            metrics: Training metrics including loss and early stopping status
        """
        # Update gates
        self.update_gates()
        
        # Forward pass and compute loss
        loss = compute_loss_fn(batch)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients and get norms
        grad_norms = self.clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Check stability
        loss_value = loss.item()
        is_stable = self.check_stability(loss_value)
        
        if not is_stable:
            print(f"[Warning] Loss spike detected at step {self.step}: {loss_value:.4f}")
            # Could implement checkpoint rollback here
        
        # Check early stopping
        should_stop = self.check_early_stopping(batch)
        
        # Update step counter
        self.step += 1
        
        # Prepare metrics
        metrics = {
            'loss': loss_value,
            'step': self.step,
            'is_stable': is_stable,
            'should_stop': should_stop,
            **grad_norms,
            **self.get_metrics(),
        }
        
        # Add early stopping metrics
        if self.baseline_performance is not None:
            metrics['baseline_performance'] = self.baseline_performance
            if self.degradation_detected_step is not None:
                metrics['steps_since_degradation'] = self.step - self.degradation_detected_step
        
        return metrics