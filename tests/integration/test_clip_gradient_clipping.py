#!/usr/bin/env python3
"""
Integration test for CLIP adapter gradient clipping.

Tests that gradients are properly clipped for adapter and main model parameters.
"""

import pytest
import torch
import torch.nn as nn
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

from glide_finetune.adapters.glide_clip_integration import (
    ClipAdapterTrainer,
    create_clip_adapter_optimizer,
)
from glide_finetune.adapters.clip_text2im_model import ClipText2ImUNet


class TestClipGradientClipping:
    @pytest.fixture
    def model_and_diffusion(self):
        """Create a small CLIP-enabled model for testing."""
        options = model_and_diffusion_defaults()
        options.update({
            'image_size': 64,
            'num_channels': 32,
            'num_res_blocks': 1,
            'num_heads': 1,
            'num_head_channels': 32,
            'attention_resolutions': '16,8',
            'channel_mult': '',
            'dropout': 0.0,
            'use_scale_shift_norm': True,
            'resblock_updown': True,
            'use_fp16': False,
            'diffusion_steps': 100,
            'noise_schedule': 'squaredcos_cap_v2',
            'timestep_respacing': '10',
            'text_ctx': 128,
            'xf_width': 512,
            'xf_layers': 8,
            'xf_heads': 8,
            'xf_final_ln': True,
            'xf_padding': True,
        })
        
        # Create ClipText2ImUNet instead of regular model
        from glide_text2im.tokenizer.bpe import get_encoder
        model = ClipText2ImUNet(
            text_ctx=options['text_ctx'],
            xf_width=options['xf_width'],
            xf_layers=options['xf_layers'],
            xf_heads=options['xf_heads'],
            xf_final_ln=options['xf_final_ln'],
            tokenizer=get_encoder(),
            xf_padding=options['xf_padding'],
            in_channels=3,
            model_channels=options['num_channels'],
            out_channels=6,
            num_res_blocks=options['num_res_blocks'],
            attention_resolutions=(16, 8),
            dropout=options['dropout'],
            channel_mult=(1, 2, 4),
            use_fp16=options['use_fp16'],
            num_heads=options['num_heads'],
            num_head_channels=options['num_head_channels'],
            num_heads_upsample=-1,
            use_scale_shift_norm=options['use_scale_shift_norm'],
            resblock_updown=options['resblock_updown'],
            use_checkpoint=False,
            cache_text_emb=False,
            # CLIP-specific
            clip_model_name="ViT-B/32",
            use_clip=True,
            clip_gate_init=0.0,
            adapter_dropout=0.1,
            use_lora=False,
            lora_rank=32,
            freeze_glide_encoder=True,
        )
        
        diffusion = create_model_and_diffusion(**options)[1]
        
        # Load pretrained weights to avoid NaN issues
        import os
        pretrained_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "glide_model_cache", "base.pt"
        )
        
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            # First need to create a model with matching architecture
            from glide_finetune.adapters.glide_clip_integration import create_clip_model_from_options
            
            # Use default options for pretrained model
            pretrained_options = model_and_diffusion_defaults()
            pretrained_options['use_fp16'] = False
            
            # Create temporary model with correct architecture
            temp_model = create_clip_model_from_options(
                pretrained_options,
                clip_model_name="ViT-B/32",
                use_clip=True,
                clip_gate_init=0.0,
                device='cpu',
            )
            
            # Load pretrained weights
            pretrained_state = torch.load(pretrained_path, map_location='cpu')
            temp_model.load_state_dict(pretrained_state, strict=False)
            
            # Copy relevant weights to our test model
            # This is a bit hacky but necessary since our test model has different architecture
            model.load_state_dict(temp_model.state_dict(), strict=False)
            print("Loaded partial pretrained weights into test model")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Test may produce NaN values without proper initialization")
        
        return model, diffusion
    
    def test_gradient_clipping_separate_norms(self, model_and_diffusion):
        """Test that adapter and main model gradients are clipped separately."""
        model, diffusion = model_and_diffusion
        device = 'cpu'
        model = model.to(device)
        
        # Create optimizer with separate parameter groups
        optimizer, optimizer_info = create_clip_adapter_optimizer(
            model,
            adapter_lr=1e-5,
            adapter_wd=0.0,
            adapter_beta2=0.98,
            main_lr=1e-4,
            main_wd=0.0,
            train_phases="full",  # Train everything
        )
        
        # Create trainer with different clip values
        trainer = ClipAdapterTrainer(
            model=model,
            diffusion=diffusion,
            optimizer=optimizer,
            warmup_steps=1000,
            stability_threshold=10.0,
            checkpoint_dir="./test_checkpoints",
            adapter_grad_clip=0.5,  # Lower clip for adapter
            main_grad_clip=2.0,     # Higher clip for main model
        )
        
        # Create dummy loss and backprop to generate gradients
        batch_size = 2
        dummy_output = torch.randn(batch_size, 6, 64, 64, requires_grad=True)
        loss = dummy_output.sum()
        loss.backward()
        
        # Manually set large gradients to test clipping
        adapter_params = list(model.get_adapter_params())
        adapter_param_set = set(adapter_params)
        for p in adapter_params:
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 10.0  # Large gradient
        
        # Set different large gradients for main model
        for p in model.parameters():
            if p.requires_grad and p not in adapter_param_set:
                p.grad = torch.randn_like(p) * 20.0  # Even larger gradient
        
        # Clip gradients
        grad_norms = trainer.clip_gradients()
        
        # Check that norms were large before clipping
        assert grad_norms['grad_norm_adapter_pre_clip'] > 0.5
        assert grad_norms['grad_norm_main_pre_clip'] > 2.0
        
        print(f"Adapter norm before clipping: {grad_norms['grad_norm_adapter_pre_clip']}")
        print(f"Main norm before clipping: {grad_norms['grad_norm_main_pre_clip']}")
        
        # Check that gradients are now clipped
        adapter_norm_after = 0.0
        for p in adapter_params:
            if p.grad is not None:
                adapter_norm_after += p.grad.data.norm(2).item() ** 2
        adapter_norm_after = adapter_norm_after ** 0.5
        
        main_norm_after = 0.0
        for p in model.parameters():
            if p.grad is not None and p not in adapter_param_set:
                main_norm_after += p.grad.data.norm(2).item() ** 2
        main_norm_after = main_norm_after ** 0.5
        
        print(f"Adapter norm after clipping: {adapter_norm_after}")
        print(f"Main norm after clipping: {main_norm_after}")
        
        # Allow small tolerance for floating point
        assert adapter_norm_after <= 0.5 + 1e-2  # 1% tolerance
        assert main_norm_after <= 2.0 + 1e-2  # 1% tolerance
    
    def test_gradient_clipping_with_no_gradients(self, model_and_diffusion):
        """Test gradient clipping when some parameters have no gradients."""
        model, diffusion = model_and_diffusion
        device = 'cpu'
        model = model.to(device)
        
        optimizer, _ = create_clip_adapter_optimizer(
            model,
            adapter_lr=1e-5,
            train_phases="adapter_only",
        )
        
        trainer = ClipAdapterTrainer(
            model=model,
            diffusion=diffusion,
            optimizer=optimizer,
            adapter_grad_clip=1.0,
            main_grad_clip=1.0,
        )
        
        # Don't create any gradients
        grad_norms = trainer.clip_gradients()
        
        # Should return 0 norms
        assert grad_norms['grad_norm_adapter_pre_clip'] == 0.0
        assert grad_norms['grad_norm_main_pre_clip'] == 0.0
    
    def test_gradient_clipping_metrics(self, model_and_diffusion):
        """Test that gradient clipping thresholds are included in metrics."""
        model, diffusion = model_and_diffusion
        device = 'cpu'
        model = model.to(device)
        
        optimizer, _ = create_clip_adapter_optimizer(model)
        
        trainer = ClipAdapterTrainer(
            model=model,
            diffusion=diffusion,
            optimizer=optimizer,
            adapter_grad_clip=0.75,
            main_grad_clip=1.5,
        )
        
        metrics = trainer.get_metrics()
        
        assert metrics['adapter_grad_clip'] == 0.75
        assert metrics['main_grad_clip'] == 1.5
        assert 'grad_norm_total' in metrics
        assert 'grad_norm_adapter' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])