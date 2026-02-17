from glide_text2im.gaussian_diffusion import get_named_beta_schedule
from glide_text2im.respace import SpacedDiffusion, space_timesteps
from glide_text2im.text2im_model import (
    InpaintText2ImUNet,
    LatentText2ImUNet,
    SuperResInpaintText2ImUnet,
    SuperResText2ImUNet,
    Text2ImUNet,
)
from glide_text2im.tokenizer.bpe import get_encoder


def model_and_diffusion_defaults():
    return dict(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
    )


def model_and_diffusion_defaults_upsampler():
    result = model_and_diffusion_defaults()
    result.update(
        dict(
            image_size=256,
            num_res_blocks=2,
            noise_schedule="linear",
            super_res=True,
        )
    )
    return result


def model_and_diffusion_defaults_latent():
    result = model_and_diffusion_defaults()
    result.update(
        dict(
            image_size=32,
            in_channels=4,
            out_channels=8,
            channel_mult="1,2,4",
            attention_resolutions="16,8",
            noise_schedule="linear_latent",
            latent_mode=True,
            clip_dim=768,
            clip_tokens=4,
        )
    )
    return result


def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    latent_mode=False,
    clip_dim=768,
    clip_tokens=4,
    **kwargs,  # absorb extra keys from defaults dicts
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
        latent_mode=latent_mode,
        clip_dim=clip_dim,
        clip_tokens=clip_tokens,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    latent_mode=False,
    clip_dim=768,
    clip_tokens=4,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if latent_mode:
        model_cls = LatentText2ImUNet
    elif inpaint and super_res:
        model_cls = SuperResInpaintText2ImUnet
    elif inpaint:
        model_cls = InpaintText2ImUNet
    elif super_res:
        model_cls = SuperResText2ImUNet
    else:
        model_cls = Text2ImUNet

    # Base kwargs common to all model classes
    model_kwargs = dict(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        tokenizer=get_encoder(),
        xf_padding=xf_padding,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
    )

    if latent_mode:
        # LatentText2ImUNet forces in_channels=4, out_channels=8 internally
        model_kwargs["in_channels"] = 4
        model_kwargs["out_channels"] = 8
        model_kwargs["clip_dim"] = clip_dim
        model_kwargs["clip_tokens"] = clip_tokens
    else:
        model_kwargs["in_channels"] = 3
        model_kwargs["out_channels"] = 6

    return model_cls(**model_kwargs)


def create_gaussian_diffusion(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )
