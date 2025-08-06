bash scripts/finetune-glide-clip.sh \
	--data-dir '/mnt/usb_nvme_2tb/Data/laion400m-dat-release/' \
	--clip-model 'ViT-L/14' \
	--clip-cache-dir '/mnt/usb_nvme_2tb/Data/laion400m-dat-release/clip_cache/' \
	--checkpoint-dir '/mnt/usb_nvme_2tb/Checkpoints/laion400m-dat-release-pt' \
	--epochs 5 \
	--eval-prompts 'examples/trippy_prompts_16.txt' \
	--resume glide_model_cache/base.pt
