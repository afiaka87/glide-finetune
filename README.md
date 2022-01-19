# (WIP) GLIDE-Finetune

Finetune the base 64 px GLIDE-text2im model from OpenAI on your own image-text dataset.

Known issues:
- batching isn't handled in the dataloader
- NaN/Inf errors
- some of the code is messy, needs refactoring.