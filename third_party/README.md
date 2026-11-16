# Third-party code

Fetched and verified at pinned commits by `scripts/pipeline/stage_all.sh`. Not redistributed in this repository.

| Directory | Source | Commit | License | Used for |
|---|---|---|---|---|
| `nightshade-release/` | https://github.com/Shawn-Shan/nightshade-release | `28e4ee9115ed07a7fd463f297fefcea08c5ee0bf` | GPL-3.0 | generating the reference poisons; the adaptive attacks re-optimise its latent objective |
| `robust-style-mimicry/` | https://github.com/ethz-spylab/robust-style-mimicry | `b4bf72b803c9bc793db085da2d33009f0fb7da7f` | MIT | Gaussian noising (M1) and noisy upscaling (M2) baselines |

Model weights are downloaded from the Hugging Face Hub at pinned revisions by `scripts/prefetch_models.py` and
`scripts/stage_sd15.py` (Stable Diffusion 1.5 and 2.1 VAE, SD x4 upscaler, CLIP ViT-B/32 and ViT-L/14);
the ImageNet judges come from torchvision.
