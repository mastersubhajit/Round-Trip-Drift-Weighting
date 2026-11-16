#!/usr/bin/env python3
"""Pre-download every auxiliary model used offline on compute nodes, and record exact revisions.

HF models go to $HF_HOME (snapshot pinned to the resolved commit, written to $POISON_STORE/assets/prefetch.json);
torchvision ImageNet weights go to $TORCH_HOME. Run on a host WITH internet (e.g. a transfer node).
"""
import json, os
from pathlib import Path

HF = ['openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14']


def main():
    from huggingface_hub import HfApi, snapshot_download
    import torchvision
    store = Path(os.environ['POISON_STORE'])
    rec = {'hf': {}, 'torchvision': {}}
    api = HfApi()
    for repo in HF:
        # revision='main' writes refs/main, which offline from_pretrained(repo) resolves; the exact
        # commit it resolved to is returned and recorded. Weights: model.safetensors when the repo has it, otherwise
        # pytorch_model.bin (openai/clip-vit-base-patch32 has NO safetensors on main; fetching only *.safetensors left
        # it without weights and the offline load failed).
        files = {s.rfilename for s in api.model_info(repo, revision='main').siblings}
        weights = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
        if weights not in files: raise SystemExit(f'{repo}: neither model.safetensors nor pytorch_model.bin on main')
        path = snapshot_download(repo, revision='main', allow_patterns=['*.json', '*.txt', weights])
        if not (Path(path) / weights).is_file() or (Path(path) / weights).stat().st_size == 0:
            raise SystemExit(f'{repo}: {weights} missing after download ({path})')
        sha = Path(path).name
        rec['hf'][repo] = dict(revision=sha, weights=weights)
        print('hf', repo, sha, weights, flush=True)
    for name, w in [('convnext_base', torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
                    ('inception_v3', torchvision.models.Inception_V3_Weights.IMAGENET1K_V1),
                    ('vit_b_16', torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)]:   # second judge
        getattr(torchvision.models, name)(weights=w)
        rec['torchvision'][name] = w.url
        print('torchvision', name, w.url, flush=True)
    # Noisy-upscaling defense (arm M2): public, ungated, pinned revision, local copy for offline nodes
    up = snapshot_download('stabilityai/stable-diffusion-x4-upscaler', revision='572c99286543a273bfd17fac263db5a77be12c4c',
                           local_dir=str(store / 'assets/sd-x4-upscaler'),
                           allow_patterns=['*.json', '*.txt', '*/diffusion_pytorch_model.safetensors', 'text_encoder/model.safetensors'])
    rec['hf']['stabilityai/stable-diffusion-x4-upscaler'] = '572c99286543a273bfd17fac263db5a77be12c4c'
    (store / 'assets').mkdir(parents=True, exist_ok=True)
    (store / 'assets/prefetch.json').write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
