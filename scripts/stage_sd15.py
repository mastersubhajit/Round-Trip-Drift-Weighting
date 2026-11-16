#!/usr/bin/env python3
"""Download Stable Diffusion 1.5 (the trained model) at a pinned revision and record file hashes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REV = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
ASSETS = Path(os.environ.get('POISON_ASSET_ROOT', str(ROOT / 'assets')))
MODEL = ASSETS / 'sd15'
FILES = ['README.md', 'model_index.json', 'scheduler/scheduler_config.json',
         'tokenizer/merges.txt', 'tokenizer/vocab.json', 'tokenizer/tokenizer_config.json',
         'tokenizer/special_tokens_map.json', 'text_encoder/config.json',
         'text_encoder/model.safetensors', 'unet/config.json',
         'unet/diffusion_pytorch_model.safetensors', 'vae/config.json',
         'vae/diffusion_pytorch_model.safetensors', 'safety_checker/config.json',
         'safety_checker/model.safetensors', 'feature_extractor/preprocessor_config.json']


def download(item):
    url, dest = item
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as f:
            while chunk := response.read(4 * 1024 * 1024):
                f.write(chunk)
        tmp.rename(dest)
    digest = hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()
    print(json.dumps({'file': str(dest), 'bytes': dest.stat().st_size, 'sha256': digest}), flush=True)
    return {'path': str(dest), 'url': url, 'sha256': digest}


if __name__ == '__main__':
    MODEL.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        provenance = list(pool.map(download, [
            (f'https://huggingface.co/{REPO}/resolve/{REV}/{name}', MODEL / name) for name in FILES]))
    (MODEL / 'provenance.json').write_text(json.dumps({
        'repo': REPO, 'revision': REV, 'license': 'creativeml-openrail-m',
        'files': provenance}, indent=2))
