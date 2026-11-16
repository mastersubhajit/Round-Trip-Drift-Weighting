#!/usr/bin/env python3
"""Pin the accessible SD2.1 community mirror for the official Nightshade optimizer."""
import concurrent.futures
import json
import os
from pathlib import Path
from stage_sd15 import download

REPO = 'sd2-community/stable-diffusion-2-1'
REV = 'bb2154823665391b4fb29b0b9cf82a198964ee05'
DEST = Path(os.environ['POISON_ASSET_ROOT']) / 'sd21'
FILES = ['README.md', 'model_index.json', 'scheduler/scheduler_config.json',
         'tokenizer/merges.txt', 'tokenizer/vocab.json', 'tokenizer/tokenizer_config.json',
         'tokenizer/special_tokens_map.json', 'text_encoder/config.json',
         'text_encoder/model.fp16.safetensors', 'unet/config.json',
         'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/config.json',
         'vae/diffusion_pytorch_model.fp16.safetensors', 'feature_extractor/preprocessor_config.json']
if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, [(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}', DEST/name) for name in FILES]))
    (DEST/'provenance.json').write_text(json.dumps(dict(repo=REPO, revision=REV, license='openrail++',
        original_upstream_model='stabilityai/stable-diffusion-2-1',
        original_endpoint_status='HTTP 401 observed; community mirror is a recorded dependency change', files=records), indent=2))
