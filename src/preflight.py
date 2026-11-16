#!/usr/bin/env python3
"""Offline input audit. Never downloads weights or constructs a diffusion model."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def inspect_output(path):
    output = Path(path).resolve()
    parent = output
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f'Output ancestor is not a directory: {parent}')
    try:
        with tempfile.TemporaryFile(dir=parent):
            pass
    except OSError as error:
        raise ValueError(f'Output filesystem is not writable: {parent}: {error}') from error
    return {'path': str(output), 'writable': True}


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dataset_fingerprint(path):
    from PIL import Image
    root = Path(path).resolve()
    metadata = root / 'metadata.jsonl'
    if not metadata.is_file():
        raise ValueError(f'Missing training metadata: {metadata}')
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty training dataset: {root}')
    audit_path = root.parent / 'audit.json'
    if not audit_path.is_file():
        raise ValueError(f'Missing rights/provenance audit: {audit_path}; export an audited manifest first')
    audit = json.loads(audit_path.read_text())
    provenance = audit.get('record_provenance', [])
    rights = {r['id']: r for r in provenance}
    if len(rights) != len(provenance) or len(rows) != audit.get('train_records'):
        raise ValueError('Dataset audit count or duplicate provenance mismatch')
    hashes, seen = [], set()
    for row in rows:
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Each training caption must be a nonempty string')
        identity = row.get('id')
        if identity in seen or identity not in rights:
            raise ValueError(f'Duplicate ID or missing rights record: {identity}')
        seen.add(identity)
        record = rights[identity]
        if record.get('permission') is not True or not record.get('license') or not record.get('source_id'):
            raise ValueError(f'Unverified rights/provenance: {identity}')
        image = (root / row['file_name']).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f'Invalid image path: {image}')
        with Image.open(image) as im:
            im.verify()
        digest = file_hash(image)
        if row.get('image_sha256', digest) != digest:
            raise ValueError(f'Image hash mismatch: {image}')
        hashes.append((row, digest))
    fingerprint = hashlib.sha256(json.dumps([hashes, audit], sort_keys=True).encode()).hexdigest()
    return {'records': len(rows), 'fingerprint': fingerprint, 'image_hashes': [h for _, h in hashes],
            'audit': audit}


def inspect_inputs(model, train_dir, trusted_dir=None, negative_map=None, require_gpu=False, output_dir=None):
    output = inspect_output(output_dir) if output_dir else None
    model = Path(model).resolve()
    if not model.is_dir():
        raise ValueError(f'Missing local model directory: {model}; stage an authorized SD1.x checkpoint')
    expected = ['model_index.json', 'unet/config.json', 'vae/config.json',
                'text_encoder/config.json', 'scheduler/scheduler_config.json',
                'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt']
    for name in expected:
        if not (model / name).is_file():
            raise ValueError(f'Missing model file: {model / name}')
    for part in ['unet', 'vae', 'text_encoder']:
        if not list((model / part).glob('*.safetensors')) and not list((model / part).glob('*.bin')):
            raise ValueError(f'Missing model weights: {model / part}')
    card = (model / 'README.md').read_text() if (model / 'README.md').is_file() else ''
    if 'license:' not in card and not (model / 'LICENSE').is_file():
        raise ValueError('Missing model license documentation (README.md license field or LICENSE)')
    scheduler = json.loads((model / 'scheduler/scheduler_config.json').read_text())
    if scheduler.get('prediction_type', 'epsilon') not in {'epsilon', 'v_prediction'}:
        raise ValueError('Unsupported scheduler prediction_type')
    unet = json.loads((model / 'unet/config.json').read_text())
    if unet.get('cross_attention_dim') != 768:
        raise ValueError('This trainer supports SD1.x (cross_attention_dim=768)')
    train = dataset_fingerprint(train_dir)
    trusted = dataset_fingerprint(trusted_dir) if trusted_dir else None
    if train['audit'].get('split') not in {'train', 'trusted'}:
        raise ValueError('Training on validation/test records is forbidden')
    if trusted:
        if trusted['audit'].get('split') != 'trusted':
            raise ValueError('Anchor dataset must use the trusted split')
        if set(train['image_hashes']) & set(trusted['image_hashes']):
            raise ValueError('Training and trusted images overlap')
    if negative_map:
        from data_utils import ImageMetadataDataset, anchor_negative_texts, collate
        mapping = json.loads(Path(negative_map).read_text())
        data = ImageMetadataDataset(trusted_dir, size=32)
        for item in data:
            anchor_negative_texts(collate([item]), mapping)
    # Include full actual weight hashes, not merely filenames, in resume identity.
    model_files = sorted(p for p in model.rglob('*') if p.is_file() and p.suffix in {'.json', '.txt', '.bin', '.safetensors', '.md'})
    model_hashes = {str(p.relative_to(model)): file_hash(p) for p in model_files if p.name != 'provenance.json'}
    report = {'model': str(model), 'model_fingerprint': hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest(),
              'model_files_sha256': model_hashes, 'train': train, 'trusted': trusted,
              'prediction_type': scheduler.get('prediction_type', 'epsilon'),
              'vae_scaling_factor': json.loads((model / 'vae/config.json').read_text()).get('scaling_factor', 0.18215)}
    if output:
        report['output'] = output
    if require_gpu:
        import os
        import torch
        if not torch.cuda.is_available():
            raise ValueError('GPU training needs a CUDA device')
        device = torch.cuda.current_device()
        report['gpu'] = {'name': torch.cuda.get_device_name(device),
                         'capability': torch.cuda.get_device_capability(device),
                         'bf16': torch.cuda.is_bf16_supported(),
                         'vram_bytes': torch.cuda.get_device_properties(device).total_memory,
                         'visible_devices': torch.cuda.device_count(),
                         'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                         'slurm_partition': os.environ.get('SLURM_JOB_PARTITION')}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--train-dir', required=True)
    p.add_argument('--trusted-dir')
    p.add_argument('--output-dir', help='Check training output filesystem writability without leaving files')
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--out')
    a = p.parse_args()
    report = inspect_inputs(a.model, a.train_dir, a.trusted_dir, require_gpu=a.require_gpu, output_dir=a.output_dir)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
