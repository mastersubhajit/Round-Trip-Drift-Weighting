#!/usr/bin/env python3
"""Round-trip semantic drift d(x) = 1 - cos(phi(x), phi(D(E(x)))) for an ImageFolder train dir.

E/D = the TRAINING model's own frozen VAE (SD1.5), phi = CLIP ViT-L/14 image encoder. Uses the exact
512 resize/center-crop of data_utils.ImageMetadataDataset, i.e. the pixels the trainer encodes.
No captions, labels or attacker-target information are used. `is_poisoned` is copied to the output
for evaluation only (AUROC); the weighting code never reads it.

weights subcommand: w_i = sigmoid((tau - d_i)/s), tau = q-quantile of drift on the trusted dir,
s = (tau - median_trusted)/4 ; or --unsupervised: tau = median + 3*MAD of the train dir itself.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def load_rows(d):
    return [json.loads(l) for l in (Path(d) / 'metadata.jsonl').read_text().splitlines() if l.strip()]


def score(a):
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL
    from transformers import CLIPModel, CLIPImageProcessor
    from data_utils import ImageMetadataDataset
    dev = 'cuda'
    vae = AutoencoderKL.from_pretrained(a.model, subfolder='vae', local_files_only=True).to(dev, torch.float32).eval()
    clip = CLIPModel.from_pretrained(a.clip).to(dev, torch.float16).eval()
    proc = CLIPImageProcessor.from_pretrained(a.clip)
    ds = ImageMetadataDataset(a.dir, 512)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    trip = Path(a.triplets) if a.triplets else None
    if trip: trip.mkdir(parents=True, exist_ok=True)
    saved = 0
    with out.open('w') as f, torch.no_grad():
        for start in range(0, len(ds), a.batch):
            items = [ds[i] for i in range(start, min(len(ds), start + a.batch))]
            rows = ds.rows[start:start + len(items)]
            x = torch.stack([it['pixel_values'] for it in items]).to(dev)
            lat = vae.encode(x).latent_dist.mean
            rec = vae.decode(lat).sample.clamp(-1, 1)
            to_pil = lambda t: [Image.fromarray(((u.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype('uint8')) for u in t]
            src, rt = to_pil(x), to_pil(rec)
            px = proc(images=src + rt, return_tensors='pt')['pixel_values'].to(dev, torch.float16)
            e = clip.get_image_features(pixel_values=px).float(); e = e / e.norm(dim=-1, keepdim=True)
            n = len(items)
            cos = (e[:n] * e[n:]).sum(-1).cpu().numpy()
            mse = ((rec - x) ** 2).flatten(1).mean(1).cpu().numpy()
            for r, c, m, s_img, r_img in zip(rows, cos, mse, src, rt):
                f.write(json.dumps(dict(id=r.get('id'), file_name=r['file_name'], concept=r.get('concept'),
                    is_poisoned=bool(r.get('is_poisoned')), drift=float(1 - c), pixel_mse=float(m))) + '\n')
                if trip and r.get('is_poisoned') and saved < a.n_triplets:
                    s_img.save(trip / f"{r.get('id')}_input.png"); r_img.save(trip / f"{r.get('id')}_roundtrip.png"); saved += 1


def weights(a):
    tr = [json.loads(l) for l in Path(a.scores).read_text().splitlines() if l.strip()]
    d = np.array([r['drift'] for r in tr])
    if a.unsupervised:
        med = float(np.median(d)); mad = float(np.median(np.abs(d - med))) * 1.4826
        tau, s = med + 3 * mad, max(mad, 1e-6) * 0.75
        basis = 'train median + 3*MAD (no trusted data)'
    else:
        cal = np.array([json.loads(l)['drift'] for l in Path(a.calibration).read_text().splitlines() if l.strip()])
        tau = float(np.quantile(cal, a.quantile)); s = max((tau - float(np.median(cal))) / 4, 1e-6)
        basis = f'trusted q{a.quantile}'
    w = {r['id']: 1 / (1 + math.exp(-(tau - r['drift']) / s)) for r in tr}
    if a.hard:
        w = {k: float(v >= 0.5) for k, v in w.items()}
    Path(a.out).write_text(json.dumps(dict(tau=tau, scale=s, basis=basis, hard=a.hard, weights=w), indent=2))
    pois = [w[r['id']] for r in tr if r['is_poisoned']]; clean = [w[r['id']] for r in tr if not r['is_poisoned']]
    print(json.dumps(dict(tau=tau, scale=s, basis=basis, mean_w_poison=float(np.mean(pois)) if pois else None,
                          mean_w_clean=float(np.mean(clean)), n_poison=len(pois), n_clean=len(clean))))


def filter_dir(a):
    """Export a filtered copy (hard detect-and-remove baselines I and J) (metadata + copied images)."""
    import os
    src = Path(a.dir); dst = Path(a.out) / 'train'; dst.mkdir(parents=True, exist_ok=False)
    keep = json.loads(Path(a.weights).read_text())['weights']
    n = 0
    with (dst / 'metadata.jsonl').open('w') as f:
        for r in load_rows(src):
            if keep.get(r['id'], 1.0) >= 0.5:
                import shutil  # real copies: preflight requires every image to live inside the training dir
                shutil.copyfile(src / r['file_name'], dst / r['file_name']); f.write(json.dumps(r) + '\n'); n += 1
    print(json.dumps(dict(kept=n)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score'); s.add_argument('--dir', required=True); s.add_argument('--model', required=True)
    s.add_argument('--clip', default='openai/clip-vit-large-patch14'); s.add_argument('--out', required=True)
    s.add_argument('--batch', type=int, default=16); s.add_argument('--triplets'); s.add_argument('--n-triplets', type=int, default=12)
    w = sub.add_parser('weights'); w.add_argument('--scores', required=True); w.add_argument('--calibration')
    w.add_argument('--quantile', type=float, default=0.99); w.add_argument('--unsupervised', action='store_true')
    w.add_argument('--hard', action='store_true'); w.add_argument('--out', required=True)
    fl = sub.add_parser('filter'); fl.add_argument('--dir', required=True); fl.add_argument('--weights', required=True); fl.add_argument('--out', required=True)
    a = p.parse_args(); {'score': score, 'weights': weights, 'filter': filter_dir}[a.cmd](a)
