#!/usr/bin/env python3
"""Arm LF score: per-record denoising loss of the PRETRAINED SD1.5 under the record's own caption.

Re-implementation of the "filter high-loss training data" countermeasure evaluated in the Nightshade paper
(Shan et al., IEEE S&P 2024, Sec. 7): a poison's latent should fit its caption badly under a clean model, so the
defender drops the highest-loss fraction (FILTER_DROP in run_arm.py, pre-registered 10%, same as arm I). Not the
authors' code (none was released for it). Deterministic: fixed timesteps and a per-record noise seed derived from
the record id, so the score does not depend on GPU count, batch order or which arm computes it first.
`is_poisoned` is copied for evaluation only (AUROC); the filter never reads it.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

TIMESTEPS = (100, 300, 500, 700, 900)


def record_seed(rid):
    return int(hashlib.sha256(str(rid).encode()).hexdigest()[:8], 16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True); p.add_argument('--model', required=True); p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=8)
    a = p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from data_utils import ImageMetadataDataset
    from train_full_unet import get_target
    from training_objectives import per_example_denoising_loss
    dev, dt = 'cuda', torch.bfloat16
    kw = dict(local_files_only=True)
    vae = AutoencoderKL.from_pretrained(a.model, subfolder='vae', **kw).to(dev, torch.float32).eval()
    unet = UNet2DConditionModel.from_pretrained(a.model, subfolder='unet', **kw).to(dev, dt).eval()
    enc = CLIPTextModel.from_pretrained(a.model, subfolder='text_encoder', **kw).to(dev, dt).eval()
    tok = CLIPTokenizer.from_pretrained(a.model, subfolder='tokenizer', **kw)
    sched = DDPMScheduler.from_pretrained(a.model, subfolder='scheduler', **kw)
    ds = ImageMetadataDataset(a.dir, 512)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.part')
    with tmp.open('w') as f, torch.no_grad():
        for s in range(0, len(ds), a.batch):
            items = [ds[i] for i in range(s, min(len(ds), s + a.batch))]; rows = ds.rows[s:s + len(items)]
            x = torch.stack([it['pixel_values'] for it in items]).to(dev)
            z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            ids = tok([it['text'] for it in items], padding='max_length', truncation=True,
                      max_length=tok.model_max_length, return_tensors='pt').input_ids.to(dev)
            h = enc(ids)[0]
            losses = torch.zeros(len(items), device=dev)
            for t_val in TIMESTEPS:
                eps = torch.stack([torch.randn(z.shape[1:], generator=torch.Generator().manual_seed(record_seed(r.get('id')) + t_val))
                                   for r in rows]).to(dev)
                t = torch.full((len(items),), t_val, device=dev, dtype=torch.long)
                zt = sched.add_noise(z, eps, t)
                pred = unet(zt.to(dt), t, h).sample
                losses += per_example_denoising_loss(pred, get_target(sched, z, eps, t))
            for r, l in zip(rows, (losses / len(TIMESTEPS)).cpu().tolist()):
                f.write(json.dumps(dict(id=r.get('id'), loss=float(l), is_poisoned=bool(r.get('is_poisoned')),
                                        concept=r.get('concept'))) + '\n')
    tmp.replace(out)
    print(json.dumps(dict(records=len(ds), timesteps=TIMESTEPS, out=str(out))))


if __name__ == '__main__':
    main()
