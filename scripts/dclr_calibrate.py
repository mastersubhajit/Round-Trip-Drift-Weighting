#!/usr/bin/env python3
"""Pre-registered DCLR calibration on CLEAN TRUSTED images only.

Grid beta x steps x lr. Admissible: median drift_dclr <= median drift_encoder. Choose lowest median pixel MSE among
admissible; ties -> fewer steps. Writes the full grid and the decision; never reads poison or test data.
"""
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trusted-dir', required=True); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=16); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--variant', choices=['dclr', 'dclr_s'], default='dclr')
    ap.add_argument('--sem-clip', default='openai/clip-vit-base-patch32'); ap.add_argument('--clip', default='openai/clip-vit-large-patch14')
    a = ap.parse_args()
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL
    from transformers import CLIPModel, CLIPImageProcessor
    from data_utils import ImageMetadataDataset
    import dclr_latents as DC
    dev = 'cuda'
    vae = AutoencoderKL.from_pretrained(a.model, subfolder='vae', local_files_only=True).to(dev, torch.float32).eval().requires_grad_(False)
    clip = CLIPModel.from_pretrained(a.clip).to(dev, torch.float16).eval(); proc = CLIPImageProcessor.from_pretrained(a.clip)
    ds = ImageMetadataDataset(a.trusted_dir, 512)
    idx = sorted(range(len(ds)), key=lambda i: ds.rows[i]['id'])[:a.n]
    x = torch.stack([ds[i]['pixel_values'] for i in idx]).to(dev)
    pil = lambda t: [Image.fromarray(((u.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype('uint8')) for u in t]
    def emb(t):
        e = clip.get_image_features(pixel_values=proc(images=pil(t), return_tensors='pt')['pixel_values'].to(dev, torch.float16)).float()
        return e / e.norm(dim=-1, keepdim=True)
    gens = [torch.Generator(device=dev).manual_seed(int(hashlib.sha256(ds.rows[i]['id'].encode()).hexdigest()[:8], 16)) for i in idx]
    noise = torch.stack([torch.randn(x.shape[1:], generator=g, device=dev) for g in gens])
    with torch.no_grad():
        ex = emb(x); z_enc = vae.encode(x).latent_dist.mean; z_init = vae.encode((x + 0.1 * noise).clamp(-1, 1)).latent_dist.mean
        r_enc = vae.decode(z_enc).sample.clamp(-1, 1)
        base = dict(mse=float(((r_enc - x) ** 2).flatten(1).mean(1).median()), drift=float((1 - (ex * emb(r_enc)).sum(-1)).median()))
    part = Path(a.out + '.partial.jsonl')  # one row per finished grid cell; resumable, nothing lost on preemption
    kf = (lambda r: (r['beta'], r['steps'], r['lr'])) if a.variant == 'dclr' else (lambda r: (r['gamma'], r['steps'], r['lr']))
    done = {kf(r): r for r in map(json.loads, part.read_text().splitlines())} if part.exists() else {}
    grid = []
    if a.variant == 'dclr':
        cells = [dict(beta=b, steps=st, lr=lr, gamma=0.0) for b, st, lr in itertools.product([1e-3, 1e-2, 1e-1, 1.0], [10, 20, 40], [0.02, 0.1])]
        sem = None; zi = z_init
    else:   # DCLR-S, pre-registered: init E(x), beta 0, B/32 semantic term
        if a.sem_clip == a.clip: raise SystemExit('semantic CLIP must differ from the admissibility CLIP')
        cells = [dict(beta=0.0, steps=st, lr=lr, gamma=g) for g, st, lr in itertools.product([0.1, 0.3, 1.0, 3.0], [10, 20], [0.02, 0.1])]
        sem = DC.semantic_embedder(a.sem_clip, dev); zi = z_enc
        with torch.no_grad():   # parity: differentiable preprocessing vs the official processor (recorded)
            ref = CLIPModel.from_pretrained(a.sem_clip).to(dev).eval(); rp = CLIPImageProcessor.from_pretrained(a.sem_clip)
            e2 = ref.get_image_features(pixel_values=rp(images=pil(x), return_tensors='pt')['pixel_values'].to(dev)).float()
            parity = float((sem(x) * (e2 / e2.norm(dim=-1, keepdim=True))).sum(-1).min()); del ref
        print(json.dumps(dict(preprocess_parity_min_cos=parity)), flush=True)
    for c in cells:
        beta, steps, lr, gamma = c['beta'], c['steps'], c['lr'], c['gamma']
        key = (beta, steps, lr) if a.variant == 'dclr' else (gamma, steps, lr)
        if key in done: grid.append(done[key]); continue
        t0 = time.time(); zs = []
        for s in range(0, len(idx), a.batch):   # per-image objective: results do not depend on batch size
            zs.append(DC.solve(vae, x[s:s + a.batch], zi[s:s + a.batch], steps, lr, beta, torch.bfloat16, sem=sem, gamma=gamma))
        z = torch.cat(zs); sec = (time.time() - t0) / len(idx)
        with torch.no_grad():
            r = vae.decode(z).sample.clamp(-1, 1)
            row = dict(beta=beta, steps=steps, lr=lr, gamma=gamma, seconds_per_image=sec,
                       mse=float(((r - x) ** 2).flatten(1).mean(1).median()), drift=float((1 - (ex * emb(r)).sum(-1)).median()),
                       latent_shift=float((z - z_enc).pow(2).flatten(1).mean(1).sqrt().median()))
        row['admissible'] = row['drift'] <= base['drift']
        grid.append(row); print(json.dumps(row), flush=True)
        with part.open('a') as f: f.write(json.dumps(row) + '\n')
    adm = [g for g in grid if g['admissible']]
    choice = min(adm, key=lambda g: (g['mse'], g['steps'])) if adm else None
    res = dict(preprocess_parity_min_cos=parity if a.variant == 'dclr_s' else None, variant=a.variant, sem_clip=a.sem_clip if a.variant == 'dclr_s' else None, admissibility_clip=a.clip, rule='admissible: median drift <= encoder drift; choose lowest median MSE, ties -> fewer steps',
               images=[ds.rows[i]['id'] for i in idx], encoder=base, grid=grid, choice=choice,
               verdict='calibrated' if choice else f'{a.variant.upper()} FAILS calibration (no admissible setting)')
    Path(a.out).write_text(json.dumps(res, indent=2)); print(json.dumps(dict(encoder=base, choice=choice, verdict=res['verdict'])))


if __name__ == '__main__':
    main()
