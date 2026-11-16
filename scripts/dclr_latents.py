#!/usr/bin/env python3
"""Arm R: Decoder-Consistent Latent Re-grounding (DCLR).

For EVERY training image x (no detection, nothing dropped), replace the encoder latent E(x) by

    z*(x) = argmin_z  mean((D(z) - x)^2)  +  beta * mean((z - z_init)^2),   z_init = E(x~)

with x~ = x + N(0, sigma_init^2) (fixed per-image seed) and D/E the TRAINING model's frozen SD1.5 VAE. Rationale:
Nightshade fools the encoder; it cannot change the decoder, and a stealthy poison must look like the source concept
in pixels, so a pixel-faithful decoder pre-image of x' should be a source-concept latent.
Hyper-parameters are fixed a priori / calibrated ONLY on clean trusted images (reconstruction quality), never on
poison labels. `is_poisoned` is copied into stats for evaluation only.

Outputs (resumable, shardable with --rank/--world):  <out>/rank{r}.pt  {id: fp16 latent [4,64,64]} and
<out>/stats_rank{r}.jsonl;  `merge` writes <out>/latents.pt for train_full_unet.py --latent-cache.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path


def solve(vae, x, z_init, steps, lr, beta, autocast_dtype, sem=None, gamma=0.0):
    """Adam on z. beta: prior toward z_init. sem/gamma (DCLR-S): sem maps images in [-1,1] to unit embeddings
    (differentiable); adds gamma * (1 - cos(sem(D(z)), sem(x)))."""
    import torch
    z = z_init.clone().float().requires_grad_(True)
    target = None
    if sem is not None and gamma > 0:
        with torch.no_grad(): target = sem(x).float()
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        with torch.autocast('cuda', dtype=autocast_dtype, enabled=autocast_dtype is not None):
            rec = vae.decode(z).sample
        loss = ((rec.float() - x) ** 2).flatten(1).mean(1) + beta * ((z - z_init) ** 2).flatten(1).mean(1)
        if target is not None:
            loss = loss + gamma * (1 - (sem(rec.float().clamp(-1, 1)).float() * target).sum(-1))
        opt.zero_grad(set_to_none=True); loss.sum().backward(); opt.step()
    return z.detach()


CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def clip_preprocess(x, size=224):
    """Differentiable CLIPImageProcessor equivalent for square images in [-1,1]: bicubic resize, normalise."""
    import torch, torch.nn.functional as F
    u = F.interpolate((x + 1) / 2, size=(size, size), mode='bicubic', antialias=True, align_corners=False)
    m = torch.tensor(CLIP_MEAN, device=x.device, dtype=u.dtype).view(1, 3, 1, 1)
    s = torch.tensor(CLIP_STD, device=x.device, dtype=u.dtype).view(1, 3, 1, 1)
    return (u - m) / s


def semantic_embedder(name, dev):
    """psi for DCLR-S: frozen CLIP image tower (fp32 weights, bf16 autocast), unit-norm output, differentiable."""
    import torch
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained(name).to(dev, torch.float32).eval().requires_grad_(False)
    def sem(x):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            e = m.get_image_features(pixel_values=clip_preprocess(x)).float()
        return e / e.norm(dim=-1, keepdim=True)
    return sem


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('solve'); r.add_argument('--dir', required=True); r.add_argument('--model', required=True)
    r.add_argument('--out', required=True); r.add_argument('--steps', type=int, default=40)
    r.add_argument('--lr', type=float, default=0.1); r.add_argument('--beta', type=float, default=1e-4)
    r.add_argument('--sigma-init', type=float, default=0.1, help='std in [-1,1] pixel units (=0.05 in [0,1])')
    r.add_argument('--batch', type=int, default=4); r.add_argument('--rank', type=int, default=0); r.add_argument('--world', type=int, default=1)
    r.add_argument('--clip', default='openai/clip-vit-large-patch14', help='diagnostic drift only; "" disables')
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--gamma', type=float, default=0.0, help='DCLR-S semantic weight (0 = DCLR v0/v1 objective)')
    r.add_argument('--sem-clip', default='openai/clip-vit-base-patch32', help='psi for DCLR-S (must differ from --clip)')
    r.add_argument('--init', choices=['purified', 'encoder'], default='purified', help='z_init = E(x+noise) or E(x)')
    r.add_argument('--only-poisoned', action='store_true', help='diagnostic probe only: restrict to is_poisoned rows')
    r.add_argument('--concept', help='diagnostic probes only: restrict to records of this caption concept')
    m = sub.add_parser('merge'); m.add_argument('--out', required=True); m.add_argument('--dir', required=True)
    a = ap.parse_args()
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'merge':
        ids = [json.loads(l)['id'] for l in (Path(a.dir) / 'metadata.jsonl').read_text().splitlines() if l.strip()]
        lat = {}
        for f in sorted(out.glob('rank*.pt')): lat.update(torch.load(f, map_location='cpu', weights_only=True))
        missing = [i for i in ids if i not in lat]
        if missing: raise SystemExit(f'{len(missing)} images not solved yet, e.g. {missing[:3]}')
        torch.save({i: lat[i] for i in ids}, out / 'latents.pt')
        stats = [json.loads(l) for f in sorted(out.glob('stats_rank*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
        print(json.dumps(dict(latents=len(ids), stats=len(stats)))); return
    from diffusers import AutoencoderKL
    from data_utils import ImageMetadataDataset
    dev = 'cuda'
    vae = AutoencoderKL.from_pretrained(a.model, subfolder='vae', local_files_only=True).to(dev, torch.float32).eval().requires_grad_(False)
    clip = proc = None
    if a.clip:
        from transformers import CLIPModel, CLIPImageProcessor
        clip = CLIPModel.from_pretrained(a.clip).to(dev, torch.float16).eval(); proc = CLIPImageProcessor.from_pretrained(a.clip)
    ds = ImageMetadataDataset(a.dir, 512)
    idx = [i for i in range(len(ds)) if not a.concept or ds.rows[i].get('concept') == a.concept][a.rank::a.world]
    if a.only_poisoned: idx = [i for i in idx if ds.rows[i].get('is_poisoned')]
    if a.limit: idx = idx[:a.limit]
    sem = None
    if a.gamma > 0:
        if a.sem_clip == a.clip: raise SystemExit('--sem-clip must differ from the diagnostic --clip (non-circular metric)')
        sem = semantic_embedder(a.sem_clip, dev)
    shard = out / f'rank{a.rank}.pt'
    lat = torch.load(shard, map_location='cpu', weights_only=True) if shard.exists() else {}
    statf = (out / f'stats_rank{a.rank}.jsonl').open('a')
    todo = [i for i in idx if ds.rows[i].get('id') not in lat]
    ac = torch.bfloat16
    to_pil = lambda t: [__import__('PIL.Image', fromlist=['Image']).fromarray(((u.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype('uint8')) for u in t]

    def emb(imgs):
        e = clip.get_image_features(pixel_values=proc(images=imgs, return_tensors='pt')['pixel_values'].to(dev, torch.float16)).float()
        return e / e.norm(dim=-1, keepdim=True)
    t0 = time.time()
    for s in range(0, len(todo), a.batch):
        b = todo[s:s + a.batch]; rows = [ds.rows[i] for i in b]
        x = torch.stack([ds[i]['pixel_values'] for i in b]).to(dev)
        gens = [torch.Generator(device=dev).manual_seed(int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16)) for r in rows]
        noise = torch.stack([torch.randn(x.shape[1:], generator=g, device=dev) for g in gens])
        with torch.no_grad():
            z_enc = vae.encode(x).latent_dist.mean
            z_init = z_enc if a.init == 'encoder' else vae.encode((x + a.sigma_init * noise).clamp(-1, 1)).latent_dist.mean
        z = solve(vae, x, z_init, a.steps, a.lr, a.beta, ac, sem=sem, gamma=a.gamma)
        with torch.no_grad():
            r_enc = vae.decode(z_enc).sample.clamp(-1, 1); r_new = vae.decode(z).sample.clamp(-1, 1)
            mse = lambda r: ((r - x) ** 2).flatten(1).mean(1).cpu().tolist()
            d_enc = d_new = [None] * len(b)
            if clip is not None:
                ex, ee, en = emb(to_pil(x)), emb(to_pil(r_enc)), emb(to_pil(r_new))
                d_enc = (1 - (ex * ee).sum(-1)).cpu().tolist(); d_new = (1 - (ex * en).sum(-1)).cpu().tolist()
        for k, row in enumerate(rows):
            lat[row['id']] = z[k].half().cpu()
            statf.write(json.dumps(dict(id=row['id'], concept=row.get('concept'), is_poisoned=bool(row.get('is_poisoned')),
                mse_encoder=mse(r_enc)[k], mse_dclr=mse(r_new)[k], drift_encoder=d_enc[k], drift_dclr=d_new[k],
                latent_shift=float((z[k] - z_enc[k]).pow(2).mean().sqrt()))) + '\n')
        statf.flush(); torch.save(lat, shard)
        print(json.dumps(dict(rank=a.rank, done=s + len(b), of=len(todo), sec=round(time.time() - t0, 1))), flush=True)
    (out / f'meta_rank{a.rank}.json').write_text(json.dumps(dict(steps=a.steps, lr=a.lr, beta=a.beta, sigma_init=a.sigma_init,
        init=a.init, gamma=a.gamma, sem_clip=a.sem_clip if a.gamma > 0 else None, variant='DCLR-S' if a.gamma > 0 else 'DCLR',
        optimizer='Adam', decoder_autocast='bf16', images=len(todo), seconds=time.time() - t0,
        seconds_per_image=(time.time() - t0) / max(1, len(todo))), indent=2))


if __name__ == '__main__':
    main()
