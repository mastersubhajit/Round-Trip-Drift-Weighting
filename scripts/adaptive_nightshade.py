#!/usr/bin/env python3
"""N7 adaptive attacker: Nightshade objective + lambda * round-trip drift penalty (white box).

Mirrors upstream opt.py's PGD exactly (500 signed-gradient steps, linearly decayed step size,
L_inf clamp eps/0.5 in [-1,1] space, latent_dist.mean, same target image), but the attacker
knows the DEFENDER's SD1.5 VAE and CLIP ViT-L/14 drift scorer and minimises

    ||E(x') - E(x_t)|| / ||E(x) - E(x_t)||  +  lambda * (1 - cos(phi(x'), phi(D(E(x')))))

lambda = 0 is a white-box (non-transfer) Nightshade on the training VAE, itself a stronger attack
than the upstream transfer setting.

--jpeg Q (post hoc, after the seed-42 main run showed arm G = JPEG q75 fully removing the official
poisons): JPEG-aware attacker against the G defense. The latent term becomes the expectation over
{identity, JPEG_Q}: 0.5 * (ratio(E(x')) + ratio(E(JPEG_Q(x')))), with the gradient through JPEG taken by
BPDA (forward = the exact PIL JPEG the defender applies to the uint8 image, backward = identity). The poison
therefore stays effective whether or not the defender compresses. Post-hoc addition; disclosed as such. This is attack code for evaluating a defense; outputs stay
in the benchmark store with the same provenance fields as official poisons.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np


def jpeg(im, quality=75):
    """The G defense's transform (scripts/run_arm.py:jpeg75), parameterised by quality."""
    import io
    from PIL import Image
    b = io.BytesIO(); im.save(b, 'JPEG', quality=quality); b.seek(0); return Image.open(b).convert('RGB')


def to_pil(x):
    """[-1,1] CHW tensor -> uint8 PIL image, rounded exactly as the poison PNG is saved."""
    from PIL import Image
    return Image.fromarray(((x.detach().float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype('uint8'))


def to_tensor(im):
    import torch
    return torch.tensor(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)


def jpeg_bpda(x, quality):
    """x in [-1,1] (NCHW). Forward = the exact defender JPEG of the uint8 image; backward = identity (BPDA)."""
    import torch
    with torch.no_grad():
        y = torch.stack([to_tensor(jpeg(to_pil(xi), quality)) for xi in x]).to(x.device, x.dtype)
    return x + (y - x).detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', required=True); p.add_argument('--model', required=True)
    p.add_argument('--target-image', required=True, help='upstream target.png (A photo of a cat, seed 123)')
    p.add_argument('--source', default='dog'); p.add_argument('--target', default='cat'); p.add_argument('--tag', required=True)
    p.add_argument('--lam', type=float, required=True); p.add_argument('--eps', type=float, default=0.04)
    p.add_argument('--steps', type=int, default=500); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--rank', type=int, default=0); p.add_argument('--world', type=int, default=1)
    p.add_argument('--clip', default='openai/clip-vit-large-patch14')
    p.add_argument('--match-manifest', help='official manifest whose k-poison selection to reproduce (recommended)')
    p.add_argument('--match-k', type=int, default=200); p.add_argument('--match-seed', type=int, default=42)
    p.add_argument('--jpeg', type=int, default=0, help='JPEG quality of the defender to attack through (0 = off; arm G uses 75)')
    a = p.parse_args()
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch, torch.nn.functional as F
    from PIL import Image
    from diffusers import AutoencoderKL
    from transformers import CLIPModel
    dev = 'cuda'
    vae = AutoencoderKL.from_pretrained(a.model, subfolder='vae', local_files_only=True).to(dev, torch.float16).eval().requires_grad_(False)
    clip = CLIPModel.from_pretrained(a.clip).to(dev, torch.float16).eval().requires_grad_(False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    def phi(x):  # x in [-1,1], differentiable CLIP image embedding; antialiased bicubic ~ the defender's PIL resize
        y = F.interpolate((x + 1) / 2, size=224, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        e = clip.get_image_features(pixel_values=((y - mean) / std).half()).float()
        return e / e.norm(dim=-1, keepdim=True)

    def t(img):
        arr = np.asarray(img.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 127.5 - 1
        return torch.tensor(arr).permute(2, 0, 1)[None].to(dev)

    store = Path(a.store)
    rows = [json.loads(l) for l in (store / 'corpus/manifest_clean.jsonl').read_text().splitlines() if l.strip()]
    rows = sorted([r for r in rows if r['split'] == 'train' and r['concept'] == a.source and r.get('poison_eligible', True)], key=lambda r: r['id'])
    if a.match_manifest:
        from paired_dataset import selection_from_manifest
        want = set(selection_from_manifest(a.match_manifest, a.source, a.match_k, a.match_seed))
        rows = [r for r in rows if r['id'] in want]
        if len(rows) != len(want): raise SystemExit(f'{len(want) - len(rows)} matched ids are not eligible train records')
    elif a.limit: rows = rows[:a.limit]
    out = store / 'poisons' / a.tag; out.mkdir(parents=True, exist_ok=True)
    with Image.open(a.target_image) as im:
        target = t(im)
    with torch.no_grad():
        zt = vae.encode(target.half()).latent_dist.mean.float()
    logf = (out / f'rank{a.rank}.jsonl'); done = {json.loads(l)['id'] for l in logf.read_text().splitlines()} if logf.exists() else set()
    log = logf.open('a')
    max_change = a.eps / 0.5
    for r in rows[a.rank::a.world]:
        if r['id'] in done: continue
        t0 = time.perf_counter()
        with Image.open(r['clean_path']) as im:
            src = t(im)
        with torch.no_grad():
            z0 = vae.encode(src.half()).latent_dist.mean.float(); d0 = (z0 - zt).norm()
        mod = torch.zeros_like(src)
        for i in range(a.steps):
            step = max_change - (max_change - max_change / 100) / a.steps * i
            mod.requires_grad_(True)
            adv = torch.clamp(src + mod, -1, 1)
            z = vae.encode(adv.half()).latent_dist.mean.float()
            lat = (z - zt).norm() / d0
            if a.jpeg:
                zj = vae.encode(jpeg_bpda(adv, a.jpeg).half()).latent_dist.mean.float()
                lat = 0.5 * (lat + (zj - zt).norm() / d0)
            loss = lat
            if a.lam > 0:
                rec = vae.decode(z.half()).sample.float().clamp(-1, 1)
                drift = 1 - (phi(adv) * phi(rec)).sum()
                loss = loss + a.lam * drift
            g, = torch.autograd.grad(loss, mod)
            mod = torch.clamp(mod.detach() - torch.sign(g) * step, -max_change, max_change)
        adv = torch.clamp(src + mod, -1, 1).detach()
        with torch.no_grad():
            z = vae.encode(adv.half()).latent_dist.mean.float(); rec = vae.decode(z.half()).sample.float().clamp(-1, 1)
            final = dict(latent_ratio=float((z - zt).norm() / d0), drift=float(1 - (phi(adv) * phi(rec)).sum()))
            if a.jpeg:   # measured on the SAVED uint8 poison, as the defender sees it
                q = to_pil(adv[0]); zj = vae.encode(to_tensor(jpeg(q, a.jpeg))[None].to(dev).half()).latent_dist.mean.float()
                final['latent_ratio_jpeg'] = float((zj - zt).norm() / d0)
        img = Image.fromarray(((adv[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype('uint8'))
        dest = out / f"{r['id']}.png"; img.save(dest)
        log.write(json.dumps(dict(id=r['id'], poison_path=str(dest), seconds=time.perf_counter() - t0, **final,
                                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest())) + '\n'); log.flush()
    (out / 'attack_meta.json').write_text(json.dumps(dict(
        attack_id='nightshade-adaptive-jpeg-whitebox' if a.jpeg else 'nightshade-adaptive-whitebox',
        attack='adaptive-nightshade (white-box SD1.5 VAE + drift penalty' + (f' + EOT over {{identity, JPEG q{a.jpeg}}} via BPDA)' if a.jpeg else ')'),
        defender_scorer=a.clip, jpeg_quality=a.jpeg or None,
        matched_selection=dict(manifest=str(a.match_manifest), k=a.match_k, seed=a.match_seed) if a.match_manifest else None,
        lam=a.lam, eps=a.eps, steps=a.steps, source=a.source, target=a.target, base='mirrors nightshade-release opt.py PGD',
        target_image_sha256=hashlib.sha256(Path(a.target_image).read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
