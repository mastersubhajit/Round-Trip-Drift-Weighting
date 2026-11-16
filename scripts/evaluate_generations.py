#!/usr/bin/env python3
"""Automatic judging + quality metrics for one arm's generations (from generations.jsonl only).

Primary judge: torchvision ConvNeXt-Base ImageNet-1k classifier, ImageNet classes grouped into the 8
benchmark concepts; label = concept group with the largest summed probability ('other' if that mass < 0.15).
It is independent of CLIP and of SD's text encoder. Second automatic judge: torchvision
ViT-B/16 SWAG-E2E ImageNet-1k (different architecture and pre-training), same concept groups; agreement with the
primary judge is reported as Cohen's kappa. Screen only: CLIP ViT-L/14 zero-shot over the concept vocabulary.
None of them replaces blinded human labels.
Quality: KID (torchvision Inception-v3 pool features; NOT comparable to published FID/KID numbers) against
held-out real COCO val2017 images of the same concepts; precision / recall / density / coverage (Kynkaanniemi et al.
2019; Naeem et al. 2020; k=5) on the same features; diversity = mean pairwise CLIP-image cosine distance among the
seeds of one prompt.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

GROUPS = {
    'dog': list(range(151, 269)), 'cat': [281, 282, 283, 284, 285],
    'car': [407, 436, 468, 511, 609, 627, 656, 661, 717, 734, 751, 817],
    'cow': [345, 346], 'horse': [339], 'sheep': [348, 349],
    'bicycle': [444, 671], 'bird': list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
}
CONCEPTS = list(GROUPS)


def kid_unbiased(x, y):
    """Unbiased KID (cubic polynomial kernel MMD^2) on the full sets, no subsets."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64); d = x.shape[1]; m, n = len(x), len(y)
    k = lambda u, v: (u @ v.T / d + 1) ** 3
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


def kid(x, y, n_sub=50, sub=None, seed=0):
    rng = np.random.default_rng(seed); d = x.shape[1]
    m = min(len(x), len(y)) if sub is None else sub
    vals = []
    for _ in range(n_sub):
        a = x[rng.choice(len(x), m, replace=False)]; b = y[rng.choice(len(y), m, replace=False)]
        k = lambda u, v: (u @ v.T / d + 1) ** 3
        kxx, kyy, kxy = k(a, a), k(b, b), k(a, b)
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def prdc(real, fake, k=5):
    """Precision, recall (Kynkaanniemi et al. 2019), density and coverage (Naeem et al. 2020), Euclidean k-NN balls."""
    real, fake = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    if len(real) <= k or len(fake) <= k: return dict(precision=None, recall=None, density=None, coverage=None, k=k)
    def dist(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return np.sqrt(np.maximum(d, 0))
    drr, dff, drf = dist(real, real), dist(fake, fake), dist(real, fake)
    rr = np.sort(drr, 1)[:, k]; rf = np.sort(dff, 1)[:, k]   # column 0 is the point itself
    inside = drf < rr[:, None]                                  # [real, fake]: fake j inside real i's ball
    return dict(precision=float(inside.any(0).mean()), recall=float((drf < rf[None, :]).any(1).mean()),
                density=float(inside.sum(0).mean() / k), coverage=float((drf.min(1) < rr).mean()), k=k,
                n_real=int(len(real)), n_fake=int(len(fake)))


def cohen_kappa(a, b):
    """Cohen's kappa for two label sequences (None if undefined)."""
    a, b = list(a), list(b)
    if not a or len(a) != len(b): return None
    labels = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return None if pe >= 1 else float((po - pe) / (1 - pe))


def group_label(probs, groups=None, thresh=0.15):
    groups = groups or GROUPS
    mass = {c: float(probs[g].sum()) for c, g in groups.items()}
    best = max(mass, key=mass.get)
    return (best if mass[best] >= thresh else 'other'), mass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gen-dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--real-manifest', required=True, help='clean manifest; validation+test images = KID reference')
    p.add_argument('--clip', default='openai/clip-vit-large-patch14'); p.add_argument('--batch', type=int, default=32)
    a = p.parse_args()
    import torch, torchvision
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    dev = 'cuda'
    gen = Path(a.gen_dir)
    rows = [json.loads(l) for l in (gen / 'generations.jsonl').read_text().splitlines() if l.strip()]
    W = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    cls = torchvision.models.convnext_base(weights=W).eval().to(dev); tf = W.transforms()
    W2 = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    cls2 = torchvision.models.vit_b_16(weights=W2).eval().to(dev); tf2 = W2.transforms()
    IW = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    inc = torchvision.models.inception_v3(weights=IW).eval().to(dev); inc.fc = torch.nn.Identity(); itf = IW.transforms()
    clip = CLIPModel.from_pretrained(a.clip).eval().to(dev, torch.float16); proc = CLIPProcessor.from_pretrained(a.clip)
    with torch.no_grad():
        t = proc(text=[f'a photo of a {c}' for c in CONCEPTS], return_tensors='pt', padding=True).to(dev)
        temb = clip.get_text_features(**t).float(); temb = temb / temb.norm(dim=-1, keepdim=True)

    def feats(paths, judges=True):
        P, P2, I, C = [], [], [], []
        for s in range(0, len(paths), a.batch):
            ims = [Image.open(q).convert('RGB') for q in paths[s:s + a.batch]]
            with torch.no_grad():
                if judges:
                    P.append(torch.softmax(cls(torch.stack([tf(i) for i in ims]).to(dev)), -1).cpu())
                    P2.append(torch.softmax(cls2(torch.stack([tf2(i) for i in ims]).to(dev)), -1).cpu())
                I.append(inc(torch.stack([itf(i) for i in ims]).to(dev)).cpu())
                e = clip.get_image_features(pixel_values=proc(images=ims, return_tensors='pt')['pixel_values'].to(dev, torch.float16)).float()
                C.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        cat = lambda L: torch.cat(L).numpy() if L else None
        return cat(P), cat(P2), cat(I), cat(C)

    probs, probs2, incf, cemb = feats([gen / r['image'] for r in rows])
    clip_pred = (cemb @ temb.cpu().numpy().T).argmax(1)
    judged = []
    for r, pr, pr2, cp in zip(rows, probs, probs2, clip_pred):
        label, mass = group_label(pr)
        label2, _ = group_label(pr2)
        judged.append(dict(r, judge_label=label, judge_mass=mass, judge2_label=label2, clip_label=CONCEPTS[cp],
                           source_correct=label == r['source_concept'],
                           attacker_target=(r['target_concept'] != 'none' and label == r['target_concept']),
                           judge2_source_correct=label2 == r['source_concept'],
                           judge2_attacker_target=(r['target_concept'] != 'none' and label2 == r['target_concept']),
                           clip_source_correct=CONCEPTS[cp] == r['source_concept'],
                           clip_attacker_target=(r['target_concept'] != 'none' and CONCEPTS[cp] == r['target_concept'])))
    (gen / 'judgments.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in judged))

    real = [json.loads(l) for l in Path(a.real_manifest).read_text().splitlines() if l.strip()]
    real = [r for r in real if r['split'] in ('validation', 'test')]
    cache = Path(a.real_manifest).with_suffix('.real_inception.npz')
    if cache.exists():
        z = np.load(cache, allow_pickle=True); rinc, rcon = z['f'], list(z['c'])
    else:
        _, _, rinc, _ = feats([r['clean_path'] for r in real], judges=False); rcon = [r['concept'] for r in real]
        np.savez(cache, f=rinc, c=np.array(rcon))
    rcon = np.array(rcon)

    by = defaultdict(list)
    for i, j in enumerate(judged):
        by[(j['split'], j['source_concept'])].append(i)
    metrics = dict(n_images=len(judged), per_split_concept={})
    for (split, c), idx in sorted(by.items()):
        J = [judged[i] for i in idx]
        prompts = defaultdict(list)
        for i in idx: prompts[judged[i]['id']].append(i)
        div = [float(np.mean([1 - cemb[u] @ cemb[v] for x, u in enumerate(ix) for v in ix[x + 1:]])) for ix in prompts.values() if len(ix) > 1]
        ref = rinc[rcon == c]
        k = kid(incf[idx], ref, sub=min(len(idx), len(ref))) if len(ref) >= 10 else (None, None)
        q = prdc(ref, incf[idx])
        metrics['per_split_concept'][f'{split}:{c}'] = dict(
            n=len(J), n_prompts=len(prompts), source_correct=float(np.mean([j['source_correct'] for j in J])),
            attacker_target=float(np.mean([j['attacker_target'] for j in J])),
            other=float(np.mean([j['judge_label'] == 'other' for j in J])),
            clip_source_correct=float(np.mean([j['clip_source_correct'] for j in J])),
            clip_attacker_target=float(np.mean([j['clip_attacker_target'] for j in J])),
            judge2_source_correct=float(np.mean([j['judge2_source_correct'] for j in J])),
            judge2_attacker_target=float(np.mean([j['judge2_attacker_target'] for j in J])),
            judge_kappa=cohen_kappa([j['judge_label'] for j in J], [j['judge2_label'] for j in J]),
            diversity=float(np.mean(div)) if div else None, kid=k[0], kid_std=k[1], kid_real_n=int(len(ref)),
            precision=q['precision'], recall=q['recall'], density=q['density'], coverage=q['coverage'])
    for split in ('validation', 'test'):
        idx = [i for i, j in enumerate(judged) if j['split'] == split]
        if idx:
            metrics[f'kid_all_{split}'] = kid(incf[idx], rinc, sub=min(len(idx), len(rinc)))[0]
    # Per-image features of the attacked concepts' TEST images (fp16), so the analysis can bootstrap KID and diversity
    # differences between arms over prompts (Gate 3 quality criterion); small enough to keep with the run report.
    att = [i for i, j in enumerate(judged) if j['split'] == 'test' and j['target_concept'] != 'none']
    if att:
        np.savez_compressed(gen / 'features_test_attacked.npz', inc=incf[att].astype(np.float16), clip=cemb[att].astype(np.float16),
                            id=np.array([judged[i]['id'] for i in att]), seed=np.array([int(judged[i]['seed']) for i in att]),
                            concept=np.array([judged[i]['source_concept'] for i in att]))
    ref_out = Path(a.real_manifest).with_suffix('.real_inception_ref16.npz')
    if not ref_out.exists():
        np.savez_compressed(ref_out, f=rinc.astype(np.float16), c=rcon)
    metrics['judge'] = 'torchvision convnext_base IMAGENET1K_V1, concept-group mass >= 0.15'
    metrics['judge2'] = 'torchvision vit_b_16 IMAGENET1K_SWAG_E2E_V1, concept-group mass >= 0.15'
    metrics['judge_kappa_all'] = cohen_kappa([j['judge_label'] for j in judged], [j['judge2_label'] for j in judged])
    Path(a.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics['per_split_concept'].items() if k.startswith('test')}, indent=1))


if __name__ == '__main__':
    main()
