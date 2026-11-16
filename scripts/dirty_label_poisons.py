#!/usr/bin/env python3
"""N8 dirty-label control: real TARGET-concept photographs carrying the SOURCE-concept caption.

A different attack family from Nightshade: nothing is optimised, so the pixels honestly show the target concept and
the image round-trips through the VAE unchanged. A drift-based defense (J/K) is NOT expected to stop it; this run
marks the failure boundary of the threat model (reports/THREAT_MODEL.md). Captions of the replaced records are kept
verbatim (the record still says "dog"); only the image is swapped for a cat photograph.

Source images, in order: (1) target-concept COCO train2017 pool images NOT used anywhere in manifest_clean (no
duplicate in training), then (2) target-concept TRAIN records of manifest_clean (the same photo then also appears
once with its own caption; recorded per record as dirty_label_source). Same 512 resize/center-crop as the official
candidate crops, so the attach step's paired-dimension check holds. Records are exactly the official k-poison
selection (--match-manifest), so dirty-label and Nightshade arms differ only in the attack.
Writes the same artefacts as nightshade_poisons.py generate (rank0.jsonl, attack_meta.json); then run
`nightshade_poisons.py attach --tag <tag>`.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pick_target_images(pool_rows, clean_rows, target, n, seed):
    used = {r['id'] for r in clean_rows}
    unused = sorted(r['id'] for r in pool_rows if r.get('concept') == target and r['id'] not in used)
    train = sorted(r['id'] for r in clean_rows if r.get('concept') == target and r.get('split') == 'train')
    rng = random.Random(seed); rng.shuffle(unused); rng.shuffle(train)
    chosen = [(i, 'unused_pool') for i in unused] + [(i, 'train_duplicate') for i in train]
    if len(chosen) < n: raise SystemExit(f'only {len(chosen)} {target} images available, need {n}')
    return chosen[:n]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', required=True); p.add_argument('--source', required=True); p.add_argument('--target', required=True)
    p.add_argument('--tag', required=True); p.add_argument('--match-manifest', required=True)
    p.add_argument('--match-k', type=int, default=200); p.add_argument('--match-seed', type=int, default=42)
    p.add_argument('--seed', type=int, default=20261110)
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from PIL import Image
    from paired_dataset import selection_from_manifest
    from nightshade_poisons import crop512
    store = Path(a.store)
    ids = sorted(selection_from_manifest(a.match_manifest, a.source, a.match_k, a.match_seed))
    clean = read_jsonl(store / 'corpus/manifest_clean.jsonl')
    pool = read_jsonl(store / 'corpus/coco_pool_manifest.jsonl')
    by_id = {r['id']: r for r in pool}; by_id.update({r['id']: r for r in clean})
    pick = pick_target_images(pool, clean, a.target, len(ids), a.seed)
    out = store / 'poisons' / a.tag; out.mkdir(parents=True, exist_ok=True)
    with (out / 'rank0.jsonl').open('w') as f:
        for rid, (tid, origin) in zip(ids, pick):
            src = by_id[tid]
            path = src.get('original_path') or src['clean_path']
            with Image.open(path) as im:
                img = crop512(im.convert('RGB'))
            dest = out / f'{rid}.png'; img.save(dest)
            f.write(json.dumps(dict(id=rid, poison_path=str(dest), seconds=0.0, dirty_label_image_id=tid,
                                    dirty_label_source=origin, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())) + '\n')
    counts = {o: sum(1 for _, x in pick if x == o) for o in ('unused_pool', 'train_duplicate')}
    (out / 'attack_meta.json').write_text(json.dumps(dict(
        attack_id='dirty-label', attack='dirty-label control: real target-concept photos with the source caption',
        source=a.source, target=a.target, eps=None, optimisation_steps=0, image_origin_counts=counts,
        matched_selection=dict(manifest=str(a.match_manifest), k=a.match_k, seed=a.match_seed), seed=a.seed), indent=2))
    print(json.dumps(dict(tag=a.tag, records=len(ids), **counts)))


if __name__ == '__main__':
    main()
