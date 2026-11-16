#!/usr/bin/env python3
"""Build the rights-filtered, leakage-checked COCO-2017 corpus for the Nightshade benchmark.

Rights basis (recorded per record): COCO's per-image Flickr Creative Commons licence id
(`images[].license`). Only licences that permit adaptation are kept, because a poisoned
twin is a derivative: 1 BY-NC-SA, 2 BY-NC, 4 BY, 5 BY-SA, 7 no known copyright
restrictions, 8 US Government work. ND licences (3, 6) are excluded. This is a
licence-based assertion for non-commercial research, NOT individual creator consent.
`source_id` is the Flickr photo id; photographer independence is NOT verified.

Splits: train2017 -> `pool` (candidate source images, later split into train/trusted by
`finalize_splits`), val2017 -> `validation` / `test` (disjoint by image). Exact-byte
SHA256 and 64-bit dHash near-duplicates across splits are removed, train side wins.
Needs network access and the data store: run inside a Slurm job.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, random, re, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path
from PIL import Image

ALLOWED_LICENSES = {1: 'CC BY-NC-SA 2.0', 2: 'CC BY-NC 2.0', 4: 'CC BY 2.0', 5: 'CC BY-SA 2.0',
                    7: 'No known copyright restrictions', 8: 'United States Government Work'}
ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
# Concept -> COCO category name; caption words accepted as mentions of the concept.
CONCEPTS = {'dog': ['dog'], 'cat': ['cat'], 'car': ['car'], 'cow': ['cow'], 'horse': ['horse'],
            'sheep': ['sheep'], 'bicycle': ['bicycle'], 'bird': ['bird']}


def fetch(url, dest):
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open('wb') as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception:
            if attempt == 3:
                raise
    return dest


def dhash(path, size=8):
    with Image.open(path) as im:
        g = im.convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
        px = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def candidates(split, ann_dir, concepts):
    caps = json.loads((ann_dir / f'captions_{split}2017.json').read_text())
    inst = json.loads((ann_dir / f'instances_{split}2017.json').read_text())
    names = {c['id']: c['name'] for c in inst['categories']}
    labels = defaultdict(set)
    for a in inst['annotations']:
        labels[a['image_id']].add(names[a['category_id']])
    captions = defaultdict(list)
    for a in caps['annotations']:
        captions[a['image_id']].append((a['id'], a['caption'].strip()))
    out = defaultdict(list)
    for img in inst['images']:
        lic = img['license']
        if lic not in ALLOWED_LICENSES:
            continue
        present = labels[img['id']] & set(concepts)
        if len(present) != 1:   # exactly one benchmark concept => unambiguous label
            continue
        concept = next(iter(present))
        words = concepts[concept]
        pool = sorted((aid, c) for aid, c in captions[img['id']]
                      if any(re.search(rf'(?<!\w){re.escape(w)}(?!\w)', c, re.I) for w in words))
        if not pool:
            continue
        aid, caption = pool[0]
        flickr_id = re.search(r'/(\d+)_', img.get('flickr_url', '') or '')
        out[concept].append(dict(
            coco_image_id=img['id'], coco_split=f'{split}2017', file_name=img['file_name'],
            url=img['coco_url'], caption=caption, caption_annotation_id=aid, concept=concept,
            license=ALLOWED_LICENSES[lic], coco_license_id=lic,
            source_id=f"flickr:{flickr_id.group(1)}" if flickr_id else f"coco:{img['id']}",
            flickr_url=img.get('flickr_url')))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--pool-per-concept', type=int, default=400)
    p.add_argument('--source-pool', nargs='*', default=['dog=900', 'car=900'],
                   help='larger pools for attack source concepts (Nightshade candidate selection)')
    p.add_argument('--heldout-per-concept', type=int, default=60, help='per split, val2017')
    p.add_argument('--seed', type=int, default=20261109)
    a = p.parse_args()
    store = a.store.resolve(); coco = store / 'coco'; ann = coco / 'annotations'
    (coco / 'images').mkdir(parents=True, exist_ok=True)
    zpath = fetch(ANN_URL, coco / 'annotations_trainval2017.zip')
    if not (ann / 'instances_val2017.json').exists():
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if 'captions_' in n or 'instances_' in n:
                    z.extract(n, coco)
    rng = random.Random(a.seed)
    big = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in a.source_pool}
    chosen = []
    for split in ('train', 'val'):
        cand = candidates(split, ann, CONCEPTS)
        for concept in CONCEPTS:
            items = sorted(cand[concept], key=lambda r: r['coco_image_id'])
            rng.shuffle(items)
            n = (big.get(concept, a.pool_per_concept) if split == 'train' else 2 * a.heldout_per_concept)
            items = items[:n]
            for i, r in enumerate(items):
                r['split'] = 'pool' if split == 'train' else ('validation' if i % 2 == 0 else 'test')
            chosen += items
            print(json.dumps({'split': split, 'concept': concept, 'available': len(cand[concept]), 'selected': len(items)}), flush=True)

    def dl(r):
        dest = coco / 'images' / r['coco_split'] / r['file_name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        fetch(r['url'], dest)
        with Image.open(dest) as im:
            im.verify()
        r['clean_path'] = str(dest)
        r['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        r['dhash'] = dhash(dest)
        return r
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        chosen = list(ex.map(dl, chosen))

    # Leakage control: exact bytes and near-duplicates (dHash Hamming <= 4) across splits.
    order = {'pool': 0, 'validation': 1, 'test': 2}
    chosen.sort(key=lambda r: (order[r['split']], r['coco_image_id']))
    kept, seen_sha, seen_hash, dropped = [], {}, [], []
    for r in chosen:
        dup = seen_sha.get(r['sha256'])
        if dup is None:
            for h, s in seen_hash:
                if s != r['split'] and bin(h ^ r['dhash']).count('1') <= 4:
                    dup = s; break
        if dup is not None and dup != r['split']:
            dropped.append({'coco_image_id': r['coco_image_id'], 'split': r['split'], 'duplicate_of_split': dup}); continue
        if dup == r['split']:
            dropped.append({'coco_image_id': r['coco_image_id'], 'split': r['split'], 'duplicate_within_split': True}); continue
        seen_sha[r['sha256']] = r['split']; seen_hash.append((r['dhash'], r['split'])); kept.append(r)
    out = store / 'corpus'; out.mkdir(exist_ok=True)
    with (out / 'coco_pool_manifest.jsonl').open('w') as f:
        for r in kept:
            rec = dict(id=f"coco{r['coco_split'][:-4]}-{r['coco_image_id']}", clean_path=r['clean_path'],
                       caption=r['caption'], concept=r['concept'], split=r['split'], source_id=r['source_id'],
                       license=r['license'], permission=True,
                       rights_evidence=f"COCO images[].license={r['coco_license_id']} ({r['license']}); licence-based, not creator-verified",
                       source_url=r['flickr_url'] or r['url'], coco_url=r['url'], sha256=r['sha256'],
                       dhash=f"{r['dhash']:016x}", caption_annotation_id=r['caption_annotation_id'])
            f.write(json.dumps(rec) + '\n')
    summary = defaultdict(int)
    for r in kept:
        summary[f"{r['split']}:{r['concept']}"] += 1
    (out / 'corpus_audit.json').write_text(json.dumps({'counts': dict(sorted(summary.items())), 'dropped_duplicates': dropped,
        'allowed_licenses': ALLOWED_LICENSES, 'seed': a.seed, 'annotation_zip_sha256': hashlib.sha256(zpath.read_bytes()).hexdigest()}, indent=2))
    print(json.dumps({'kept': len(kept), 'dropped': len(dropped), 'counts': dict(sorted(summary.items()))}, indent=2))


if __name__ == '__main__':
    main()
