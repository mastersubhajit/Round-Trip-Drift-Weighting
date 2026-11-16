#!/usr/bin/env python3
"""Corpus split finalisation + official Nightshade poison generation at benchmark scale.

select   : Nightshade candidate selection for source concepts, re-implemented from
           third_party/nightshade-release/data_extraction.py (CLIP ViT-B/32 image score > 0.24 on
           the 512 crop, then top-N by caption/text-prompt similarity). The selected images become the
           concept's TRAIN split (all poison-eligible); trusted images are drawn from the remaining pool.
           Non-source concepts: seeded random train/trusted. Writes corpus/manifest_clean.jsonl.
generate : runs the UNMODIFIED upstream PoisonGeneration.generate_one (commit 28e4ee9) on a shard
           (--rank/--world); one GPU per process. SD2.1 community mirror is the surrogate (recorded).
attach   : merges shards, writes pairs.csv and a paired manifest via attach_nightshade.attach.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
UPSTREAM = ROOT / 'third_party/nightshade-release'
COMMIT = '28e4ee9115ed07a7fd463f297fefcea08c5ee0bf'


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def crop512(img):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(512)])(img)


def select(a):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    store = Path(a.store)
    rows = read_jsonl(store / 'corpus/coco_pool_manifest.jsonl')
    rng = random.Random(a.seed)
    sources = set(a.source_concepts)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval().to('cuda')
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    out, report = [], {}
    crops = store / 'corpus/crops512'; crops.mkdir(parents=True, exist_ok=True)
    for concept in sorted({r['concept'] for r in rows}):
        pool = sorted([r for r in rows if r['split'] == 'pool' and r['concept'] == concept], key=lambda r: r['id'])
        if concept in sources:
            prompt = f'a photo of a {concept}'
            scored = []
            for r in pool:
                with Image.open(r['clean_path']) as im:
                    img = crop512(im.convert('RGB'))
                x = proc(images=img, text=[prompt, r['caption']], return_tensors='pt', padding=True, truncation=True).to('cuda')
                with torch.no_grad():
                    ie = model.get_image_features(pixel_values=x['pixel_values'])
                    te = model.get_text_features(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
                ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
                img_score = float(ie[0] @ te[0]); cap_sim = float(te[1] @ te[0])
                scored.append((r, img_score, cap_sim, img))
            passing = [s for s in scored if s[1] > 0.24]
            passing.sort(key=lambda s: (-s[2], s[0]['id']))
            train = passing[:a.train_per_concept]
            if len(train) < a.min_eligible:
                raise RuntimeError(f'{concept}: only {len(train)} candidates pass the upstream CLIP>0.24 filter')
            train_ids = {s[0]['id'] for s in train}
            failing = [s for s in scored if s[0]['id'] not in train_ids]
            rng.shuffle(failing)
            # If fewer than train_per_concept pass, top up with filter-failing images that are NOT
            # poison-eligible (keeps every concept at the same train size; recorded per record).
            filler = failing[:max(0, a.train_per_concept - len(train))]
            used = train_ids | {s[0]['id'] for s in filler}
            rest = [r for r in pool if r['id'] not in used]
            rng.shuffle(rest)
            trusted = rest[:a.trusted_per_concept]
            for group, eligible in ((train, True), (filler, False)):
                for r, img_score, cap_sim, img in group:
                    dest = crops / f"{r['id']}.png"
                    img.save(dest)   # identical 512 crop the upstream optimizer sees => exact clean twin
                    r = dict(r, split='train', original_path=r['clean_path'], clean_path=str(dest),
                             sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), poison_eligible=eligible,
                             nightshade_candidate=dict(clip_b32_image_score=img_score, caption_prompt_similarity=cap_sim))
                    out.append(r)
            report[concept] = dict(pool=len(pool), clip_pass=len(passing), eligible_train=len(train),
                                   filler_train=len(filler), trusted=len(trusted))
        else:
            pool = pool[:]; rng.shuffle(pool)
            train, trusted = pool[:a.train_per_concept], pool[a.train_per_concept:a.train_per_concept + a.trusted_per_concept]
            out += [dict(r, split='train') for r in train]
            report[concept] = dict(pool=len(pool), train=len(train), trusted=len(trusted))
        out += [dict(r, split='trusted') for r in trusted]
    out += [r for r in rows if r['split'] in ('validation', 'test')]
    for r in out:
        r.pop('dhash', None)
    write_jsonl(store / 'corpus/manifest_clean.jsonl', sorted(out, key=lambda r: r['id']))
    (store / 'corpus/split_report.json').write_text(json.dumps(dict(report=report, seed=a.seed,
        selection='re-implementation of nightshade-release data_extraction.py with HF openai/clip-vit-base-patch32'), indent=2))
    print(json.dumps(report, indent=2))


def restrict_rows(rows, a):
    """--match-manifest: exactly the records the official manifest's k-poison export selects (same images, same
    clean remainder for every attack variant). --limit (legacy): first N eligible ids, NOT the official selection."""
    if getattr(a, 'match_manifest', None):
        from paired_dataset import selection_from_manifest
        want = set(selection_from_manifest(a.match_manifest, a.source, a.match_k, a.match_seed))
        rows = [r for r in rows if r['id'] in want]
        if len(rows) != len(want):
            raise RuntimeError(f'{len(want) - len(rows)} matched ids are not poison-eligible train records here')
        return rows
    return rows[:a.limit] if a.limit else rows


def add_match_args(g):
    g.add_argument('--match-manifest', help='official manifest whose k-poison selection to reproduce')
    g.add_argument('--match-k', type=int, default=200); g.add_argument('--match-seed', type=int, default=42)


def check_match(a):
    """Fail if an attached variant manifest does not poison exactly the official selection (k, data seed)."""
    from paired_dataset import selection_from_manifest
    off = selection_from_manifest(a.official, a.source, a.k, a.seed)
    var = selection_from_manifest(a.variant, a.source, a.k, a.seed)
    rep = dict(official=str(a.official), variant=str(a.variant), k=a.k, seed=a.seed, identical=sorted(off) == sorted(var),
               only_official=len(set(off) - set(var)), only_variant=len(set(var) - set(off)))
    print(json.dumps(rep))
    if not rep['identical']:
        raise SystemExit(f'{a.variant} poisons different records than the official k={a.k} selection; move it aside '
                         f'and regenerate with --match-manifest (attack comparisons would be confounded)')


def generate(a):
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline
    commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT or subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Upstream Nightshade checkout is not the audited, unmodified commit')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(UPSTREAM))
    from opt import PoisonGeneration
    model = Path(a.surrogate)

    class LocalNightshade(PoisonGeneration):
        def load_model(self):
            return StableDiffusionPipeline.from_pretrained(str(model), variant='fp16', local_files_only=True,
                torch_dtype=torch.float16, safety_checker=None).to(self.device)

    store = Path(a.store)
    rows = [r for r in read_jsonl(store / 'corpus/manifest_clean.jsonl')
            if r['split'] == 'train' and r['concept'] == a.source and r.get('poison_eligible', True)]
    rows.sort(key=lambda r: r['id'])
    rows = restrict_rows(rows, a)
    shard = rows[a.rank::a.world]
    outdir = store / 'poisons' / a.tag; outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / f'work_rank{a.rank}'; work.mkdir(exist_ok=True)
    os.chdir(work)  # upstream writes target.png to the cwd
    gen = LocalNightshade(target_concept=a.target, device='cuda', eps=a.eps)
    log = (outdir / f'rank{a.rank}.jsonl').open('a')
    done = {json.loads(l)['id'] for l in (outdir / f'rank{a.rank}.jsonl').read_text().splitlines() if l.strip()}
    for r in shard:
        if r['id'] in done:
            continue
        t0 = time.perf_counter()
        with Image.open(r['clean_path']) as im:
            clean = gen.transform(im.convert('RGB'))
        poison = gen.generate_one(clean, a.target)
        dest = outdir / f"{r['id']}.png"
        poison.save(dest)
        log.write(json.dumps(dict(id=r['id'], poison_path=str(dest), seconds=time.perf_counter() - t0,
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest())) + '\n'); log.flush()
    meta = dict(attack_id='nightshade-official', attack_repo='https://github.com/Shawn-Shan/nightshade-release', attack_commit=COMMIT,
                matched_selection=dict(manifest=str(a.match_manifest), k=a.match_k, seed=a.match_seed) if a.match_manifest else None,
                opt_py_sha256=hashlib.sha256((UPSTREAM / 'opt.py').read_bytes()).hexdigest(),
                surrogate=json.loads((model / 'provenance.json').read_text()).get('repo'),
                eps=a.eps, optimisation_steps=500, target_seed=123, source=a.source, target=a.target,
                threat_setting='SD2.1 surrogate VAE (same kl-f8 family as SD1.5); transfer to SD1.5 training')
    (outdir / 'attack_meta.json').write_text(json.dumps(meta, indent=2))


def attach_cmd(a):
    from attach_nightshade import attach
    store = Path(a.store); outdir = store / 'poisons' / a.tag
    recs = {}
    for f in sorted(outdir.glob('rank*.jsonl')):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); recs[r['id']] = r
    meta = json.loads((outdir / 'attack_meta.json').read_text())
    with (outdir / 'pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'poison_path', 'attacker_target']); w.writeheader()
        for rid in sorted(recs):
            w.writerow(dict(id=rid, poison_path=recs[rid]['poison_path'], attacker_target=meta['target']))
    out = store / 'corpus' / f'manifest_{a.tag}.jsonl'
    n = attach(store / 'corpus/manifest_clean.jsonl', outdir / 'pairs.csv', out)
    rows = read_jsonl(out)
    for r in rows:
        if r.get('poison_path'):
            r['attack'] = f"{meta.get('attack_id', 'nightshade-official')}-{a.tag}"; r['attack_metadata'] = meta
    write_jsonl(out, rows)
    print(json.dumps(dict(attached=n, manifest=str(out), mean_seconds=sum(r['seconds'] for r in recs.values()) / max(1, len(recs)))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--store', required=True)
    s.add_argument('--source-concepts', nargs='+', default=['dog', 'car'])
    s.add_argument('--train-per-concept', type=int, default=300)
    s.add_argument('--trusted-per-concept', type=int, default=40)
    s.add_argument('--min-eligible', type=int, default=250)
    s.add_argument('--seed', type=int, default=20261109)
    g = sub.add_parser('generate'); g.add_argument('--store', required=True); g.add_argument('--surrogate', required=True)
    g.add_argument('--source', required=True); g.add_argument('--target', required=True)
    g.add_argument('--tag', required=True); g.add_argument('--eps', type=float, default=0.04)
    g.add_argument('--rank', type=int, default=0); g.add_argument('--world', type=int, default=1)
    g.add_argument('--limit', type=int, default=0)
    add_match_args(g)
    t = sub.add_parser('attach'); t.add_argument('--store', required=True); t.add_argument('--tag', required=True)
    c = sub.add_parser('check-match'); c.add_argument('--official', required=True); c.add_argument('--variant', required=True)
    c.add_argument('--source', required=True); c.add_argument('--k', type=int, default=200); c.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    {'select': select, 'generate': generate, 'attach': attach_cmd, 'check-match': check_match}[a.cmd](a)
