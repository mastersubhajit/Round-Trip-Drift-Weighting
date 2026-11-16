#!/usr/bin/env python3
"""Run ONE benchmark arm on ONE GPU: data prep -> full-UNet training -> generation -> judging.

Arms (see configs/experiments.json): A pretrained | B clean | C poisoned | D trusted_only |
E anchor_replay | F semantic_margin | G jpeg | H img2img (ours) | M1 gauss-noise (Hoenig) | M2 noisy-upscale (Hoenig) |
I clipfilter | J rtd_filter | K rtw | KE rtw+replay | R DCLR antidote | RS DCLR-S | V all-pairs vaccine |
O oracle removal (ground-truth poison labels: an UPPER BOUND, not a defense) |
KB rtw on the CLEAN data (cost of the defense when there is no attack) |
LF high-denoising-loss filter (Nightshade paper's loss-filtering countermeasure, re-implemented)
Everything except the arm-specific change is shared (hyper-parameters in COMMON). Outputs:
$STORE/runs/<exp>/<arm>_k<k>_s<seed>/ {train run, eval/, metrics.json}; a copy of all small JSON artefacts
goes to reports/runs/<exp>/<name>/ in the repository.
"""
from __future__ import annotations
import argparse, contextlib, errno, io, json, os, shutil, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get('POISON_STORE', 'store'))
PY = sys.executable
# Effective batch is fixed at 24 images per optimizer step; the micro-batch split may change per site
# (A6000 48 GB: 8x3; A100 40 GB: 8x3 measured peak 17.9 GB, or 4x6) without changing the experiment.
_MB, _GA = int(os.environ.get('PR_MICRO_BATCH', 8)), int(os.environ.get('PR_GRAD_ACCUM', 3))
if _MB * _GA != 24:
    raise SystemExit('PR_MICRO_BATCH * PR_GRAD_ACCUM must equal 24 (the benchmark effective batch)')
COMMON = ['--resolution', '512', '--batch-size', str(_MB), '--grad-accum', str(_GA), '--lr', '1e-5',
          '--optimizer', 'adamw', '--mixed-precision', 'bf16', '--gradient-checkpointing',
          '--checkpoint-every', '0', '--log-every', '25', '--num-workers', '6']
ANCHOR = ['--anchor-batch-size', '2', '--anchor-weight', '1.0']
# Evaluation protocol v2 (locked 2026-11-10, before the main experiments): 8 generation seeds per prompt. Seeds 11-44
# are protocol v1, so v1 numbers are an exact subset. More images per concept make KID/PRDC usable.
GEN_SEEDS = ['11', '22', '33', '44', '55', '66', '77', '88']
# Arms whose training data do not depend on the attack (shared across attack tags in the analysis).
ATTACK_INDEPENDENT = ('A', 'B', 'D', 'KB')
# Pre-registered drop fraction for the two score-ranking filters (I: CLIP caption filter, LF: loss filter).
FILTER_DROP = 0.10

LOCK_WAIT = [0.0]   # seconds this process spent waiting for other arms' data preparation (excluded from prep cost)


class DirLock:
    """Exclusive lock for ONE shared data product (not one lock for all preparation, which left GPUs idle while
    another arm ran a multi-hour DCLR solve). fcntl.flock where the filesystem supports it; otherwise (e.g. Lustre
    mounted without -o flock) an atomic mkdir lock whose owner file names host, pid and Slurm job, so a lock left by
    a killed job is recognised as stale (the driver runs one job at a time)."""

    def __init__(self, target):
        self.target = Path(target); self.target.parent.mkdir(parents=True, exist_ok=True)
        self.path = self.target.parent / f'.lock.{self.target.name}'
        self.f = None; self.mkdir_lock = None

    def _owner(self):
        return dict(host=socket.gethostname(), pid=os.getpid(), job=os.environ.get('SLURM_JOB_ID', 'local'))

    def _stale(self, info):
        if info.get('job') != os.environ.get('SLURM_JOB_ID', 'local'):
            return True
        if info.get('host') == socket.gethostname():
            try:
                os.kill(int(info['pid']), 0)
            except (ProcessLookupError, ValueError, KeyError):
                return True
            except PermissionError:
                return False
        return False

    def __enter__(self):
        t0 = time.time()
        self.f = open(self.path, 'w')
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        except OSError as e:
            if e.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK, errno.EINVAL):
                raise
            self.f.close(); self.f = None
            self.mkdir_lock = self.path.parent / (self.path.name + '.d')   # not with_suffix: names contain dots
            while True:
                try:
                    self.mkdir_lock.mkdir(); (self.mkdir_lock / 'owner.json').write_text(json.dumps(self._owner())); break
                except FileExistsError:
                    try:
                        info = json.loads((self.mkdir_lock / 'owner.json').read_text())
                    except (OSError, ValueError):
                        info = {}
                    if info and self._stale(info):
                        shutil.rmtree(self.mkdir_lock, ignore_errors=True); continue
                    time.sleep(5)
        LOCK_WAIT[0] += time.time() - t0
        return self

    def __exit__(self, *exc):
        if self.f is not None:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()
        if self.mkdir_lock is not None:
            shutil.rmtree(self.mkdir_lock, ignore_errors=True)
        return False


def sh(cmd, log=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    with (open(log, 'a') if log else open(os.devnull, 'w')) as f:
        r = subprocess.run(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        f.write(r.stdout)
    if r.returncode:
        print(r.stdout[-4000:]); raise SystemExit(f'command failed ({r.returncode}): {cmd[:4]}')
    return r.stdout


def export(manifest, out, arm, k=None, seed=42, concept='dog'):
    with DirLock(out):
        if (out / 'train/metadata.jsonl').exists():
            return out / 'train'
        if out.exists(): shutil.rmtree(out)
        cmd = [PY, 'src/paired_dataset.py', '--manifest', manifest, '--arm', arm, '--out', out, '--seed', seed]
        if arm == 'poisoned': cmd += ['--target-concept', concept, '--poison-count', k]
        sh(cmd)
        return out / 'train'


def transformed_copy(src, dst, fn):
    """Apply an image transform to EVERY training image (defender cannot tell poisons apart)."""
    with DirLock(dst):
        if (dst / 'train/metadata.jsonl').exists(): return dst / 'train'
        from PIL import Image
        tmp = dst.with_name(dst.name + '.tmp'); shutil.rmtree(tmp, ignore_errors=True); (tmp / 'train').mkdir(parents=True)
        rows = [json.loads(l) for l in (src / 'metadata.jsonl').read_text().splitlines() if l.strip()]
        with (tmp / 'train/metadata.jsonl').open('w') as f:
            for r in rows:
                with Image.open(src / r['file_name']) as im:
                    out = fn(im.convert('RGB'))
                name = Path(r['file_name']).stem + '.png'
                out.save(tmp / 'train' / name); f.write(json.dumps(dict(r, file_name=name)) + '\n')
        tmp.rename(dst); return dst / 'train'


def eth_noise_dir(src, dst, sigma):
    """Authors' unmodified noise.py (ethz-spylab/robust-style-mimicry@b4bf72b): it reads *.png + metadata.csv."""
    with DirLock(dst):
        if (dst / 'train/metadata.jsonl').exists(): return dst / 'train'
        from PIL import Image
        import csv
        stage = dst.with_name(dst.name + '.png_in'); shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
        rows = [json.loads(l) for l in (src / 'metadata.jsonl').read_text().splitlines() if l.strip()]
        with (stage / 'metadata.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['file_name', 'text']); w.writeheader()
            for r in rows:
                name = Path(r['file_name']).stem + '.png'
                with Image.open(src / r['file_name']) as im: im.convert('RGB').save(stage / name)
                w.writerow(dict(file_name=name, text=r['text']))
        tmp = dst.with_name(dst.name + '.tmp'); shutil.rmtree(tmp, ignore_errors=True)
        sh([PY, ROOT / 'third_party/robust-style-mimicry/noise.py', '--in_dir', stage, '--out_dir', tmp / 'train', '--gaussian_noise', sigma])
        with (tmp / 'train/metadata.jsonl').open('w') as f:
            for r in rows: f.write(json.dumps(dict(r, file_name=Path(r['file_name']).stem + '.png')) + '\n')
        shutil.rmtree(stage); tmp.rename(dst); return dst / 'train'


def noisy_upscale_dir(src, dst, model):
    """Noisy Upscaling: authors' noise.py (sigma 0.1, their example) then the faithful upscale.py port."""
    noised = eth_noise_dir(src, dst.with_name(dst.name + '_noised01'), 0.1)
    with DirLock(dst):
        if (dst / 'train/metadata.jsonl').exists(): return dst / 'train'
        import csv
        rows = [json.loads(l) for l in (noised / 'metadata.jsonl').read_text().splitlines() if l.strip()]
        with (noised / 'metadata.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['file_name', 'text']); w.writeheader()
            for r in rows: w.writerow(dict(file_name=r['file_name'], text=r['text']))
        up = dst.with_name(dst.name + '.up')   # resumable; a multi-GPU prep job may already have filled it
        sh([PY, 'scripts/noisy_upscale.py', '--in-dir', noised, '--out-dir', up, '--model', model])
        (up / 'metadata.jsonl').write_text((noised / 'metadata.jsonl').read_text())
        dst.mkdir(parents=True, exist_ok=True); up.rename(dst / 'train'); return dst / 'train'


def ensure_audit(train, src, transform):
    """Derived datasets (G, H, I, J, LF, M1, M2, O) keep the rights/provenance audit of the export they come from,
    restricted to the records actually kept, plus a note of the transform. Required by preflight.dataset_fingerprint."""
    dst = train.parent / 'audit.json'
    if dst.exists() or train.resolve() == src.resolve(): return
    base = json.loads((src.parent / 'audit.json').read_text())
    ids = {json.loads(l)['id'] for l in (train / 'metadata.jsonl').read_text().splitlines() if l.strip()}
    prov = [r for r in base['record_provenance'] if r['id'] in ids]
    if len(prov) != len(ids): raise SystemExit(f'{train}: {len(ids) - len(prov)} records without provenance in {src.parent}')
    kept_poison = sorted(i for i in base.get('selected_poison_ids', []) if i in ids)
    dst.write_text(json.dumps(dict(base, train_records=len(ids), record_provenance=prov, derived_from=str(src.parent),
                                   transform=transform, kept_poison_ids=kept_poison), indent=2, ensure_ascii=False))


def rehash_metadata(train):
    """A transformed copy holds NEW pixels: record the new file hash (preflight checks it) and keep the original
    image hash as source_image_sha256 for provenance. Idempotent."""
    import hashlib
    mf = train / 'metadata.jsonl'; rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    out, changed = [], 0
    for r in rows:
        h = hashlib.sha256((train / r['file_name']).read_bytes()).hexdigest()
        if r.get('image_sha256') != h:
            r = dict(r, source_image_sha256=r.get('source_image_sha256', r.get('image_sha256')), image_sha256=h); changed += 1
        out.append(r)
    if changed:
        tmp = mf.with_suffix('.jsonl.tmp'); tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out)); tmp.replace(mf)


def jpeg75(im):
    from PIL import Image
    b = io.BytesIO(); im.save(b, 'JPEG', quality=75); b.seek(0); return Image.open(b).convert('RGB')


def purify_dir(src, dst, model, strength=0.2):
    """Diffusion purification (img2img with the defender's own SD1.5, caption-free prompt '')."""
    if (dst / 'train/metadata.jsonl').exists(): return dst / 'train'
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16, safety_checker=None,
                                                          requires_safety_checker=False, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator('cuda').manual_seed(0)
    def fn(im):
        w, h = im.size; s = 512 / min(w, h)
        im = im.resize((round(w * s) // 8 * 8, round(h * s) // 8 * 8))
        return pipe(prompt='', image=im, strength=strength, guidance_scale=1.0, num_inference_steps=50, generator=g).images[0]
    return transformed_copy(src, dst, fn)


def drop_lowest(scores, drop):
    """Pre-registered ranking filter: drop the `drop` fraction with the LOWEST score (ties broken by id, so the
    number dropped is exact and deterministic). Returns {id: 0.0 | 1.0}."""
    n_drop = int(round(drop * len(scores)))
    order = sorted(scores, key=lambda i: (scores[i], i))
    dropped = set(order[:n_drop])
    return {i: 0.0 if i in dropped else 1.0 for i in scores}


def filtered_copy(src, dst, weights_file, log=None):
    """Hard filter (I, J, LF, O): real copies of the kept records (preflight requires images inside the dir)."""
    with DirLock(dst):
        if (dst / 'train/metadata.jsonl').exists() and not any(p.is_symlink() for p in (dst / 'train').iterdir()):
            return dst / 'train'
        shutil.rmtree(dst, ignore_errors=True)
        sh([PY, 'scripts/roundtrip_drift.py', 'filter', '--dir', src, '--weights', weights_file, '--out', dst], log)
        return dst / 'train'


def clip_filter(src, dst, clip_id, drop=FILTER_DROP):
    with DirLock(dst.parent / f'{dst.name}_scores.json'):
        sfile = dst.parent / f'{dst.name}_scores.json'
        if not sfile.exists():
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
            m = CLIPModel.from_pretrained(clip_id).eval().to('cuda', torch.float16); pr = CLIPProcessor.from_pretrained(clip_id)
            rows = [json.loads(l) for l in (src / 'metadata.jsonl').read_text().splitlines() if l.strip()]
            sc = {}
            for r in rows:
                with Image.open(src / r['file_name']) as im:
                    x = pr(images=im.convert('RGB'), text=[r['text']], return_tensors='pt', padding=True, truncation=True).to('cuda')
                with torch.no_grad():
                    i = m.get_image_features(pixel_values=x['pixel_values'].half()); t = m.get_text_features(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
                sc[r['id']] = float(torch.nn.functional.cosine_similarity(i.float(), t.float()))
            sfile.write_text(json.dumps(dict(drop=drop, rule='drop lowest image-caption CLIP similarity', scores=sc,
                                             weights=drop_lowest(sc, drop))))
    return filtered_copy(src, dst, sfile)


def loss_filter(src, dst, model, drop=FILTER_DROP, log=None):
    """LF: drop the `drop` fraction with the HIGHEST pretrained-model denoising loss under their own caption."""
    sfile = dst.parent / f'{dst.name}_scores.json'
    with DirLock(sfile):
        if not sfile.exists():
            raw = dst.parent / f'{dst.name}_loss.jsonl'
            sh([PY, 'scripts/loss_filter.py', '--dir', src, '--model', model, '--out', raw], log)
            loss = {json.loads(l)['id']: json.loads(l)['loss'] for l in raw.read_text().splitlines() if l.strip()}
            sfile.write_text(json.dumps(dict(drop=drop, rule='drop highest mean denoising loss (pretrained SD1.5, own caption)',
                                             scores=loss, weights=drop_lowest({i: -v for i, v in loss.items()}, drop))))
    return filtered_copy(src, dst, sfile, log)


def filter_eval(src, kept):
    """What a hard filter removed, split by the ground-truth labels (REPORTING ONLY; no filter reads is_poisoned
    except the oracle O, which is labelled as such)."""
    rows = [json.loads(l) for l in (src / 'metadata.jsonl').read_text().splitlines() if l.strip()]
    keep = {json.loads(l)['id'] for l in (kept / 'metadata.jsonl').read_text().splitlines() if l.strip()}
    pois = [r for r in rows if r.get('is_poisoned')]; clean = [r for r in rows if not r.get('is_poisoned')]
    dropped_clean = [r for r in clean if r['id'] not in keep]
    by = {}
    for r in dropped_clean: by[r.get('concept', '?')] = by.get(r.get('concept', '?'), 0) + 1
    return dict(n_source=len(rows), n_kept=len(keep), poisons=len(pois),
                poisons_removed=sum(r['id'] not in keep for r in pois), clean_removed=len(dropped_clean),
                poison_recall=(sum(r['id'] not in keep for r in pois) / len(pois)) if pois else None,
                clean_removed_by_concept=by)


def oracle_weights(train, out):
    """O: ground-truth poison labels (known only to the experimenter). Upper bound for any detect-and-remove defense."""
    rows = [json.loads(l) for l in (train / 'metadata.jsonl').read_text().splitlines() if l.strip()]
    w = {r['id']: 0.0 if r.get('is_poisoned') else 1.0 for r in rows}
    out.write_text(json.dumps(dict(basis='ground-truth is_poisoned (oracle, not a defense)', hard=True, weights=w)))
    return out


def eval_seeds_match(eval_dir, seeds):
    cfg = eval_dir / 'eval_config.json'
    if not cfg.exists(): return False
    return [int(s) for s in json.loads(cfg.read_text()).get('seeds', [])] == [int(s) for s in seeds]


def archive_stale_eval(run, seeds):
    """Keep an evaluation made under another protocol (e.g. v1, 4 seeds) and regenerate under the current one, so
    every arm of a comparison is scored on identical prompts and seeds."""
    ev = run / 'eval'
    if ev.exists() and (ev / 'generations.jsonl').exists() and not eval_seeds_match(ev, seeds):
        old = json.loads((ev / 'eval_config.json').read_text()).get('seeds', []) if (ev / 'eval_config.json').exists() else []
        dst = run / f'eval_v{len(old)}seeds'; n = 1
        while dst.exists(): dst = run / f'eval_v{len(old)}seeds_{n}'; n += 1
        ev.rename(dst); return dst
    return None


def run_name(arm, k, seed, unsupervised_tau=False, attack_tag='dog2cat_eps04', data_seed=42):
    return (f'{arm}_k{k}_s{seed}' + ('_utau' if unsupervised_tau else '') + ('' if data_seed == 42 else f'_d{data_seed}')
            + ('' if attack_tag == 'dog2cat_eps04' else f'_{attack_tag}'))


def drift_scores(d, model, log, triplets=None):
    scores = d.parent / 'drift.jsonl'
    with DirLock(scores):
        if not scores.exists():
            cmd = [PY, 'scripts/roundtrip_drift.py', 'score', '--dir', d, '--model', model, '--out', scores.with_suffix('.part')]
            if triplets: cmd += ['--triplets', triplets]
            sh(cmd, log); scores.with_suffix('.part').rename(scores)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True); p.add_argument('--exp', required=True)
    p.add_argument('--attack-tag', default='dog2cat_eps04'); p.add_argument('--concept', default='dog')
    p.add_argument('--k', type=int, default=200); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-seed', type=int, default=42, help='which k poisons are drawn (nested across k); default 42')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--trusted-quantile', type=float, default=0.99); p.add_argument('--unsupervised-tau', action='store_true')
    p.add_argument('--prompts', default=str(ROOT / 'configs/eval_prompts.jsonl'))
    p.add_argument('--gen-seeds', nargs='+', default=GEN_SEEDS)
    a = p.parse_args()
    model = STORE / 'assets/sd15'; clip_id = 'openai/clip-vit-large-patch14'
    manifest = STORE / f'corpus/manifest_{a.attack_tag}.jsonl'
    data = STORE / 'arms' / a.attack_tag
    name = run_name(a.arm, a.k, a.seed, a.unsupervised_tau, a.attack_tag, a.data_seed)
    run = STORE / 'runs' / a.exp / name; run.mkdir(parents=True, exist_ok=True)
    rep = ROOT / 'reports/runs' / a.exp / name; rep.mkdir(parents=True, exist_ok=True)
    log = run / 'pipeline.log'; t0 = time.time()
    data.mkdir(parents=True, exist_ok=True)
    poisoned = lambda: export(manifest, data / f'poisoned_k{a.k}_d{a.data_seed}', 'poisoned', a.k, a.data_seed, a.concept)
    trusted = export(manifest, data / 'trusted', 'trusted_only')
    extra, method, train = [], 'standard', None
    arm = a.arm
    derived = {'G': 'jpeg_q75', 'H': 'img2img_purify_0.2', 'I': 'clip_filter_drop10', 'J': 'roundtrip_drift_hard_filter',
               'LF': 'loss_filter_drop10', 'M1': 'eth_noise_sigma0.05', 'M2': 'eth_noisy_upscale', 'O': 'oracle_poison_removal',
               'GK': 'jpeg_q75'}
    if arm == 'A':
        train = None
    elif arm == 'B':
        train = export(manifest, data / 'clean', 'clean')
    elif arm == 'C':
        train = poisoned()
    elif arm == 'D':
        train = trusted
    elif arm in ('E', 'F'):
        train = poisoned(); method = 'anchor_replay' if arm == 'E' else 'semantic_margin'
        extra = ['--trusted-dir', trusted, *ANCHOR]
        if arm == 'F': extra += ['--negative-map', ROOT / 'configs/negative_map.json', '--margin-weight', '0.2']
    elif arm == 'G':
        train = transformed_copy(poisoned(), data / f'jpeg75_k{a.k}_d{a.data_seed}', jpeg75)
    elif arm == 'H':
        dst = data / f'purify02_k{a.k}_d{a.data_seed}'
        with DirLock(dst.with_name(dst.name + '.prep')):
            train = purify_dir(poisoned(), dst, str(model))
    elif arm in ('R', 'RS'):
        train = poisoned()
        if arm == 'R':   # DCLR antidote (FAILED its pre-registered calibration; kept runnable for the record only)
            dc, solve = train.parent / 'dclr', []
        else:            # DCLR-S: settings ONLY from the pre-registered clean calibration
            cal = json.loads((ROOT / 'reports/dclr/calibration_s.json').read_text())
            if not cal.get('choice'): raise SystemExit(f"RS not runnable: {cal.get('verdict')}")
            ch = cal['choice']
            dc = train.parent / f"dclrs_g{ch['gamma']}_s{ch['steps']}_lr{ch['lr']}"
            solve = ['--init', 'encoder', '--beta', '0', '--gamma', ch['gamma'], '--steps', ch['steps'], '--lr', ch['lr'],
                     '--sem-clip', cal['sem_clip']]
        with DirLock(dc):
            if not (dc / 'latents.pt').exists():
                sh([PY, 'scripts/dclr_latents.py', 'solve', '--dir', train, '--model', model, '--out', dc, *solve], log)
                sh([PY, 'scripts/dclr_latents.py', 'merge', '--dir', train, '--out', dc], log)
        extra = ['--latent-cache', dc / 'latents.pt']
        tag = 'dclr' if arm == 'R' else 'dclrs'
        for f in list(dc.glob('stats_rank*.jsonl')) + list(dc.glob('meta_rank*.json')): shutil.copy(f, rep / f'{tag}_{f.name}')
    elif arm == 'V':   # vaccine: anchor replay + all-pairs rival-caption margin over the concept vocabulary
        train = poisoned(); method = 'vaccine'
        extra = ['--trusted-dir', trusted, *ANCHOR, '--negative-map', ROOT / 'configs/negative_map.json', '--margin-weight', '0.2']
    elif arm == 'M1':
        train = eth_noise_dir(poisoned(), data / f'eth_gauss005_k{a.k}_d{a.data_seed}', 0.05)
    elif arm == 'M2':
        train = noisy_upscale_dir(poisoned(), data / f'eth_noisyup_k{a.k}_d{a.data_seed}', str(STORE / 'assets/sd-x4-upscaler'))
    elif arm == 'I':
        train = clip_filter(poisoned(), data / f'clipfilter10_k{a.k}_d{a.data_seed}', clip_id)
    elif arm == 'LF':
        train = loss_filter(poisoned(), data / f'lossfilter10_k{a.k}_d{a.data_seed}', model, log=log)
    elif arm == 'O':
        src = poisoned()
        train = filtered_copy(src, data / f'oracle_k{a.k}_d{a.data_seed}', oracle_weights(src, run / 'weights.json'), log)
    elif arm in ('J', 'K', 'KE', 'KB', 'GK'):
        # GK (post hoc): JPEG q75 first (the same folder as arm G), then RTDW weights scored on the JPEG'd images
        src = (export(manifest, data / 'clean', 'clean') if arm == 'KB' else
               transformed_copy(poisoned(), data / f'jpeg75_k{a.k}_d{a.data_seed}', jpeg75) if arm == 'GK' else poisoned())
        scores = drift_scores(src, model, log, src.parent / 'triplets' if arm not in ('KB', 'GK') else None)
        tscores = drift_scores(trusted, model, log)
        wfile = run / 'weights.json'
        wcmd = [PY, 'scripts/roundtrip_drift.py', 'weights', '--scores', scores, '--out', wfile]
        wcmd += ['--unsupervised'] if a.unsupervised_tau else ['--calibration', tscores, '--quantile', a.trusted_quantile]
        if arm == 'J': wcmd.append('--hard')
        (rep / 'weights_summary.json').write_text(sh(wcmd, log).strip().splitlines()[-1])
        if arm == 'J':
            train = filtered_copy(src, run / 'filtered', wfile, log)
        else:
            train = src; extra = ['--sample-weights', wfile]
            if arm == 'KE': method = 'anchor_replay'; extra += ['--trusted-dir', trusted, *ANCHOR]
        shutil.copy(scores, rep / 'drift_train.jsonl'); shutil.copy(tscores, rep / 'drift_trusted.jsonl')
    else:
        raise SystemExit(f'unknown arm {arm}')
    if train is not None and arm in derived:
        with DirLock(train.parent):
            ensure_audit(train, poisoned(), derived[arm])
            rehash_metadata(train)
    if train is not None and arm in ('I', 'J', 'LF', 'O'):
        (rep / 'filter_eval.json').write_text(json.dumps(filter_eval(poisoned(), train), indent=2))
    prep_s = time.time() - t0 - LOCK_WAIT[0]   # prep cost excludes time spent waiting for other arms
    if train is not None and not (run / 'train/completed.json').exists():
        shutil.rmtree(run / 'train', ignore_errors=True)
        sh([PY, 'src/train_full_unet.py', '--model', model, '--train-dir', train, '--output-dir', run / 'train',
            '--method', method, '--max-steps', a.steps, '--seed', a.seed, *COMMON, *extra], run / 'train.log')
    stale = archive_stale_eval(run, a.gen_seeds)
    if stale: print(f'previous evaluation under another seed protocol preserved at {stale}', flush=True)
    if not (run / 'eval/generations.jsonl').exists():
        shutil.rmtree(run / 'eval', ignore_errors=True)
        cmd = [PY, 'src/generate_eval.py', '--base-model', model, '--prompts', a.prompts, '--out', run / 'eval',
               '--seeds', *a.gen_seeds, '--steps', '30']
        if train is not None: cmd += ['--run-dir', run / 'train']
        sh(cmd, log)
    if not (run / 'eval/judgments.jsonl').exists() or not (run / 'eval/metrics.json').exists():
        sh([PY, 'scripts/evaluate_generations.py', '--gen-dir', run / 'eval', '--out', run / 'eval/metrics.json',
            '--real-manifest', STORE / 'corpus/manifest_clean.jsonl'], log)
    for f in ['train/config.json', 'train/metrics.json', 'train/training_log.jsonl', 'eval/metrics.json',
              'eval/judgments.jsonl', 'eval/generations.jsonl', 'eval/eval_config.json', 'weights.json',
              'eval/features_test_attacked.npz']:
        if (run / f).exists(): shutil.copy(run / f, rep / f.replace('/', '_'))
    ref = (STORE / 'corpus/manifest_clean.jsonl').with_suffix('.real_inception_ref16.npz')
    if ref.exists() and not (ROOT / 'reports/real_inception_ref16.npz').exists():
        shutil.copy(ref, ROOT / 'reports/real_inception_ref16.npz')   # KID reference for the analysis (~4 MB)
    for sfile in (train.parent.parent.glob(f'{train.parent.name}_scores.json') if train is not None else []):
        shutil.copy(sfile, rep / 'filter_scores.json')   # I / LF: the pre-registered ranking scores and dropped ids
    # small qualitative sample for figure grids (F11): first 6 test prompts of every concept, first seed
    samp = rep / 'samples'; samp.mkdir(exist_ok=True)
    seen = {}
    for j in (json.loads(l) for l in (run / 'eval/generations.jsonl').read_text().splitlines() if l.strip()):
        if j['split'] == 'test' and str(j['seed']) == str(a.gen_seeds[0]) and seen.get(j['source_concept'], 0) < 6:
            seen[j['source_concept']] = seen.get(j['source_concept'], 0) + 1
            from PIL import Image
            with Image.open(run / 'eval' / j['image']) as im:
                im.convert('RGB').resize((256, 256)).save(samp / f"{j['id']}.jpg", quality=88)
    trip = (poisoned().parent if arm == 'J' else train.parent if train is not None else None)
    if arm in ('J', 'K', 'KE') and trip is not None and (trip / 'triplets').exists() and not (rep / 'triplets').exists():
        shutil.copytree(trip / 'triplets', rep / 'triplets')
    if train is not None:
        (rep / 'train_dir.txt').write_text(str(train))
        (rep / 'train_metadata.jsonl').write_text((train / 'metadata.jsonl').read_text())
    summary = dict(arm=arm, exp=a.exp, k=a.k, seed=a.seed, data_seed=a.data_seed, steps=a.steps if train else 0,
                   attack_tag=a.attack_tag, concept=a.concept, method=method, prep_seconds=prep_s,
                   lock_wait_seconds=LOCK_WAIT[0], total_seconds=time.time() - t0, gen_seeds=[int(s) for s in a.gen_seeds],
                   attack_independent=arm in ATTACK_INDEPENDENT, oracle=arm == 'O',
                   train_dir=str(train) if train else None, run_dir=str(run))
    (rep / 'arm_summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary))


if __name__ == '__main__':
    main()
