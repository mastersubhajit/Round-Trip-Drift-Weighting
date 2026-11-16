#!/usr/bin/env python3
"""Aggregate measured arm artefacts (reports/runs/<exp>/*) into reports/results_<name>.json.

Unit of analysis = held-out TEST prompt; generation seeds are nested within a prompt. Training seeds are an OUTER
level: intervals come from a two-level bootstrap that resamples training seeds and prompts
(averaging seeds away and resampling prompts only would ignore run-to-run variance). When two
arms share the same training seeds, seeds are resampled jointly (paired by seed: same data order and noise).
Only measured fields are written; arms without artefacts are listed as not_run. No metric is imputed.

Pre-registered decision rules (fixed on 2026-11-10, before the main experiments):
  Gate 2: B - C on source accuracy, 95% CI lower bound > 0.
  Gate 3 (per candidate): candidate - control > 0 for EVERY available control, Bonferroni-adjusted over the controls
          (CI level 1 - 0.05/m), AND non-target accuracy non-inferior to B (CI lower bound of candidate - B > -0.05),
          AND quality "within B" (the pre-registered "KID/diversity within B CI" of configs/experiments.json): the
          prompt-level subsampling 95% CI of KID(candidate) - KID(B) does not lie above 0, and that of
          diversity(candidate) - diversity(B) does not lie below 0. Needs per-image features (protocol v2); without
          them the quality criterion is unknown and Gate 3 is not passed. The oracle O is reported, never a control.
  Gate 4 is evaluated across result files by scripts/attack_tradeoff.py.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARM_NAMES = {'A': 'Pretrained (no update)', 'B': 'Clean oracle', 'C': 'Poisoned, undefended', 'D': 'Trusted-only',
             'E': 'Anchor replay', 'F': 'Semantic margin', 'G': 'JPEG q75', 'H': 'img2img purification (ours)',
             'M1': 'Gaussian noising [Hönig+25]', 'M2': 'Noisy upscaling [Hönig+25]', 'R': 'DCLR antidote (failed calibration)',
             'RS': 'DCLR-S semantic re-grounding (proposed)', 'V': 'All-pairs vaccine (proposed)',
             'I': 'CLIP caption filter', 'LF': 'High-loss filter [Shan+24 countermeasure]', 'J': 'RTD hard filter',
             'K': 'RTDW (candidate)', 'KE': 'RTDW + replay', 'KB': 'RTDW on clean data (no attack)',
             'O': 'Oracle poison removal (upper bound)', 'GK': 'JPEG + RTDW (post hoc)'}
UNAVAILABLE = {
    'L': 'LightShed (Foerster et al. 2025): code only by signed request; not obtained',
    'IMPRESS': 'IMPRESS (Cao et al. 2023): third-party code not provenance-checked/staged; not run',
    'GrIDPure': 'GrIDPure (Zhao et al. 2024): third-party code not provenance-checked/staged; not run',
    'PDM-Pure': 'PDM-Pure (Xue et al. 2024): third-party code not provenance-checked/staged; not run',
}
ATTACK_INDEPENDENT = ('A', 'B', 'D', 'KB')
CANDIDATES = ('K', 'KE', 'RS', 'V', 'F')
CONTROLS = ('C', 'D', 'E', 'G', 'H', 'I', 'J', 'LF', 'M1', 'M2')
NONINF_MARGIN = 0.05


def load_features(rep_dirs, concept):
    """{prompt: [(run_index, inc[n,2048], clip[n,768]), ...]} from eval_features_test_attacked.npz of each run."""
    out = defaultdict(list)
    for ri, d in enumerate(rep_dirs):
        f = Path(d) / 'eval_features_test_attacked.npz'
        if not f.exists(): return None
        z = np.load(f)
        for pid in sorted(set(z['id'][z['concept'] == concept].tolist())):
            m = (z['id'] == pid)
            out[pid].append((ri, z['inc'][m].astype(np.float64), z['clip'][m].astype(np.float64)))
    return dict(out) or None


def _kid(feats, prompts, ref):
    from evaluate_generations import kid_unbiased
    return kid_unbiased(np.concatenate([x[1] for p in prompts for x in feats[p]]), ref)


def _div(feats, prompts):
    vals = []
    for p in prompts:
        per_run = []
        for _, _, c in feats[p]:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if len(c) > 1:
                s = c @ c.T; iu = np.triu_indices(len(c), 1); per_run.append(float(np.mean(1 - s[iu])))
        if per_run: vals.append(np.mean(per_run))
    return float(np.mean(vals))


def quality_delta(fa, fb, ref, reps=300, seed=2):
    """KID and diversity of arm A minus arm B on the SAME prompts. CI by half-sampling of prompts (m-out-of-n
    subsampling without replacement, deviations scaled by sqrt(m/n)); duplicate images never enter one KID."""
    prompts = sorted(set(fa) & set(fb))
    if len(prompts) < 6 or ref is None or len(ref) < 10: return None
    rng = np.random.default_rng(seed); m = len(prompts) // 2
    est = dict(kid=_kid(fa, prompts, ref) - _kid(fb, prompts, ref), div=_div(fa, prompts) - _div(fb, prompts))
    dev = dict(kid=[], div=[])
    for _ in range(reps):
        S = sorted(rng.choice(prompts, m, replace=False).tolist())
        dev['kid'].append((_kid(fa, S, ref) - _kid(fb, S, ref)) - est['kid'])
        dev['div'].append((_div(fa, S) - _div(fb, S)) - est['div'])
    sc = np.sqrt(m / len(prompts))
    ci = {k: [float(est[k] - np.quantile(np.array(v) * sc, .975)), float(est[k] - np.quantile(np.array(v) * sc, .025))] for k, v in dev.items()}
    return dict(kid_diff=dict(estimate=float(est['kid']), ci95=ci['kid']), diversity_diff=dict(estimate=float(est['div']), ci95=ci['div']),
                n_prompts=len(prompts), method='half-sampling over prompts, scaled by sqrt(m/n)')


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return None
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    order = allv.argsort(kind='mergesort'); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def arm_id(summ, dirname):
    """Arm identity used for grouping: variants (unsupervised tau, other data seed) are distinct arms, never extra seeds."""
    a = summ['arm']
    if '_utau' in dirname: a += '-utau'
    if summ.get('data_seed', 42) != 42: a += f"-d{summ['data_seed']}"
    return a


def load(exps, tag=None):
    runs = {}
    for d in sorted(x for exp in exps for x in (ROOT / 'reports/runs' / exp).glob('*')):
        s = d / 'arm_summary.json'
        if not s.exists() or not (d / 'eval_judgments.jsonl').exists(): continue
        summ = json.loads(s.read_text()); summ['arm_id'] = arm_id(summ, d.name)
        # attack-independent controls (pretrained, clean, trusted-only, RTDW-on-clean) are shared across attack tags
        if tag and summ.get('attack_tag') != tag and summ['arm'] not in ATTACK_INDEPENDENT: continue
        if d.name in runs: continue
        if summ['arm'] in ATTACK_INDEPENDENT and any(r['summary']['arm_id'] == summ['arm_id'] and r['summary']['seed'] == summ['seed']
                                                     for r in runs.values()):
            continue  # same control trained once; another experiment's copy is not an extra seed
        runs[d.name] = dict(summary=summ, dir=d, judg=jl(d / 'eval_judgments.jsonl'),
                            metrics=json.loads((d / 'eval_metrics.json').read_text()),
                            train=json.loads((d / 'train_metrics.json').read_text()) if (d / 'train_metrics.json').exists() else None)
    return runs


def per_prompt(judg, concepts, field, split='test'):
    concepts = {concepts} if isinstance(concepts, str) else set(concepts)
    by = defaultdict(list)
    for j in judg:
        if j['split'] == split and j['source_concept'] in concepts and field in j: by[j['id']].append(float(j[field]))
    return {k: float(np.mean(v)) for k, v in by.items()}


def group_key(s):
    return (s['arm_id'], s['k'], None if s['arm'] in ATTACK_INDEPENDENT else s.get('attack_tag'))


def _matrix(mats, prompts):
    return np.array([[m[p] for p in prompts] for m in mats])


def boot(mats, reps=4000, seed=0):
    """mats: list over training seeds of {prompt: value}. Mean and 95% CI from a 2-level bootstrap (seeds, prompts)."""
    prompts = sorted(set.intersection(*[set(m) for m in mats]))
    X = _matrix(mats, prompts)
    rng = np.random.default_rng(seed)
    si = rng.integers(0, X.shape[0], (reps, X.shape[0])); pi = rng.integers(0, X.shape[1], (reps, X.shape[1]))
    b = np.array([X[si[r]][:, pi[r]].mean() for r in range(reps)])
    return dict(estimate=float(X.mean()), ci95=[float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                n_prompts=len(prompts), n_train_seeds=X.shape[0]), prompts, X


def paired(mA, mB, seedsA=None, seedsB=None, reps=4000, seed=1, alpha=0.05):
    """A - B, paired over prompts; training seeds resampled jointly when both arms have the same >= 2 seeds, else
    independently per arm (a single run contributes no seed variance, which the note states)."""
    prompts = sorted(set.intersection(*[set(m) for m in mA + mB]))
    XA, XB = _matrix(mA, prompts), _matrix(mB, prompts)
    rng = np.random.default_rng(seed)
    pairs = bool(seedsA and seedsB and sorted(seedsA) == sorted(seedsB) and len(seedsA) >= 2)
    if pairs:   # align rows by seed
        XB = XB[[list(seedsB).index(s) for s in seedsA]]
    pi = rng.integers(0, len(prompts), (reps, len(prompts)))
    sa = rng.integers(0, XA.shape[0], (reps, XA.shape[0]))
    sb = sa if pairs else rng.integers(0, XB.shape[0], (reps, XB.shape[0]))
    b = np.array([XA[sa[r]][:, pi[r]].mean() - XB[sb[r]][:, pi[r]].mean() for r in range(reps)])
    lo, hi = alpha / 2, 1 - alpha / 2
    per_seed = ({int(s): float((XA[i] - XB[i]).mean()) for i, s in enumerate(seedsA)} if pairs else None)
    n_seed = min(XA.shape[0], XB.shape[0])
    return dict(estimate=float(XA.mean() - XB.mean()), ci=[float(np.quantile(b, lo)), float(np.quantile(b, hi))],
                ci95=[float(np.quantile(b, .025)), float(np.quantile(b, .975))], level=1 - alpha,
                n_prompts=len(prompts), train_seeds_a=len(mA), train_seeds_b=len(mB), seed_paired=pairs,
                per_seed=per_seed,
                note=('two-level bootstrap: prompts paired; training seeds ' + ('paired' if pairs else 'resampled per arm')
                      + ('' if n_seed >= 2 else '; SINGLE training run on at least one side: run-to-run variance not captured')))


def fp_cost(rep_dir):
    """False-positive cost of a drift/oracle weighting: weight lost by CLEAN records, per concept. Poison labels are
    used only to separate clean from poisoned records in this report, never by the defense."""
    w = rep_dir / 'weights.json'; md = rep_dir / 'train_metadata.jsonl'
    if not w.exists() or not md.exists(): return None
    W = json.loads(w.read_text())['weights']; rows = jl(md)
    per = defaultdict(list); pois = []
    for r in rows:
        if r['id'] not in W: continue
        (pois if r.get('is_poisoned') else per[r.get('concept', '?')]).append(float(W[r['id']]))
    allc = [x for v in per.values() for x in v]
    return dict(clean_mean_weight=float(np.mean(allc)) if allc else None,
                clean_frac_below_half=float(np.mean(np.array(allc) < 0.5)) if allc else None,
                clean_weight_mass_lost=float(np.sum(1 - np.array(allc))) if allc else None,
                poison_mean_weight=float(np.mean(pois)) if pois else None, n_poison=len(pois), n_clean=len(allc),
                by_concept={c: dict(n=len(v), mean_weight=float(np.mean(v)), frac_below_half=float(np.mean(np.array(v) < 0.5)))
                            for c, v in sorted(per.items())})


def concept_metric(m, concept, field):
    v = m['per_split_concept'].get(f'test:{concept}', {}).get(field)
    return v


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp', nargs='+', required=True); ap.add_argument('--concept', default='dog'); ap.add_argument('--target', default='cat')
    ap.add_argument('--attack-tag', help='keep only this attack (plus shared attack-independent controls)')
    ap.add_argument('--name', help='output name (default: first exp)')
    ap.add_argument('--reps', type=int, default=4000)
    a = ap.parse_args()
    runs = load(a.exp, a.attack_tag)
    groups = defaultdict(list)
    for name, r in runs.items(): groups[group_key(r['summary'])].append(name)
    out = dict(exp=a.exp, attack_tag=a.attack_tag, concept=a.concept, target=a.target, arms={}, deltas={}, not_run=[],
               unavailable=UNAVAILABLE, drift={}, gates={}, protocol=dict(
                   unit='test prompt', ci='two-level bootstrap over training seeds and prompts', reps=a.reps,
                   gate3=f'Bonferroni over available controls; non-target non-inferiority margin {NONINF_MARGIN}; KID <= B + 2 sd'))
    concepts = sorted({j['source_concept'] for r in runs.values() for j in r['judg']})
    nontarget = [c for c in concepts if c != a.concept]
    for (aid, k, tag), names in sorted(groups.items(), key=lambda x: (str(x[0][1]), x[0][0])):
        key = f'{aid}@k{k}'
        R = [runs[n] for n in names]
        arm = R[0]['summary']['arm']
        pp = lambda f, cs=a.concept: [per_prompt(r['judg'], cs, f) for r in R]
        acc, _, _ = boot(pp('source_correct'), a.reps)
        asr, _, _ = boot(pp('attacker_target'), a.reps)
        cacc, _, _ = boot(pp('clip_source_correct'), a.reps)
        casr, _, _ = boot(pp('clip_attacker_target'), a.reps)
        j2 = boot(pp('judge2_source_correct'), a.reps)[0] if all(pp('judge2_source_correct')) else None
        nt = boot(pp('source_correct', nontarget), a.reps)[0] if nontarget else None
        seeds_acc = [float(np.mean(list(m.values()))) for m in pp('source_correct')]
        seeds_asr = [float(np.mean(list(m.values()))) for m in pp('attacker_target')]
        coll = {c: float(np.mean([np.mean(list(per_prompt(r['judg'], c, 'source_correct').values())) for r in R])) for c in concepts}
        m = [r['metrics'] for r in R]
        tr = [r['train'] for r in R if r['train']]
        out['arms'][key] = dict(
            arm=arm, arm_id=aid, label=ARM_NAMES.get(arm, arm) + ('' if aid == arm else f' [{aid[len(arm) + 1:]}]'), k=k,
            attack_tag=tag, runs=names, train_seeds=[r['summary']['seed'] for r in R],
            gen_seeds=sorted({int(j['seed']) for r in R for j in r['judg']}),
            source_correct=acc, attacker_target=asr, clip_source_correct=cacc, clip_attacker_target=casr,
            judge2_source_correct=j2, nontarget_source_correct=nt,
            per_seed_source_correct=seeds_acc, per_seed_attacker_target=seeds_asr, accuracy_by_concept=coll,
            judge_kappa=mean_or_none([concept_metric(x, a.concept, 'judge_kappa') for x in m]),
            kid_all_test=float(np.mean([x['kid_all_test'] for x in m])),
            kid_concept=mean_or_none([concept_metric(x, a.concept, 'kid') for x in m]),
            kid_concept_std=mean_or_none([concept_metric(x, a.concept, 'kid_std') for x in m]),
            precision_concept=mean_or_none([concept_metric(x, a.concept, 'precision') for x in m]),
            recall_concept=mean_or_none([concept_metric(x, a.concept, 'recall') for x in m]),
            density_concept=mean_or_none([concept_metric(x, a.concept, 'density') for x in m]),
            coverage_concept=mean_or_none([concept_metric(x, a.concept, 'coverage') for x in m]),
            diversity_concept=mean_or_none([concept_metric(x, a.concept, 'diversity') for x in m]),
            gpu_hours=float(np.mean([t['gpu_hours'] for t in tr])) if tr else 0.0,
            prep_gpu_hours=float(np.mean([r['summary']['prep_seconds'] for r in R])) / 3600,
            train_exposures=float(np.mean([t['train_exposures'] for t in tr])) if tr else 0,
            trusted_exposures=float(np.mean([t['trusted_exposures'] for t in tr])) if tr else 0,
            denoiser_forward_examples=float(np.mean([t['denoiser_forward_examples'] for t in tr])) if tr else 0,
            peak_vram_gib=float(np.mean([t['peak_vram_bytes'] for t in tr])) / 2**30 if tr else None,
            optimizer_steps=int(np.mean([t['optimizer_steps'] for t in tr])) if tr else 0,
            weight_cost=fp_cost(R[0]['dir']) if arm in ('J', 'K', 'KE', 'KB', 'O', 'GK') else None)
        d = R[0]['dir']
        if (d / 'drift_train.jsonl').exists():
            dr = jl(d / 'drift_train.jsonl'); trd = jl(d / 'drift_trusted.jsonl')
            out['drift'][key] = dict(auroc=auroc([x['drift'] for x in dr if x['is_poisoned']], [x['drift'] for x in dr if not x['is_poisoned']]),
                poison=[x['drift'] for x in dr if x['is_poisoned']], clean=[x['drift'] for x in dr if not x['is_poisoned']],
                trusted=[x['drift'] for x in trd],
                weights_summary=json.loads((d / 'weights_summary.json').read_text()) if (d / 'weights_summary.json').exists() else None)
        if (d / 'filter_eval.json').exists():   # I / LF / J / O: what the hard filter actually removed
            out['arms'][key]['filter_eval'] = json.loads((d / 'filter_eval.json').read_text())
    for arm in ARM_NAMES:
        if not any(v['arm'] == arm for v in out['arms'].values()): out['not_run'].append(arm)
    # paired deltas at each dose against undefended and clean, plus candidates vs every control
    keys = list(out['arms'])
    def pp(key, f='source_correct', cs=None):
        return [per_prompt(runs[n]['judg'], cs or a.concept, f) for n in out['arms'][key]['runs']]
    def delta(ka, kb, alpha=0.05):
        A, B = out['arms'][ka], out['arms'][kb]
        d = dict(source_correct=paired(pp(ka), pp(kb), A['train_seeds'], B['train_seeds'], a.reps, alpha=alpha),
                 attacker_target=paired(pp(ka, 'attacker_target'), pp(kb, 'attacker_target'), A['train_seeds'], B['train_seeds'], a.reps, alpha=alpha))
        if A['judge2_source_correct'] and B['judge2_source_correct']:
            d['judge2_source_correct'] = paired(pp(ka, 'judge2_source_correct'), pp(kb, 'judge2_source_correct'),
                                                A['train_seeds'], B['train_seeds'], a.reps, alpha=alpha)
        if nontarget:
            d['nontarget_source_correct'] = paired(pp(ka, cs=nontarget), pp(kb, cs=nontarget), A['train_seeds'], B['train_seeds'], a.reps, alpha=alpha)
        return d
    for ka in keys:
        for kb in keys:
            if ka == kb: continue
            A, B = out['arms'][ka], out['arms'][kb]
            if A['k'] != B['k'] and not (A['arm'] in ATTACK_INDEPENDENT or B['arm'] in ATTACK_INDEPENDENT): continue
            if B['arm'] in ('C', 'B', 'E', 'O') or (A['arm'] in CANDIDATES and B['arm'] not in CANDIDATES):
                out['deltas'][f'{ka} - {kb}'] = delta(ka, kb)
    # ---- pre-registered gates -------------------------------------------------------------------------------------
    def find(aid, k):
        for kk, v in out['arms'].items():
            if v['arm_id'] == aid and (v['k'] == k or v['arm'] in ATTACK_INDEPENDENT): return kk
        return None
    for k in sorted({v['k'] for v in out['arms'].values() if v['arm'] == 'C'}):
        kb_, kc = find('B', k), find('C', k)
        if not (kb_ and kc): continue
        g2 = out['deltas'].get(f'{kb_} - {kc}') or delta(kb_, kc)
        gate = dict(rule='B - C source accuracy, 95% CI lower bound > 0', estimate=g2['source_correct']['estimate'],
                    ci95=g2['source_correct']['ci95'], train_seeds=[len(out['arms'][kb_]['train_seeds']), len(out['arms'][kc]['train_seeds'])],
                    passed=g2['source_correct']['ci95'][0] > 0)
        out['gates'][f'gate2@k{k}'] = gate
        if not gate['passed']: continue
        for cand in CANDIDATES:
            kx = find(cand, k)
            if not kx: continue
            ctrls = [c for c in CONTROLS if find(c, k)]
            alpha = 0.05 / max(1, len(ctrls))
            comps = {}
            for c in ctrls:
                dd = delta(kx, find(c, k), alpha)['source_correct']
                comps[c] = dict(estimate=dd['estimate'], ci=dd['ci'], level=dd['level'], passed=dd['ci'][0] > 0)
            nonint = delta(kx, kb_)['nontarget_source_correct'] if nontarget else None
            fx = load_features([runs[n]['dir'] for n in out['arms'][kx]['runs']], a.concept)
            fb = load_features([runs[n]['dir'] for n in out['arms'][kb_]['runs']], a.concept)
            refp = ROOT / 'reports/real_inception_ref16.npz'
            ref = None
            if refp.exists():
                z = np.load(refp, allow_pickle=True); ref = z['f'][z['c'] == a.concept].astype(np.float64)
            q = quality_delta(fx, fb, ref, reps=max(100, a.reps // 20)) if (fx and fb) else None
            kid_ok = None if q is None else bool(q['kid_diff']['ci95'][0] <= 0 and q['diversity_diff']['ci95'][1] >= 0)
            orc = find('O', k)
            out['gates'][f'gate3@k{k}:{cand}'] = dict(
                controls=comps, bonferroni_level=1 - alpha, all_controls_beaten=all(v['passed'] for v in comps.values()) if comps else False,
                missing_controls=[c for c in CONTROLS if c not in ctrls],
                nontarget_vs_B=dict(estimate=nonint['estimate'], ci95=nonint['ci95'], margin=-NONINF_MARGIN,
                                    passed=nonint['ci95'][0] > -NONINF_MARGIN) if nonint else None,
                quality_vs_B=q, quality_within_B=kid_ok,
                gap_to_oracle=(delta(kx, orc)['source_correct'] if orc else None),
                passed=bool(comps) and all(v['passed'] for v in comps.values()) and bool(nonint and nonint['ci95'][0] > -NONINF_MARGIN)
                       and kid_ok is True)
    dst = ROOT / 'reports' / f'results_{a.name or a.exp[0]}.json'
    dst.write_text(json.dumps(out, indent=2))
    for k, v in out['arms'].items():
        print(f"{k:12s} {v['label'][:34]:34s} acc={v['source_correct']['estimate']:.3f} {np.round(v['source_correct']['ci95'], 3).tolist()} "
              f"asr={v['attacker_target']['estimate']:.3f} KID={v['kid_all_test']:.4f} seeds={v['train_seeds']}")
    for g, v in out['gates'].items(): print(g, 'PASSED' if v['passed'] else 'not passed')
    print('not_run:', out['not_run'])


if __name__ == '__main__':
    main()
