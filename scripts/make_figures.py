#!/usr/bin/env python3
"""Paper figures F1-F14 from MEASURED artefacts only (reports/results_<exp>.json + reports/runs/<exp>).

A figure whose inputs do not exist is not drawn; it is listed as not_run in reports/figures/FIGURES.md.
Colour encodes arm identity with a fixed mapping (dataviz reference palette, validated): the four key arms
get categorical slots, all other arms a neutral gray with direct labels, so identity is never colour-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_labels import LONG, SHORT, plain  # noqa: E402
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reports'; FIG = REP / 'figures'   # --reports overrides both
COL = {'B': '#2a78d6', 'C': '#eb6834', 'E': '#1baf7a', 'K': '#6250d6', 'KE': '#6250d6', 'R': '#e87ba4', 'RS': '#e87ba4', 'V': '#eda100'}
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
ORDER = ['A', 'B', 'O', 'C', 'D', 'E', 'F', 'G', 'H', 'M1', 'M2', 'I', 'LF', 'J', 'K', 'KE', 'KB', 'V', 'R', 'RS']
rank = lambda arm: ORDER.index(arm) if arm in ORDER else len(ORDER)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b5b4ae',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': '#e8e7e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False})
col = lambda arm: COL.get(arm, GRAY)


def save(fig, name, made):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png'); fig.savefig(FIG / f'{name}.pdf'); plt.close(fig); made.append(name)


def err(v):
    return [[v['estimate'] - v['ci95'][0]], [v['ci95'][1] - v['estimate']]]


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--exp', nargs='+', required=True)
    ap.add_argument('--main-k', type=int, default=200)
    ap.add_argument('--reports', default=None, help='folder holding results_*.json (default: reports/)')
    a = ap.parse_args()
    global REP, FIG
    if a.reports: REP = Path(a.reports); FIG = REP / 'figures'
    R = {}
    for e in a.exp:
        R.update({k: dict(v, exp=e, label=plain(v['arm']) + ('' if v['k'] in (200, None) or v['arm'] in ('A', 'B') else f" (k={v['k']})")) for k, v in json.loads((REP / f'results_{e}.json').read_text())['arms'].items()})
    res0 = json.loads((REP / f'results_{a.exp[0]}.json').read_text())
    concept, target = res0['concept'], res0['target']
    made, missing = [], []
    main = [k for k in sorted(R, key=lambda k: rank(R[k]['arm'])) if R[k]['k'] == a.main_k or R[k]['arm'] in ('A', 'B')]
    main = list(dict.fromkeys(main))

    # F1: accuracy + ASR per arm with 95% CI (horizontal: one row per arm, readable with ~20 arms)
    if main:
        n = len(main); ys = np.arange(n)[::-1]
        fig, axs = plt.subplots(1, 2, figsize=(10, 0.27 * n + 1.3), sharey=True); fig.subplots_adjust(wspace=0.06)
        for ax, f, title in [(axs[0], 'source_correct', f'a  "{concept}" prompts judged {concept} (higher = better)'),
                             (axs[1], 'attacker_target', f'b  "{concept}" prompts judged {target} = attack success (lower = better)')]:
            for y, k in zip(ys, main):
                v = R[k][f]; ax.barh(y, v['estimate'], color=col(R[k]['arm']), height=0.7)
                ax.errorbar(v['estimate'], y, xerr=err(v), color=INK, capsize=1.5, lw=0.8)
                ax.text(min(v['ci95'][1] + 0.02, 1.02), y, f"{v['estimate']:.2f}", va='center', fontsize=6.5, color=INK)
            ax.set_xlim(0, 1.12); ax.set_title(title, fontsize=9, loc='left'); ax.grid(axis='y', visible=False)
        seeds = lambda k: len(R[k]['train_seeds'])
        axs[0].set_yticks(ys, [R[k]['label'] + ('' if seeds(k) >= 3 else f'  [{seeds(k)} seed]') for k in main], fontsize=7.5)
        fig.subplots_adjust(top=1 - 0.55 / (0.27 * n + 1.3))
        fig.suptitle(f'F1  Held-out test prompts, k={a.main_k} poisons; independent ImageNet judge; 95% CI over prompts; '
                     '[n seed] = fewer than 3 training seeds', x=0.01, y=1.0, ha='left', fontsize=10)
        save(fig, 'F1_accuracy_asr_by_arm', made)
    else: missing.append('F1')

    # F2: dose response
    ks = sorted({R[k]['k'] for k in R if R[k]['arm'] in ('C', 'K', 'E')})
    if len(ks) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4)); fig.subplots_adjust(wspace=0.3)
        for arm in ['B', 'C', 'E', 'K', 'R', 'RS', 'V']:
            pts = sorted([(R[k]['k'] if arm != 'B' else None, R[k]) for k in R if R[k]['arm'] == arm], key=lambda x: x[0] or 0)
            for ax, f in zip(axs, ['source_correct', 'attacker_target']):
                if arm == 'B' and pts:
                    v = pts[0][1][f]; ax.axhspan(*v['ci95'], color=col('B'), alpha=0.12); ax.axhline(v['estimate'], color=col('B'), lw=1.5, ls='--')
                    ax.text(max(ks), v['estimate'], ' clean', color=INK, fontsize=7, va='bottom', ha='right')
                elif pts:
                    x = [p[0] for p in pts]; y = [p[1][f]['estimate'] for p in pts]
                    lo = [p[1][f]['ci95'][0] for p in pts]; hi = [p[1][f]['ci95'][1] for p in pts]
                    ax.fill_between(x, lo, hi, color=col(arm), alpha=0.12, lw=0)
                    ax.plot(x, y, '-o', color=col(arm), lw=2, ms=4, label=plain(arm))
        for ax, t in zip(axs, [f'{concept} accuracy', f'attack success ({target})']):
            ax.set_xlabel('# poisoned images for the attacked concept (k)'); ax.set_title(t, loc='left', fontsize=9); ax.set_ylim(-0.02, 1.05)
        axs[0].legend(fontsize=7, loc='lower left'); fig.suptitle('F2  Dose-response (shaded = 95% CI)', x=0.01, ha='left', fontsize=10)
        save(fig, 'F2_dose_response', made)
    else: missing.append('F2 (needs >=2 poison doses)')

    # F3: forest of paired deltas at k=main_k only: (a) every arm minus poisoned C, (b) RTDW minus every other arm
    deltas = {}
    for e in a.exp:
        deltas.update(json.loads((REP / f'results_{e}.json').read_text())['deltas'])
    kk = f'@k{a.main_k}'
    pa = [d for d in deltas if d.count(kk) == 2 and d.endswith(f' - C{kk}') and not d.startswith(f'C{kk}')]
    pb = [d for d in deltas if d.count(kk) == 2 and d.startswith(f'K{kk} - ')]
    if pa or pb:
        names = {k: R[k]['label'] for k in R}
        panels = [(p, t) for p, t in [(pa, 'a  defense − Poisoned'), (pb, 'b  RTDW − other defense')] if p]
        h = 0.26 * max(len(p) for p, _ in panels) + 1.4
        fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels) + 1.5, h)); axs = np.atleast_1d(axs)
        fig.subplots_adjust(wspace=0.9)
        for ax, (sel, title) in zip(axs, panels):
            sel = sorted(sel, key=lambda d: deltas[d]['source_correct']['estimate'])
            for i, d in enumerate(sel):
                v = deltas[d]['source_correct']; lo, hi = v['ci95']; c = col(d.split('@')[0] if title.startswith('a') else d.split(' - ')[1].split('@')[0])
                ax.errorbar(v['estimate'], i, xerr=err(v), fmt='o', color=c, ms=4, capsize=2, mfc=c if lo > 0 or hi < 0 else 'white')
            other = (lambda d: d.split(' - ')[0]) if title.startswith('a') else (lambda d: d.split(' - ')[1])
            ax.set_yticks(range(len(sel)), [names.get(other(d).strip(), other(d)) for d in sel], fontsize=7.5)
            ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-1.05, 1.05); ax.set_title(title, loc='left', fontsize=9)
            ax.set_xlabel(f'Δ {concept} accuracy (paired, 95% CI over prompts)')
        fig.suptitle(f'F3  Paired differences at k={a.main_k}; filled = CI excludes 0', x=0.01, ha='left', fontsize=10)
        save(fig, 'F3_paired_deltas_forest', made)
    else: missing.append('F3')

    # F4 Pareto: accuracy vs KID ; F5 cost effectiveness. (a) all arms, labelled by arm code; (b) the defended arms
    # (accuracy >= 0.9) as a sorted dot plot with full names, so the crowded top cluster stays readable.
    for name, xf, xl in [('F4_pareto_accuracy_vs_kid', 'kid_all_test', 'KID vs held-out real COCO (lower = better)'),
                         ('F5_accuracy_vs_compute', 'gpu_total', 'GPU-hours (training + one-off preprocessing)')]:
        if not main: missing.append(name); continue
        xv = lambda v: v['gpu_hours'] + v['prep_gpu_hours'] if xf == 'gpu_total' else v[xf]
        good = sorted([k for k in main if R[k]['source_correct']['estimate'] >= 0.9], key=lambda k: xv(R[k]))
        fig, axs = plt.subplots(1, 2, figsize=(11, max(3.8, 0.24 * len(good) + 1.4)), gridspec_kw=dict(width_ratios=[1.2, 1]))
        fig.subplots_adjust(wspace=0.75)
        ax = axs[0]; ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        for k in main:
            v = R[k]; x = xv(v); y = v['source_correct']['estimate']
            ax.errorbar(x, y, yerr=err(v['source_correct']), fmt='o', color=col(v['arm']), ms=5, capsize=2, mec='white', mew=1)
        low = sorted([k for k in main if R[k]['source_correct']['estimate'] < 0.9], key=lambda k: xv(R[k]))
        lo_, hi_ = ax.get_xlim(); sp = hi_ - lo_
        lx = np.linspace(lo_ + 0.12 * sp, hi_ - 0.05 * sp, len(low)) if len(low) > 1 else [xv(R[k]) for k in low]
        for i, k in enumerate(low):   # undefended arms crowd near accuracy 0: names spread evenly, leader lines to points
            ax.annotate(plain(R[k]['arm'], short=True), (xv(R[k]), R[k]['source_correct']['estimate']),
                        xytext=(lx[i], 0.34 + 0.1 * (i % 3)), textcoords='data', fontsize=7, color=INK, ha='center',
                        arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.5, shrinkA=1, shrinkB=2))
        ax.set_xlim(lo_, hi_)
        if good:
            gx = [xv(R[k]) for k in good]
            ax.add_patch(plt.Rectangle((min(gx), 0.9), max(gx) - min(gx), 0.12, fill=False, ec=MUTED, ls='--', lw=0.8))
            ax.text(min(gx), 0.88, 'defended arms: see b', fontsize=7, color=MUTED, va='top')
        lo_, hi_ = ax.get_xlim(); ax.set_xlim(lo_ - 0.04 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
        ax.set_xlabel(xl); ax.set_ylabel(f'{concept} accuracy'); ax.set_ylim(0, 1.05); ax.set_title('a  all arms', loc='left', fontsize=9)
        ax = axs[1]
        for i, k in enumerate(good):
            ax.plot(xv(R[k]), i, 'o', color=col(R[k]['arm']), ms=6, mec='white', mew=1)
        ax.set_yticks(range(len(good)), [f"{R[k]['label']}  ({R[k]['source_correct']['estimate']:.2f})" for k in good], fontsize=7.5)
        ax.set_xlabel(xl); ax.invert_yaxis(); ax.set_title(f'b  arms with {concept} accuracy ≥ 0.9 (accuracy in brackets)', loc='left', fontsize=9)
        fig.suptitle(name.split('_')[0] + ('  Robustness vs image quality' if 'kid' in xf else '  Robustness vs compute') + f' (k={a.main_k})',
                     x=0.01, ha='left', fontsize=10)
        save(fig, name, made)

    # F6: drift distributions + ROC
    drift = {}
    for e in a.exp:
        drift.update(json.loads((REP / f'results_{e}.json').read_text())['drift'])
    if drift:
        key = next((k for k in drift if f'k{a.main_k}' in k), next(iter(drift))); d = drift[key]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        bins = np.linspace(0, max(max(d['poison'] or [0.1]), max(d['clean'])) * 1.05, 50)
        axs[0].hist(d['clean'], bins, color=col('B'), alpha=0.75, label=f"clean train (n={len(d['clean'])})", density=True)
        axs[0].hist(d['poison'], bins, color=col('C'), alpha=0.75, label=f"poisoned train (n={len(d['poison'])})", density=True)
        ws = d.get('weights_summary') or {}
        if ws.get('tau'): axs[0].axvline(ws['tau'], color=INK, ls='--', lw=1); axs[0].text(ws['tau'], axs[0].get_ylim()[1] * 0.9, ' τ (trusted q.99)', fontsize=7)
        axs[0].set_xlabel('round-trip semantic drift d(x)'); axs[0].legend(fontsize=7); axs[0].set_title('distribution', loc='left', fontsize=9)
        for kk, dd in drift.items():
            if not dd['poison']: continue
            th = np.unique(np.concatenate([dd['poison'], dd['clean']]))[::-1]
            tpr = [np.mean(np.array(dd['poison']) >= t) for t in th]; fpr = [np.mean(np.array(dd['clean']) >= t) for t in th]
            axs[1].plot([0] + fpr, [0] + tpr, lw=2, label=f"{plain(kk.split('@')[0])}, k={kk.split('@k')[-1]}: AUROC={dd['auroc']:.3f}", color=col('K'))
        axs[1].plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=':'); axs[1].set_xlabel('false-positive rate (clean)'); axs[1].set_ylabel('true-positive rate (poison)')
        axs[1].legend(fontsize=7, loc='lower right'); axs[1].set_title('ROC (evaluation only: labels never used in training)', loc='left', fontsize=9)
        fig.suptitle('F6  Does the VAE round trip expose the poisons?', x=0.01, ha='left', fontsize=10)
        save(fig, 'F6_roundtrip_drift_detection', made)
    else: missing.append('F6')
    # F7 (adaptive trade-off) and F14 (trusted-threshold ablation): scripts/make_comparison_figures.py

    # F8 heatmap
    if main:
        cs = sorted(next(iter(R.values()))['accuracy_by_concept'])
        M = np.array([[R[k]['accuracy_by_concept'][c] for c in cs] for k in main])
        fig, ax = plt.subplots(figsize=(1 + 0.75 * len(cs), 0.35 * len(main) + 1.2))
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(cs)), cs); ax.set_yticks(range(len(main)), [R[k]['label'] for k in main]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03, label='concept accuracy')
        ax.set_title('F8  Collateral effects: test accuracy per concept', loc='left', fontsize=10)
        save(fig, 'F8_concept_accuracy_heatmap', made)

    # F9 training curves
    curves = []
    for k in main:
        for e in a.exp:
            f = REP / 'runs' / e / R[k]['runs'][0] / 'train_training_log.jsonl'
            if f.exists(): curves.append((k, [json.loads(l) for l in f.read_text().splitlines() if l.strip()]))
    if curves:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for k, log in curves:
            y = np.array([e['untrusted_loss'] for e in log]); w = min(50, len(y))
            ys = np.convolve(y, np.ones(w) / w, mode='valid')
            ax.plot(np.arange(len(ys)) + w, ys, color=col(R[k]['arm']), lw=1.5 if R[k]['arm'] in COL else 0.8)
            ax.text(len(ys) + w, ys[-1], ' ' + R[k]['label'], fontsize=6, color=INK, va='center')
        ax.set_xlabel('optimizer step'); ax.set_ylabel('training-batch denoising loss (moving avg)')
        ax.set_title('F9  Training curves (loss is NOT semantic accuracy)', loc='left', fontsize=10)
        save(fig, 'F9_training_curves', made)
    missing.append('F10 per-timestep loss diagnostics (N11b not_run)')

    # F11 grid
    grid_arms = [k for k in main if any((REP / 'runs' / e / R[k]['runs'][0] / 'samples').exists() for e in a.exp)]
    if grid_arms:
        from PIL import Image
        cols_ = sorted((REP / 'runs' / R[grid_arms[0]]['exp'] / R[grid_arms[0]]['runs'][0] / 'samples').glob(f'test_{concept}_*.jpg'))[:6]
        fig, axs = plt.subplots(len(grid_arms), len(cols_), figsize=(1.3 * len(cols_), 1.35 * len(grid_arms)))
        axs = np.atleast_2d(axs)
        for i, k in enumerate(grid_arms):
            for j, c in enumerate(cols_):
                p = REP / 'runs' / R[k]['exp'] / R[k]['runs'][0] / 'samples' / c.name
                axs[i, j].imshow(Image.open(p)) if p.exists() else None; axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                if j == 0: axs[i, j].set_ylabel(R[k]['label'], fontsize=7, rotation=0, ha='right', va='center')
        fig.suptitle(f'F11  Same "{concept}" test prompt and noise seed across arms', x=0.01, ha='left', fontsize=10)
        save(fig, 'F11_image_grid', made)

    # F12 triplets
    trip = sorted(p for e in a.exp for p in (REP / 'runs' / e).glob('*/triplets'))
    if trip:
        from PIL import Image
        ids = sorted({p.name.rsplit('_', 1)[0] for p in trip[0].glob('*_input.png')})[:5]
        fig, axs = plt.subplots(2, len(ids), figsize=(1.8 * len(ids), 3.8)); axs = np.atleast_2d(axs)
        for j, i in enumerate(ids):
            for r, s in enumerate(['input', 'roundtrip']):
                axs[r, j].imshow(Image.open(trip[0] / f'{i}_{s}.png')); axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
        axs[0, 0].set_ylabel('poisoned\nimage x′', fontsize=8); axs[1, 0].set_ylabel('VAE round trip\nD(E(x′))', fontsize=8)
        fig.suptitle('F12  What the UNet actually learns from: decoded latents of poisoned training images', x=0.01, ha='left', fontsize=10)
        save(fig, 'F12_poison_roundtrip_triplets', made)
    else: missing.append('F12')

    # F13 seed variability
    multi = [k for k in R if len(R[k]['train_seeds']) >= 2]
    if multi:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for i, k in enumerate(multi):
            ys = R[k]['per_seed_source_correct']; ax.scatter([i] * len(ys), ys, color=col(R[k]['arm']), s=18, zorder=3)
            ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color=INK, lw=1.5)
        ax.set_xticks(range(len(multi)), [R[k]['label'] for k in multi], rotation=30, ha='right'); ax.set_ylabel(f'{concept} accuracy'); ax.set_ylim(0, 1.05)
        ax.set_title('F13  Training-seed variability (dots = seeds, bar = mean)', loc='left', fontsize=10)
        save(fig, 'F13_seed_variability', made)
    else: missing.append('F13 (needs >=2 training seeds)')

    lines = ['# Figure index (generated by scripts/make_figures.py from measured artefacts)', '', '## Drawn', '']
    lines += [f'- `{m}.png` / `.pdf`' for m in made] + ['', '## Not drawn (inputs not_run)', ''] + [f'- {m}' for m in missing]
    lines += ['', 'F7, F14-F17 and table T1: see `COMPARISON.md` (scripts/make_comparison_figures.py).']
    (FIG / 'FIGURES.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
