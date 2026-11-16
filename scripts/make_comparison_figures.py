#!/usr/bin/env python3
"""Cross-attack comparison figures and tables, built only from MEASURED analysis outputs.

Inputs: <reports>/results_<name>.json (scripts/analyze_results.py) and <reports>/results_tradeoff.json
(scripts/attack_tradeoff.py). Nothing is recomputed; a panel whose input file or arm is missing is left out and listed
under "Not drawn" in <reports>/figures/COMPARISON.md.

  F7   adaptive-attack trade-off: accuracy of C/G/J/K and poison drift detectability vs the drift penalty lambda
  F14  trusted-set ablation: trusted-calibrated threshold vs unsupervised threshold (N4 ran these two variants)
  F15  attack x defense heatmap of source-concept accuracy
  F16  paired differences K-C, G-C, K-G per attack (95% CI over prompts, from results_tradeoff.json)
  F17  per-training-seed accuracy for the multi-seed attack conditions
  F18/F19  qualitative grids from the saved generations (<reports>/runs/*/*/samples) with the judge's label per image
  T1   the F15 numbers as CSV and LaTeX (estimate, 95% CI, number of training seeds)
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_labels import LONG, SHORT, plain  # noqa: E402
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
COL = {'B': '#2a78d6', 'C': '#eb6834', 'E': '#1baf7a', 'G': '#c28a00', 'J': '#a79cf0', 'K': '#6250d6', 'O': '#52514e'}
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
col = lambda arm: COL.get(arm, GRAY)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b5b4ae',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': '#e8e7e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False})

# (results name, row label) in display order. Labels state concept -> target and the attack.
CONDITIONS = [('dog2cat_eps02', 'dog→cat, ε=0.02'), ('dog', 'dog→cat, ε=0.04'), ('dog2cat_eps08', 'dog→cat, ε=0.08'),
              ('dog2cat_adapt0', 'dog→cat, adaptive λ=0'), ('dog2cat_adapt0p05', 'dog→cat, adaptive λ=0.05'),
              ('dog2cat_adaptjpeg', 'dog→cat, JPEG-aware'), ('dog2cat_dirty', 'dog→cat, dirty-label'),
              ('car', 'car→cow, ε=0.04'), ('car_eps08', 'car→cow, ε=0.08')]
LAMBDAS = [('dog2cat_adapt0', 0.0), ('dog2cat_adapt0p05', 0.05), ('dog2cat_adapt0p1', 0.1), ('dog2cat_adapt0p25', 0.25),
           ('dog2cat_adapt0p5', 0.5), ('dog2cat_adapt1', 1.0), ('dog2cat_adapt4', 4.0), ('dog2cat_adapt16', 16.0)]
HEAT_ARMS = ['B', 'O', 'C', 'E', 'G', 'M1', 'M2', 'D', 'J', 'K', 'GK']
ATTACK_INDEPENDENT = ('A', 'B', 'D', 'KB')   # trained without poisons: one run serves every attack
ARM_SHORT = SHORT


class Reports:
    def __init__(self, d: Path, k: int):
        self.d, self.k, self._c = d, k, {}
        p = d / 'results_tradeoff.json'
        self.trade = json.loads(p.read_text()) if p.exists() else {'rows': {}}

    def res(self, name):
        if name not in self._c:
            p = self.d / f'results_{name}.json'
            self._c[name] = json.loads(p.read_text()) if p.exists() else None
        return self._c[name]

    def arm(self, name, arm_id):
        r = self.res(name)
        return (r or {}).get('arms', {}).get(f'{arm_id}@k{self.k}')

    def row(self, name):
        return self.trade.get('rows', {}).get(name) or {}

    def effective(self, name):
        g2 = self.row(name).get('gate2')
        return None if not g2 else bool(g2.get('passed'))


def save(fig, out, name, made):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png'); fig.savefig(out / f'{name}.pdf'); plt.close(fig); made.append(name)


def fig7(R, out, made, missing):
    pts = [(lam, n) for n, lam in LAMBDAS if R.res(n)]
    if len(pts) < 2: missing.append('F7 adaptive trade-off (needs >=2 lambda results)'); return
    x = np.array([p[0] for p in pts]); xs = np.where(x == 0, 0.02, x)   # lambda=0 drawn at 0.02 on the log axis
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6)); fig.subplots_adjust(wspace=0.28)
    for ax in axs:
        for (lam, n), xi in zip(pts, xs):
            if R.effective(n) is False: ax.axvspan(xi / 1.25, xi * 1.25, color='#e8e7e3', alpha=0.8, lw=0)
    for arm in ['C', 'G', 'J', 'K']:
        y, lo, hi, xx = [], [], [], []
        for (lam, n), xi in zip(pts, xs):
            v = R.arm(n, arm)
            if v: xx.append(xi); y.append(v['source_correct']['estimate']); lo.append(v['source_correct']['ci95'][0]); hi.append(v['source_correct']['ci95'][1])
        if xx:
            axs[0].fill_between(xx, lo, hi, color=col(arm), alpha=0.12, lw=0)
            axs[0].plot(xx, y, '-o', color=col(arm), lw=2, ms=4, label=ARM_SHORT[arm])
    b = R.arm(pts[0][1], 'B')
    if b: axs[0].axhline(b['source_correct']['estimate'], color=col('B'), ls='--', lw=1.2, label='Clean')
    axs[0].set_ylabel('dog accuracy (test prompts)'); axs[0].set_ylim(-0.02, 1.05)
    axs[0].set_title('a  accuracy; grey band = attack ineffective (Gate 2 not passed)', loc='left', fontsize=9)
    axs[0].legend(fontsize=7, loc='center right')
    au = [R.row(n).get('drift_auroc') for _, n in pts]; pm = [R.row(n).get('poison_drift_median') for _, n in pts]
    cm = [R.row(n).get('clean_drift_median') for _, n in pts]
    ok = [i for i, v in enumerate(au) if v is not None]
    axs[1].plot(xs[ok], [au[i] for i in ok], '-o', color=col('K'), lw=2, ms=4, label='poison-vs-clean AUROC')
    axs[1].axhline(0.5, color=GRAY, ls=':', lw=1); axs[1].text(xs[-1], 0.51, 'chance', fontsize=7, ha='right', color=MUTED)
    axs[1].set_ylim(-0.02, 1.05); axs[1].set_ylabel('drift AUROC')
    t = axs[1].twinx(); t.spines['right'].set_visible(True); t.grid(False)
    okm = [i for i, v in enumerate(pm) if v is not None]
    t.plot(xs[okm], [pm[i] for i in okm], '--s', color=col('C'), lw=1.5, ms=3, label='poison drift median')
    okc = [i for i, v in enumerate(cm) if v is not None]
    if okc: t.plot(xs[okc], [cm[i] for i in okc], ':', color=col('B'), lw=1.5, label='clean drift median')
    t.set_ylabel('round-trip drift d(x)')
    h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = t.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')
    axs[1].set_title('b  is the poison still detectable by round-trip drift?', loc='left', fontsize=9)
    for ax in axs:
        ax.set_xscale('log'); ax.set_xticks(xs, ['0' if v == 0 else f'{v:g}' for v in x]); ax.minorticks_off()
        ax.set_xlabel('drift penalty λ in the adaptive attack objective')
    seeds = sorted({len((R.arm(n, 'K') or {}).get('train_seeds', [])) for _, n in pts} - {0})
    fig.suptitle('F7  Adaptive attacker: evading round-trip drift vs keeping the poison effective (dog→cat, ε=0.04, k=200; '
                 f"training seeds per λ: {'/'.join(map(str, seeds))}; shaded = 95% CI over prompts)", x=0.01, y=1.02, ha='left', fontsize=10)
    save(fig, out, 'F7_adaptive_tradeoff', made)


def fig14(R, out, made, missing):
    ids = [('K', 'RTDW\n(trusted τ)'), ('K-utau', 'RTDW\n(unsupervised τ)'), ('J', 'Drift filter\n(trusted τ)'),
           ('J-utau', 'Drift filter\n(unsupervised τ)')]
    have = [(i, l, R.arm('trusted', i)) for i, l in ids]; have = [h for h in have if h[2]]
    if len(have) < 2: missing.append('F14 trusted-set ablation (results_trusted.json missing)'); return
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2)); fig.subplots_adjust(wspace=0.3)
    b = R.arm('trusted', 'B')
    for j, (f, lab) in enumerate([('source_correct', 'dog accuracy'), ('kid_all_test', 'KID, all test prompts (lower = better)')]):
        ax = axs[j]
        for i, (aid, l, v) in enumerate(have):
            arm = aid.split('-')[0]; hatch = '//' if 'utau' in aid else None
            if f == 'source_correct':
                e = v[f]; ax.bar(i, e['estimate'], color=col(arm), hatch=hatch, edgecolor='white', width=0.7)
                ax.errorbar(i, e['estimate'], yerr=[[e['estimate'] - e['ci95'][0]], [e['ci95'][1] - e['estimate']]], color=INK, capsize=2, lw=1)
                ax.text(i, e['ci95'][1] + 0.02, f"{e['estimate']:.3f}\n{len(v['train_seeds'])} seed(s)", ha='center', fontsize=6.5)
            else:
                ax.bar(i, v[f], color=col(arm), hatch=hatch, edgecolor='white', width=0.7)
                ax.text(i, v[f], f'{v[f]:.4f}', ha='center', va='bottom', fontsize=6.5)
        if b:
            ref = b[f]['estimate'] if f == 'source_correct' else b[f]
            ax.axhline(ref, color=col('B'), ls='--', lw=1.2); ax.text(-0.45, ref, 'Clean ', fontsize=7, va='bottom', ha='left', color=col('B'))
        ax.set_xticks(range(len(have)), [h[1] for h in have], fontsize=7); ax.set_title(lab, loc='left', fontsize=9); ax.grid(axis='x', visible=False)
        if f == 'source_correct': ax.set_ylim(0, 1.15)
    fig.suptitle('F14  Does RTDW need the trusted set to set its threshold? (dog→cat, ε=0.04, k=200)', x=0.01, y=1.04, ha='left', fontsize=10)
    save(fig, out, 'F14_trusted_threshold_ablation', made)


def table(R):
    rows = []
    for n, label in CONDITIONS:
        if not R.res(n): continue
        cells = {}
        for arm in HEAT_ARMS:
            v = R.arm(n, arm)
            if v: cells[arm] = dict(est=v['source_correct']['estimate'], ci=v['source_correct']['ci95'], seeds=len(v['train_seeds']))
        rows.append(dict(name=n, label=label, effective=R.effective(n), cells=cells, concept=R.res(n)['concept']))
    return rows


def fig15(rows, out, made, missing):
    if not rows: missing.append('F15 attack x defense heatmap (no results)'); return
    M = np.full((len(rows), len(HEAT_ARMS)), np.nan)
    for i, r in enumerate(rows):
        for j, arm in enumerate(HEAT_ARMS):
            if arm in r['cells']: M[i, j] = r['cells'][arm]['est']
    cmap = LinearSegmentedColormap.from_list('acc', ['#eb6834', '#f6efe6', '#2a78d6']); cmap.set_bad('#f4f3f0')
    fig, ax = plt.subplots(figsize=(1.0 + 0.78 * len(HEAT_ARMS), 0.9 + 0.42 * len(rows)))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto'); ax.grid(False)
    for i, r in enumerate(rows):
        for j, arm in enumerate(HEAT_ARMS):
            c = r['cells'].get(arm)
            if c is None: ax.text(j, i, '–', ha='center', va='center', fontsize=7, color=GRAY); continue
            ax.text(j, i, f"{c['est']:.2f}" + ('' if c['seeds'] >= 3 else '¹' if c['seeds'] == 1 else f"^{c['seeds']}"),
                    ha='center', va='center', fontsize=7.5, color=INK)
    ylab = [r['label'] + ('  (attack ineffective)' if r['effective'] is False else '') for r in rows]
    ax.set_yticks(range(len(rows)), ylab, fontsize=7.5); ax.set_xticks(range(len(HEAT_ARMS)), [ARM_SHORT[a] for a in HEAT_ARMS], rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title('F15  Source-concept accuracy under each attack (blue = defended, orange = poisoned); k=200;\n'
                 'mean over training seeds (3 unless marked ¹ = 1 seed); – = not run', loc='left', fontsize=9)
    save(fig, out, 'F15_attack_defense_heatmap', made)


def fig16(R, out, made, missing):
    names = [(n, l) for n, l in CONDITIONS if R.row(n).get('status') == 'measured']
    if not names: missing.append('F16 paired-difference forest (results_tradeoff.json missing)'); return
    keys = [('K_minus_C', 'RTDW − Poisoned', col('K')), ('G_minus_C', 'JPEG − Poisoned', col('G')),
            ('K_minus_G', 'RTDW − JPEG', INK)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 0.35 * len(names) + 1.4), sharey=True); fig.subplots_adjust(wspace=0.08)
    for ax, (k, title, c) in zip(axs, keys):
        for i, (n, _) in enumerate(names):
            v = R.row(n).get(k)
            if not v: continue
            lo, hi = v['ci95']; sig = lo > 0 or hi < 0
            ax.errorbar(v['estimate'], i, xerr=[[v['estimate'] - lo], [hi - v['estimate']]], fmt='o', color=c, ms=4, capsize=2,
                        mfc=c if sig else 'white')
        ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-1.05, 1.05); ax.set_title(title, loc='left', fontsize=9)
        ax.set_xlabel('Δ accuracy, 95% CI')
    seedtag = lambda n: '' if (R.row(n).get('K') or {}).get('seeds', 3) >= 3 else f"  [{R.row(n)['K']['seeds']} seed]"
    axs[0].set_yticks(range(len(names)), [l + seedtag(n) + ('  (attack ineffective)' if R.effective(n) is False else '') for n, l in names], fontsize=7.5)
    axs[0].invert_yaxis()
    fig.suptitle('F16  Paired differences per attack (filled = CI excludes 0); k=200', x=0.01, ha='left', fontsize=10)
    save(fig, out, 'F16_paired_differences_by_attack', made)


def fig17(R, out, made, missing):
    conds = [(n, l) for n, l in CONDITIONS if R.res(n) and any(len((R.arm(n, a) or {}).get('train_seeds', [])) >= 2 for a in 'CGJK')]
    if not conds: missing.append('F17 per-seed accuracy (no multi-seed attack conditions)'); return
    arms = ['C', 'E', 'G', 'J', 'K']
    fig, ax = plt.subplots(figsize=(1.6 * len(conds) + 1.5, 3.4)); w = 0.16
    for i, (n, _) in enumerate(conds):
        for j, arm in enumerate(arms):
            v = R.arm(n, arm)
            if not v: continue
            xpos = i + (j - (len(arms) - 1) / 2) * w; ys = v['per_seed_source_correct']
            ax.scatter([xpos] * len(ys), ys, color=col(arm), s=16, zorder=3, label=ARM_SHORT[arm] if i == 0 or not ax.get_legend_handles_labels()[1].count(ARM_SHORT[arm]) else None)
            ax.hlines(np.mean(ys), xpos - w * 0.4, xpos + w * 0.4, color=col(arm), lw=2)
    ax.set_xticks(range(len(conds)), [l for _, l in conds], rotation=20, ha='right', fontsize=7.5); ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel('source-concept accuracy'); ax.legend(fontsize=7, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax.set_title('F17  Per-training-seed accuracy (dots = seeds, bar = mean)', loc='left', fontsize=10, pad=22)
    save(fig, out, 'F17_per_seed_by_attack', made)


def _run_dir(rep, arm, tag, seed=42):
    """Folder of one arm's run under <reports>/runs/<exp>/, or None. Attack-independent arms (B) live in main."""
    name = f'{arm}_k200_s{seed}' + (f'_{tag}' if tag else '')
    names = [name] + ([f'{arm}_k200_s{seed}'] if arm in ATTACK_INDEPENDENT else [])   # never borrow another attack's run
    for n in names:
        for exp in ('main', 'attacks', 'car', 'trusted'):
            d = rep / 'runs' / exp / n
            if (d / 'samples').is_dir(): return d
    return None


def _sample(d, concept, i):
    """i-th saved test sample for a concept and the primary judge's label for that same image (first generation seed)."""
    files = sorted((d / 'samples').glob(f'*_{concept}_*.jpg'))
    if i >= len(files): return None, None
    f = files[i]; label = None
    cfg, jud = d / 'eval_eval_config.json', d / 'eval_judgments.jsonl'
    if cfg.exists() and jud.exists():
        s0 = json.loads(cfg.read_text())['seeds'][0]
        for l in jud.read_text().splitlines():
            j = json.loads(l) if l.strip() else None
            if j and j['id'] == f.stem and str(j['seed']) == str(s0): label = j['judge_label']; break
    return f, label


def _tile(ax, f, label, concept):
    from PIL import Image
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color('#d6d5d0')
    if f is None: ax.text(0.5, 0.5, 'not run', ha='center', va='center', fontsize=7, color=GRAY, transform=ax.transAxes); return
    ax.imshow(Image.open(f))
    if label:
        ok = label == concept
        ax.text(0.03, 0.04, label, transform=ax.transAxes, fontsize=6.5, color='white', fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc=col('B') if ok else col('C'), ec='none'))


def fig18(R, out, made, missing):
    arms = ['B', 'C', 'E', 'G', 'J', 'K']
    dirs = {a: _run_dir(R.d, a, None) for a in arms}
    if not any(dirs.values()): missing.append('F18 qualitative grid (no reports/runs/*/samples)'); return
    ncol = 5
    fig, axs = plt.subplots(len(arms), ncol, figsize=(1.25 * ncol + 1.3, 1.25 * len(arms) + 0.5)); fig.subplots_adjust(wspace=0.04, hspace=0.06)
    for i, a in enumerate(arms):
        for j in range(ncol):
            f, lab = _sample(dirs[a], 'dog', j) if dirs[a] else (None, None)
            _tile(axs[i, j], f, lab, 'dog')
        axs[i, 0].set_ylabel(ARM_SHORT[a], fontsize=8, rotation=0, ha='right', va='center')
    fig.suptitle('F18  Held-out "dog" test prompts, same noise seed in every row (dog→cat, ε=0.04, k=200, training seed 42);\n'
                 'tag = primary judge label for that image', x=0.01, ha='left', fontsize=9)
    save(fig, out, 'F18_qualitative_dog', made)


def fig19(R, out, made, missing):
    conds = [('dog2cat_eps08', 'dog', 'dog→cat\nε=0.08'), ('dog2cat_adapt0p05', 'dog', 'dog→cat\nadaptive λ=0.05'),
             ('car2cow_eps08', 'car', 'car→cow\nε=0.08')]
    arms = ['C', 'G', 'J', 'K']
    have = [(t, c, l) for t, c, l in conds if any(_run_dir(R.d, a, t) for a in arms)]
    if not have: missing.append('F19 qualitative attacks (no attack samples)'); return
    per = 2
    fig, axs = plt.subplots(len(have), len(arms) * per, figsize=(1.15 * len(arms) * per + 1.3, 1.25 * len(have) + 0.7))
    axs = np.atleast_2d(axs); fig.subplots_adjust(wspace=0.04, hspace=0.08)
    for i, (t, c, l) in enumerate(have):
        for j, a in enumerate(arms):
            d = _run_dir(R.d, a, t)
            for q in range(per):
                f, lab = _sample(d, c, q) if d else (None, None)
                ax = axs[i, j * per + q]; _tile(ax, f, lab, c)
                if i == 0 and q == 0: ax.set_title(ARM_SHORT[a], fontsize=8, loc='left')
        axs[i, 0].set_ylabel(l, fontsize=8, rotation=0, ha='right', va='center')
    fig.suptitle('F19  Where each defense fails (training seed 42; two test prompts per arm; tag = primary judge label)',
                 x=0.01, ha='left', fontsize=9)
    save(fig, out, 'F19_qualitative_attacks', made)


def write_table(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'T1_attack_defense_accuracy.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['condition', 'results_name', 'attack_effective', 'arm', 'accuracy', 'ci95_lo', 'ci95_hi', 'train_seeds'])
        for r in rows:
            for arm in HEAT_ARMS:
                c = r['cells'].get(arm)
                if c: w.writerow([r['label'], r['name'], r['effective'], arm, f"{c['est']:.4f}", f"{c['ci'][0]:.4f}", f"{c['ci'][1]:.4f}", c['seeds']])
    tex = ['% generated by scripts/make_comparison_figures.py from results_*.json; do not edit by hand',
           '\\begin{tabular}{l' + 'c' * len(HEAT_ARMS) + '}', '\\toprule',
           'Attack & ' + ' & '.join(ARM_SHORT[x].replace('†', '$^\\dagger$') for x in HEAT_ARMS) + ' \\\\', '\\midrule']
    for r in rows:
        cells = []
        for arm in HEAT_ARMS:
            c = r['cells'].get(arm)
            cells.append('--' if not c else f"{c['est']:.2f}" + ('' if c['seeds'] >= 3 else f"$^{{{c['seeds']}}}$"))
        lab = r['label'].replace('→', '$\\to$').replace('ε', '$\\epsilon$').replace('λ', '$\\lambda$')
        tex.append(lab + ('$^\\dagger$' if r['effective'] is False else '') + ' & ' + ' & '.join(cells) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}', '% superscript = number of training seeds when < 3; $\\dagger$ = Gate 2 not passed (attack ineffective)']
    (out / 'T1_attack_defense_accuracy.tex').write_text('\n'.join(tex) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports', default=str(ROOT / 'reports'), help='folder holding results_*.json')
    ap.add_argument('--k', type=int, default=200)
    a = ap.parse_args()
    rep = Path(a.reports); out = rep / 'figures'
    R = Reports(rep, a.k); made, missing = [], []
    fig7(R, out, made, missing); fig14(R, out, made, missing)
    rows = table(R); fig15(rows, out, made, missing); fig16(R, out, made, missing); fig17(R, out, made, missing)
    fig18(R, out, made, missing); fig19(R, out, made, missing)
    if rows: write_table(rows, out); made.append('T1_attack_defense_accuracy (.csv, .tex)')
    else: missing.append('T1 (no results)')
    lines = ['# Comparison figures (generated by scripts/make_comparison_figures.py from measured results)', '', '## Drawn', '']
    lines += [f'- `{m}`' for m in made] + ['', '## Not drawn', ''] + [f'- {m}' for m in missing]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'COMPARISON.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
