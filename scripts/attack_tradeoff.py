#!/usr/bin/env python3
"""Cross-experiment summary: adaptive-attack trade-off (N7), attack variants (N6, N8) and the Gate 4 verdict.

Reads reports/results_<name>.json written by analyze_results.py; writes reports/results_tradeoff.json and
reports/GATES.md. Only measured values; a missing result file is reported as not_run.

N7 prediction (pre-registered): if the attacker lowers the round-trip drift of its
poisons (larger lambda), the undefended attack weakens. Either outcome is reported: the attack keeps its strength
while drift falls (defense broken) or loses it (trade-off holds).
Gate 4 (pre-registered): the K candidate passes Gate 3 on dog with >= 3 training seeds (N10) and on car (N5), and
under every adaptive lambda either K - C has a 95% CI above 0 or the adaptive attack itself fails Gate 2 (N7).

Post-hoc addition (NOT part of Gate 4): the seed-42 main run showed G (JPEG q75) recovering as fully
as K. Every variant now also reports G - C and K - G, and dog2cat_adaptjpeg (attacker optimising through G's
JPEG) gets its own verdict for G and for K, so the paper can say which defense survives which adaptive attacker.
Post-hoc addition 2 (after the attacks phase): lambda 0 poisons and lambda 1 does not, so the grid is
extended with lambda in {0.05, 0.1, 0.25, 0.5}. Gate 4 keeps its pre-registered grid; `gate4_extended` applies the same
rule to the union of both grids and is reported next to it. A defense that fails only on the new values fails there.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / 'reports' / f'results_{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def arm(res, aid, k=200):
    for v in res['arms'].values():
        if v['arm_id'] == aid and v['k'] == k: return v
    return None


def row(name, res):
    if res is None: return dict(name=name, status='not_run')
    out = dict(name=name, attack_tag=res.get('attack_tag'), status='measured')
    for aid in ('B', 'C', 'G', 'J', 'K', 'RS', 'E', 'O', 'GK'):
        v = arm(res, aid)
        if v: out[aid] = dict(acc=v['source_correct']['estimate'], acc_ci=v['source_correct']['ci95'],
                              asr=v['attacker_target']['estimate'], seeds=len(v['train_seeds']))
    dk = next((d for k, d in res.get('drift', {}).items() if k.startswith('K@k200') and ':' not in k), None)
    if dk and dk.get('poison'):
        out['poison_drift_median'] = float(np.median(dk['poison'])); out['clean_drift_median'] = float(np.median(dk['clean']))
        out['drift_auroc'] = dk['auroc']
    kc = res['deltas'].get('K@k200 - C@k200')
    if kc: out['K_minus_C'] = dict(estimate=kc['source_correct']['estimate'], ci95=kc['source_correct']['ci95'])
    for key, name in (('G@k200 - C@k200', 'G_minus_C'), ('K@k200 - G@k200', 'K_minus_G'), ('GK@k200 - C@k200', 'GK_minus_C')):
        d = res['deltas'].get(key)
        if d: out[name] = dict(estimate=d['source_correct']['estimate'], ci95=d['source_correct']['ci95'])
    out['gate2'] = res.get('gates', {}).get('gate2@k200')
    out['gate3_K'] = res.get('gates', {}).get('gate3@k200:K')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--adaptive', nargs='+', default=['dog2cat_adapt0', 'dog2cat_adapt1', 'dog2cat_adapt4', 'dog2cat_adapt16'])
    ap.add_argument('--adaptive-extended', nargs='+',
                    default=['dog2cat_adapt0p05', 'dog2cat_adapt0p1', 'dog2cat_adapt0p25', 'dog2cat_adapt0p5'])
    a = ap.parse_args()
    rows = {n: row(n, load(n)) for n in a.names + ['car'] if n not in ()}
    dog, car = rows.get('dog', {}), rows.get('car', {})
    g10 = bool(dog.get('gate3_K', {}) and dog['gate3_K'].get('passed') and dog.get('K', {}).get('seeds', 0) >= 3
               and dog.get('C', {}).get('seeds', 0) >= 3)
    g5 = bool(car.get('gate3_K') and car['gate3_K'].get('passed'))
    def n7_of(tags):
        res = {}
        for t in tags:
            r = rows.get(t) or row(t, load(t))
            if r['status'] != 'measured': res[t] = 'not_run'; continue
            g2 = r.get('gate2') or {}
            if g2 and not g2.get('passed'): res[t] = 'attack_ineffective'
            elif r.get('K_minus_C') and r['K_minus_C']['ci95'][0] > 0: res[t] = 'defense_holds'
            elif r.get('K_minus_C'): res[t] = 'DEFENSE_FAILS'
            else: res[t] = 'incomplete'
        return res
    n7 = n7_of(a.adaptive)
    n7x = n7_of(a.adaptive_extended)
    ok = lambda d: bool(d) and all(v in ('defense_holds', 'attack_ineffective') for v in d.values())
    def verdict(r, key):
        if r['status'] != 'measured': return 'not_run'
        g2 = r.get('gate2') or {}
        if g2 and not g2.get('passed'): return 'attack_ineffective'
        if not r.get(key): return 'incomplete'
        return 'defense_holds' if r[key]['ci95'][0] > 0 else 'DEFENSE_FAILS'
    rj = rows.get('dog2cat_adaptjpeg') or row('dog2cat_adaptjpeg', load('dog2cat_adaptjpeg'))
    jpeg_adaptive = dict(G=verdict(rj, 'G_minus_C'), K=verdict(rj, 'K_minus_C'), post_hoc=True)
    gate4 = dict(N10_dog_3seeds=g10, N5_car=g5, N7_adaptive=n7,
                 passed=g10 and g5 and bool(n7) and all(v in ('defense_holds', 'attack_ineffective') for v in n7.values()))
    gate4_extended = dict(N7_adaptive_extended=n7x, passed=g10 and g5 and ok(n7) and ok(n7x), post_hoc=True)
    out = dict(rows=rows, gate4=gate4, gate4_extended=gate4_extended, jpeg_adaptive=jpeg_adaptive)
    (ROOT / 'reports/results_tradeoff.json').write_text(json.dumps(out, indent=2))
    L = ['# Gates (generated by scripts/attack_tradeoff.py from reports/results_*.json; do not edit by hand)', '',
         '| Result file | Gate 2 (B−C > 0) | Gate 3 (K) | K − C [95% CI] | G − C [95% CI] | K − G [95% CI] | poison drift median | drift AUROC |',
         '|---|---|---|---|---|---|---|---|']
    ci = lambda d: '–' if not d else '%+.3f [%.3f, %.3f]' % (d['estimate'], *d['ci95'])
    for n, r in rows.items():
        if r['status'] != 'measured': L.append(f'| {n} | not_run | not_run | – | – | – | – | – |'); continue
        g2 = r.get('gate2'); g3 = r.get('gate3_K'); kc = r.get('K_minus_C')
        L.append(f"| {n} | {'–' if not g2 else ('PASSED' if g2['passed'] else 'not passed')} "
                 f"| {'–' if not g3 else ('PASSED' if g3['passed'] else 'not passed')} "
                 f"| {ci(kc)} | {ci(r.get('G_minus_C'))} | {ci(r.get('K_minus_G'))} "
                 f"| {r.get('poison_drift_median', '–') if not isinstance(r.get('poison_drift_median'), float) else '%.3f' % r['poison_drift_median']} "
                 f"| {r.get('drift_auroc', '–') if not isinstance(r.get('drift_auroc'), float) else '%.4f' % r['drift_auroc']} |")
    L += ['', f"**Gate 4**: {'PASSED' if gate4['passed'] else 'not passed'} — N10 dog ≥3 seeds: {g10}; N5 car: {g5}; "
          f"N7 adaptive: {', '.join(f'{k}={v}' for k, v in n7.items()) or 'not_run'}",
          '', f"**Gate 4 with the extended lambda grid** (post-hoc): {'PASSED' if gate4_extended['passed'] else 'not passed'} — "
          f"{', '.join(f'{k}={v}' for k, v in n7x.items()) or 'not_run'}",
          '', f"**JPEG-aware attacker** (dog2cat_adaptjpeg; post-hoc, not part of Gate 4): G {jpeg_adaptive['G']}, K {jpeg_adaptive['K']}"]
    (ROOT / 'reports/GATES.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
