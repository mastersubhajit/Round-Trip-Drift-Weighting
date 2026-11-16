"""make_comparison_figures.py: cross-attack figures/tables drawn only from measured results files."""
import csv, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _arm(arm, acc, seeds=3, aid=None, kid=0.025):
    return {f'{aid or arm}@k200': dict(arm=arm, arm_id=aid or arm, k=200, label=arm, train_seeds=list(range(42, 42 + seeds)),
                                       source_correct=dict(estimate=acc, ci95=[max(0, acc - 0.05), min(1, acc + 0.05)]),
                                       per_seed_source_correct=[acc] * seeds, kid_all_test=kid)}


def _res(d, name, arms, concept='dog'):
    a = {}
    for x in arms: a.update(x)
    (d / f'results_{name}.json').write_text(json.dumps(dict(concept=concept, target='cat', arms=a)))


def _run(d):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/make_comparison_figures.py'), '--reports', str(d)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return (d / 'figures/COMPARISON.md').read_text()


def test_draws_every_figure_and_table_from_measured_files(tmp_path):
    base = [_arm('B', 0.98), _arm('C', 0.03), _arm('G', 0.98), _arm('J', 0.99), _arm('K', 0.97)]
    _res(tmp_path, 'dog', base + [_arm('M1', 0.98, 1)])
    _res(tmp_path, 'dog2cat_adapt0', [_arm('B', 0.98), _arm('C', 0.16, 1), _arm('K', 0.96, 1)])
    _res(tmp_path, 'dog2cat_adapt0p05', [_arm('B', 0.98), _arm('C', 0.04, 1), _arm('K', 0.06, 1)])
    _res(tmp_path, 'trusted', base + [_arm('K', 0.988, 1, 'K-utau')])
    rows = {n: dict(status='measured', gate2=dict(passed=True), drift_auroc=au, poison_drift_median=0.1, clean_drift_median=0.02,
                    K=dict(seeds=s), K_minus_C=dict(estimate=0.5, ci95=[0.4, 0.6]))
            for n, au, s in [('dog', 1.0, 3), ('dog2cat_adapt0', 0.99, 1), ('dog2cat_adapt0p05', 0.42, 1)]}
    (tmp_path / 'results_tradeoff.json').write_text(json.dumps(dict(rows=rows)))
    md = _run(tmp_path)
    for f in ('F7_adaptive_tradeoff', 'F14_trusted_threshold_ablation', 'F15_attack_defense_heatmap',
              'F16_paired_differences_by_attack', 'F17_per_seed_by_attack'):
        assert (tmp_path / f'figures/{f}.png').exists() and (tmp_path / f'figures/{f}.pdf').exists(), f
    rows = list(csv.DictReader((tmp_path / 'figures/T1_attack_defense_accuracy.csv').open()))
    got = {(r['results_name'], r['arm']): (float(r['accuracy']), int(r['train_seeds'])) for r in rows}
    assert got[('dog', 'K')] == (0.97, 3) and got[('dog', 'M1')] == (0.98, 1) and got[('dog2cat_adapt0p05', 'K')] == (0.06, 1)
    assert ('dog2cat_adapt0p05', 'G') not in got          # arms that were not run are not invented
    tex = (tmp_path / 'figures/T1_attack_defense_accuracy.tex').read_text()
    assert '0.06$^{1}$' in tex and '--' in tex
    not_drawn = md.split('## Not drawn')[1]
    assert all(l.startswith(('- F18', '- F19')) for l in not_drawn.strip().splitlines())   # only the image grids (no samples here)


def test_missing_inputs_are_listed_not_drawn(tmp_path):
    md = _run(tmp_path)
    assert not list((tmp_path / 'figures').glob('*.png'))
    for f in ('F7', 'F14', 'F15', 'F16', 'F17', 'T1'):
        assert f'- {f}' in md


def test_wired_into_pr_sh_and_make_figures():
    s = (ROOT / 'scripts/pipeline/experiments.sh').read_text()
    assert s.count('scripts/make_comparison_figures.py') == 2 and '  figures)' in s
    m = (ROOT / 'scripts/make_figures.py').read_text()
    assert 'N7 not_run' not in m and 'N4 not_run' not in m
