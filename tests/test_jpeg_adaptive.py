"""Post-hoc JPEG additions: JPEG-aware adaptive attacker, G on every attack variant, tradeoff verdicts."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'tests'))
from test_analysis_and_eval import _fake_run  # noqa: E402


def test_attack_jpeg_is_exactly_the_defenders_jpeg():
    from PIL import Image
    import adaptive_nightshade as AN, run_arm
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    assert np.array_equal(np.asarray(AN.jpeg(im, 75)), np.asarray(run_arm.jpeg75(im)))
    # uint8 -> [-1,1] tensor -> uint8 is lossless, so the attacker's forward pass sees the saved PNG exactly
    assert np.array_equal(np.asarray(AN.to_pil(AN.to_tensor(im))), np.asarray(im))


def test_bpda_forward_is_jpeg_and_backward_is_identity():
    torch = pytest.importorskip('torch')
    import adaptive_nightshade as AN
    x = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_(True)
    y = AN.jpeg_bpda(x, 75)
    want = torch.stack([AN.to_tensor(AN.jpeg(AN.to_pil(xi), 75)) for xi in x])
    assert torch.allclose(y.detach(), want)
    w = torch.randn_like(x); (y * w).sum().backward()
    assert torch.allclose(x.grad, w)
    # quality matters: a different quality gives a different forward
    assert not torch.allclose(AN.jpeg_bpda(x, 30).detach(), y.detach())


def _phase_vars():
    src = (ROOT / 'scripts/pipeline/experiments.sh').read_text()
    seg = src[src.index('S="42"'):src.index('declare -A PHASE')]
    out = subprocess.run(['bash', '-c', seg + '\necho "$seed_arms"; echo "$car_seed_arms"; echo "$attack_arms"'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [l.split() for l in out]


def test_g_runs_on_every_attack_variant_and_on_seeds_and_car():
    seeds, car_seeds, attacks = _phase_vars()
    assert {'G:200:43', 'G:200:44'} <= set(seeds) and {'G:200:43', 'G:200:44'} <= set(car_seeds)
    tags = {a.split(':')[3] for a in attacks}
    assert 'dog2cat_adaptjpeg' in tags
    for t in tags:
        arms = {a.split(':')[0] for a in attacks if a.split(':')[3] == t}
        assert {'C', 'G', 'J', 'K'} <= arms, t
    assert len(attacks) == len(set(attacks))
    src = (ROOT / 'scripts/pipeline/experiments.sh').read_text()
    assert 'dog2cat_adaptjpeg:dog:cat:0.04:adaptive_jpeg:200:0' in src
    assert 'G:200:$S RS:200:$S V:200:$S|--attack-tag car2cow_eps04' in src


def test_poisons_sh_passes_jpeg_only_for_adaptive_jpeg(tmp_path):
    store = tmp_path / 'store'; (store / 'corpus').mkdir(parents=True)
    for f in ('manifest_clean.jsonl', 'split_report.json', 'corpus_audit.json', 'manifest_dog2cat_eps04.jsonl'):
        (store / 'corpus' / f).write_text('{}\n')
    t = store / 'poisons/dog2cat_eps04/work_rank0'; t.mkdir(parents=True); (t / 'target.png').write_bytes(b'x')
    bin_ = tmp_path / 'bin'; bin_.mkdir()
    (bin_ / 'nvidia-smi').write_text('#!/bin/bash\n[[ $1 == -L ]] && echo "GPU 0" || echo "0, fake, 1 MiB"\n')
    calls = tmp_path / 'calls.txt'
    (bin_ / 'fakepy').write_text(f'#!/bin/bash\necho "$*" >> {calls}\n')
    for f in bin_.iterdir(): f.chmod(0o755)
    env = dict(os.environ, PATH=f'{bin_}:{os.environ["PATH"]}', POISON_STORE=str(store), PR_PYTHON=str(bin_ / 'fakepy'),
               SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID='testjpeg',
               POISON_SPECS='dog2cat_adapt4:dog:cat:0.04:adaptive:200:4 dog2cat_adaptjpeg:dog:cat:0.04:adaptive_jpeg:200:0')
    try:
        r = subprocess.run(['bash', str(ROOT / 'scripts/pipeline/poisons.sh')], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        gen = [l for l in calls.read_text().splitlines() if 'adaptive_nightshade.py' in l]
        a4 = next(l for l in gen if '--tag dog2cat_adapt4 ' in l); aj = next(l for l in gen if '--tag dog2cat_adaptjpeg ' in l)
        assert '--jpeg' not in a4 and '--lam 4' in a4
        assert '--jpeg 75' in aj and '--lam 0' in aj and '--match-k 200' in aj
    finally:
        import shutil; shutil.rmtree(ROOT / 'reports/poisons-testjpeg', ignore_errors=True)


def test_tradeoff_reports_g_and_jpeg_attacker_verdicts(tmp_path, monkeypatch):
    import analyze_results as AR, attack_tradeoff as AT
    tag = 'dog2cat_adaptjpeg'
    for s in (42, 43, 44):
        _fake_run(tmp_path, 'main', f'B_k200_s{s}', 'B', s, 0.98, 0.0)
        _fake_run(tmp_path, 'attacks', f'C_k200_s{s}_{tag}', 'C', s, 0.10, 0.60, tag=tag)
        _fake_run(tmp_path, 'attacks', f'G_k200_s{s}_{tag}', 'G', s, 0.12, 0.55, tag=tag)   # JPEG no longer helps
        _fake_run(tmp_path, 'attacks', f'K_k200_s{s}_{tag}', 'K', s, 0.95, 0.0, tag=tag)
    monkeypatch.setattr(AR, 'ROOT', tmp_path); monkeypatch.setattr(AT, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--exp', 'attacks', 'main', '--attack-tag', tag, '--name', tag, '--reps', '300'])
    AR.main()
    monkeypatch.setattr(sys, 'argv', ['x', '--names', 'dog', tag])
    AT.main()
    t = json.loads((tmp_path / 'reports/results_tradeoff.json').read_text())
    r = t['rows'][tag]
    assert r['G']['acc'] < 0.3 and 'G_minus_C' in r and 'K_minus_G' in r
    assert r['K_minus_G']['ci95'][0] > 0
    assert t['jpeg_adaptive'] == dict(G='DEFENSE_FAILS', K='defense_holds', post_hoc=True)
    assert set(t['gate4']) == {'N10_dog_3seeds', 'N5_car', 'N7_adaptive', 'passed'}   # Gate 4 definition unchanged
    md = (tmp_path / 'reports/GATES.md').read_text()
    assert 'K − G' in md and 'JPEG-aware attacker' in md and 'G DEFENSE_FAILS, K defense_holds' in md


def test_attacks2_lambda_sweep_and_eps08_seeds():
    src = (ROOT / 'scripts/pipeline/experiments.sh').read_text()
    seg = src[src.index('S="42"'):src.index('declare -A PHASE')]
    out = subprocess.run(['bash', '-c', seg + '\necho "$attack2_poisons"; echo "$attack2_arms"; echo "$attack_arms"'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    poisons, arms, old = (l.split() for l in out)
    assert poisons == ['dog2cat_adapt0p05:dog:cat:0.04:adaptive:200:0.05', 'dog2cat_adapt0p1:dog:cat:0.04:adaptive:200:0.1',
                       'dog2cat_adapt0p25:dog:cat:0.04:adaptive:200:0.25', 'dog2cat_adapt0p5:dog:cat:0.04:adaptive:200:0.5']
    assert len(arms) == 24 and len(set(arms)) == 24 and not set(arms) & set(old)   # nothing re-run from `attacks`
    for t in ('0p05', '0p1', '0p25', '0p5'):
        assert {a.split(':')[0] for a in arms if a.endswith(f'dog2cat_adapt{t}')} == {'C', 'G', 'J', 'K'}
    assert {a for a in arms if a.endswith('dog2cat_eps08')} == {f'{x}:200:{s}:dog2cat_eps08' for x in 'CGJK' for s in (43, 44)}
    order = src[src.index('ORDER="'):].split('"')[1].split()
    assert order.index('antipoison') < order.index('poisons_attacks2') == order.index('attacks2') - 1
    assert '[poisons_attacks2]="poisons|$attack2_poisons"' in src and '[attacks2]="arms|attacks|$attack2_arms|"' in src


def test_gate4_extended_reports_a_failure_on_the_new_lambdas(tmp_path, monkeypatch):
    import analyze_results as AR, attack_tradeoff as AT
    for s in (42, 43, 44):
        _fake_run(tmp_path, 'main', f'B_k200_s{s}', 'B', s, 0.98, 0.0)
    for tag, c_acc, k_acc in (('dog2cat_adapt0p1', 0.10, 0.12), ('dog2cat_adapt0p5', 0.97, 0.97)):
        for s in (42, 43, 44):
            _fake_run(tmp_path, 'attacks', f'C_k200_s{s}_{tag}', 'C', s, c_acc, 0.6 if c_acc < 0.5 else 0.0, tag=tag)
            _fake_run(tmp_path, 'attacks', f'K_k200_s{s}_{tag}', 'K', s, k_acc, 0.6 if k_acc < 0.5 else 0.0, tag=tag)
        monkeypatch.setattr(AR, 'ROOT', tmp_path)
        monkeypatch.setattr(sys, 'argv', ['x', '--exp', 'attacks', 'main', '--attack-tag', tag, '--name', tag, '--reps', '300'])
        AR.main()
    monkeypatch.setattr(AT, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--names', 'dog', 'dog2cat_adapt0p1', 'dog2cat_adapt0p5'])
    AT.main()
    t = json.loads((tmp_path / 'reports/results_tradeoff.json').read_text())
    n7x = t['gate4_extended']['N7_adaptive_extended']
    assert n7x['dog2cat_adapt0p1'] == 'DEFENSE_FAILS' and n7x['dog2cat_adapt0p5'] == 'attack_ineffective'
    assert n7x['dog2cat_adapt0p05'] == 'not_run' and t['gate4_extended']['passed'] is False
    assert set(t['gate4']['N7_adaptive']) == {'dog2cat_adapt0', 'dog2cat_adapt1', 'dog2cat_adapt4', 'dog2cat_adapt16'}
    assert 'extended lambda grid' in (tmp_path / 'reports/GATES.md').read_text()


def test_car_eps08_phases_match_official_selection_and_are_last():
    src = (ROOT / 'scripts/pipeline/experiments.sh').read_text()
    assert '[poisons_car2]="poisons|car2cow_eps08:car:cow:0.08:official:200"' in src
    assert '[car2]="arms|car|C:200:$S G:200:$S J:200:$S K:200:$S O:200:$S|--attack-tag car2cow_eps08 --concept car"' in src
    order = src[src.index('ORDER="'):].split('"')[1].split()
    assert order.index('poisons_car2') == order.index('car2') - 1 and order.index('attacks2') < order.index('poisons_car2')
    assert len(order) == len(set(order))
    assert '--attack-tag car2cow_eps08 --concept car --target cow --name car_eps08' in src
