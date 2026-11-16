"""Regression tests for the data, statistics and evaluation pipeline. CPU only."""
import errno, json, os, shutil, subprocess, sys, textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import run_arm as RA  # noqa: E402
import analyze_results as AR  # noqa: E402
import evaluate_generations as EG  # noqa: E402
import nightshade_poisons as NP  # noqa: E402
import dirty_label_poisons as DL  # noqa: E402
from paired_dataset import select_poison_ids, selection_from_manifest  # noqa: E402


# ---------------------------------------------------------------- R1: a failed arm can no longer be lost
def _fake_runner(tmp_path):
    r = tmp_path / 'fake_arm.sh'
    r.write_text(textwrap.dedent('''\
        #!/bin/bash
        echo "$@" >> "$FAKE_LOG"
        arm=""; while [[ $# -gt 0 ]]; do [[ $1 == --arm ]] && arm=$2; shift; done
        sleep 0.2
        [[ $arm == C ]] && exit 3
        exit 0
        '''))
    r.chmod(0o755)
    return r


def _run_arms(tmp_path, arms, gpus):
    env = dict(os.environ, POISON_STORE=str(tmp_path / 'store'), EXP='t', ARMS=arms, PR_FAKE_GPUS=str(gpus),
               PR_ARM_RUNNER=str(_fake_runner(tmp_path)), PR_LAUNCH_GAP='0', PR_POLL_SECONDS='0.1',
               SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID='99', FAKE_LOG=str(tmp_path / 'calls.txt'))
    return subprocess.run(['bash', str(ROOT / 'scripts/pipeline/arms.sh')], env=env, capture_output=True, text=True)


def test_arms_job_fails_when_an_early_arm_on_a_reused_gpu_fails(tmp_path):
    # one GPU: C runs first and fails, then B reuses the slot and succeeds -> the job must still fail
    r = _run_arms(tmp_path, 'C:200:42 B:200:42', 1)
    assert r.returncode == 1, r.stdout + r.stderr
    status = (tmp_path / 'logs/arm-status-99.tsv').read_text().splitlines()
    codes = {l.split('\t')[0]: l.split('\t')[1] for l in status}
    assert codes == {'C:200:42': '3', 'B:200:42': '0'}


def test_arms_job_succeeds_and_passes_per_arm_attack_tag(tmp_path):
    r = _run_arms(tmp_path, 'B:200:42 K:200:42:dog2cat_adapt4', 2)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = (tmp_path / 'calls.txt').read_text()
    assert '--arm K' in calls and '--attack-tag dog2cat_adapt4' in calls
    assert '--attack-tag' not in [l for l in calls.splitlines() if '--arm B' in l][0]


# ---------------------------------------------------------------- R2: per-product locks, Lustre-safe fallback
def test_dirlock_mkdir_fallback_and_stale_lock(tmp_path, monkeypatch):
    import fcntl
    def nosys(*a, **k): raise OSError(errno.ENOSYS, 'flock not supported')
    monkeypatch.setattr(fcntl, 'flock', nosys)
    target = tmp_path / 'poisoned_k200_d42'
    stale = tmp_path / '.lock.poisoned_k200_d42.d'; stale.mkdir()
    (stale / 'owner.json').write_text(json.dumps(dict(host='other', pid=1, job='an-earlier-job')))
    monkeypatch.setenv('SLURM_JOB_ID', 'now')
    with RA.DirLock(target) as L:
        assert L.mkdir_lock.exists() and json.loads((L.mkdir_lock / 'owner.json').read_text())['job'] == 'now'
    assert not stale.exists()


def test_dirlock_flock_is_per_product(tmp_path):
    with RA.DirLock(tmp_path / 'a'):
        with RA.DirLock(tmp_path / 'b'):   # a different product must not block
            pass


# ---------------------------------------------------------------- R3: new arms O / LF / I bookkeeping
def _export_dir(root, poisoned=(0, 1)):
    tr = root / 'train'; tr.mkdir(parents=True)
    rows = [dict(file_name=f'{i}.png', text='a dog', id=f'id{i}', concept='dog' if i < 4 else 'cat',
                 is_poisoned=i in poisoned) for i in range(6)]
    (tr / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return tr


def test_drop_lowest_is_exact_and_deterministic():
    sc = {'a': 0.5, 'b': 0.1, 'c': 0.1, 'd': 0.9, 'e': 0.3, 'f': 0.7, 'g': 0.2, 'h': 0.8, 'i': 0.4, 'j': 0.6}
    w = RA.drop_lowest(sc, 0.2)
    assert sorted(k for k, v in w.items() if v == 0) == ['b', 'c'] and sum(w.values()) == 8
    assert RA.drop_lowest(sc, 0.0) == {k: 1.0 for k in sc}


def test_oracle_weights_and_filter_eval(tmp_path):
    src = _export_dir(tmp_path / 'p')
    w = json.loads(RA.oracle_weights(src, tmp_path / 'w.json').read_text())['weights']
    assert w == {'id0': 0.0, 'id1': 0.0, 'id2': 1.0, 'id3': 1.0, 'id4': 1.0, 'id5': 1.0}
    kept = tmp_path / 'k' / 'train'; kept.mkdir(parents=True)
    rows = [json.loads(l) for l in (src / 'metadata.jsonl').read_text().splitlines()]
    (kept / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows if r['id'] not in ('id0', 'id4')))
    ev = RA.filter_eval(src, kept)
    assert ev['poisons_removed'] == 1 and ev['poison_recall'] == 0.5 and ev['clean_removed'] == 1
    assert ev['clean_removed_by_concept'] == {'cat': 1}


def test_stale_eval_protocol_is_archived_not_mixed(tmp_path):
    ev = tmp_path / 'eval'; ev.mkdir()
    (ev / 'generations.jsonl').write_text('{}\n'); (ev / 'eval_config.json').write_text(json.dumps(dict(seeds=[11, 22, 33, 44])))
    assert RA.archive_stale_eval(tmp_path, RA.GEN_SEEDS) == tmp_path / 'eval_v4seeds'
    assert not ev.exists() and (tmp_path / 'eval_v4seeds/generations.jsonl').exists()
    ev.mkdir(); (ev / 'generations.jsonl').write_text('{}\n')
    (ev / 'eval_config.json').write_text(json.dumps(dict(seeds=[int(s) for s in RA.GEN_SEEDS])))
    assert RA.archive_stale_eval(tmp_path, RA.GEN_SEEDS) is None and ev.exists()
    assert RA.GEN_SEEDS[:4] == ['11', '22', '33', '44']   # protocol v1 is an exact subset of v2


def test_run_names_keep_variants_distinct():
    assert RA.run_name('K', 200, 42) == 'K_k200_s42'
    assert RA.run_name('K', 200, 42, True) == 'K_k200_s42_utau'
    assert RA.run_name('C', 200, 42, attack_tag='dog2cat_adapt4') == 'C_k200_s42_dog2cat_adapt4'
    assert RA.run_name('C', 200, 43, data_seed=7) == 'C_k200_s43_d7'


# ---------------------------------------------------------------- R4: attack variants poison the SAME records
def _manifest(path, ids_with_poison, n=30):
    rows = [dict(id=f'd{i:02d}', split='train', concept='dog', poison_path=(f'p{i}.png' if i in ids_with_poison else None))
            for i in range(n)]
    rows.append(dict(id='t0', split='test', concept='dog'))
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows)); return path


def test_variant_selection_matches_official_export(tmp_path):
    off = _manifest(tmp_path / 'official.jsonl', set(range(30)))           # official: every eligible record poisoned
    want = selection_from_manifest(off, 'dog', 10, 42)
    assert want == select_poison_ids([f'd{i:02d}' for i in range(30)], 10, 42)
    # a variant generated for exactly those records reproduces the official k=10 selection ...
    var = _manifest(tmp_path / 'variant.jsonl', {int(i[1:]) for i in want})
    assert sorted(selection_from_manifest(var, 'dog', 10, 42)) == sorted(want)
    NP.check_match(type('A', (), dict(official=off, variant=var, source='dog', k=10, seed=42))())
    # ... the legacy "first N ids" rule does not, and is rejected
    legacy = _manifest(tmp_path / 'legacy.jsonl', set(range(10)))
    with pytest.raises(SystemExit):
        NP.check_match(type('A', (), dict(official=off, variant=legacy, source='dog', k=10, seed=42))())


def test_generate_restricts_rows_to_matched_ids(tmp_path):
    off = _manifest(tmp_path / 'official.jsonl', set(range(30)))
    rows = [dict(id=f'd{i:02d}') for i in range(30)]
    a = type('A', (), dict(match_manifest=off, match_k=5, match_seed=42, source='dog', limit=0))()
    got = NP.restrict_rows(rows, a)
    assert sorted(r['id'] for r in got) == sorted(selection_from_manifest(off, 'dog', 5, 42))


def test_dirty_label_prefers_unused_images_then_train_duplicates():
    pool = [dict(id=f'u{i}', concept='cat', split='pool') for i in range(3)] + [dict(id='c0', concept='cat', split='pool')]
    clean = [dict(id='c0', concept='cat', split='train'), dict(id='c1', concept='cat', split='train'),
             dict(id='c2', concept='cat', split='trusted'), dict(id='v0', concept='cat', split='test')]
    pick = DL.pick_target_images(pool, clean, 'cat', 5, seed=1)
    assert [o for _, o in pick] == ['unused_pool'] * 3 + ['train_duplicate'] * 2
    assert {i for i, _ in pick} == {'u0', 'u1', 'u2', 'c0', 'c1'}      # never trusted / test images
    with pytest.raises(SystemExit):
        DL.pick_target_images(pool, clean, 'cat', 6, seed=1)


# ---------------------------------------------------------------- R5: evaluation metrics
def test_prdc_identical_and_disjoint_sets():
    rng = np.random.default_rng(0); x = rng.normal(size=(60, 8))
    q = EG.prdc(x, x.copy())
    assert q['precision'] == 1 and q['recall'] == 1 and q['coverage'] == 1
    far = EG.prdc(x, x + 100)
    assert far['precision'] == 0 and far['recall'] == 0 and far['coverage'] == 0 and far['density'] == 0
    assert EG.prdc(x[:3], x[:3])['precision'] is None


def test_cohen_kappa_and_group_label():
    assert EG.cohen_kappa(list('aabb'), list('aabb')) == pytest.approx(1.0)
    assert EG.cohen_kappa(list('aabb'), list('abab')) == pytest.approx(0.0)
    p = np.zeros(1000); p[EG.GROUPS['dog'][0]] = 0.10; p[EG.GROUPS['cat'][0]] = 0.05
    assert EG.group_label(p)[0] == 'other'          # below the 0.15 mass threshold
    p[EG.GROUPS['dog'][1]] = 0.10
    assert EG.group_label(p)[0] == 'dog'


# ---------------------------------------------------------------- R6: statistics and gates
def test_seed_paired_bootstrap_is_tighter_and_reports_per_seed():
    rng = np.random.default_rng(3); prompts = [f'p{i}' for i in range(20)]
    base = {s: rng.normal(0.5, 0.2) for s in (42, 43, 44)}          # large run-to-run variation, shared by both arms
    A = [{p: base[s] + 0.1 + rng.normal(0, 0.01) for p in prompts} for s in (42, 43, 44)]
    B = [{p: base[s] + rng.normal(0, 0.01) for p in prompts} for s in (42, 43, 44)]
    pd = AR.paired(A, B, [42, 43, 44], [42, 43, 44], reps=2000)
    un = AR.paired(A, B, [42, 43, 44], [44, 43, 45], reps=2000)    # different seed sets -> unpaired
    assert pd['seed_paired'] and not un['seed_paired']
    assert pd['estimate'] == pytest.approx(0.1, abs=0.01) and pd['ci95'][0] > 0.05
    assert (pd['ci95'][1] - pd['ci95'][0]) < (un['ci95'][1] - un['ci95'][0])
    assert set(pd['per_seed']) == {42, 43, 44}
    single = AR.paired(A[:1], B[:1], [42], [42], reps=500)
    assert 'SINGLE training run' in single['note']


def _fake_run(root, exp, name, arm, seed, acc_dog, asr, acc_other=1.0, tag='dog2cat_eps04', kid=0.03, collapse=False):
    d = root / 'reports/runs' / exp / name; d.mkdir(parents=True)
    rng = np.random.default_rng(seed * 7 + len(arm))
    ref = root / 'reports/real_inception_ref16.npz'
    if not ref.exists():
        np.savez_compressed(ref, f=np.random.default_rng(0).normal(size=(60, 16)).astype(np.float16), c=np.array(['dog'] * 60))
    ids = np.repeat([f'test_dog_{p}' for p in range(20)], 4); seeds = np.tile([11, 22, 33, 44], 20)
    clip = rng.normal(size=(80, 8))
    if collapse: clip = np.repeat(clip[::4], 4, axis=0)                 # every seed of a prompt gives the same image
    np.savez_compressed(d / 'eval_features_test_attacked.npz', inc=rng.normal(size=(80, 16)).astype(np.float16),
                        clip=clip.astype(np.float16), id=ids, seed=seeds, concept=np.array(['dog'] * 80))
    J = []
    for p in range(20):
        for g in (11, 22, 33, 44):
            ok = rng.random() < acc_dog
            J.append(dict(id=f'test_dog_{p}', split='test', seed=g, source_concept='dog', target_concept='cat',
                          source_correct=ok, attacker_target=(not ok) and rng.random() < asr / max(1e-9, 1 - acc_dog),
                          clip_source_correct=ok, clip_attacker_target=False, judge2_source_correct=ok))
    for p in range(8):
        for g in (11, 22, 33, 44):
            ok = rng.random() < acc_other
            J.append(dict(id=f'test_cat_{p}', split='test', seed=g, source_concept='cat', target_concept='none',
                          source_correct=ok, attacker_target=False, clip_source_correct=ok, clip_attacker_target=False,
                          judge2_source_correct=ok))
    (d / 'eval_judgments.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in J))
    (d / 'eval_metrics.json').write_text(json.dumps(dict(kid_all_test=kid, per_split_concept={
        'test:dog': dict(kid=kid, kid_std=0.005, precision=0.7, recall=0.6, density=0.9, coverage=0.8, diversity=0.2, judge_kappa=0.9)})))
    (d / 'arm_summary.json').write_text(json.dumps(dict(arm=arm, k=200, seed=seed, data_seed=42, attack_tag=tag, prep_seconds=0)))
    (d / 'train_metrics.json').write_text(json.dumps(dict(gpu_hours=1, train_exposures=24000, trusted_exposures=0,
                                                          denoiser_forward_examples=24000, peak_vram_bytes=2**34, optimizer_steps=1000)))


def test_analysis_gates_end_to_end(tmp_path, monkeypatch):
    for s in (42, 43, 44):
        _fake_run(tmp_path, 'main', f'B_k200_s{s}', 'B', s, 0.98, 0.0)
        _fake_run(tmp_path, 'main', f'C_k200_s{s}', 'C', s, 0.12, 0.55)
        _fake_run(tmp_path, 'main', f'E_k200_s{s}', 'E', s, 0.10, 0.60)
        _fake_run(tmp_path, 'main', f'K_k200_s{s}', 'K', s, 0.96, 0.0)
        _fake_run(tmp_path, 'main', f'O_k200_s{s}', 'O', s, 0.97, 0.0)
    _fake_run(tmp_path, 'trusted', 'K_k200_s42_utau', 'K', 42, 0.5, 0.2)    # a variant must NOT become a 4th K seed
    monkeypatch.setattr(AR, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--exp', 'main', 'trusted', '--name', 'dog', '--reps', '300'])
    AR.main()
    res = json.loads((tmp_path / 'reports/results_dog.json').read_text())
    assert len(res['arms']['K@k200']['train_seeds']) == 3 and 'K-utau@k200' in res['arms']
    assert res['gates']['gate2@k200']['passed']
    g3 = res['gates']['gate3@k200:K']
    assert set(g3['controls']) == {'C', 'E'} and g3['all_controls_beaten'] and g3['bonferroni_level'] == pytest.approx(0.975)
    assert g3['nontarget_vs_B']['passed'] and g3['quality_within_B'] is True and g3['passed']
    assert g3['quality_vs_B']['n_prompts'] == 20
    assert 'O' not in g3['controls'] and g3['gap_to_oracle'] is not None
    assert res['deltas']['K@k200 - C@k200']['source_correct']['seed_paired']
    assert 'O' not in res['not_run'] and 'LF' in res['not_run'] and 'L' in res['unavailable']
    # Gate 4 summary: adaptive results absent -> not_run, so Gate 4 cannot pass
    import attack_tradeoff as AT
    monkeypatch.setattr(AT, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--names', 'dog', 'dog2cat_adapt4'])
    AT.main()
    t = json.loads((tmp_path / 'reports/results_tradeoff.json').read_text())
    assert t['gate4']['N10_dog_3seeds'] is True and t['gate4']['passed'] is False
    assert t['gate4']['N7_adaptive']['dog2cat_adapt4'] == 'not_run'
    assert 'Gate 4' in (tmp_path / 'reports/GATES.md').read_text()


def test_gate3_fails_when_candidate_collapses_diversity(tmp_path, monkeypatch):
    for s in (42, 43, 44):
        _fake_run(tmp_path, 'main', f'B_k200_s{s}', 'B', s, 0.98, 0.0)
        _fake_run(tmp_path, 'main', f'C_k200_s{s}', 'C', s, 0.12, 0.55)
        _fake_run(tmp_path, 'main', f'K_k200_s{s}', 'K', s, 0.96, 0.0, collapse=True)
    monkeypatch.setattr(AR, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--exp', 'main', '--name', 'dog', '--reps', '300'])
    AR.main()
    g3 = json.loads((tmp_path / 'reports/results_dog.json').read_text())['gates']['gate3@k200:K']
    assert g3['all_controls_beaten'] and g3['quality_vs_B']['diversity_diff']['ci95'][1] < 0
    assert g3['quality_within_B'] is False and not g3['passed']


def test_gate3_fails_when_candidate_does_not_beat_replay(tmp_path, monkeypatch):
    for s in (42, 43, 44):
        _fake_run(tmp_path, 'main', f'B_k200_s{s}', 'B', s, 0.98, 0.0)
        _fake_run(tmp_path, 'main', f'C_k200_s{s}', 'C', s, 0.12, 0.55)
        _fake_run(tmp_path, 'main', f'E_k200_s{s}', 'E', s, 0.90, 0.02)
        _fake_run(tmp_path, 'main', f'K_k200_s{s}', 'K', s, 0.90, 0.02)
    monkeypatch.setattr(AR, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--exp', 'main', '--name', 'dog', '--reps', '300'])
    AR.main()
    g3 = json.loads((tmp_path / 'reports/results_dog.json').read_text())['gates']['gate3@k200:K']
    assert g3['controls']['C']['passed'] and not g3['controls']['E']['passed'] and not g3['passed']


# ---------------------------------------------------------------- R7: blinded human review
