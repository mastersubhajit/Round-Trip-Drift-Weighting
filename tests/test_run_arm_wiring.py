"""Drive scripts/run_arm.py main() for EVERY arm on CPU with faked subprocesses (no GPU, no models).

Catches wiring errors (wrong paths, missing files, NameErrors in rarely used branches) before they cost GPU hours.
Each faked step creates exactly the files the real step would create; the test then checks the arm's training
input and bookkeeping. Arm H (needs a CUDA img2img pipeline in-process) is the only arm not driven here.
"""
import json, shutil, sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import run_arm as RA  # noqa: E402


def _arg(cmd, flag):
    cmd = list(map(str, cmd)); return cmd[cmd.index(flag) + 1]


def _rows(d):
    return [json.loads(l) for l in (Path(d) / 'metadata.jsonl').read_text().splitlines() if l.strip()]


def _write_rows(d, rows):
    (Path(d) / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


def fake_sh(calls):
    def sh(cmd, log=None):
        cmd = list(map(str, cmd)); calls.append(cmd); prog = Path(cmd[1]).name
        if prog == 'paired_dataset.py':
            out = Path(_arg(cmd, '--out')); arm = _arg(cmd, '--arm'); tr = out / 'train'; tr.mkdir(parents=True)
            k = int(_arg(cmd, '--poison-count')) if arm == 'poisoned' else 0
            concepts = ['dog'] * 6 + ['cat'] * 4 if arm != 'trusted_only' else ['dog', 'cat']
            rows = []
            for i, c in enumerate(concepts):
                f = tr / f'{arm}{i}.png'; Image.new('RGB', (8, 8), (i * 20, 0, 0)).save(f)
                rows.append(dict(file_name=f.name, text=f'a photo of a {c}', id=f'{arm[0]}{i}', concept=c,
                                 is_poisoned=c == 'dog' and i < k, image_sha256='x'))
            _write_rows(tr, rows)
            (out / 'audit.json').write_text(json.dumps(dict(record_provenance=[dict(id=r['id']) for r in rows],
                                                            selected_poison_ids=[r['id'] for r in rows if r['is_poisoned']])))
        elif prog == 'roundtrip_drift.py':
            sub = cmd[2]
            if sub == 'score':
                rows = _rows(_arg(cmd, '--dir'))
                Path(_arg(cmd, '--out')).write_text(''.join(json.dumps(dict(id=r['id'], drift=0.3 if r['is_poisoned'] else 0.02,
                                                                                 is_poisoned=r['is_poisoned'])) + '\n' for r in rows))
            elif sub == 'weights':
                rows = [json.loads(l) for l in Path(_arg(cmd, '--scores')).read_text().splitlines()]
                w = {r['id']: 0.0 if r['drift'] > 0.1 else 1.0 for r in rows}
                Path(_arg(cmd, '--out')).write_text(json.dumps(dict(tau=0.1, weights=w)))
                return json.dumps(dict(tau=0.1)) + '\n'
            elif sub == 'filter':
                src = Path(_arg(cmd, '--dir')); dst = Path(_arg(cmd, '--out')) / 'train'; dst.mkdir(parents=True)
                keep = json.loads(Path(_arg(cmd, '--weights')).read_text())['weights']; kept = []
                for r in _rows(src):
                    if keep.get(r['id'], 1.0) >= 0.5: shutil.copyfile(src / r['file_name'], dst / r['file_name']); kept.append(r)
                _write_rows(dst, kept)
        elif prog == 'loss_filter.py':
            rows = _rows(_arg(cmd, '--dir'))
            Path(_arg(cmd, '--out')).write_text(''.join(json.dumps(dict(id=r['id'], loss=0.9 if r['is_poisoned'] else 0.1)) + '\n' for r in rows))
        elif prog == 'dclr_latents.py':
            out = Path(_arg(cmd, '--out')); out.mkdir(parents=True, exist_ok=True)
            if cmd[2] == 'merge': (out / 'latents.pt').write_bytes(b'fake')
        elif prog == 'noise.py':
            src = Path(_arg(cmd, '--in_dir')); dst = Path(_arg(cmd, '--out_dir')); dst.mkdir(parents=True)
            for f in src.glob('*.png'): shutil.copyfile(f, dst / f.name)
        elif prog == 'noisy_upscale.py':
            src = Path(_arg(cmd, '--in-dir')); dst = Path(_arg(cmd, '--out-dir')); dst.mkdir(parents=True)
            for f in src.glob('*.png'): shutil.copyfile(f, dst / f.name)
        elif prog == 'train_full_unet.py':
            out = Path(_arg(cmd, '--output-dir')); out.mkdir(parents=True)
            for n in ('completed.json', 'metrics.json', 'config.json'): (out / n).write_text('{}')
        elif prog == 'generate_eval.py':
            out = Path(_arg(cmd, '--out')); out.mkdir(parents=True)
            seeds = [int(s) for s in cmd[cmd.index('--seeds') + 1:cmd.index('--steps')]]
            Image.new('RGB', (8, 8)).save(out / 'g.png')
            (out / 'generations.jsonl').write_text(''.join(json.dumps(dict(image='g.png', id='test_dog_00', split='test', seed=s,
                                                                           source_concept='dog', target_concept='cat')) + '\n' for s in seeds))
            (out / 'eval_config.json').write_text(json.dumps(dict(seeds=seeds)))
        elif prog == 'evaluate_generations.py':
            g = Path(_arg(cmd, '--gen-dir'))
            shutil.copyfile(g / 'generations.jsonl', g / 'judgments.jsonl'); Path(_arg(cmd, '--out')).write_text('{}')
        else:
            raise AssertionError(f'unexpected subprocess {cmd}')
        return ''
    return sh


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / 'repo'; (root / 'configs').mkdir(parents=True); (root / 'reports/dclr').mkdir(parents=True)
    shutil.copy(ROOT / 'configs/negative_map.json', root / 'configs')
    shutil.copy(ROOT / 'reports/dclr/calibration_s.json', root / 'reports/dclr')
    store = tmp_path / 'store'; (store / 'corpus').mkdir(parents=True)
    calls = []
    monkeypatch.setattr(RA, 'ROOT', root); monkeypatch.setattr(RA, 'STORE', store); monkeypatch.setattr(RA, 'sh', fake_sh(calls))
    return dict(root=root, store=store, calls=calls, mp=monkeypatch)


def run(env, arm, *extra):
    env['mp'].setattr(sys, 'argv', ['run_arm.py', '--exp', 'main', '--arm', arm, '--k', '3', *extra])
    env['calls'].clear(); RA.main()
    name = RA.run_name(arm, 3, 42, '--unsupervised-tau' in extra, dict(zip(extra[::2], extra[1::2])).get('--attack-tag', 'dog2cat_eps04'))
    rep = env['root'] / 'reports/runs/main' / name
    return json.loads((rep / 'arm_summary.json').read_text()), rep


def train_call(env):
    t = [c for c in env['calls'] if Path(c[1]).name == 'train_full_unet.py']
    return t[0] if t else None


@pytest.mark.parametrize('arm', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'KE', 'KB', 'LF', 'M1', 'M2', 'O', 'R', 'RS', 'V'])
def test_every_arm_runs_end_to_end_with_faked_steps(env, arm):
    if arm == 'I':   # CLIP scoring needs CUDA: provide the pre-registered score file the real step writes
        d = env['store'] / 'arms/dog2cat_eps04'; d.mkdir(parents=True, exist_ok=True)
        sc = {f'p{i}': (0.0 if i < 1 else 1.0) for i in range(10)}
        (d / 'clipfilter10_k3_d42_scores.json').write_text(json.dumps(dict(scores=sc, weights=RA.drop_lowest(sc, 0.1))))
    s, rep = run(env, arm)
    assert s['arm'] == arm and s['gen_seeds'] == [int(x) for x in RA.GEN_SEEDS]
    tc = train_call(env)
    if arm == 'A':
        assert tc is None and s['train_dir'] is None; return
    train = Path(s['train_dir'])
    rows = _rows(train)
    if arm in ('O', 'J'):
        assert not any(r['is_poisoned'] for r in rows) and len(rows) == 7
        fe = json.loads((rep / 'filter_eval.json').read_text()); assert fe['poisons_removed'] == 3 and fe['clean_removed'] == 0
    if arm == 'LF':
        assert len(rows) == 9   # exactly 10% of 10 records dropped (the highest-loss one is a poison)
    if arm == 'KB':
        assert 'clean' in str(train) and not any(r['is_poisoned'] for r in rows) and '--sample-weights' in tc
        assert s['attack_independent']
    if arm in ('K', 'KE'):
        assert '--sample-weights' in tc and ('--trusted-dir' in tc) == (arm == 'KE')
    if arm in ('E', 'F', 'V', 'KE'):
        assert _arg(tc, '--trusted-dir').endswith('trusted/train')
    if arm in ('R', 'RS'):
        assert _arg(tc, '--latent-cache').endswith('latents.pt')
    if arm in ('G', 'M1', 'M2', 'I', 'J', 'LF', 'O'):
        assert (train.parent / 'audit.json').exists()
    assert _arg(tc, '--max-steps') == '1000' and _arg(tc, '--batch-size') == str(RA._MB)


def test_attack_tag_and_utau_variants_are_separate_runs(env):
    s, rep = run(env, 'K', '--attack-tag', 'dog2cat_adapt4')
    assert rep.name == 'K_k3_s42_dog2cat_adapt4' and '/arms/dog2cat_adapt4/' in s['train_dir']
    s2, rep2 = run(env, 'K', '--unsupervised-tau')
    assert rep2.name == 'K_k3_s42_utau'
    wcall = [c for c in env['calls'] if Path(c[1]).name == 'roundtrip_drift.py' and c[2] == 'weights'][0]
    assert '--unsupervised' in wcall and '--calibration' not in wcall


def test_rerun_skips_finished_work_and_regenerates_old_protocol_eval(env):
    run(env, 'C')
    run(env, 'C')
    assert train_call(env) is None                                      # finished training is reused
    assert not any(Path(c[1]).name == 'generate_eval.py' for c in env['calls'])
    ev = env['store'] / 'runs/main/C_k3_s42/eval'
    (ev / 'eval_config.json').write_text(json.dumps(dict(seeds=[11, 22, 33, 44])))   # a protocol-v1 evaluation
    run(env, 'C')
    assert any(Path(c[1]).name == 'generate_eval.py' for c in env['calls'])
    assert (env['store'] / 'runs/main/C_k3_s42/eval_v4seeds/generations.jsonl').exists()
