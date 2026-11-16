"""CPU regression tests for counter-experiment infrastructure (no GPU, no real model)."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import train_full_unet as T  # noqa: E402
import paired_dataset as PD  # noqa: E402


def _train_dir(tmp_path, ids):
    d = tmp_path / 'train'; d.mkdir()
    (d / 'metadata.jsonl').write_text(''.join(json.dumps({'id': i, 'file_name': f'{i}.png', 'text': 'x'}) + '\n' for i in ids))
    return d


def test_sample_weights_normalised_to_unit_dataset_mean(tmp_path):
    d = _train_dir(tmp_path, ['a', 'b', 'c', 'd'])
    w = tmp_path / 'w.json'; w.write_text(json.dumps({'weights': {'a': 1, 'b': 1, 'c': 1, 'd': 0}}))
    norm, mean = T.load_sample_weights(w, d)
    assert mean == pytest.approx(0.75)
    assert sum(norm.values()) / 4 == pytest.approx(1.0)
    assert norm['d'] == 0


def test_sample_weights_missing_or_invalid_fail_closed(tmp_path):
    d = _train_dir(tmp_path, ['a', 'b'])
    w = tmp_path / 'w.json'
    w.write_text(json.dumps({'weights': {'a': 1}}))
    with pytest.raises(ValueError, match='lack a sample weight'):
        T.load_sample_weights(w, d)
    w.write_text(json.dumps({'weights': {'a': 1.5, 'b': 1}}))
    with pytest.raises(ValueError):
        T.load_sample_weights(w, d)
    w.write_text(json.dumps({'weights': {'a': 0, 'b': 0}}))
    with pytest.raises(ValueError, match='zero'):
        T.load_sample_weights(w, d)


def test_weighted_loss_zero_weight_removes_example_gradient():
    pred = torch.randn(3, 4, 8, 8, requires_grad=True)
    target = torch.zeros(3, 4, 8, 8)
    from training_objectives import per_example_denoising_loss
    per = per_example_denoising_loss(pred, target)
    loss = (per * torch.tensor([1.5, 1.5, 0.0])).mean()
    loss.backward()
    assert pred.grad[2].abs().max() == 0
    assert pred.grad[0].abs().max() > 0


def test_unit_weights_equal_unweighted_loss():
    from training_objectives import per_example_denoising_loss
    pred, target = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    per = per_example_denoising_loss(pred, target)
    assert (per * torch.ones(4)).mean() == pytest.approx(float(per.mean()))


def test_poison_sets_are_nested_across_doses(tmp_path):
    rows = [dict(id=f'r{i:03d}', split='train', concept='dog', _poison=f'p{i}', _clean=f'c{i}') for i in range(50)]
    def chosen(k):
        order = sorted([r for r in rows if r['_poison']], key=lambda x: x['id'])
        import random
        random.Random(42).shuffle(order)
        return {r['id'] for r in order[:k]}
    assert chosen(5) <= chosen(10) <= chosen(25)
    # and the production rule (used by export AND by attack variants) is exactly this one, behaviourally
    from paired_dataset import select_poison_ids
    ids = [r['id'] for r in rows]
    for k in (5, 10, 25):
        assert set(select_poison_ids(list(reversed(ids)), k, 42)) == chosen(k)   # input order must not matter


def test_drift_weights_calibrated_on_trusted(tmp_path):
    import roundtrip_drift as RD
    scores = tmp_path / 's.jsonl'; cal = tmp_path / 'c.jsonl'; out = tmp_path / 'w.json'
    scores.write_text(''.join(json.dumps({'id': f'x{i}', 'drift': d, 'is_poisoned': d > 0.3}) + '\n'
                              for i, d in enumerate([0.05, 0.06, 0.07, 0.5, 0.6])))
    cal.write_text(''.join(json.dumps({'drift': d}) + '\n' for d in [0.04, 0.05, 0.06, 0.07, 0.08]))
    class A: pass
    a = A(); a.scores, a.calibration, a.quantile, a.unsupervised, a.hard, a.out = scores, cal, 0.99, False, False, out
    RD.weights(a)
    w = json.loads(out.read_text())['weights']
    assert w['x0'] > 0.9 and w['x3'] < 0.01 and w['x4'] < 0.01
    a.hard = True; RD.weights(a)
    assert set(json.loads(out.read_text())['weights'].values()) <= {0.0, 1.0}
