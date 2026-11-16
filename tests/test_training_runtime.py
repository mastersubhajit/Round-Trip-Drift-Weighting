"""CPU regressions for the actual Accelerate/trainer runtime contracts."""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_anchor_loader_does_not_force_early_optimizer_step():
    from accelerate import Accelerator
    from train_full_unet import accumulation_plugin
    accelerator = Accelerator(cpu=True, gradient_accumulation_plugin=accumulation_plugin(4))
    model = torch.nn.Linear(1, 1)
    main = DataLoader(torch.ones(6, 1), batch_size=1)
    anchors = DataLoader(torch.ones(1, 1), batch_size=1)
    model, main, anchors = accelerator.prepare(model, main, anchors)
    sync = []
    for batch in main:
        with accelerator.accumulate(model):
            next(iter(anchors))
            accelerator.backward(model(batch).sum())
            sync.append(accelerator.sync_gradients)
    assert sync == [False, False, False, True, False, False]
    accelerator.free_memory()


def test_resume_rejects_changed_data_or_settings():
    from train_full_unet import check_resume_contract
    saved = {'seed': 42, 'world_size': 1, 'grad_accum': 4, 'train_fingerprint': 'abc'}
    check_resume_contract(saved, dict(saved))
    for field, value in [('grad_accum', 2), ('train_fingerprint', 'tampered')]:
        with pytest.raises(ValueError, match=field):
            check_resume_contract(saved, {**saved, field: value})


def test_skipped_update_does_not_increment_step():
    from train_full_unet import optimizer_update_completed
    assert optimizer_update_completed(sync_gradients=True, step_was_skipped=False)
    assert not optimizer_update_completed(sync_gradients=True, step_was_skipped=True)
    assert not optimizer_update_completed(sync_gradients=False, step_was_skipped=False)


def test_preflight_rejects_missing_assets_without_downloading(tmp_path):
    from preflight import inspect_inputs
    with pytest.raises(ValueError, match='model'):
        inspect_inputs(tmp_path / 'missing-model', tmp_path / 'missing-data')


def test_export_rejects_poisoned_trusted_data(tmp_path):
    from paired_dataset import load_and_validate
    from PIL import Image
    Image.new('RGB', (16, 16), 'red').save(tmp_path / 'a.png')
    Image.new('RGB', (16, 16), 'blue').save(tmp_path / 'b.png')
    row = dict(id='a', clean_path='a.png', poison_path='b.png', caption='a cat',
               concept='cat', source_id='a', split='trusted', permission=True,
               license='CC0', attack='nightshade_research', attacker_target='dog')
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Poison cannot enter'):
        load_and_validate(path)


def test_preflight_detects_modified_image_contents(tmp_path):
    from preflight import dataset_fingerprint
    from paired_dataset import load_and_validate, export
    from PIL import Image
    image = tmp_path / 'cat.png'
    Image.new('RGB', (16, 16), 'red').save(image)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(dict(id='cat', clean_path='cat.png', caption='a cat',
        concept='cat', source_id='photographer', split='train', permission=True, license='CC0'))+'\n')
    rows, _ = load_and_validate(manifest)
    export(rows, tmp_path/'export', 'clean', '', 0, 42)
    folder = tmp_path/'export/train'
    first = dataset_fingerprint(folder)
    assert first['records'] == 1
    exported = next(p for p in folder.glob('*.png'))
    Image.new('RGB', (16, 16), 'blue').save(exported)
    with pytest.raises(ValueError, match='Image hash mismatch'):
        dataset_fingerprint(folder)


def test_prepared_loader_epoch_and_resume_offset():
    from accelerate import Accelerator
    from data_utils import EpochPermutationSampler
    accelerator = Accelerator(cpu=True)
    samples = list(range(13))
    sampler = EpochPermutationSampler(samples, seed=42)
    loader = accelerator.prepare(DataLoader(samples, batch_size=1, sampler=sampler))
    loader.set_epoch(2)
    expected = list(EpochPermutationSampler(samples, seed=44))
    remaining = accelerator.skip_first_batches(loader, 4)
    assert [int(batch[0]) for batch in remaining] == expected[4:]
    accelerator.free_memory()


def test_resume_at_epoch_boundary_never_iterates_empty_loader():
    from train_full_unet import normalize_cursor
    assert normalize_cursor(2, 6, 6) == (3, 0)
    assert normalize_cursor(2, 4, 6) == (2, 4)
    for epoch, offset in [(-1, 0), (2, -1), (2, 7)]:
        with pytest.raises(ValueError, match='cursor'):
            normalize_cursor(epoch, offset, 6)


def test_attach_preserves_paths_when_output_directory_changes(tmp_path):
    import csv
    from PIL import Image
    from attach_nightshade import attach
    from paired_dataset import load_and_validate
    source = tmp_path/'source'
    source.mkdir()
    Image.new('RGB', (16, 16), 'red').save(source/'clean.png')
    Image.new('RGB', (16, 16), 'blue').save(source/'poison.png')
    manifest = source/'manifest.jsonl'
    manifest.write_text(json.dumps(dict(id='cat', clean_path='clean.png', caption='a cat', concept='cat',
        source_id='photographer', split='train', permission=True, license='CC0'))+'\n')
    mapping = source/'pairs.csv'
    mapping.write_text('id,poison_path,attacker_target\ncat,poison.png,dog\n')
    out = tmp_path/'attached.jsonl'
    attach(manifest, mapping, out)
    rows, _ = load_and_validate(out)
    assert Path(rows[0]['_clean']) == source/'clean.png'


def test_resume_requires_fresh_output_to_preserve_existing_results(tmp_path):
    from train_full_unet import prepare_output
    run = tmp_path/'run'
    prepare_output(run)
    (run/'completed.json').write_text('{"optimizer_steps":2}')
    with pytest.raises(ValueError, match='fresh'):
        prepare_output(run)
    assert (run/'completed.json').read_text() == '{"optimizer_steps":2}'


def test_epoch_reaches_sampler_through_distributed_wrapper():
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
    from torch.utils.data import BatchSampler
    from data_utils import EpochPermutationSampler
    from train_full_unet import set_loader_epoch
    data = list(range(12))
    sampler = EpochPermutationSampler(data, seed=42)
    sharded = BatchSamplerShard(BatchSampler(sampler, 1, True), num_processes=2, process_index=0)
    loader = DataLoaderShard(data, batch_sampler=sharded, device=torch.device('cpu'))
    set_loader_epoch(loader, sampler, 2)
    expected = list(EpochPermutationSampler(data, seed=44))[::2]
    assert [int(batch[0]) for batch in loader] == expected
    assert sampler.epoch == 2


def test_empty_trusted_loader_fails_before_training():
    from train_full_unet import require_nonempty_loaders
    require_nonempty_loaders([1], [1])
    with pytest.raises(ValueError, match='trusted'):
        require_nonempty_loaders([1], [])
    with pytest.raises(ValueError, match='Training'):
        require_nonempty_loaders([], None)


def test_deterministic_validation_configuration(monkeypatch):
    from train_full_unet import configure_determinism
    old = torch.are_deterministic_algorithms_enabled()
    flash = torch.backends.cuda.flash_sdp_enabled()
    efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    try:
        configure_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        import os
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    finally:
        torch.use_deterministic_algorithms(old)
        torch.backends.cuda.enable_flash_sdp(flash)
        torch.backends.cuda.enable_mem_efficient_sdp(efficient)


def test_preflight_checks_output_permissions_without_leaving_files(tmp_path):
    from preflight import inspect_output
    parent = tmp_path/'parent'
    parent.mkdir()
    assert inspect_output(parent/'new-run')['writable']
    assert list(parent.iterdir()) == []
    parent.chmod(0o500)
    try:
        with pytest.raises(ValueError, match='not writable'):
            inspect_output(parent/'new-run')
    finally:
        parent.chmod(0o700)
