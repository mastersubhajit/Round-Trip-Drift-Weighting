"""Derived training sets (G, H, I, J, M1, M2) must pass the same rights/provenance + hash preflight as exports."""
import hashlib, json, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import run_arm as RA  # noqa: E402
import preflight  # noqa: E402


def _export(root, n=3):
    tr = root / 'train'; tr.mkdir(parents=True); rows, prov = [], []
    for i in range(n):
        f = tr / f'{i}.png'; Image.fromarray(__import__('numpy').random.default_rng(i).integers(0, 255, (32, 32, 3), dtype='uint8')).save(f)
        rows.append(dict(file_name=f.name, text=f'a photo of a dog {i}', id=f'id{i}',
                         image_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), is_poisoned=i == 0))
        prov.append(dict(id=f'id{i}', permission=True, license='cc-by', source_id=f's{i}', is_poisoned=i == 0))
    (tr / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (root / 'audit.json').write_text(json.dumps(dict(train_records=n, record_provenance=prov, selected_poison_ids=['id0'])))
    return tr


def test_transformed_and_filtered_copies_pass_preflight(tmp_path):
    src = _export(tmp_path / 'poisoned')
    preflight.dataset_fingerprint(src)
    g = RA.transformed_copy(src, tmp_path / 'jpeg', RA.jpeg75)          # new pixels
    RA.ensure_audit(g, src, 'jpeg_q75'); RA.rehash_metadata(g)
    fp = preflight.dataset_fingerprint(g)
    assert fp['records'] == 3 and fp['audit']['transform'] == 'jpeg_q75'
    rows = [json.loads(l) for l in (g / 'metadata.jsonl').read_text().splitlines()]
    assert all(r['source_image_sha256'] != r['image_sha256'] for r in rows)
    # filtered subset (J): one record dropped, audit restricted, kept poisons recorded
    j = tmp_path / 'filtered' / 'train'; j.mkdir(parents=True)
    keep = [json.loads(l) for l in (src / 'metadata.jsonl').read_text().splitlines()][1:]
    for r in keep: (j / r['file_name']).write_bytes((src / r['file_name']).read_bytes())
    (j / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in keep))
    RA.ensure_audit(j, src, 'roundtrip_drift_hard_filter'); RA.rehash_metadata(j)
    fp = preflight.dataset_fingerprint(j)
    assert fp['records'] == 2 and fp['audit']['kept_poison_ids'] == []
