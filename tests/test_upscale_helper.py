"""noisy_upscale.py --helper: extra processes fill the arm's in-progress output from the end, safely."""
import subprocess, sys, threading
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import noisy_upscale as NU  # noqa: E402

NAMES = [f'{i:03d}.png' for i in range(12)]


def _img(name):
    return Image.new('RGB', (8, 8), (int(name[:3]) * 10, 0, 0))


def test_forward_process_unchanged_and_skips_done(tmp_path):
    out = tmp_path / 'x.up'; out.mkdir()
    _img('000').save(out / '000.png')
    seen = []
    assert NU.run(NAMES, out, lambda n: (seen.append(n), _img(n))[1]) == 11
    assert seen == NAMES[1:] and not (tmp_path / 'x.up.helper').exists()


def test_helpers_go_backwards_never_duplicate_and_leave_nothing_in_out(tmp_path):
    out = tmp_path / 'x.up'; out.mkdir()
    calls, lock = [], threading.Lock()

    def up(n):
        with lock: calls.append(n)
        return _img(n)
    ts = [threading.Thread(target=NU.run, args=(NAMES, out, up, True)) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert sorted(calls) == NAMES                       # each image computed exactly once across 4 helpers
    assert sorted(p.name for p in out.iterdir()) == NAMES   # no claim/temp files inside the folder that becomes train/
    assert all(Image.open(out / n).getpixel((0, 0))[0] == int(n[:3]) * 10 for n in NAMES)
    first = []
    out2 = tmp_path / 'y.up'; out2.mkdir()
    NU.run(NAMES[:3], out2, lambda n: (first.append(n), _img(n))[1], True)
    assert first == NAMES[2::-1]                        # reverse order: meets the forward process from the far end


def test_helper_stops_when_the_arm_renames_the_folder(tmp_path):
    out = tmp_path / 'x.up'; out.mkdir()
    done = []

    def up(n):
        if len(done) == 2: out.rename(tmp_path / 'train')   # arm finished its pass meanwhile
        done.append(n); return _img(n)
    assert NU.run(NAMES, out, up, True) == 2
    assert not out.exists() and sorted(p.name for p in (tmp_path / 'train').iterdir()) == NAMES[-2:]


def test_helper_cli_refuses_to_create_a_missing_folder(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/noisy_upscale.py'), '--in-dir', str(tmp_path),
                        '--out-dir', str(tmp_path / 'gone.up'), '--model', 'm', '--helper', '0'],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'does not exist' in r.stderr + r.stdout and not (tmp_path / 'gone.up').exists()
