import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from arm_labels import LONG, SHORT, plain  # noqa: E402


def test_every_figure_name_is_plain_words_not_a_code():
    for d in (LONG, SHORT):
        for code, name in d.items():
            assert not name.startswith(code + ' '), (code, name)   # e.g. never "K RTDW" or "C poisoned"
    assert plain('C') == 'Poisoned' and plain('B') == 'Clean' and plain('K') == 'RTDW'
    assert plain('K-utau') == 'RTDW (unsupervised τ)'
    assert plain('M2', short=True) == 'Upscale' and plain('XYZ') == 'XYZ'
