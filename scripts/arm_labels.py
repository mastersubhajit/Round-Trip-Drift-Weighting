"""Plain-language arm names for every figure and diagram, so images never show arm codes (B, C, K, ...).
LONG names go on axes and row labels; SHORT names go where space is tight (legends, column heads)."""
LONG = {'A': 'Pretrained', 'B': 'Clean', 'O': 'Poisons removed (oracle)', 'C': 'Poisoned', 'D': 'Trusted-only',
        'E': 'Anchor replay', 'F': 'Semantic margin', 'G': 'JPEG', 'H': 'img2img', 'I': 'CLIP filter',
        'LF': 'High-loss filter', 'M1': 'Gaussian noise', 'M2': 'Noisy upscaling', 'J': 'Drift filter',
        'K': 'RTDW', 'KE': 'RTDW + replay', 'KB': 'RTDW, no attack', 'GK': 'JPEG + RTDW†', 'V': 'Vaccine',
        'R': 'DCLR', 'RS': 'DCLR-S'}
SHORT = dict(LONG, O='Oracle', M1='Noise', M2='Upscale', E='Replay', GK='JPEG+RTDW†')


def plain(arm: str, short: bool = False) -> str:
    """Name for an arm code; an arm id like 'K-utau' keeps its suffix in words."""
    base, _, suf = arm.partition('-')
    name = (SHORT if short else LONG).get(base, base)
    return name + (' (unsupervised τ)' if suf == 'utau' else f' ({suf})' if suf else '')
