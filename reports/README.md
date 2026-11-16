# Measured results

Written by `scripts/analyze_results.py` and `scripts/attack_tradeoff.py` from the per-arm run outputs; do not edit.

| File | Contents |
|---|---|
| `results_dog.json` | main setting (dog → cat, ε = 0.04) at every dose `k`, three training seeds for the headline arms |
| `results_dog2cat_*.json` | attack variants: ε 0.02 / 0.08, adaptive λ grid, JPEG-aware, dirty-label |
| `results_car.json`, `results_car_eps08.json` | second pair (car → cow) at ε 0.04 and 0.08 |
| `results_trusted.json` | thresholds set without trusted data (`--unsupervised-tau`) |
| `results_dog_run1.json` | first single-seed run of the main setting |
| `results_tradeoff.json`, `GATES.md` | cross-attack summary and gate outcomes |
| `dclr/` | DCLR calibration used by arm RS (fixed before training) |

Each `arms` entry is keyed `<arm>@k<k>` and holds accuracy and attack success with 95 % intervals, per-seed values,
KID, judge agreement and GPU-hours; `drift` holds the per-image drift scores and RTDW weights.
