#!/bin/bash
# Run the paper's experiments on the current machine (all visible GPUs), then the analysis.
#
#   export POISON_STORE=/path/to/storage            # models, COCO corpus, poisons, runs (~250 GB)
#   bash scripts/pipeline/stage_all.sh               # once, with internet access
#   bash scripts/pipeline/experiments.sh run         # every phase in ORDER; finished phases and arms are skipped
#   bash scripts/pipeline/experiments.sh run main    # only the named phase(s)
#   bash scripts/pipeline/experiments.sh analyze     # statistics, gates and figures -> reports/
#   bash scripts/pipeline/experiments.sh figures     # redraw figures and tables from reports/results_*.json
#
# Arm spec: ARM:K:SEED[:ATTACK_TAG]. Phases and the experiments they implement are listed in configs/experiments.json.
# Phases marked post hoc were added after the pre-registered ones.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PR_PYTHON="${PR_PYTHON:-python}" PYTHONNOUSERSITE=1
export PR_MICRO_BATCH="${PR_MICRO_BATCH:-8}" PR_GRAD_ACCUM="${PR_GRAD_ACCUM:-3}"   # effective batch 24
mkdir -p logs; STATE="$PWD/logs/phases_done.tsv"; touch "$STATE"

S="42"; SEEDS2="43 44"
# Gate-2 arms first (B clean, C poisoned), then the headline defense K with its decisive controls (O oracle upper bound,
# E replay, J hard filter, KB = K's cost with NO attack), then the remaining defenses and baselines.
main_arms="B:200:$S C:200:$S K:200:$S O:200:$S E:200:$S J:200:$S KB:200:$S A:200:$S D:200:$S LF:200:$S I:200:$S G:200:$S H:200:$S KE:200:$S F:200:$S V:200:$S RS:200:$S"
seed_arms=""; for s in $SEEDS2; do for a in B C E J K O KB G; do seed_arms+="$a:200:$s "; done; done
dose_arms=""; for k in 25 50 100 300; do for a in C E J K RS; do dose_arms+="$a:$k:$S "; done; done
car_seed_arms=""; for s in $SEEDS2; do for a in C E K O G; do car_seed_arms+="$a:200:$s "; done; done
# N6/N7/N8 in ONE job (4th field = attack tag): every variant poisons exactly the official k=200 records.
# adapt0 = white-box Nightshade on the defender's SD1.5 VAE with NO drift penalty (separates "white-box" from "adaptive").
# G (JPEG q75) runs on EVERY variant, and
# dog2cat_adaptjpeg is the attacker that optimises through G's JPEG (EOT over {identity, JPEG}, BPDA).
attack_arms=""
for t in dog2cat_eps02 dog2cat_eps08 dog2cat_adapt0 dog2cat_adapt16; do for a in C J K G; do attack_arms+="$a:200:$S:$t "; done; done
for t in dog2cat_adapt1 dog2cat_adapt4; do for a in C J K RS G; do attack_arms+="$a:200:$S:$t "; done; done
for a in C E J K O G; do attack_arms+="$a:200:$S:dog2cat_dirty "; done
for a in C G J K O; do attack_arms+="$a:200:$S:dog2cat_adaptjpeg "; done
# attacks2 (post hoc): lambda=0 poisons but lambda=1 does not, while its drift is already below clean, so the first grid
# skipped the range where an evading attack could exist: lambda in {0.05, 0.1, 0.25, 0.5}. Also seeds 43/44 at eps 0.08.
LAM2="0p05:0.05 0p1:0.1 0p25:0.25 0p5:0.5"
attack2_arms=""; attack2_poisons=""
for l in $LAM2; do t=dog2cat_adapt${l%%:*}; attack2_poisons+="$t:dog:cat:0.04:adaptive:200:${l#*:} "
  for a in C G J K; do attack2_arms+="$a:200:$S:$t "; done; done
for s in $SEEDS2; do for a in C G J K; do attack2_arms+="$a:200:$s:dog2cat_eps08 "; done; done
# extra (post hoc): training seeds 43/44 where a claim rested on one seed (the lambda=0.05 evasion; the Gate-3 ties with
# D, M1, M2), and GK = JPEG q75 then RTDW on every condition where one of G/K failed plus the main setting (exploratory).
extra_main="D:200:43 D:200:44 M1:200:43 M1:200:44 M2:200:43 M2:200:44 GK:200:42 GK:200:43 GK:200:44"
extra_attacks=""
for s in $SEEDS2; do for a in C G J K; do extra_attacks+="$a:200:$s:dog2cat_adapt0p05 "; done; done
for s in 42 $SEEDS2; do for t in dog2cat_adapt0p05 dog2cat_eps08; do extra_attacks+="GK:200:$s:$t "; done; done
declare -A PHASE=(
  [poisons]="poisons|dog2cat_eps04:dog:cat:0.04:official:0 car2cow_eps04:car:cow:0.04:official:0"
  [main]="arms|main|$main_arms|"
  [seeds]="arms|main|$seed_arms|"
  [dose]="arms|dose|$dose_arms|"
  [car]="arms|car|C:200:$S E:200:$S K:200:$S J:200:$S O:200:$S G:200:$S RS:200:$S V:200:$S|--attack-tag car2cow_eps04 --concept car"
  [car_seeds]="arms|car|$car_seed_arms|--attack-tag car2cow_eps04 --concept car"
  [poisons_attacks]="poisons|dog2cat_eps02:dog:cat:0.02:official:200 dog2cat_eps08:dog:cat:0.08:official:200 dog2cat_adapt0:dog:cat:0.04:adaptive:200:0 dog2cat_adapt1:dog:cat:0.04:adaptive:200:1 dog2cat_adapt4:dog:cat:0.04:adaptive:200:4 dog2cat_adapt16:dog:cat:0.04:adaptive:200:16 dog2cat_dirty:dog:cat:0:dirty:200 dog2cat_adaptjpeg:dog:cat:0.04:adaptive_jpeg:200:0"
  [attacks]="arms|attacks|$attack_arms|"
  [poisons_attacks2]="poisons|$attack2_poisons"
  # car2cow at eps 0.08 (post hoc: the eps 0.04 car attack is weak). Same 200 car records as the official k=200 selection.
  [poisons_car2]="poisons|car2cow_eps08:car:cow:0.08:official:200"
  [car2]="arms|car|C:200:$S G:200:$S J:200:$S K:200:$S O:200:$S|--attack-tag car2cow_eps08 --concept car"
  [attacks2]="arms|attacks|$attack2_arms|"
  [trusted]="arms|trusted|K:200:$S J:200:$S|--unsupervised-tau"
  [published]="arms|main|M1:200:$S M2:200:$S|"   # public-code published defenses (Hoenig et al. 2025)
  [antipoison]="arms|main|RS:200:43 V:200:43 RS:200:44 V:200:44|"   # proposed DCLR-S + vaccine, extra training seeds
  [extra_main]="arms|main|$extra_main|"
  [extra_attacks]="arms|attacks|$extra_attacks|"
  [extra_car]="arms|car|GK:200:$S|--attack-tag car2cow_eps08 --concept car"
)
# Order: attack works (main) -> replicates (seeds) -> survives an adaptive attacker
# (poisons_attacks, attacks) -> second pair (car, car_seeds) -> dose -> remaining baselines.
ORDER="poisons main seeds poisons_attacks attacks car car_seeds dose published trusted antipoison poisons_attacks2 attacks2 poisons_car2 car2 extra_main extra_attacks extra_car"

case "${1:-}" in
  run)
    : "${POISON_STORE:?export POISON_STORE first}"
    shift; phases="${*:-$ORDER}"
    for p in $phases; do [[ -n "${PHASE[$p]:-}" ]] || { echo "unknown phase $p (known: $ORDER)" >&2; exit 64; }; done
    for p in $phases; do
      if cut -f1 "$STATE" | grep -qx "$p"; then echo "phase $p already done"; continue; fi
      IFS='|' read -r kind rest <<< "${PHASE[$p]}"
      if [[ $kind == poisons ]]; then POISON_SPECS="$rest" bash scripts/pipeline/poisons.sh
      else IFS='|' read -r exp arms extra <<< "$rest"; EXP="$exp" ARMS="$arms" EXTRA="$extra" bash scripts/pipeline/arms.sh; fi
      printf '%s\t%s\n' "$p" "$(date -Is)" >> "$STATE"
    done ;;
  analyze)
    P="$PR_PYTHON"
    "$P" scripts/analyze_results.py --exp main dose --attack-tag dog2cat_eps04 --name dog
    "$P" scripts/analyze_results.py --exp car main --attack-tag car2cow_eps04 --concept car --target cow --name car
    "$P" scripts/analyze_results.py --exp car main --attack-tag car2cow_eps08 --concept car --target cow --name car_eps08 || true
    for t in dog2cat_eps02 dog2cat_eps08 dog2cat_adapt0 dog2cat_adapt0p05 dog2cat_adapt0p1 dog2cat_adapt0p25 dog2cat_adapt0p5 dog2cat_adapt1 dog2cat_adapt4 dog2cat_adapt16 dog2cat_dirty dog2cat_adaptjpeg; do
      "$P" scripts/analyze_results.py --exp attacks main --attack-tag $t --name $t || true; done
    "$P" scripts/analyze_results.py --exp trusted main --attack-tag dog2cat_eps04 --name trusted || true
    "$P" scripts/attack_tradeoff.py --names dog dog2cat_adapt0 dog2cat_adapt0p05 dog2cat_adapt0p1 dog2cat_adapt0p25 dog2cat_adapt0p5 dog2cat_adapt1 dog2cat_adapt4 dog2cat_adapt16 dog2cat_eps02 dog2cat_eps08 dog2cat_dirty dog2cat_adaptjpeg car_eps08 || true
    "$P" scripts/make_figures.py --exp dog
    "$P" scripts/make_comparison_figures.py
    echo "figures: reports/figures/ (FIGURES.md, COMPARISON.md) ; tables + gates: reports/results_*.json, reports/GATES.md" ;;
  figures)  # redraw all figures and tables from the existing reports/results_*.json (no re-analysis; CPU, seconds)
    "$PR_PYTHON" scripts/make_figures.py --exp dog
    "$PR_PYTHON" scripts/make_comparison_figures.py
    echo "figures: reports/figures/ (FIGURES.md, COMPARISON.md)" ;;
  *) sed -n 2,11p "$0"; exit 64 ;;
esac
