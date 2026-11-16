#!/bin/bash
# Runs offline on one GPU machine: candidate selection (once) + poison generation on all visible GPUs.
#   POISON_SPECS="tag:source:target:eps:kind:limit[:lambda] ..."   kind = official | adaptive | adaptive_jpeg | dirty
# adaptive_jpeg = adaptive attack that also optimises through the G defense (JPEG q75, BPDA; post hoc).
#   default: dog2cat_eps04:dog:cat:0.04:official:0  car2cow_eps04:car:cow:0.04:official:0
# limit 0 = every poison-eligible train image of the source concept (the official base attack, <src>2<tgt>_eps04).
# limit N>0 = EXACTLY the N records the official manifest's k=N, data-seed-42 export poisons,
# so every variant perturbs the same images as the official arm it is compared with.
# An already attached variant is re-checked and the job fails if it poisons other records. Idempotent.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
: "${POISON_STORE:?}"
PY="${PR_PYTHON:-$PWD/.conda-pr/bin/python}"
export HF_HOME="$POISON_STORE/hf_cache" TORCH_HOME="$POISON_STORE/torch_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/src" OMP_NUM_THREADS=8
NG=$(nvidia-smi -L | wc -l)
REP="$PWD/reports/poisons-${SLURM_JOB_ID:-local}"; mkdir -p "$REP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee "$REP/nvidia-smi.txt"
if [[ ! -f "$POISON_STORE/corpus/manifest_clean.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/nightshade_poisons.py select --store "$POISON_STORE" ${SELECT_ARGS:-} 2>&1 | tee "$REP/select.log"
fi
cp "$POISON_STORE/corpus/split_report.json" "$POISON_STORE/corpus/corpus_audit.json" "$REP/"
for spec in ${POISON_SPECS:-dog2cat_eps04:dog:cat:0.04:official:0 car2cow_eps04:car:cow:0.04:official:0}; do
  IFS=: read -r tag src tgt eps kind limit lam <<< "$spec"
  BASE="$POISON_STORE/corpus/manifest_${src}2${tgt}_eps04.jsonl"
  MATCH=(); if [[ ${limit:-0} != 0 ]]; then
    [[ -f $BASE ]] || { echo "$tag must match the official selection in $BASE (generate ${src}2${tgt}_eps04 first)" >&2; exit 1; }
    MATCH=(--match-manifest "$BASE" --match-k "$limit" --match-seed 42)
  fi
  if [[ -f "$POISON_STORE/corpus/manifest_$tag.jsonl" ]]; then
    echo "skip $tag (attached)"
    if [[ ${limit:-0} != 0 ]]; then "$PY" scripts/nightshade_poisons.py check-match --official "$BASE" \
        --variant "$POISON_STORE/corpus/manifest_$tag.jsonl" --source "$src" --k "$limit" | tee "$REP/match_$tag.json"; fi
    continue
  fi
  [[ $kind == official && ${limit:-0} == 0 ]] || [[ ${#MATCH[@]} -gt 0 ]] || { echo "$tag: variants need limit>0 (matched selection)" >&2; exit 1; }
  echo "$(date +%T) generating $spec on $NG GPUs"
  if [[ $kind == dirty ]]; then
    "$PY" scripts/dirty_label_poisons.py --store "$POISON_STORE" --source "$src" --target "$tgt" --tag "$tag" ${MATCH[@]+"${MATCH[@]}"} \
      > "$REP/gen_${tag}.log" 2>&1 || { tail -20 "$REP/gen_${tag}.log"; exit 1; }
  else
    for r in $(seq 0 $((NG-1))); do
      if [[ $kind == official ]]; then
        CUDA_VISIBLE_DEVICES=$r "$PY" scripts/nightshade_poisons.py generate --store "$POISON_STORE" \
          --surrogate "$POISON_STORE/assets/sd21" --source "$src" --target "$tgt" --tag "$tag" --eps "$eps" \
          --rank "$r" --world "$NG" ${MATCH[@]+"${MATCH[@]}"} > "$REP/gen_${tag}_rank$r.log" 2>&1 &
      else
        TIMG="$POISON_STORE/poisons/${src}2${tgt}_eps04/work_rank0/target.png"
        [[ -f $TIMG ]] || { echo "adaptive attack needs the official target image $TIMG (generate ${src}2${tgt}_eps04 first)" >&2; exit 1; }
        JPG=(); [[ $kind == adaptive_jpeg ]] && JPG=(--jpeg 75)
        CUDA_VISIBLE_DEVICES=$r "$PY" scripts/adaptive_nightshade.py --store "$POISON_STORE" --model "$POISON_STORE/assets/sd15" \
          --target-image "$TIMG" --source "$src" --target "$tgt" --tag "$tag" --lam "${lam:-1}" --eps "$eps" ${MATCH[@]+"${MATCH[@]}"} ${JPG[@]+"${JPG[@]}"} \
          --rank "$r" --world "$NG" > "$REP/gen_${tag}_rank$r.log" 2>&1 &
      fi
    done
    GFAIL=0; for j in $(jobs -p); do wait "$j" || GFAIL=1; done
    [[ $GFAIL == 0 ]] || { echo "poison generation failed for $tag; see $REP/gen_${tag}_rank*.log" >&2; exit 1; }
  fi
  "$PY" scripts/nightshade_poisons.py attach --store "$POISON_STORE" --tag "$tag" | tee "$REP/attach_$tag.log"
  if [[ ${limit:-0} != 0 ]]; then "$PY" scripts/nightshade_poisons.py check-match --official "$BASE" \
      --variant "$POISON_STORE/corpus/manifest_$tag.jsonl" --source "$src" --k "$limit" | tee "$REP/match_$tag.json"; fi
  "$PY" src/paired_dataset.py --manifest "$POISON_STORE/corpus/manifest_$tag.jsonl" --audit-only > "$REP/manifest_audit_$tag.json"
  "$PY" src/audit_attack.py --manifest "$POISON_STORE/corpus/manifest_$tag.jsonl" --out "$REP/pair_audit_$tag.json"
done
echo '{"status":"poisons finished"}' > "$REP/done.json"
