#!/bin/bash
# Runs offline on one GPU machine: run a queue of arms, one arm per visible GPU at a time.
#   EXP=main ARMS="B:200:42 C:200:42 C:200:42:dog2cat_eps02 ..." [STEPS=1000] [EXTRA="--attack-tag car2cow_eps04 --concept car"]
# An optional 4th field is that arm's --attack-tag (lets several attacks share one job instead of idling GPUs).
# Every arm has world size 1 and identical hyper-parameters (scripts/run_arm.py COMMON), so results do not depend
# on how many GPUs the node has. A finished arm is skipped on resubmission (run_arm.py is idempotent).
#
# Exit status: non-zero if ANY arm failed. Each arm's exit code is collected when its GPU slot is reused and at the
# end, so a failure on a reused GPU slot cannot be lost and the phase is never recorded as done after a failure. Per-arm codes: logs/arm-status-<job>.tsv.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
: "${POISON_STORE:?}" "${EXP:?}" "${ARMS:?}"
PY="${PR_PYTHON:-$PWD/.conda-pr/bin/python}"
export HF_HOME="$POISON_STORE/hf_cache" TORCH_HOME="$POISON_STORE/torch_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/src" OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${PR_FAKE_GPUS:-}" ]]; then NG=$PR_FAKE_GPUS; else NG=$(nvidia-smi -L | wc -l); fi   # PR_FAKE_GPUS: CPU test only
JOB=${SLURM_JOB_ID:-local}
RUNNER=(${PR_ARM_RUNNER:-"$PY" scripts/run_arm.py})   # PR_ARM_RUNNER: CPU test hook only
mkdir -p logs; STATUS="logs/arm-status-$JOB.tsv"; : > "$STATUS"
echo "job $JOB gpus=$NG EXP=$EXP ARMS=$ARMS"
[[ -n "${PR_FAKE_GPUS:-}" ]] || nvidia-smi --query-gpu=index,name,memory.total --format=csv
read -ra Q <<< "$ARMS"; BUSY=(); SPEC=(); FAIL=0   # indexed by GPU slot (bash >= 3.2)

reap() {  # slot -> collect the finished arm's exit code (wait on an exited child returns its status immediately)
  local g=$1 pid=${BUSY[$1]:-} rc
  [[ -n $pid ]] || return 0
  wait "$pid"; rc=$?
  printf '%s\t%s\t%s\n' "${SPEC[$g]}" "$rc" "$(date -Is)" >> "$STATUS"
  if [[ $rc != 0 ]]; then FAIL=1; echo "$(date +%T) FAILED ${SPEC[$g]} (exit $rc)"; else echo "$(date +%T) done ${SPEC[$g]}"; fi
  unset "BUSY[$g]"
}

for spec in "${Q[@]}"; do
  IFS=: read -r arm k seed tag <<< "$spec"; ATAG=(); [[ -z ${tag:-} ]] || ATAG=(--attack-tag "$tag")
  while true; do
    for g in $(seq 0 $((NG-1))); do
      pid=${BUSY[$g]:-}
      if [[ -z $pid ]]; then slot=$g; break 2; fi
      if ! kill -0 "$pid" 2>/dev/null; then reap "$g"; slot=$g; break 2; fi
    done
    sleep "${PR_POLL_SECONDS:-15}"
  done
  echo "$(date +%T) start $spec on GPU $slot"
  CUDA_VISIBLE_DEVICES=$slot "${RUNNER[@]}" --exp "$EXP" --arm "$arm" --k "$k" --seed "$seed" \
      --steps "${STEPS:-1000}" ${EXTRA:-} ${ATAG[@]+"${ATAG[@]}"} > "logs/arm-$JOB-$EXP-${arm}_k${k}_s${seed}${tag:+_$tag}.log" 2>&1 &
  BUSY[$slot]=$!; SPEC[$slot]=$spec
  sleep "${PR_LAUNCH_GAP:-20}"
done
for g in "${!BUSY[@]}"; do reap "$g"; done
echo "$(date +%T) all arms finished (any failure: $FAIL)"
cat "$STATUS"
for f in logs/arm-$JOB-$EXP-*.log; do echo "== $f"; tail -1 "$f"; done
exit $FAIL
