#!/bin/bash
# Stage ALL network inputs into $POISON_STORE (idempotent; re-run after an interruption).
# Must run where there is internet and NO heavy compute is needed: a transfer node, or a compute job with internet access.
#   POISON_STORE=/path/to/store bash scripts/pipeline/stage_all.sh
# Downloads (~12 GB): SD1.5 @451f4fe (training model), SD2.1 community mirror @bb21548 (Nightshade surrogate),
# CLIP ViT-B/32 + ViT-L/14, torchvision ConvNeXt-Base + Inception-v3, COCO-2017 annotations + ~4.6k rights-filtered
# images; verifies the Nightshade checkout is the audited commit.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${POISON_STORE:?set POISON_STORE to large project storage (not home)}"
PY="${PR_PYTHON:-$PWD/.conda-pr/bin/python}"
export HF_HOME="$POISON_STORE/hf_cache" TORCH_HOME="$POISON_STORE/torch_cache" POISON_ASSET_ROOT="$POISON_STORE/assets"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src:$PWD/scripts" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
mkdir -p "$POISON_STORE/logs"
LOG="$POISON_STORE/logs/stage_all.log"
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

COMMIT=28e4ee9115ed07a7fd463f297fefcea08c5ee0bf
if [[ ! -d third_party/nightshade-release/.git ]]; then
  step "clone nightshade-release"
  git clone https://github.com/Shawn-Shan/nightshade-release.git third_party/nightshade-release
fi
git -C third_party/nightshade-release checkout -q "$COMMIT"
[[ "$(git -C third_party/nightshade-release rev-parse HEAD)" == "$COMMIT" ]] || { echo "nightshade commit mismatch" >&2; exit 1; }
[[ -z "$(git -C third_party/nightshade-release status --porcelain)" ]] || { echo "nightshade checkout has local changes" >&2; exit 1; }
step "nightshade-release OK at $COMMIT"
ETH=b4bf72b803c9bc793db085da2d33009f0fb7da7f   # Hoenig et al. 2025 robust mimicry (MIT): arms M1/M2
[[ -d third_party/robust-style-mimicry/.git ]] || git clone https://github.com/ethz-spylab/robust-style-mimicry.git third_party/robust-style-mimicry
git -C third_party/robust-style-mimicry checkout -q "$ETH"
step "robust-style-mimicry OK at $ETH"

step "SD1.5"; "$PY" scripts/stage_sd15.py >> "$LOG" 2>&1
step "SD2.1 surrogate"; "$PY" scripts/stage_nightshade_model.py >> "$LOG" 2>&1
step "aux models"; "$PY" scripts/prefetch_models.py 2>&1 | tee -a "$LOG"
if [[ ! -f "$POISON_STORE/corpus/coco_pool_manifest.jsonl" ]]; then
  step "COCO corpus"; "$PY" scripts/build_coco_corpus.py --store "$POISON_STORE" ${CORPUS_ARGS:-} 2>&1 | grep -v Deprecation | tee -a "$LOG"
fi
for f in assets/sd15/unet/diffusion_pytorch_model.safetensors assets/sd21/vae/diffusion_pytorch_model.fp16.safetensors \
         assets/prefetch.json corpus/coco_pool_manifest.jsonl corpus/corpus_audit.json; do
  [[ -s "$POISON_STORE/$f" ]] || { echo "MISSING after staging: $f" >&2; exit 1; }
done
step "STAGE_COMPLETE store=$POISON_STORE"
