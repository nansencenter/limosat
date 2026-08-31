#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/arktalas_vae/bin/python
ROOT=/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020
OUTPUT="$ROOT/fair_orb_aliked_runtime_v3_20260819"
ORB_CONFIG=experiments/configs/limosat_dense_30000_full70_local.yaml
PAIR_ONE="$ROOT/learned_feature_pilots/dense_pair_10245_10341_aliked1024_fixedgrid_cache_fullcontext_dev_marchseq_v1_20260818"
PAIR_TWO="$ROOT/learned_feature_pilots/dense_pair_10341_10352_aliked1024_fixedgrid_cache_fullcontext_dev_marchseq_v1_20260818"
LABEL=warm_rep4
ORB_PARENT="$OUTPUT/orb/$LABEL"
ALIKED_OUTPUT="$OUTPUT/aliked/$LABEL"
export MPLCONFIGDIR="$OUTPUT/matplotlib"

if find "$ORB_PARENT" -name run_manifest.json -type f -maxdepth 3 2>/dev/null | grep -q . || [[ -s "$ALIKED_OUTPUT/run_manifest.json" ]]; then
  echo "Replacement output already exists; refusing to overwrite $LABEL." >&2
  exit 2
fi

mkdir -p "$ORB_PARENT" "$ALIKED_OUTPUT" "$MPLCONFIGDIR"
"$PYTHON" experiments/run_operational_baseline.py \
  --config "$ORB_CONFIG" \
  --mode dense_operational \
  --catalog-image-ids 10245,10341,10352 \
  --model-estimator legacy_homography \
  --model-coordinate-scale-m 1000 \
  --model-threshold-m 15000 \
  --border-matched 24 \
  --border-interpolated 48 \
  --pattern-matching-subpixel-method quadratic \
  --template-sampling bilinear \
  --grid-cache-dir "$OUTPUT/caches/orb_warm" \
  --output-root "$ORB_PARENT/runs" \
  --storage-root "$ORB_PARENT/storage" \
  --log-dir "$ORB_PARENT/logs" \
  --run-suffix fair_runtime_${LABEL} \
  --detailed-timing

"$PYTHON" experiments/run_aliked_selected_sequence.py \
  --pair-run-dir "$PAIR_ONE" \
  --pair-run-dir "$PAIR_TWO" \
  --output-dir "$ALIKED_OUTPUT" \
  --feature-cache-dir "$OUTPUT/caches/aliked_warm" \
  --features-per-tile 1024 \
  --sequential-prior \
  --sequential-prior-uncertainty-m 15000 \
  --maximum-radius-m 6000 \
  --candidate-count 12 \
  --minimum-selected-vectors 8 \
  --consensus-radius-m 1000 \
  --device cpu
