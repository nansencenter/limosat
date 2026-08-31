#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/arktalas_vae/bin/python
ROOT=/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020
OUTPUT="$ROOT/fair_orb_aliked_runtime_v3_20260819"
MANIFEST=experiments/configs/fair_orb_aliked_runtime_20260819.json
ORB_CONFIG=experiments/configs/limosat_dense_30000_full70_local.yaml
PAIR_ONE="$ROOT/learned_feature_pilots/dense_pair_10245_10341_aliked1024_fixedgrid_cache_fullcontext_dev_marchseq_v1_20260818"
PAIR_TWO="$ROOT/learned_feature_pilots/dense_pair_10341_10352_aliked1024_fixedgrid_cache_fullcontext_dev_marchseq_v1_20260818"
export MPLCONFIGDIR="$OUTPUT/matplotlib"

mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cp "$MANIFEST" "$OUTPUT/frozen_benchmark_manifest.json"

run_orb() {
  local label="$1"
  local cache="$2"
  local parent="$OUTPUT/orb/$label"
  local suffix="fair_runtime_${label}"
  mkdir -p "$parent"
  if find "$parent" -name run_manifest.json -type f -maxdepth 3 | grep -q .; then
    echo "SKIP complete ORB $label"
    return
  fi
  mkdir -p "$cache"
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
    --grid-cache-dir "$cache" \
    --output-root "$parent/runs" \
    --storage-root "$parent/storage" \
    --log-dir "$parent/logs" \
    --run-suffix "$suffix" \
    --detailed-timing
}

run_aliked() {
  local label="$1"
  local cache="$2"
  local out="$OUTPUT/aliked/$label"
  if [[ -s "$out/run_manifest.json" ]]; then
    echo "SKIP complete ALIKED $label"
    return
  fi
  mkdir -p "$out" "$cache"
  "$PYTHON" experiments/run_aliked_selected_sequence.py \
    --pair-run-dir "$PAIR_ONE" \
    --pair-run-dir "$PAIR_TWO" \
    --output-dir "$out" \
    --feature-cache-dir "$cache" \
    --features-per-tile 1024 \
    --sequential-prior \
    --sequential-prior-uncertainty-m 15000 \
    --maximum-radius-m 6000 \
    --candidate-count 12 \
    --minimum-selected-vectors 8 \
    --consensus-radius-m 1000 \
    --device cpu
}

run_orb setup "$OUTPUT/caches/orb_warm"
run_aliked setup "$OUTPUT/caches/aliked_warm"

for repetition in 1 2 3; do
  run_orb "cold_rep${repetition}" "$OUTPUT/caches/orb_cold_rep${repetition}"
  run_aliked "cold_rep${repetition}" "$OUTPUT/caches/aliked_cold_rep${repetition}"
done

for repetition in 1 2 3; do
  run_orb "warm_rep${repetition}" "$OUTPUT/caches/orb_warm"
  run_aliked "warm_rep${repetition}" "$OUTPUT/caches/aliked_warm"
done
