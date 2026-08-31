#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/limosat_scaling/bin/python
ROOT=/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020
DATABASE="$ROOT/operational_baseline/limosat_dense_30000__pm_q24_i48_quadratic_bilinear_full70_20260818.sqlite"
TABLE=limosat_dense_30000__pm_q24_i48_quadratic_bilinear_full70_20260818
FIELD="$ROOT/learned_feature_pilots/dense_pair_10245_10352_nearest12_flipnode_q4_v1_20260818/flip_rejected_field.csv"
ICESAT_RAW="$ROOT/icesat2_validation/raw"
ICESAT_OUTPUT="$ROOT/icesat2_validation/results/selected_fold_rejected_v3/pair_10245_10352"
CRYOSAT_OUTPUT="$ROOT/cryosat2_validation/results/selected_fold_rejected_v2/pair_10245_10352"
PAIR_START=2020-03-28T12:13:29Z
PAIR_END=2020-03-29T11:16:05Z

for specification in \
  "0040 1000 $ICESAT_RAW/ATL07/007/2020/03/29/ATL07-01_20200329033603_00400701_007_01.h5" \
  "0040 4000 $ICESAT_RAW/ATL07/007/2020/03/29/ATL07-01_20200329033603_00400701_007_01.h5" \
  "0044 1000 $ICESAT_RAW/ATL07/007/2020/03/29/ATL07-01_20200329095312_00440701_007_01.h5" \
  "0044 4000 $ICESAT_RAW/ATL07/007/2020/03/29/ATL07-01_20200329095312_00440701_007_01.h5"
do
  read -r track resolution product <<< "$specification"
  "$PYTHON" experiments/validate_icesat2_deformation.py \
    --atl07 "$product" \
    --orb-database "$DATABASE" \
    --orb-table "$TABLE" \
    --orb-source-image-id 50 \
    --orb-target-image-id 53 \
    --aliked-field "$FIELD" \
    --pair-start "$PAIR_START" \
    --pair-end "$PAIR_END" \
    --bin-size-m "$resolution" \
    --orb-endpoint-error-p90-m 1310 \
    --aliked-endpoint-error-p90-m 1301 \
    --sar-source-product-id 10245 \
    --sar-target-product-id 10352 \
    --candidate-inclusion-reason "Frozen existing event; correction replaces the audited pre-final ALIKED field with the selected fixed-point fold-rejected field without changing thresholds or support rules." \
    --analysis-role development \
    --output-dir "$ICESAT_OUTPUT/atl07_${track}_${resolution}m"
done

"$PYTHON" experiments/validate_cryosat2_deformation.py \
  --cryosat-dir "$ROOT/cryosat2_validation/raw/RDWES1B/1/2020/03" \
  --orb-database "$DATABASE" \
  --orb-table "$TABLE" \
  --orb-source-image-id 50 \
  --orb-target-image-id 53 \
  --aliked-field "$FIELD" \
  --pair-start "$PAIR_START" \
  --pair-end "$PAIR_END" \
  --orb-endpoint-error-p90-m 1310 \
  --aliked-endpoint-error-p90-m 1301 \
  --null-repetitions 999 \
  --sar-source-product-id 10245 \
  --sar-target-product-id 10352 \
  --candidate-inclusion-reason "Frozen existing event; correction replaces the audited pre-final ALIKED field with the selected fixed-point fold-rejected field without changing thresholds or support rules." \
  --analysis-role development \
  --output-dir "$CRYOSAT_OUTPUT"
