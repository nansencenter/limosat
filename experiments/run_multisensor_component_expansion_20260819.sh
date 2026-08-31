#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/limosat_scaling/bin/python
ROOT=/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020
RAW="$ROOT/icesat2_validation/raw"
DATABASE="$ROOT/operational_baseline/limosat_dense_30000__pm_q24_i48_quadratic_bilinear_full70_20260818.sqlite"
TABLE=limosat_dense_30000__pm_q24_i48_quadratic_bilinear_full70_20260818
FIELD="$ROOT/learned_feature_pilots/dense_pair_10245_10341_nearest12_flipnode_q4_v1_20260819/flip_rejected_field.csv"
OUTPUT="$ROOT/icesat2_validation/results/selected_expansion_v1/pair_10245_10341"
PAIR_START=2020-03-28T12:13:29Z
PAIR_END=2020-03-29T07:59:46Z
REASON="Frozen 2026-08-19 component-pair application: every newly retrieved granule intersecting 10245-to-10341 is retained, independent of deformation association or altimetry outcome."

for specification in \
  "0030 1000 $RAW/ATL07/007/2020/03/28/ATL07-01_20200328115309_00300701_007_01.h5" \
  "0030 4000 $RAW/ATL07/007/2020/03/28/ATL07-01_20200328115309_00300701_007_01.h5" \
  "0041 1000 $RAW/ATL07/007/2020/03/29/ATL07-01_20200329051020_00410701_007_01.h5" \
  "0041 4000 $RAW/ATL07/007/2020/03/29/ATL07-01_20200329051020_00410701_007_01.h5"
do
  read -r track resolution product_path <<< "$specification"
  "$PYTHON" experiments/validate_icesat2_deformation.py \
    --atl07 "$product_path" --orb-database "$DATABASE" --orb-table "$TABLE" \
    --orb-source-image-id 50 --orb-target-image-id 52 --aliked-field "$FIELD" \
    --pair-start "$PAIR_START" --pair-end "$PAIR_END" --bin-size-m "$resolution" \
    --orb-endpoint-error-p90-m 1310 --aliked-endpoint-error-p90-m 1301 \
    --sar-source-product-id 10245 --sar-target-product-id 10341 \
    --candidate-inclusion-reason "$REASON" --analysis-role confirmation \
    --output-dir "$OUTPUT/atl07_${track}_${resolution}m"
done

for specification in \
  "0030 $RAW/ATL10/007/2020/03/28/ATL10-01_20200328115309_00300701_007_01.h5" \
  "0039 $RAW/ATL10/007/2020/03/29/ATL10-01_20200329020145_00390701_007_01.h5" \
  "0040 $RAW/ATL10/007/2020/03/29/ATL10-01_20200329033603_00400701_007_01.h5" \
  "0041 $RAW/ATL10/007/2020/03/29/ATL10-01_20200329051020_00410701_007_01.h5"
do
  read -r track product_path <<< "$specification"
  "$PYTHON" experiments/validate_icesat2_deformation.py \
    --atl10 "$product_path" --orb-database "$DATABASE" --orb-table "$TABLE" \
    --orb-source-image-id 50 --orb-target-image-id 52 --aliked-field "$FIELD" \
    --pair-start "$PAIR_START" --pair-end "$PAIR_END" --bin-size-m 4000 \
    --orb-endpoint-error-p90-m 1310 --aliked-endpoint-error-p90-m 1301 \
    --minimum-spatial-shift-m 20000 \
    --sar-source-product-id 10245 --sar-target-product-id 10341 \
    --candidate-inclusion-reason "$REASON" --analysis-role confirmation \
    --output-dir "$OUTPUT/atl10_${track}_4000m"
done
