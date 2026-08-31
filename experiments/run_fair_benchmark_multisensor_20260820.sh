#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/limosat_scaling/bin/python
ROOT=/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020
BENCHMARK="$ROOT/fair_orb_aliked_runtime_v3_20260819"
ORB_PARENT="$BENCHMARK/orb/warm_rep1"
ORB_DATABASE="$ORB_PARENT/storage/limosat_dense_30000__fair_runtime_warm_rep1.sqlite"
ORB_TABLE=limosat_dense_30000__fair_runtime_warm_rep1
ALIKED_FIELD="$BENCHMARK/aliked/warm_rep1/pair_10245_10341/field_nearest12_fold_rejected.csv"
ATL07="$ROOT/icesat2_validation/raw/ATL07/007/2020/03/29/ATL07-01_20200329020145_00390701_007_01.h5"
ATL10="$ROOT/icesat2_validation/raw/ATL10/007/2020/03/29/ATL10-01_20200329020145_00390701_007_01.h5"
OUTPUT="$BENCHMARK/comparison/multisensor_warm_rep1"
REASON="Frozen runtime-gate carry-forward of prequalified RGT 0039: selected from acquisition time and overlap before inspecting this benchmark's association."
export MPLCONFIGDIR="$BENCHMARK/matplotlib"

for required in "$ORB_DATABASE" "$ALIKED_FIELD" "$ATL07" "$ATL10"; do
  [[ -s "$required" ]] || { echo "Missing benchmark input: $required" >&2; exit 2; }
done
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"

"$PYTHON" experiments/validate_icesat2_deformation.py \
  --atl07 "$ATL07" \
  --orb-database "$ORB_DATABASE" --orb-table "$ORB_TABLE" \
  --orb-source-image-id 1 --orb-target-image-id 2 \
  --aliked-field "$ALIKED_FIELD" \
  --pair-start 2020-03-28T12:13:29Z --pair-end 2020-03-29T07:59:46Z \
  --bin-size-m 4000 --minimum-spatial-shift-m 20000 \
  --orb-endpoint-error-p90-m 1310 --aliked-endpoint-error-p90-m 1301 \
  --sar-source-product-id 10245 --sar-target-product-id 10341 \
  --candidate-inclusion-reason "$REASON" --analysis-role development \
  --output-dir "$OUTPUT/atl07_0039_4000m"

"$PYTHON" experiments/validate_icesat2_deformation.py \
  --atl10 "$ATL10" \
  --orb-database "$ORB_DATABASE" --orb-table "$ORB_TABLE" \
  --orb-source-image-id 1 --orb-target-image-id 2 \
  --aliked-field "$ALIKED_FIELD" \
  --pair-start 2020-03-28T12:13:29Z --pair-end 2020-03-29T07:59:46Z \
  --bin-size-m 4000 --minimum-spatial-shift-m 20000 \
  --orb-endpoint-error-p90-m 1310 --aliked-endpoint-error-p90-m 1301 \
  --sar-source-product-id 10245 --sar-target-product-id 10341 \
  --candidate-inclusion-reason "$REASON" --analysis-role development \
  --output-dir "$OUTPUT/atl10_0039_4000m"
