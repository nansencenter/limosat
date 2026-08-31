#!/usr/bin/env bash
set -euo pipefail

# Frozen LiMOSAT image input: balanced model, fixed q2..q98, post-VAE CLAHE 2.5.
MODE="${MODE:-pilot}"
MONTH="${MONTH:-01}"
WORKERS="${WORKERS:-2}"

DATA_ROOT="/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020"
RAW_ROOT="$DATA_ROOT/sentinel1/raw/2020"
OUTPUT_ROOT="$DATA_ROOT/sentinel1/standard_vae/2020"
LOG_ROOT="$DATA_ROOT/logs/standard_vae_balanced_q2q98_clahe25"
AUX_ROOT="$DATA_ROOT/auxiliary"
VAE_ROOT="/Users/seachu/projects/arktalas_vae"
PYTHON="/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/arktalas_vae/bin/python"
WEIGHTS="$VAE_ROOT/reports/generalist_balanced_20260307/weights/vae_64_2_24_ELU_generalist_balanced_20260307_selected_best.pth"
NORM="$VAE_ROOT/reports/all_out_combo_20260305/data/mean_std_allout_64.npz"
MOD44W="/Users/seachu/data/arktalas_vae/data/MOD44W"
PYTHESINT_SEED="/Users/seachu/.local/share/pythesint/json"
S1DENOISE_SEED="/Users/seachu/results/arktalas_vae/local_state/.xdg-data/.s1denoise"

EXPECTED_WEIGHTS_SHA256="1a8a2f1738c780ee97b819c2b75bb47612f05fbd93338731a7e0a79674638de6"
EXPECTED_NORM_SHA256="1a4463b62ad3e45fef618beaffd8de85b2948268d3ec6601dc63b5fa266555a9"
FIXED_EXPORT_LO="-1.4830764532089233"
FIXED_EXPORT_HI="1.2564970254898071"

if [[ ! -d /Volumes/KINGSTON ]] || [[ ! -d "$RAW_ROOT" ]]; then
  echo "KINGSTON raw-data root is not mounted: $RAW_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]] || [[ ! -f "$WEIGHTS" ]] || [[ ! -f "$NORM" ]]; then
  echo "Frozen VAE runtime, weights, or normalization file is missing." >&2
  exit 2
fi
if [[ ! -d "$MOD44W" ]] || [[ ! -d "$PYTHESINT_SEED" ]] || [[ ! -d "$S1DENOISE_SEED" ]]; then
  echo "Required MOD44W, pythesint, or s1denoise auxiliary data is missing." >&2
  exit 2
fi

actual_weights_sha256="$(shasum -a 256 "$WEIGHTS" | awk '{print $1}')"
actual_norm_sha256="$(shasum -a 256 "$NORM" | awk '{print $1}')"
if [[ "$actual_weights_sha256" != "$EXPECTED_WEIGHTS_SHA256" ]]; then
  echo "Frozen VAE weights checksum mismatch." >&2
  exit 2
fi
if [[ "$actual_norm_sha256" != "$EXPECTED_NORM_SHA256" ]]; then
  echo "Frozen VAE normalization checksum mismatch." >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT_ROOT" \
  "$LOG_ROOT" \
  "$AUX_ROOT/xdg_data/pythesint/json" \
  "$AUX_ROOT/xdg_data/.s1denoise" \
  "$AUX_ROOT/matplotlib" \
  "$AUX_ROOT/tmp"
cp "$PYTHESINT_SEED"/*.json "$AUX_ROOT/xdg_data/pythesint/json/"
cp -R "$S1DENOISE_SEED"/. "$AUX_ROOT/xdg_data/.s1denoise/"
export MOD44WPATH="$MOD44W"
export XDG_DATA_HOME="$AUX_ROOT/xdg_data"
export MPLCONFIGDIR="$AUX_ROOT/matplotlib"
export TMPDIR="$AUX_ROOT/tmp"

run_month() {
  local month="$1"
  local month_workers="$2"
  local max_files="$3"
  local input_root="$4"
  local output_root="$5"
  local log_slug="$6"
  local input_dir="$input_root/$month"
  local output_dir="$output_root/$month"
  local log_path="$LOG_ROOT/${log_slug}_2020_${month}.log"
  mkdir -p "$output_dir"

  vae_command=(
    "$PYTHON" "$VAE_ROOT/preprocess.py"
    "$input_dir" "$output_dir"
    --weights-file "$WEIGHTS"
    --norm-file "$NORM"
    --grid-size 64
    --n-layers 2
    --hidden-size 24
    --export-scaling-mode fixed_range
    --fixed-export-lo "$FIXED_EXPORT_LO"
    --fixed-export-hi "$FIXED_EXPORT_HI"
    --post-vae-clahe
    --clahe-clip-limit 2.5
    --hv-thermal-denoise auto
    --device cpu
    --workers "$month_workers"
    --compression LZW
    --compression-threads 2
    --gdal-cache 512
    --torch-threads 2
    --resume
    --log-file "$log_path"
  )
  if [[ "$max_files" -gt 0 ]]; then
    vae_command+=(--max-files "$max_files")
  fi
  (
    cd "$LOG_ROOT"
    "${vae_command[@]}"
  )
  expected_count="$(find "$input_dir" -maxdepth 1 -type f -name 'S1?_EW_GRDM_1SDH_*.zip' ! -name '._*' | wc -l | tr -d ' ')"
  if [[ "$max_files" -gt 0 ]] && [[ "$expected_count" -gt "$max_files" ]]; then
    expected_count="$max_files"
  fi
  actual_count="$(find "$output_dir" -maxdepth 1 -type f -name 'S1?_EW_GRDM_1SDH_*.tiff' ! -name '._*' | wc -l | tr -d ' ')"
  if [[ "$actual_count" -lt "$expected_count" ]]; then
    echo "VAE output count is incomplete for 2020-$month: $actual_count / $expected_count" >&2
    exit 1
  fi
}

case "$MODE" in
  pilot)
    run_month "$MONTH" 1 1 "$RAW_ROOT" "$OUTPUT_ROOT" "full70"
    ;;
  full)
    for month in 01 02 03 04; do
      run_month "$month" "$WORKERS" 0 "$RAW_ROOT" "$OUTPUT_ROOT" "full70"
    done
    ;;
  controls)
    control_raw_root="$DATA_ROOT/sentinel1/repeat_publication_controls/raw/2020"
    control_output_root="$DATA_ROOT/sentinel1/repeat_publication_controls/standard_vae/2020"
    for month in 03 04; do
      run_month \
        "$month" \
        "$WORKERS" \
        0 \
        "$control_raw_root" \
        "$control_output_root" \
        "repeat_publication_controls"
    done
    ;;
  *)
    echo "MODE must be pilot, full, or controls, got: $MODE" >&2
    exit 2
    ;;
esac
