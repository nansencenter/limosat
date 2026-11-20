#!/usr/bin/env bash
#SBATCH --nodes=1
# One stage of the chained Parquet run on Olivia; each stage submits the next:
#   prepare (CPU: plan, catalogue, scene checksums) -> primary-pairs (GPU array) -> primary-compose (CPU)
# Each GPU task runs LIMOSAT_PROCESSES_PER_GPU matcher processes: pair work is mostly CPU-side tile sampling and
# field fitting, so one process keeps an H200 only ~15 % busy. Olivia GPUs are in Exclusive_Process mode, so the
# processes share the GPU through an MPS daemon started by the task. On the January 2020 test week, 8 processes ran
# 5.5x faster than one with bit-identical pair products; 10 ran 6.3x (GPU 86 % busy, 60 GB), 12 and 14 no faster.
# GPU arrays are sized and time-limited from their pair count (see submit_chain_olivia.sh).
# STOP_AFTER=primary-pairs (at submission) matches only; compose later with --start-at primary-compose.

set -euo pipefail

: "${LIMOSAT_METHOD_ROOT:?}"
: "${LIMOSAT_EXPECTED_REVISION:?}"
: "${LIMOSAT_RUN_ID:?}"
: "${LIMOSAT_RUN_ROOT:?}"
: "${LIMOSAT_CONFIG:?}"
: "${LIMOSAT_CONFIG_SHA256:?}"
: "${LIMOSAT_CATALOGUE:?}"
: "${LIMOSAT_CATALOGUE_SHA256:?}"
: "${LIMOSAT_DATA_BINDS:?}"
: "${LIMOSAT_OFFICIAL_REPOSITORY:?}"
: "${LIMOSAT_OFFICIAL_REVISION:?}"
: "${LIMOSAT_CHECKPOINT:?}"
: "${LIMOSAT_CHECKPOINT_SHA256:?}"
: "${LIMOSAT_CONTAINER:?}"
: "${LIMOSAT_OVERLAY:?}"
: "${LIMOSAT_STAGE:?}"
: "${LIMOSAT_PROCESSES_PER_GPU:?}"
: "${LIMOSAT_SUBMIT:?}"

finish() {
  code=$?
  echo "job_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$code"
}
trap finish EXIT

check_revision() {
  local actual
  actual=$(git -C "$1" rev-parse HEAD)
  [[ "$actual" == "$2" ]] || { echo "$3 revision mismatch: expected $2, found $actual" >&2; exit 2; }
  [[ -z "$(git -C "$1" status --porcelain)" ]] || { echo "$3 checkout changed after submission" >&2; exit 2; }
}
check_sha256() {
  local actual
  actual=$(sha256sum "$1" | awk '{print $1}')
  [[ "$actual" == "$2" ]] || { echo "$3 SHA256 changed after submission" >&2; exit 2; }
}
check_revision "$LIMOSAT_METHOD_ROOT" "$LIMOSAT_EXPECTED_REVISION" "LiMOSAT EfficientLoFTR"
check_revision "$LIMOSAT_OFFICIAL_REPOSITORY" "$LIMOSAT_OFFICIAL_REVISION" "EfficientLoFTR"
check_sha256 "$LIMOSAT_CONFIG" "$LIMOSAT_CONFIG_SHA256" "configuration"
check_sha256 "$LIMOSAT_CATALOGUE" "$LIMOSAT_CATALOGUE_SHA256" "catalogue"
check_sha256 "$LIMOSAT_CHECKPOINT" "$LIMOSAT_CHECKPOINT_SHA256" "checkpoint"

checkpoint_directory=$(dirname "$LIMOSAT_CHECKPOINT")
mps="${TMPDIR:-/tmp}/limosat-mps-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$mps/pipe" "$mps/log"
run_in_container() {
  local -a options=(exec --cleanenv --no-home)
  [[ "$LIMOSAT_STAGE" != *-pairs ]] || options+=(--nv)
  local path
  # Inputs (scenes, OSI-SAF, AMSR2 ASI, land raster) are bound read-only from a colon-separated list.
  IFS=: read -r -a data_binds <<< "$LIMOSAT_DATA_BINDS"
  for path in "${data_binds[@]}"; do options+=(--bind "$path:$path:ro"); done
  [[ -z "${LIMOSAT_PAIR_STORE:-}" ]] || options+=(--bind "$LIMOSAT_PAIR_STORE:$LIMOSAT_PAIR_STORE")
  env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
    apptainer "${options[@]}" \
    --overlay "$LIMOSAT_OVERLAY:ro" \
    --bind "$LIMOSAT_METHOD_ROOT:$LIMOSAT_METHOD_ROOT:ro" \
    --bind "$LIMOSAT_OFFICIAL_REPOSITORY:$LIMOSAT_OFFICIAL_REPOSITORY:ro" \
    --bind "$checkpoint_directory:$checkpoint_directory:ro" \
    --bind "$LIMOSAT_RUN_ROOT:$LIMOSAT_RUN_ROOT" \
    --env PYTHONUSERBASE=/home/apptainer/user_software \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH="$LIMOSAT_METHOD_ROOT" \
    --bind "$mps:$mps" --env CUDA_MPS_PIPE_DIRECTORY="$mps/pipe" \
    "$LIMOSAT_CONTAINER" "$@"
}
limosat() { run_in_container python -m limosat "$@"; }

array_index="${SLURM_ARRAY_TASK_ID:-0}"
control="$LIMOSAT_RUN_ROOT/control"
outputs="$LIMOSAT_RUN_ROOT/outputs"
ledger="$control/chain.log"
run_json() {
  local destination="$1"
  shift
  "$@" > "$destination.tmp.${SLURM_JOB_ID:-local}.$array_index"
  mv "$destination.tmp.${SLURM_JOB_ID:-local}.$array_index" "$destination"
}
# The submitter script, re-entered with a stage name, prints the new job ID.
submit() { "$LIMOSAT_SUBMIT" --stage "$1" --after "${2:-}" --pairs "${3:-0}"; }

echo "job_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) stage=$LIMOSAT_STAGE"
echo "method_revision=$LIMOSAT_EXPECTED_REVISION official_revision=$LIMOSAT_OFFICIAL_REVISION"

case "$LIMOSAT_STAGE" in
  prepare)
    run_in_container python - "$LIMOSAT_CONFIG" <<'PY'
import sys
from dataclasses import replace

import pyarrow

from limosat.catalog import load_catalogue
from limosat.config import load_config
from limosat.efficientloftr import EfficientLoFTR
from limosat.surface import surface_gate

config = load_config(sys.argv[1])
EfficientLoFTR(replace(config.matcher, device="cpu"))
gate = surface_gate(config)
assert gate is not None and gate.land is not None, "land masking is not configured"
# Reading one day's ASI file checks that the container's GDAL can open the product.
first_day = load_catalogue(config.catalogue, config.analysis_epsg).chronological()[0].time_utc
assert gate.concentration._day(first_day) is not None, "no ASI concentration for %s" % first_day
print("pyarrow=%s surface=%s" % (pyarrow.__version__, config.surface))
PY
    run_json "$control/plan.json" limosat plan "$LIMOSAT_CONFIG"
    run_json "$control/prepare-stage.json" limosat prepare "$LIMOSAT_CONFIG"
    pairs=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["primary_pairs"])' "$control/plan.json")
    gpu=$(submit primary-pairs "" "$pairs")
    # STOP_AFTER=primary-pairs matches only, e.g. while composition is being changed; compose later by hand.
    [[ "${LIMOSAT_STOP_AFTER:-}" == primary-pairs ]] || submit primary-compose "$gpu"
    ;;
  primary-pairs)
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv,noheader
    export CUDA_MPS_PIPE_DIRECTORY="$mps/pipe" CUDA_MPS_LOG_DIRECTORY="$mps/log"
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control; finish' EXIT
    tasks="${SLURM_ARRAY_TASK_COUNT:-1}"
    workers=()
    for (( k = 0; k < LIMOSAT_PROCESSES_PER_GPU; k++ )); do
      batch=$(( array_index * LIMOSAT_PROCESSES_PER_GPU + k ))
      run_json "$control/primary-pairs-${batch}.json" \
        limosat pairs "$LIMOSAT_CONFIG" --batch-index "$batch" \
        --batch-count $(( tasks * LIMOSAT_PROCESSES_PER_GPU )) &
      workers+=($!)
    done
    for worker in "${workers[@]}"; do wait "$worker"; done
    ;;
  primary-compose)
    run_json "$control/primary-compose.json" \
      limosat compose-parquet "$LIMOSAT_CONFIG" "$outputs/primary-composition"
    echo "$(date -u +%FT%TZ) RUN COMPLETE: primary composed" >> "$ledger"
    ;;
  *)
    echo "unknown LIMOSAT_STAGE: $LIMOSAT_STAGE" >&2
    exit 2
    ;;
esac
