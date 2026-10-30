#!/usr/bin/env bash
#SBATCH --nodes=1
# One stage of the chained Parquet run on Olivia. Pass controllers submit the next stage themselves, so the number
# of continuation passes follows the stop rule instead of being fixed at submission:
#   prepare -> primary-pairs -> primary-compose -> pass 1 -> pass 2 -> ... -> finalize
# A pass is: pass (CPU: base + targets + stop rule) -> recovery-pairs (GPU array) -> pass-compose (CPU).
# Each GPU task runs LIMOSAT_PROCESSES_PER_GPU matcher processes: pair work is mostly CPU-side tile sampling and
# field fitting, so one process keeps an H200 only ~15 % busy. GPU arrays are sized and time-limited from their
# pair count (see submit_chain_olivia.sh) so that small passes backfill quickly.
# Stop when a pass targets fewer than LIMOSAT_MINIMUM_PASS_PAIRS pairs, when its accepted continuation links per
# matcher tile fall below LIMOSAT_MINIMUM_GAIN_SHARE of pass 1's, or after LIMOSAT_MAXIMUM_PASSES.

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
LIMOSAT_PASS="${LIMOSAT_PASS:-0}"
LIMOSAT_MINIMUM_PASS_PAIRS="${LIMOSAT_MINIMUM_PASS_PAIRS:-50}"
LIMOSAT_MINIMUM_GAIN_SHARE="${LIMOSAT_MINIMUM_GAIN_SHARE:-0.3}"
LIMOSAT_MAXIMUM_PASSES="${LIMOSAT_MAXIMUM_PASSES:-12}"

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
run_in_container() {
  local -a options=(exec --cleanenv --no-home)
  [[ "$LIMOSAT_STAGE" != *-pairs ]] || options+=(--nv)
  local path
  # Inputs (scenes, OSI-SAF, AMSR2 ASI, land raster) are bound read-only from a colon-separated list.
  IFS=: read -r -a data_binds <<< "$LIMOSAT_DATA_BINDS"
  for path in "${data_binds[@]}"; do options+=(--bind "$path:$path:ro"); done
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
    "$LIMOSAT_CONTAINER" "$@"
}
limosat() { run_in_container python -m limosat "$@"; }

array_index="${SLURM_ARRAY_TASK_ID:-0}"
control="$LIMOSAT_RUN_ROOT/control"
outputs="$LIMOSAT_RUN_ROOT/outputs"
passes="$LIMOSAT_RUN_ROOT/passes"
ledger="$control/passes.log"
run_json() {
  local destination="$1"
  shift
  "$@" > "$destination.tmp.${SLURM_JOB_ID:-local}.$array_index"
  mv "$destination.tmp.${SLURM_JOB_ID:-local}.$array_index" "$destination"
}
# The submitter script, re-entered with a stage name, prints the new job ID.
submit() { LIMOSAT_PASS="$2" "$LIMOSAT_SUBMIT" --stage "$1" --after "${3:-}" --pairs "${4:-0}"; }

echo "job_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) stage=$LIMOSAT_STAGE pass=$LIMOSAT_PASS"
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
    gpu=$(submit primary-pairs 0 "" "$pairs")
    submit primary-compose 0 "$gpu"
    ;;
  primary-pairs|recovery-pairs)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    kind="${LIMOSAT_STAGE%-pairs}"
    extra=()
    [[ "$kind" == primary ]] || extra=(--recovery-targets "$passes/p$LIMOSAT_PASS/targets")
    tasks="${SLURM_ARRAY_TASK_COUNT:-1}"
    workers=()
    for (( k = 0; k < LIMOSAT_PROCESSES_PER_GPU; k++ )); do
      batch=$(( array_index * LIMOSAT_PROCESSES_PER_GPU + k ))
      run_json "$control/${kind}-pairs-p${LIMOSAT_PASS}-${batch}.json" \
        limosat pairs "$LIMOSAT_CONFIG" --kind "$kind" --batch-index "$batch" \
        --batch-count $(( tasks * LIMOSAT_PROCESSES_PER_GPU )) "${extra[@]}" &
      workers+=($!)
    done
    for worker in "${workers[@]}"; do wait "$worker"; done
    ;;
  primary-compose)
    run_json "$control/primary-compose.json" \
      limosat compose-parquet "$LIMOSAT_CONFIG" "$outputs/primary-composition"
    echo "$(date -u +%FT%TZ) primary composed" >> "$ledger"
    submit pass 1
    ;;
  pass)
    n="$LIMOSAT_PASS"
    pass_root="$passes/p$n"
    mkdir -p "$pass_root"
    if (( n == 1 )); then
      base="$outputs/primary-composition"
    else
      base="$pass_root/base"
      limosat merge-recovery-base "$LIMOSAT_CONFIG" "$(cat "$passes/p$((n - 1))/BASE")" \
        "$passes/p$((n - 1))/aug" "$base" --pass-index $((n - 1))
    fi
    echo "$base" > "$pass_root/BASE"
    limosat prepare-recovery-parquet "$LIMOSAT_CONFIG" "$base" "$pass_root/targets"
    pairs=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["targeted_recovery_pairs"])' \
      "$pass_root/targets/recovery-targets-manifest-v1.json")
    echo "$(date -u +%FT%TZ) pass $n targeted_pairs $pairs" >> "$ledger"
    if (( pairs < LIMOSAT_MINIMUM_PASS_PAIRS )) || (( n > LIMOSAT_MAXIMUM_PASSES )); then
      echo "$(date -u +%FT%TZ) STOP before pass $n: pairs $pairs" >> "$ledger"
      submit finalize $((n - 1))
      exit 0
    fi
    gpu=$(submit recovery-pairs "$n" "" "$pairs")
    submit pass-compose "$n" "$gpu"
    ;;
  pass-compose)
    n="$LIMOSAT_PASS"
    pass_root="$passes/p$n"
    limosat compose-recovery-parquet "$LIMOSAT_CONFIG" "$(cat "$pass_root/BASE")" \
      "$pass_root/targets" "$pass_root/aug"
    # Gain: accepted continuation links per matcher tile spent on this pass's pairs.
    decision=$(python3 - "$passes" "$n" "$LIMOSAT_RUN_ROOT" "$LIMOSAT_MINIMUM_GAIN_SHARE" <<'PY'
import json, sys
from pathlib import Path

passes, n, run_root, share = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
manifest = json.load(open(passes / f"p{n}/aug/recovery-composition-manifest-v1.json"))
links = manifest["counts"]["trajectory_augmentation_extension_kinds"].get("recovery", 0)
# Target markers and recovery products share the per-pair file stem.
tiles = 0
for target in (passes / f"p{n}/targets/pairs").glob("*.json"):
    path = run_root / "work/pair-products/recovery" / target.name
    if path.is_file():
        tiles += json.load(open(path))["diagnostics"].get("matcher_tile_evaluations", 0)
gain = links / max(tiles, 1)
record = {"pass": n, "links": links, "matcher_tiles": tiles, "gain": gain}
(passes / f"p{n}/GAIN.json").write_text(json.dumps(record))
first = json.load(open(passes / "p1/GAIN.json"))["gain"]
print("stop" if n > 1 and gain < share * first else "continue", json.dumps(record))
PY
)
    echo "$(date -u +%FT%TZ) pass $n ${decision#* } -> ${decision%% *}" >> "$ledger"
    if [[ "${decision%% *}" == stop ]]; then
      submit finalize "$n"
    else
      submit pass $((n + 1))
    fi
    ;;
  finalize)
    n="$LIMOSAT_PASS"
    final="$outputs/final-base"
    if (( n == 0 )); then
      ln -sfn "$outputs/primary-composition" "$final"
    else
      limosat merge-recovery-base "$LIMOSAT_CONFIG" "$(cat "$passes/p$n/BASE")" \
        "$passes/p$n/aug" "$final" --pass-index "$n"
    fi
    limosat annotate-links "$LIMOSAT_CONFIG" "$final" "$outputs/annotated"
    echo "$(date -u +%FT%TZ) RUN COMPLETE after pass $n" >> "$ledger"
    ;;
  *)
    echo "unknown LIMOSAT_STAGE: $LIMOSAT_STAGE" >&2
    exit 2
    ;;
esac
