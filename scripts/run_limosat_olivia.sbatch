#!/usr/bin/env bash
#SBATCH --nodes=1

set -euo pipefail

: "${LIMOSAT_METHOD_ROOT:?}"
: "${LIMOSAT_EXPECTED_REVISION:?}"
: "${LIMOSAT_RUN_ID:?}"
: "${LIMOSAT_RUN_ROOT:?}"
: "${LIMOSAT_CONFIG:?}"
: "${LIMOSAT_CONFIG_SHA256:?}"
: "${LIMOSAT_CATALOGUE:?}"
: "${LIMOSAT_CATALOGUE_SHA256:?}"
: "${LIMOSAT_SIC_ROOT:?}"
: "${LIMOSAT_SCENE_ROOT:?}"
: "${LIMOSAT_OFFICIAL_REPOSITORY:?}"
: "${LIMOSAT_OFFICIAL_REVISION:?}"
: "${LIMOSAT_CHECKPOINT:?}"
: "${LIMOSAT_CHECKPOINT_SHA256:?}"
: "${LIMOSAT_CONTAINER:?}"
: "${LIMOSAT_OVERLAY:?}"
: "${LIMOSAT_READY_MARKER:?}"
: "${LIMOSAT_CPU_AUDIT:?}"
: "${LIMOSAT_EXPECTED_IMAGES:?}"
: "${LIMOSAT_STAGE:?}"
: "${LIMOSAT_GPU_WORKERS:?}"

finish() {
  code=$?
  echo "job_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$code"
}
trap finish EXIT

check_revision() {
  local repository="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual=$(git -C "$repository" rev-parse HEAD)
  [[ "$actual" == "$expected" ]] || {
    echo "$label revision mismatch: expected $expected, found $actual" >&2
    exit 2
  }
}

check_revision "$LIMOSAT_METHOD_ROOT" "$LIMOSAT_EXPECTED_REVISION" "LiMOSAT EfficientLoFTR"
check_revision "$LIMOSAT_OFFICIAL_REPOSITORY" "$LIMOSAT_OFFICIAL_REVISION" "EfficientLoFTR"
[[ -z "$(git -C "$LIMOSAT_METHOD_ROOT" status --porcelain)" ]] || {
  echo "LiMOSAT EfficientLoFTR checkout changed after submission" >&2
  exit 2
}
[[ -z "$(git -C "$LIMOSAT_OFFICIAL_REPOSITORY" status --porcelain)" ]] || {
  echo "official EfficientLoFTR checkout changed after submission" >&2
  exit 2
}
check_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA256 changed after submission" >&2
    exit 2
  }
}

for required in \
  "$LIMOSAT_CONFIG" "$LIMOSAT_CATALOGUE" "$LIMOSAT_SIC_ROOT" \
  "$LIMOSAT_SCENE_ROOT" "$LIMOSAT_CHECKPOINT" \
  "$LIMOSAT_CONTAINER" "$LIMOSAT_OVERLAY" "$LIMOSAT_READY_MARKER" \
  "$LIMOSAT_CPU_AUDIT"; do
  [[ -e "$required" ]] || { echo "missing run prerequisite: $required" >&2; exit 2; }
done
check_sha256 "$LIMOSAT_CONFIG" "$LIMOSAT_CONFIG_SHA256" "configuration"
check_sha256 "$LIMOSAT_CATALOGUE" "$LIMOSAT_CATALOGUE_SHA256" "catalogue"
check_sha256 "$LIMOSAT_CHECKPOINT" "$LIMOSAT_CHECKPOINT_SHA256" "checkpoint"

checkpoint_directory=$(dirname "$LIMOSAT_CHECKPOINT")
run_in_container() {
  local -a container_options=(exec --cleanenv --no-home)
  if [[ "$LIMOSAT_STAGE" == "primary-pairs" || "$LIMOSAT_STAGE" == "recovery-pairs" ]]; then
    container_options+=(--nv)
  fi
  env -u APPTAINER_BIND -u APPTAINER_BINDPATH \
    -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
    apptainer "${container_options[@]}" \
    --overlay "$LIMOSAT_OVERLAY:ro" \
    --bind "$LIMOSAT_METHOD_ROOT:$LIMOSAT_METHOD_ROOT:ro" \
    --bind "$LIMOSAT_OFFICIAL_REPOSITORY:$LIMOSAT_OFFICIAL_REPOSITORY:ro" \
    --bind "$checkpoint_directory:$checkpoint_directory:ro" \
    --bind "$LIMOSAT_SCENE_ROOT:$LIMOSAT_SCENE_ROOT:ro" \
    --bind "$LIMOSAT_RUN_ROOT:$LIMOSAT_RUN_ROOT" \
    --bind "$LIMOSAT_SIC_ROOT:$LIMOSAT_SIC_ROOT:ro" \
    --env PYTHONUSERBASE=/home/apptainer/user_software \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH="$LIMOSAT_METHOD_ROOT" \
    "$LIMOSAT_CONTAINER" "$@"
}

plan="$LIMOSAT_RUN_ROOT/control/april-week-full-plan.json"
array_index="${SLURM_ARRAY_TASK_ID:-0}"
temporary_suffix=".tmp.${SLURM_JOB_ID}.${array_index}"

run_json() {
  local destination="$1"
  shift
  run_in_container "$@" > "${destination}${temporary_suffix}"
  mv "${destination}${temporary_suffix}" "$destination"
}

echo "job_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "stage=$LIMOSAT_STAGE"
echo "method_revision=$LIMOSAT_EXPECTED_REVISION"
echo "official_revision=$LIMOSAT_OFFICIAL_REVISION"

case "$LIMOSAT_STAGE" in
  prepare)
    first_sic="$LIMOSAT_SIC_ROOT/ice_conc_nh_polstere-100_multi_202004011200.nc"
    run_in_container python - "$first_sic" <<'PY'
import sys
from limosat.tile_gates import load_sic_field

field = load_sic_field(sys.argv[1])
print("sic_variable=%s sic_shape=%s" % (field.variable, field.values_percent.shape))
PY
    run_json "$plan" python -m limosat plan "$LIMOSAT_CONFIG"
    run_in_container python - "$plan" "$LIMOSAT_EXPECTED_IMAGES" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    plan = json.load(stream)
expected = int(sys.argv[2])
if plan["catalogue_images"] != expected:
    raise SystemExit(
        "expected %d catalogue images, found %d"
        % (expected, plan["catalogue_images"])
    )
if plan["candidate_pairs"] < 1 or plan["primary_pairs"] < 1:
    raise SystemExit("candidate plan contains no usable primary image pairs")
print(
    "catalogue_images=%d candidate_pairs=%d primary_pairs=%d"
    % (
        plan["catalogue_images"],
        plan["candidate_pairs"],
        plan["primary_pairs"],
    )
)
PY
    run_json "$LIMOSAT_RUN_ROOT/control/prepare-stage.json" \
      python -m limosat prepare "$LIMOSAT_CONFIG"
    ;;
  primary-pairs|recovery-pairs)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    kind="${LIMOSAT_STAGE%-pairs}"
    result="$LIMOSAT_RUN_ROOT/control/${kind}-pairs-${array_index}.json"
    run_json "$result" python -m limosat pairs "$LIMOSAT_CONFIG" \
      --kind "$kind" --batch-index "$array_index" \
      --batch-count "$LIMOSAT_GPU_WORKERS"
    ;;
  primary-compose)
    run_json "$LIMOSAT_RUN_ROOT/control/primary-compose.json" \
      python -m limosat compose "$LIMOSAT_CONFIG" --phase primary
    run_json "$LIMOSAT_RUN_ROOT/control/primary-status.json" \
      python -m limosat status "$LIMOSAT_CONFIG"
    ;;
  final-compose)
    run_json "$LIMOSAT_RUN_ROOT/control/final-compose.json" \
      python -m limosat compose "$LIMOSAT_CONFIG" --phase final
    run_json "$LIMOSAT_RUN_ROOT/control/final-status.json" \
      python -m limosat status "$LIMOSAT_CONFIG"
    run_json "$LIMOSAT_RUN_ROOT/control/finalize-result.json" \
      python -m limosat finalize "$LIMOSAT_CONFIG"
    echo "run_complete=$LIMOSAT_RUN_ID"
    ;;
  *)
    echo "unknown LIMOSAT_STAGE: $LIMOSAT_STAGE" >&2
    exit 2
    ;;
esac
