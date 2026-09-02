#!/usr/bin/env bash
#SBATCH --nodes=1

set -euo pipefail

: "${LIMOSAT_METHOD_ROOT:?}"
: "${LIMOSAT_EXPECTED_REVISION:?}"
: "${LIMOSAT_RUN_ID:?}"
: "${LIMOSAT_RUN_ROOT:?}"
: "${LIMOSAT_CONFIG:?}"
: "${LIMOSAT_CATALOGUE:?}"
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

finish() {
  code=$?
  echo "job_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$code"
}
trap finish EXIT

check_revision() {
  repository="$1"
  expected="$2"
  label="$3"
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
actual_checkpoint_sha256=$(sha256sum "$LIMOSAT_CHECKPOINT" | awk '{print $1}')
[[ "$actual_checkpoint_sha256" == "$LIMOSAT_CHECKPOINT_SHA256" ]] || {
  echo "checkpoint SHA256 changed after submission" >&2
  exit 2
}
for required in \
  "$LIMOSAT_CONFIG" "$LIMOSAT_CATALOGUE" "$LIMOSAT_SIC_ROOT" \
  "$LIMOSAT_SCENE_ROOT" \
  "$LIMOSAT_CONTAINER" "$LIMOSAT_OVERLAY" "$LIMOSAT_READY_MARKER" \
  "$LIMOSAT_CPU_AUDIT"; do
  [[ -e "$required" ]] || { echo "missing run prerequisite: $required" >&2; exit 2; }
done

checkpoint_directory=$(dirname "$LIMOSAT_CHECKPOINT")
run_in_container() {
  env -u APPTAINER_BIND -u APPTAINER_BINDPATH \
    -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
    apptainer exec --cleanenv --no-home --nv \
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
run_result="$LIMOSAT_RUN_ROOT/control/run-result.json"
status_result="$LIMOSAT_RUN_ROOT/control/final-status.json"
finalize_result="$LIMOSAT_RUN_ROOT/control/finalize-result.json"
temporary_suffix=".tmp.${SLURM_JOB_ID}"

echo "job_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "method_revision=$LIMOSAT_EXPECTED_REVISION"
echo "official_revision=$LIMOSAT_OFFICIAL_REVISION"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

first_sic="$LIMOSAT_SIC_ROOT/ice_conc_nh_polstere-100_multi_202004011200.nc"
run_in_container python - "$first_sic" <<'PY'
import sys
from limosat.tile_gates import load_sic_field

field = load_sic_field(sys.argv[1])
print("sic_variable=%s sic_shape=%s" % (field.variable, field.values_percent.shape))
PY

run_in_container python -m limosat plan "$LIMOSAT_CONFIG" > "${plan}${temporary_suffix}"
mv "${plan}${temporary_suffix}" "$plan"
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

run_in_container python -m limosat run "$LIMOSAT_CONFIG" \
  > "${run_result}${temporary_suffix}"
mv "${run_result}${temporary_suffix}" "$run_result"
run_in_container python -m limosat status "$LIMOSAT_CONFIG" \
  > "${status_result}${temporary_suffix}"
mv "${status_result}${temporary_suffix}" "$status_result"
run_in_container python -m limosat finalize "$LIMOSAT_CONFIG" \
  > "${finalize_result}${temporary_suffix}"
mv "${finalize_result}${temporary_suffix}" "$finalize_result"
echo "run_complete=$LIMOSAT_RUN_ID"
