#!/usr/bin/env bash
# Submit the chained Parquet run on Olivia (see run_limosat_chain_olivia.sbatch).
#   submit_chain_olivia.sh [--dry-run]   freeze the run environment and submit prepare, which submits the rest
#   submit_chain_olivia.sh --stage S [--after J] [--pairs N]   (used by running stages) submit one stage, print its ID
# A GPU stage for N pairs gets ceil(N / PAIRS_PER_GPU_TASK) one-GPU tasks (at most MAXIMUM_GPU_TASKS) and a time
# limit of twice the expected run time at PAIRS_PER_GPU_HOUR, at least 30 minutes: short requests backfill sooner.
# RUN_ROOT must hold control/config.json and control/catalogue.json from prepare_chain_olivia.py.
set -euo pipefail

stage=""
after=""
pairs=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --stage) stage="$2"; shift 2 ;;
    --after) after="$2"; shift 2 ;;
    --pairs) pairs="$2"; shift 2 ;;
    *) echo "usage: $0 [--dry-run] | --stage STAGE [--after JOB] [--pairs N]" >&2; exit 2 ;;
  esac
done

if [[ -n "$stage" ]]; then
  # Re-entry from a running stage: everything comes from the frozen environment.
  : "${LIMOSAT_RUN_ROOT:?}"
  # shellcheck disable=SC1091
  source "$LIMOSAT_RUN_ROOT/control/submit.env"
else
  method_root="${METHOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  project_id="${PROJECT_ID:-nn9878k}"
  work_root="${WORK_ROOT:-/cluster/work/projects/$project_id/$USER}"
  project_root="${PROJECT_ROOT:-/cluster/projects/$project_id/$USER}"
  run_root="${RUN_ROOT:?set RUN_ROOT}"
  config="$run_root/control/config.json"
  catalogue="$run_root/control/catalogue.json"
  official_repository="${ELOFTR_REPO:-$project_root/EfficientLoFTR}"
  checkpoint="${CHECKPOINT:-$project_root/checkpoints/efficientloftr/eloftr_outdoor.ckpt}"
  container="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
  overlay="${OVERLAY:-$work_root/method-neutral-benchmark/environments/efficientloftr/efficientloftr-apptainer-v3.img}"
  data_binds="${DATA_BINDS:?set DATA_BINDS to the colon-separated input roots (scenes, OSI-SAF, ASI, land raster dir)}"
  for required in "$method_root/.git" "$config" "$catalogue" "$official_repository/.git" \
    "$checkpoint" "$container" "$overlay"; do
    [[ -e "$required" ]] || { echo "missing run prerequisite: $required" >&2; exit 2; }
  done
  [[ -z "$(git -C "$method_root" status --porcelain)" ]] || { echo "LiMOSAT checkout is dirty" >&2; exit 2; }
  [[ -z "$(git -C "$official_repository" status --porcelain)" ]] || { echo "EfficientLoFTR checkout is dirty" >&2; exit 2; }
  mkdir -p "$run_root/logs"
  environment="$run_root/control/submit.env"
  if [[ "$dry_run" == 1 ]]; then
    environment="$run_root/control/submit.env.dry-run"
  elif [[ -e "$environment" ]]; then
    echo "run already submitted: $environment" >&2
    exit 2
  fi
  {
    printf 'export LIMOSAT_METHOD_ROOT=%q\n' "$method_root"
    printf 'export LIMOSAT_EXPECTED_REVISION=%q\n' "$(git -C "$method_root" rev-parse HEAD)"
    printf 'export LIMOSAT_RUN_ID=%q\n' "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$config")"
    printf 'export LIMOSAT_RUN_ROOT=%q\n' "$run_root"
    printf 'export LIMOSAT_CONFIG=%q\n' "$config"
    printf 'export LIMOSAT_CONFIG_SHA256=%q\n' "$(sha256sum "$config" | awk '{print $1}')"
    printf 'export LIMOSAT_CATALOGUE=%q\n' "$catalogue"
    printf 'export LIMOSAT_CATALOGUE_SHA256=%q\n' "$(sha256sum "$catalogue" | awk '{print $1}')"
    printf 'export LIMOSAT_DATA_BINDS=%q\n' "$data_binds"
    printf 'export LIMOSAT_OFFICIAL_REPOSITORY=%q\n' "$official_repository"
    printf 'export LIMOSAT_OFFICIAL_REVISION=%q\n' "$(git -C "$official_repository" rev-parse HEAD)"
    printf 'export LIMOSAT_CHECKPOINT=%q\n' "$checkpoint"
    printf 'export LIMOSAT_CHECKPOINT_SHA256=%q\n' "$(sha256sum "$checkpoint" | awk '{print $1}')"
    printf 'export LIMOSAT_CONTAINER=%q\n' "$container"
    printf 'export LIMOSAT_OVERLAY=%q\n' "$overlay"
    printf 'export LIMOSAT_PROCESSES_PER_GPU=%q\n' "${PROCESSES_PER_GPU:-10}"
    printf 'export LIMOSAT_SUBMIT=%q\n' "$method_root/scripts/submit_chain_olivia.sh"
    printf 'export LIMOSAT_MINIMUM_PASS_PAIRS=%q\n' "${MINIMUM_PASS_PAIRS:-50}"
    printf 'export LIMOSAT_MINIMUM_GAIN_SHARE=%q\n' "${MINIMUM_GAIN_SHARE:-0.3}"
    printf 'export LIMOSAT_MAXIMUM_PASSES=%q\n' "${MAXIMUM_PASSES:-12}"
    printf 'export LIMOSAT_STOP_AFTER=%q\n' "${STOP_AFTER:-}"
    printf 'GPU_ACCOUNT=%q\nCPU_ACCOUNT=%q\n' "${GPU_ACCOUNT:-nn2993k}" "${CPU_ACCOUNT:-${GPU_ACCOUNT:-nn2993k}}"
    printf 'MAXIMUM_GPU_TASKS=%q\nPAIRS_PER_GPU_TASK=%q\nPAIRS_PER_GPU_HOUR=%q\n' \
      "${MAXIMUM_GPU_TASKS:-4}" "${PAIRS_PER_GPU_TASK:-4000}" "${PAIRS_PER_GPU_HOUR:-2500}"
    printf 'CPUS_PER_PROCESS=%q\nMEMORY_PER_PROCESS_GB=%q\n' "${CPUS_PER_PROCESS:-6}" "${MEMORY_PER_PROCESS_GB:-10}"
    printf 'CPU_PARTITION=%q\nCPU_WALL_TIME=%q\nCPU_MEMORY=%q\nCPU_CPUS=%q\n' \
      "${CPU_PARTITION:-accel}" "${CPU_WALL_TIME:-12:00:00}" "${CPU_MEMORY:-128G}" "${CPU_CPUS_PER_TASK:-16}"
  } > "$environment.tmp"
  mv "$environment.tmp" "$environment"
  # shellcheck disable=SC1090
  source "$environment"
fi

job_script="$LIMOSAT_METHOD_ROOT/scripts/run_limosat_chain_olivia.sbatch"
build_command() {
  local name="$1" dependency="$2"
  command=(sbatch --parsable --nodes=1 --ntasks=1 --job-name="eloftr-$name-p${LIMOSAT_PASS:-0}")
  if [[ "$name" == *-pairs ]]; then
    local tasks=$(( (pairs + PAIRS_PER_GPU_TASK - 1) / PAIRS_PER_GPU_TASK ))
    (( tasks >= 1 )) || tasks=1
    (( tasks <= MAXIMUM_GPU_TASKS )) || tasks=$MAXIMUM_GPU_TASKS
    local minutes=$(( 2 * 60 * pairs / (tasks * PAIRS_PER_GPU_HOUR) ))
    (( minutes >= 30 )) || minutes=30
    (( minutes <= 1440 )) || minutes=1440
    command+=(--account="$GPU_ACCOUNT" --partition=accel --gpus=1 --array="0-$((tasks - 1))"
      --cpus-per-task=$(( CPUS_PER_PROCESS * LIMOSAT_PROCESSES_PER_GPU ))
      --mem="$(( MEMORY_PER_PROCESS_GB * LIMOSAT_PROCESSES_PER_GPU ))G" --time="$minutes"
      --output="$LIMOSAT_RUN_ROOT/logs/%x-%A_%a.out" --error="$LIMOSAT_RUN_ROOT/logs/%x-%A_%a.err")
  else
    command+=(--account="$CPU_ACCOUNT" --cpus-per-task="$CPU_CPUS" --mem="$CPU_MEMORY" --time="$CPU_WALL_TIME"
      --output="$LIMOSAT_RUN_ROOT/logs/%x-%j.out" --error="$LIMOSAT_RUN_ROOT/logs/%x-%j.err")
    [[ -z "$CPU_PARTITION" ]] || command+=(--partition="$CPU_PARTITION")
    [[ "$CPU_PARTITION" != accel ]] || command+=(--gpus-per-node=0)
  fi
  [[ -z "$dependency" ]] || command+=(--dependency="afterok:$dependency")
  command+=(--export="ALL,LIMOSAT_STAGE=$name,LIMOSAT_PASS=${LIMOSAT_PASS:-0}" "$job_script")
}
submit_stage() {
  build_command "$1" "$2"
  if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}" >&2
    printf '\n' >&2
    echo "<$1>"
  else
    local submitted
    submitted=$("${command[@]}")
    echo "${submitted%%;*}"
    echo "$(date -u +%FT%TZ) submitted $1 pass ${LIMOSAT_PASS:-0} job ${submitted%%;*}" >> "$LIMOSAT_RUN_ROOT/control/passes.log"
  fi
}

if [[ -n "$stage" ]]; then
  submit_stage "$stage" "$after"
  exit 0
fi
echo "prepare_job=$(submit_stage prepare "")"
if [[ "$dry_run" == 1 ]]; then
  # Shows the GPU request prepare would make for this week's 2,418 primary pairs.
  pairs=2418
  submit_stage primary-pairs "<prepare>" > /dev/null
fi
echo "ledger=$LIMOSAT_RUN_ROOT/control/passes.log"
if [[ "$dry_run" == 1 ]]; then
  rm -f "$LIMOSAT_RUN_ROOT/control/submit.env.dry-run"
  echo "DRY RUN: no jobs submitted"
fi
