#!/usr/bin/env bash
set -euo pipefail

method_root="${METHOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
project_id="${PROJECT_ID:-nn9878k}"
gpu_account="${GPU_ACCOUNT:-nn2993k}"
cpu_account="${CPU_ACCOUNT:-$gpu_account}"
cpu_partition="${CPU_PARTITION:-accel}"
run_id="${RUN_ID:-april2020-week01-global-5pct-sic-v1}"
work_root="${WORK_ROOT:-/cluster/work/projects/$project_id/$USER}"
project_root="${PROJECT_ROOT:-/cluster/projects/$project_id/$USER}"
run_root="${RUN_ROOT:-$work_root/method-neutral-benchmark/efficientloftr-production/$run_id}"
config="${CONFIG:-$run_root/control/april-week-full.json}"
catalogue="${CATALOGUE:-$run_root/control/april-week-full-catalog.json}"
sic_root="${SIC_ROOT:-$run_root/inputs/osisaf-sic}"
scene_root="${SCENE_ROOT:-$work_root/limosat_staging/201905_202007/s1_preprocessed}"
official_repository="${ELOFTR_REPO:-$project_root/EfficientLoFTR}"
checkpoint="${CHECKPOINT:-$project_root/checkpoints/efficientloftr/eloftr_outdoor.ckpt}"
container="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
overlay="${OVERLAY:-$work_root/method-neutral-benchmark/environments/efficientloftr/efficientloftr-apptainer-v3.img}"
ready_marker="${READY_MARKER:-$overlay.ready}"
cpu_audit="${CPU_AUDIT:-$overlay.cpu-audit.json}"
wall_time="${WALL_TIME:-24:00:00}"
memory="${MEMORY:-32G}"
cpus="${CPUS_PER_TASK:-4}"
gpu_workers="${GPU_WORKERS:-1}"
cpu_wall_time="${CPU_WALL_TIME:-06:00:00}"
cpu_memory="${CPU_MEMORY:-64G}"
cpu_cpus="${CPU_CPUS_PER_TASK:-8}"
slurm_qos="${SLURM_QOS:-}"
cpu_qos="${CPU_QOS:-}"
dry_run=0

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

for required in \
  "$method_root/.git" "$config" "$catalogue" "$sic_root" "$scene_root" \
  "$official_repository/.git" "$checkpoint" "$container" "$overlay" \
  "$ready_marker" "$cpu_audit"; do
  [[ -e "$required" ]] || { echo "missing run prerequisite: $required" >&2; exit 2; }
done
command -v apptainer >/dev/null || { echo "apptainer is unavailable" >&2; exit 2; }
[[ "$gpu_workers" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU_WORKERS must be a positive integer" >&2
  exit 2
}

method_revision=$(git -C "$method_root" rev-parse HEAD)
official_revision=$(git -C "$official_repository" rev-parse HEAD)
config_sha256=$(sha256sum "$config" | awk '{print $1}')
catalogue_sha256=$(sha256sum "$catalogue" | awk '{print $1}')
checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
recovery_enabled=$(
  python3 -c \
    'import json,sys; value=(json.load(open(sys.argv[1])).get("routing") or {}).get("targeted_recovery", True); print("1" if value is True else "0" if value is False else "invalid")' \
    "$config"
)
[[ "$recovery_enabled" == "0" || "$recovery_enabled" == "1" ]] || {
  echo "routing.targeted_recovery must be a JSON boolean" >&2
  exit 2
}
if [[ -n "$(git -C "$method_root" status --porcelain)" ]]; then
  echo "LiMOSAT EfficientLoFTR checkout is dirty" >&2
  git -C "$method_root" status --short >&2
  exit 2
fi
if [[ -n "$(git -C "$official_repository" status --porcelain)" ]]; then
  echo "official EfficientLoFTR checkout is dirty" >&2
  git -C "$official_repository" status --short >&2
  exit 2
fi

mkdir -p "$run_root/logs"
job_script="$method_root/scripts/run_limosat_olivia.sbatch"
common_export="ALL,LIMOSAT_METHOD_ROOT=$method_root,LIMOSAT_EXPECTED_REVISION=$method_revision,LIMOSAT_RUN_ID=$run_id,LIMOSAT_RUN_ROOT=$run_root,LIMOSAT_CONFIG=$config,LIMOSAT_CONFIG_SHA256=$config_sha256,LIMOSAT_CATALOGUE=$catalogue,LIMOSAT_CATALOGUE_SHA256=$catalogue_sha256,LIMOSAT_SIC_ROOT=$sic_root,LIMOSAT_SCENE_ROOT=$scene_root,LIMOSAT_OFFICIAL_REPOSITORY=$official_repository,LIMOSAT_OFFICIAL_REVISION=$official_revision,LIMOSAT_CHECKPOINT=$checkpoint,LIMOSAT_CHECKPOINT_SHA256=$checkpoint_sha256,LIMOSAT_CONTAINER=$container,LIMOSAT_OVERLAY=$overlay,LIMOSAT_READY_MARKER=$ready_marker,LIMOSAT_CPU_AUDIT=$cpu_audit,LIMOSAT_EXPECTED_IMAGES=781,LIMOSAT_GPU_WORKERS=$gpu_workers"

build_command() {
  local stage="$1"
  local resource="$2"
  local dependency="$3"
  command=(sbatch --parsable --nodes=1 --ntasks=1 --job-name="eloftr-${stage}")
  if [[ "$resource" == "gpu" ]]; then
    command+=(
      --account="$gpu_account" --partition=accel --gpus=1
      --array="0-$((gpu_workers - 1))"
      --cpus-per-task="$cpus" --mem="$memory" --time="$wall_time"
      --output="$run_root/logs/%x-%A_%a.out"
      --error="$run_root/logs/%x-%A_%a.err"
    )
    [[ -z "$slurm_qos" ]] || command+=(--qos="$slurm_qos")
  else
    command+=(
      --account="$cpu_account"
      --cpus-per-task="$cpu_cpus" --mem="$cpu_memory" --time="$cpu_wall_time"
      --output="$run_root/logs/%x-%j.out"
      --error="$run_root/logs/%x-%j.err"
    )
    [[ -z "$cpu_partition" ]] || command+=(--partition="$cpu_partition")
    [[ "$cpu_partition" != "accel" ]] || command+=(--gpus-per-node=0)
    [[ -z "$cpu_qos" ]] || command+=(--qos="$cpu_qos")
  fi
  [[ -z "$dependency" ]] || command+=(--dependency="afterok:$dependency")
  command+=(--export="$common_export,LIMOSAT_STAGE=$stage" "$job_script")
}

submit_stage() {
  local submitted
  build_command "$@"
  submitted=$("${command[@]}")
  echo "${submitted%%;*}"
}

if [[ "$dry_run" == "1" ]]; then
  previous=""
  stages=("prepare cpu" "primary-pairs gpu" "primary-compose cpu")
  if [[ "$recovery_enabled" == "1" ]]; then
    stages+=("recovery-pairs gpu")
  fi
  stages+=("final-compose cpu")
  for specification in "${stages[@]}"; do
    read -r stage resource <<< "$specification"
    dependency="$previous"
    build_command "$stage" "$resource" "$dependency"
    printf '%q ' "${command[@]}"
    printf '\n'
    previous="<${stage}-job-id>"
  done
  printf 'DRY RUN: no jobs submitted\n'
  exit 0
fi

prepare_job=$(submit_stage prepare cpu "")
primary_job=$(submit_stage primary-pairs gpu "$prepare_job")
primary_compose_job=$(submit_stage primary-compose cpu "$primary_job")
if [[ "$recovery_enabled" == "1" ]]; then
  recovery_job=$(submit_stage recovery-pairs gpu "$primary_compose_job")
  final_dependency="$recovery_job"
else
  recovery_job="disabled"
  final_dependency="$primary_compose_job"
fi
final_job=$(submit_stage final-compose cpu "$final_dependency")

echo "prepare_job=$prepare_job"
echo "primary_pair_job=$primary_job"
echo "primary_compose_job=$primary_compose_job"
echo "recovery_pair_job=$recovery_job"
echo "final_compose_job=$final_job"
echo "run_root=$run_root"
echo "final_log=$run_root/logs/eloftr-final-compose-$final_job.out"
job_ids="$prepare_job,$primary_job,$primary_compose_job"
if [[ "$recovery_enabled" == "1" ]]; then
  job_ids="$job_ids,$recovery_job"
fi
squeue -j "$job_ids,$final_job"
