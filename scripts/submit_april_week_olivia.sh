#!/usr/bin/env bash
set -euo pipefail

method_root="${METHOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
project_id="${PROJECT_ID:-nn9878k}"
gpu_account="${GPU_ACCOUNT:-nn2993k}"
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
slurm_qos="${SLURM_QOS:-}"
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

method_revision=$(git -C "$method_root" rev-parse HEAD)
official_revision=$(git -C "$official_repository" rev-parse HEAD)
checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
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
common_export="ALL,LIMOSAT_METHOD_ROOT=$method_root,LIMOSAT_EXPECTED_REVISION=$method_revision,LIMOSAT_RUN_ID=$run_id,LIMOSAT_RUN_ROOT=$run_root,LIMOSAT_CONFIG=$config,LIMOSAT_CATALOGUE=$catalogue,LIMOSAT_SIC_ROOT=$sic_root,LIMOSAT_SCENE_ROOT=$scene_root,LIMOSAT_OFFICIAL_REPOSITORY=$official_repository,LIMOSAT_OFFICIAL_REVISION=$official_revision,LIMOSAT_CHECKPOINT=$checkpoint,LIMOSAT_CHECKPOINT_SHA256=$checkpoint_sha256,LIMOSAT_CONTAINER=$container,LIMOSAT_OVERLAY=$overlay,LIMOSAT_READY_MARKER=$ready_marker,LIMOSAT_CPU_AUDIT=$cpu_audit,LIMOSAT_EXPECTED_IMAGES=781"

command=(
  sbatch --parsable
  --job-name="eloftr-apr20"
  --account="$gpu_account"
  --partition=accel
  --gpus=1
  --nodes=1
  --cpus-per-task="$cpus"
  --mem="$memory"
  --time="$wall_time"
  --output="$run_root/logs/%x-%j.out"
  --error="$run_root/logs/%x-%j.err"
  --export="$common_export"
)
if [[ -n "$slurm_qos" ]]; then
  command+=(--qos="$slurm_qos")
fi
command+=("$job_script")

if [[ "$dry_run" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\nDRY RUN: no job submitted\n'
  exit 0
fi

job_id=$("${command[@]}")
echo "submitted_job=$job_id"
echo "run_root=$run_root"
echo "log=$run_root/logs/eloftr-apr20-$job_id.out"
squeue -j "$job_id"
