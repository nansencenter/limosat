#!/bin/bash
#SBATCH --job-name=limosat-overlay
#SBATCH --account=nn2993k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Build the chain's Python overlay on a Grace Hopper (aarch64) node and check that limosat and
# the EfficientLoFTR model load on the GPU with it.
#   sbatch --output=OUT.log build_overlay_olivia.sbatch OVERLAY_PATH METHOD_ROOT [EFFICIENTLOFTR_REPO CHECKPOINT]
set -euo pipefail
overlay="$1"; method="$2"
repo="${3:-/cluster/projects/nn9878k/seachu/EfficientLoFTR}"
checkpoint="${4:-/cluster/projects/nn9878k/seachu/checkpoints/efficientloftr/eloftr_outdoor.ckpt}"
container="${LIMOSAT_CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
bash "$method/scripts/overlay/build_overlay_olivia.sh" "$overlay"
apptainer exec --cleanenv --no-home --nv --overlay "$overlay:ro" \
  --env PYTHONUSERBASE=/home/apptainer/user_software --env PYTHONPATH="$method" \
  --bind "$method:$method:ro" --bind "$repo:$repo:ro" --bind "$(dirname "$checkpoint"):$(dirname "$checkpoint"):ro" \
  "$container" python - "$repo" "$checkpoint" <<'PY'
import sys
import numpy as np
from limosat.config import MatcherConfig
from limosat.efficientloftr import EfficientLoFTR
repo, checkpoint = sys.argv[1:3]
matcher = EfficientLoFTR(MatcherConfig(repository=repo, checkpoint=checkpoint, device="cuda"))
rng = np.random.default_rng(0)
tile = (rng.random((512, 512)) * 255).astype(np.uint8)
out = matcher.match_batch([tile] * 4, [np.roll(tile, 5, axis=1)] * 4)
print("model ok:", matcher.execution_mode, [len(o[0]) for o in out])
PY
