#!/usr/bin/env bash
# Build the Python overlay for the EfficientLoFTR chain on Olivia.
#
# The chain runs NVIDIA's PyTorch container (LIMOSAT_CONTAINER) with this overlay mounted
# read-only (LIMOSAT_OVERLAY); its packages live in /home/apptainer/user_software and are found
# through PYTHONUSERBASE. Build it on an aarch64 node (the container is arm64; Olivia's login
# nodes are x86), e.g. with build_overlay_olivia.sbatch, then copy it off /cluster/work (the
# work area deletes files with old modification times; the v3 overlay was lost that way on
# 22 November 2026).
#
#   build_overlay_olivia.sh OVERLAY_PATH [SIZE_MB]
#
# Writes OVERLAY_PATH, OVERLAY_PATH.base-freeze.txt (the base image's packages; its numerical
# stack is used as pip constraints so the overlay never replaces it) and OVERLAY_PATH.lock.txt (what was
# installed into the overlay, with versions). Commit the lock file next to this script.
set -euo pipefail
overlay="${1:?usage: $0 OVERLAY_PATH [SIZE_MB]}"
size="${2:-6144}"
container="${LIMOSAT_CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
here="$(cd "$(dirname "$0")" && pwd)"
[[ "$(uname -m)" == aarch64 ]] || { echo "run on an aarch64 node (the container is arm64)" >&2; exit 2; }
[[ ! -e "$overlay" ]] || { echo "exists, not overwriting: $overlay" >&2; exit 2; }
mkdir -p "$(dirname "$overlay")"
apptainer overlay create --size "$size" --create-dir /home/apptainer/user_software "$overlay"
run() {
  apptainer exec --cleanenv --no-home --overlay "$overlay" \
    --env PYTHONUSERBASE=/home/apptainer/user_software --env PIP_DISABLE_PIP_VERSION_CHECK=1 \
    --bind "$here:$here:ro" --bind "$(dirname "$overlay"):$(dirname "$overlay")" \
    "$container" "$@"
}
run python -m pip freeze --all > "$overlay.base-freeze.txt"
# Constraints: pin the base image's numerical stack (torch and NVIDIA libraries, numpy, scipy,
# pyarrow, pillow) so the overlay never replaces it. Pinning every base package over-constrains
# pip (on 23 Nov 2026 it fell back to a rasterio source build); the rest may be added alongside.
grep -E -i '^(torch|torchvision|triton|nvidia-[a-z0-9-]+|numpy|scipy|pyarrow|pillow)==' "$overlay.base-freeze.txt" > "$overlay.constraints.txt"
run python -m pip install --user --no-cache-dir --only-binary=:all: -c "$overlay.constraints.txt" -r "$here/requirements.txt"
run python -m pip freeze --user > "$overlay.lock.txt"
run python - <<'PY'
import importlib, sys
for name in ("torch", "numpy", "scipy", "cv2", "kornia", "einops", "yacs", "loguru", "pyproj", "yaml",
             "rasterio", "shapely", "pyarrow"):
    module = importlib.import_module(name)
    print(f"{name:10s} {getattr(module, '__version__', '?')}")
print("python", sys.version.split()[0])
PY
echo "built $overlay ($(du -h "$overlay" | cut -f1)); lock file $overlay.lock.txt"
