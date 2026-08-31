#!/usr/bin/env bash
set -euo pipefail

: "${LIMOSAT_HPC_PYTHON:?Set LIMOSAT_HPC_PYTHON to the CUDA environment Python executable}"
: "${LIMOSAT_HPC_PAIR_ROOT:?Set LIMOSAT_HPC_PAIR_ROOT to the directory containing the two frozen pair runs}"
: "${LIMOSAT_HPC_STANDARD_VAE_ROOT:?Set LIMOSAT_HPC_STANDARD_VAE_ROOT to the mirrored standard-VAE TIFF tree}"
: "${LIMOSAT_HPC_MODEL_CACHE:?Set LIMOSAT_HPC_MODEL_CACHE to a writable model cache containing the installed ALIKED weights}"
: "${LIMOSAT_HPC_OUTPUT:?Set LIMOSAT_HPC_OUTPUT to a new writable output directory}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAIR_ONE="$LIMOSAT_HPC_PAIR_ROOT/dense_pair_10245_10341_aliked1024_fixedgrid_cache_fullcontext_dev_marchseq_v1_20260818"
PAIR_TWO="$LIMOSAT_HPC_PAIR_ROOT/dense_pair_10341_10352_aliked1024_fixedgrid_cache_fullcontext_dev_marchseq_v1_20260818"
STAGED="$LIMOSAT_HPC_OUTPUT/staged_pair_inputs"
export MPLCONFIGDIR="$LIMOSAT_HPC_OUTPUT/matplotlib"

if [[ -e "$LIMOSAT_HPC_OUTPUT/run_complete.json" ]]; then
  echo "CUDA benchmark is already complete: $LIMOSAT_HPC_OUTPUT" >&2
  exit 2
fi
if [[ -d "$LIMOSAT_HPC_OUTPUT/aliked" ]]; then
  echo "Refusing a partially populated output; choose a new LIMOSAT_HPC_OUTPUT." >&2
  exit 2
fi
for required in "$PAIR_ONE/run_manifest.json" "$PAIR_TWO/run_manifest.json"; do
  [[ -s "$required" ]] || { echo "Missing frozen input: $required" >&2; exit 2; }
done
mkdir -p "$STAGED" "$MPLCONFIGDIR"

cd "$REPO_ROOT"
"$LIMOSAT_HPC_PYTHON" - "$PAIR_ONE" "$PAIR_TWO" "$LIMOSAT_HPC_STANDARD_VAE_ROOT" "$LIMOSAT_HPC_MODEL_CACHE" "$STAGED" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

pair_dirs = [Path(sys.argv[1]), Path(sys.argv[2])]
image_root = Path(sys.argv[3])
model_cache = Path(sys.argv[4])
output = Path(sys.argv[5])
expected_manifests = [
    "bff5b9948a1359ad0b82f901610839af3b6a81e84e0843e95abfd5bab0fa9f97",
    "c34555864c494dcb437dee4257a5d13a600c26c58c5875a4abc55eb939e11c2a",
]
expected_images = {
    "S1B_EW_GRDM_1SDH_20200328T121329_20200328T121429_020892_0279E9_C789.tiff": "0175c9cd54e57ff5ffceace5f471dc98374d5a9db15d5f7ec98d36f23f859a0f",
    "S1B_EW_GRDM_1SDH_20200329T075946_20200329T080046_020904_027A43_B256.tiff": "47485c5303aecfd5d029efb6c72ef4d4cd414523b298bb470281bdb42f88c1d9",
    "S1B_EW_GRDM_1SDH_20200329T111605_20200329T111705_020906_027A5A_7263.tiff": "37a9ed3cde3449d8a96387438979f1e22dd6a178e8bfd4ed72b16f790ae34413",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

resolved_images = {}
for name, expected in expected_images.items():
    matches = list(image_root.rglob(name))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {name} below {image_root}, found {len(matches)}")
    actual = sha256(matches[0])
    if actual != expected:
        raise SystemExit(f"Image hash mismatch for {matches[0]}: {actual} != {expected}")
    resolved_images[name] = str(matches[0].resolve())

for index, (pair_dir, expected_manifest) in enumerate(zip(pair_dirs, expected_manifests, strict=True), 1):
    source_manifest = pair_dir / "run_manifest.json"
    actual_manifest = sha256(source_manifest)
    if actual_manifest != expected_manifest:
        raise SystemExit(
            f"Pair-manifest hash mismatch for {source_manifest}: "
            f"{actual_manifest} != {expected_manifest}"
        )
    manifest = json.loads(source_manifest.read_text())
    manifest["source_image_filepath"] = resolved_images[Path(manifest["source_image_filepath"]).name]
    manifest["target_image_filepath"] = resolved_images[Path(manifest["target_image_filepath"]).name]
    manifest["parameters"]["model_cache"] = str(model_cache.resolve())
    staged_pair = output / f"pair_{index}"
    staged_pair.mkdir(parents=True, exist_ok=False)
    (staged_pair / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    buoy = pair_dir / "buoy_results.csv"
    if buoy.is_file():
        shutil.copy2(buoy, staged_pair / buoy.name)

(output / "staging_audit.json").write_text(
    json.dumps(
        {
            "source_pair_manifest_sha256": expected_manifests,
            "standard_vae_image_sha256": expected_images,
            "resolved_standard_vae_images": resolved_images,
            "change": "Only absolute image and model-cache paths were remapped; scientific parameters are unchanged.",
        },
        indent=2,
    )
    + "\n"
)
PY

"$LIMOSAT_HPC_PYTHON" - "$LIMOSAT_HPC_OUTPUT/cuda_environment.json" <<'PY'
import json
from pathlib import Path
import platform
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
details = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
}
Path(sys.argv[1]).write_text(json.dumps(details, indent=2) + "\n")
PY

run_aliked() {
  local label="$1"
  local cache="$2"
  local output="$LIMOSAT_HPC_OUTPUT/aliked/$label"
  mkdir -p "$output" "$cache"
  "$LIMOSAT_HPC_PYTHON" experiments/run_aliked_selected_sequence.py \
    --pair-run-dir "$STAGED/pair_1" \
    --pair-run-dir "$STAGED/pair_2" \
    --output-dir "$output" \
    --feature-cache-dir "$cache" \
    --features-per-tile 1024 \
    --sequential-prior \
    --sequential-prior-uncertainty-m 15000 \
    --maximum-radius-m 6000 \
    --candidate-count 12 \
    --minimum-selected-vectors 8 \
    --consensus-radius-m 1000 \
    --device cuda
}

run_aliked setup "$LIMOSAT_HPC_OUTPUT/caches/aliked_warm"
for repetition in 1 2 3; do
  run_aliked "cold_rep${repetition}" "$LIMOSAT_HPC_OUTPUT/caches/aliked_cold_rep${repetition}"
done
for repetition in 1 2 3; do
  run_aliked "warm_rep${repetition}" "$LIMOSAT_HPC_OUTPUT/caches/aliked_warm"
done

"$LIMOSAT_HPC_PYTHON" - "$LIMOSAT_HPC_OUTPUT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
labels = [f"cold_rep{i}" for i in range(1, 4)] + [f"warm_rep{i}" for i in range(1, 4)]
records = []
for label in labels:
    summary_path = root / "aliked" / label / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("device") != "cuda":
        raise SystemExit(f"Incomplete CUDA repetition: {summary_path}")
    records.append(
        {
            "label": label,
            "elapsed_seconds": summary["elapsed_seconds"],
            "prior_audits": summary["prior_audits"],
        }
    )
(root / "run_complete.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "interpretation": "Measured CUDA handoff; compare fields and accuracy with the frozen CPU gate before operational use.",
            "repetitions": records,
        },
        indent=2,
    )
    + "\n"
)
PY
