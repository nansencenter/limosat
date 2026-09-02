"""Versioned, checksummed run manifests."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .catalog import ImageCatalogue
from .config import RunConfig
from .pair_artifacts import PAIR_PRODUCT_SCHEMA_VERSION
from .store import RunStore, file_sha256


MANIFEST_SCHEMA_VERSION = 4


def write_manifest(
    config: RunConfig,
    catalogue: ImageCatalogue,
    store: RunStore,
    started_utc: datetime,
    completed_utc: datetime,
    runtime_seconds: float,
    command: Sequence[str],
) -> tuple[Path, str]:
    """Atomically write the complete resolved run manifest."""
    rows = store.manifest_rows()
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": "complete",
        "method": "EfficientLoFTR",
        "product_schemas": {
            "sqlite": 4,
            "pair_displacement_field": 1,
            "pair_match_archive": 1,
            "pair_worker_product": PAIR_PRODUCT_SCHEMA_VERSION,
            "lagrangian_trajectory": 4,
            "deformation_cell": 1,
        },
        "coordinates": {
            "crs": "EPSG:3413",
            "dtype": "float64",
            "distance_unit": "metre",
            "time_unit": "second",
            "strain_rate_unit": "s-1",
        },
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "implementation_sha256": store.implementation_sha256,
        "checkpoint_sha256": store.model_sha256,
        "git": {
            "limosat": _git_state(),
            "efficientloftr": (
                _git_state(Path(config.matcher.repository))
                if config.matcher.repository
                and Path(config.matcher.repository).is_dir()
                else None
            ),
        },
        "command": list(command),
        "started_utc": started_utc.isoformat(),
        "completed_utc": completed_utc.isoformat(),
        "runtime_seconds": float(runtime_seconds),
        "compute_planning_labels": {
            name: [image.image_id for image in images]
            for name, images in catalogue.components().items()
        },
        "images": rows["images"],
        "candidate_pairs": rows["candidate_pairs"],
        "candidate_pair_planning_counts": rows["planning_counts"],
        "ancillary_inputs": rows["ancillary_inputs"],
        "pairs": rows["pairs"],
        "product_counts": rows["counts"],
        "pair_match_retention": {
            "enabled": config.retain_pair_matches,
            "stage": "post-gate, pre-field-consensus selected hypothesis",
            "storage": "checksummed compressed SQLite BLOB per completed pair",
        },
        "execution_architecture": {
            "pair_workers_write_sqlite": False,
            "pair_products": "immutable atomic NPZ plus JSON completion marker",
            "trajectory_composition": "single-writer streamed CPU composition",
        },
        "recovery_deformation_policy": (
            "non-consecutive recovery pair fields reconnect trajectories only; "
            "only primary pair fields produce deformation cells"
        ),
    }
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run-manifest-v4.json"
    temporary = output / ".run-manifest-v4.json.writing"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, file_sha256(path)


def _git_state(repository: Path | None = None) -> dict[str, str | bool | None]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                [
                    "git",
                    *([] if repository is None else ["-C", str(repository)]),
                    *arguments,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"revision": revision, "dirty": None if status is None else bool(status)}
