"""Versioned, checksummed run manifests."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .catalog import ImageCatalogue
from .config import RunConfig
from .store import RunStore, file_sha256


MANIFEST_SCHEMA_VERSION = 1


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
            "sqlite": 1,
            "pair_displacement_field": 1,
            "lagrangian_trajectory": 2,
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
        "checkpoint_sha256": (
            file_sha256(config.matcher.checkpoint)
            if config.matcher.checkpoint and Path(config.matcher.checkpoint).is_file()
            else None
        ),
        "git": _git_state(),
        "command": list(command),
        "started_utc": started_utc.isoformat(),
        "completed_utc": completed_utc.isoformat(),
        "runtime_seconds": float(runtime_seconds),
        "components": {
            name: [image.image_id for image in images]
            for name, images in catalogue.components().items()
        },
        "images": rows["images"],
        "pairs": rows["pairs"],
        "product_counts": rows["counts"],
        "recovery_deformation_policy": "recovery fields reconnect trajectories only",
    }
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run-manifest-v1.json"
    temporary = output / ".run-manifest-v1.json.writing"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, file_sha256(path)


def _git_state() -> dict[str, str | bool | None]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"revision": revision, "dirty": None if status is None else bool(status)}
