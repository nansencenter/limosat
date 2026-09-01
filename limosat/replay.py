"""Read-only adapters for replaying completed production displacement fields."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .catalog import ImageRecord
from .models import DisplacementField, FieldEdge
from .store import file_sha256


@dataclass(frozen=True)
class ReplayFieldSource:
    edge_id: str
    shard_id: str
    path: Path
    product_sha256: str
    overlap_fraction: float
    started_utc: str
    completed_utc: str
    node_count: int
    available_node_count: int


@dataclass(frozen=True)
class ProductionFieldReplay:
    root: Path
    images: tuple[ImageRecord, ...]
    edges: tuple[FieldEdge, ...]
    fields: tuple[ReplayFieldSource, ...]
    scene_inventory: tuple[dict, ...]
    source_metadata: dict


def load_production_field_replay(
    root: str | Path, *, verify_checksums: bool = True
) -> ProductionFieldReplay:
    """Load completed primary CSV fields without reading source imagery."""
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - analysis environment
        raise RuntimeError("pandas is required for production field replay") from error

    replay_root = Path(root).resolve()
    state_path = replay_root / "control" / "state.sqlite"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    with sqlite3.connect(state_path) as connection:
        connection.row_factory = sqlite3.Row
        scene_rows = connection.execute(
            "SELECT * FROM scenes ORDER BY time_utc,scene_id"
        ).fetchall()
        edge_rows = connection.execute(
            """
            SELECT e.*,a.started_utc,a.finished_utc,a.product_relative_path,
                   a.product_sha256
            FROM edges e JOIN edge_attempts a ON e.edge_id=a.edge_id
            WHERE e.role='primary' AND a.status='completed'
            ORDER BY e.edge_id
            """
        ).fetchall()
        metadata = {
            row["key"]: json.loads(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
    scene_components: dict[str, str] = {}
    for row in edge_rows:
        for scene_id in (row["source_scene_id"], row["target_scene_id"]):
            previous = scene_components.setdefault(scene_id, row["shard_id"])
            if previous != row["shard_id"]:
                raise ValueError(
                    f"scene {scene_id} belongs to multiple primary compute labels"
                )
    scenes = {row["scene_id"]: row for row in scene_rows}
    images = tuple(
        ImageRecord(
            image_id=row["scene_id"],
            path=replay_root / row["relative_path"],
            time_utc=_utc(row["time_utc"]),
            component_id=scene_components.get(row["scene_id"], "unpaired"),
        )
        for row in scene_rows
    )
    edges = []
    sources = []
    for index, row in enumerate(edge_rows, start=1):
        path = replay_root / row["product_relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_checksums:
            actual = file_sha256(path)
            if actual != row["product_sha256"]:
                raise ValueError(
                    f"source field checksum mismatch for {row['edge_id']}: {actual}"
                )
        frame = pd.read_csv(path)
        available = (
            frame["available"].astype(str).str.lower().eq("true").to_numpy()
        )
        displacement = frame[
            ["proposal_dx_m", "proposal_dy_m"]
        ].to_numpy(dtype=np.float64)
        displacement[~available] = np.nan
        source_scene = scenes[row["source_scene_id"]]
        target_scene = scenes[row["target_scene_id"]]
        field = DisplacementField(
            pair_id=row["edge_id"],
            source_image_id=row["source_scene_id"],
            target_image_id=row["target_scene_id"],
            source_time_utc=_utc(source_scene["time_utc"]),
            target_time_utc=_utc(target_scene["time_utc"]),
            grid_row=frame["grid_row"].to_numpy(dtype=np.int32),
            grid_column=frame["grid_column"].to_numpy(dtype=np.int32),
            source_xy_m=frame[["source_x", "source_y"]].to_numpy(
                dtype=np.float64
            ),
            displacement_m=displacement,
            available=available,
            selected_matches=frame["selected_vectors"]
            .fillna(0)
            .to_numpy(dtype=np.int32),
            candidate_matches=frame["candidate_count"]
            .fillna(0)
            .to_numpy(dtype=np.int32),
            support_radius_m=frame["support_radius_m"].to_numpy(
                dtype=np.float64
            ),
            maximum_residual_m=frame[
                "maximum_vector_residual_m"
            ].to_numpy(dtype=np.float64),
        )
        edges.append(FieldEdge(field))
        sources.append(
            ReplayFieldSource(
                edge_id=row["edge_id"],
                shard_id=row["shard_id"],
                path=path,
                product_sha256=row["product_sha256"],
                overlap_fraction=float(row["overlap_fraction"]),
                started_utc=row["started_utc"],
                completed_utc=row["finished_utc"],
                node_count=len(field),
                available_node_count=int(field.available.sum()),
            )
        )
        if index % 50 == 0:
            print(f"loaded {index}/{len(edge_rows)} completed primary fields")
    environment_path = replay_root / "control" / "environment.json"
    source_metadata = {
        "state_sqlite": {
            "path": str(state_path),
            "sha256": file_sha256(state_path),
        },
        "week_plan": _checksummed_file(replay_root / "control" / "week_plan.json"),
        "week_plan_report": _checksummed_file(
            replay_root / "control" / "week_plan_report.json"
        ),
        "environment": (
            {
                **_checksummed_file(environment_path),
                "content": json.loads(environment_path.read_text()),
            }
            if environment_path.is_file()
            else None
        ),
        "state_metadata": metadata,
        "verified_field_checksums": bool(verify_checksums),
    }
    return ProductionFieldReplay(
        replay_root,
        images,
        tuple(edges),
        tuple(sources),
        tuple(dict(row) for row in scene_rows),
        source_metadata,
    )


def replay_field_set_sha256(fields: tuple[ReplayFieldSource, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for field in fields:
        digest.update(field.edge_id.encode())
        digest.update(field.product_sha256.encode())
    return digest.hexdigest()


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"source timestamp is not timezone-aware: {value}")
    return parsed


def _checksummed_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": file_sha256(path)}
