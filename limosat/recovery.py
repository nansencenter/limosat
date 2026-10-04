"""SQLite-free targeting for measured-loss recovery pair products."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import shapely

from .catalog import ImagePair, ImageRecord, load_catalogue
from .config import RunConfig
from .pair_queue import loss_targeted_pairs
from .planning import PlannedPair, build_candidate_plan, recovery_candidates
from .store import file_sha256
from .trajectory import TrajectoryPoint


RECOVERY_TARGET_SCHEMA_VERSION = 1


def load_recovery_target_manifest(
    config: RunConfig, directory: str | Path
) -> dict:
    """Load and validate the identity and pair index of a completed target set."""
    path = Path(directory) / "recovery-targets-manifest-v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "recovery_target_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_sha256": config.sha256,
    }
    changed = [
        name for name, value in expected.items() if manifest.get(name) != value
    ]
    if changed:
        raise ValueError(f"recovery target manifest identity changed: {changed}")
    records = manifest.get("pairs")
    if not isinstance(records, list):
        raise ValueError("recovery target manifest pairs must be a list")
    pair_ids = [record.get("pair_id") for record in records]
    if (
        any(not isinstance(pair_id, str) for pair_id in pair_ids)
        or len(pair_ids) != len(set(pair_ids))
        or manifest.get("targeted_recovery_pairs") != len(pair_ids)
    ):
        raise ValueError("recovery target manifest pair index is invalid")
    return manifest


class RecoveryTargetStore:
    """Immutable per-pair source coordinates selected from primary trajectories."""

    def __init__(self, config: RunConfig, root: str | Path) -> None:
        self.config = config
        self.root = Path(root)

    def save(
        self,
        pair: ImagePair,
        positions_xy_m: np.ndarray,
        trajectory_ids: Sequence[str] | None = None,
    ) -> Path:
        positions = _positions(positions_xy_m)
        if not len(positions):
            raise ValueError("recovery targets cannot be empty")
        identities = (
            _trajectory_ids(trajectory_ids, len(positions))
            if trajectory_ids is not None else None
        )
        data_path, marker_path = self.paths(pair.pair_id)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        positions_sha256 = _positions_sha256(positions)
        if marker_path.exists():
            loaded = self.load(pair)
            if _positions_sha256(loaded) != positions_sha256:
                raise ValueError(f"immutable recovery targets differ: {pair.pair_id}")
            saved = json.loads(marker_path.read_text(encoding="utf-8"))
            if saved.get("trajectory_ids") != (
                list(identities) if identities is not None else None
            ):
                raise ValueError(f"immutable recovery identities differ: {pair.pair_id}")
            return marker_path

        temporary = _temporary(data_path)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, positions_xy_m=positions)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, data_path)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {
            "recovery_target_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "config_sha256": self.config.sha256,
            "pair_id": pair.pair_id,
            "source_image_id": pair.source.image_id,
            "target_image_id": pair.target.image_id,
            "source_time_utc": pair.source.time_utc.isoformat(),
            "target_time_utc": pair.target.time_utc.isoformat(),
            "target_count": len(positions),
            "trajectory_ids": list(identities) if identities is not None else None,
            "positions_sha256": positions_sha256,
            "data_file": data_path.name,
            "data_sha256": file_sha256(data_path),
            "data_size_bytes": data_path.stat().st_size,
        }
        marker_temporary = _temporary(marker_path)
        marker_temporary.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(marker_temporary, marker_path)
        return marker_path

    def load(self, pair: ImagePair) -> np.ndarray | None:
        data_path, marker_path = self.paths(pair.pair_id)
        if not marker_path.exists():
            return None
        metadata = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = {
            "recovery_target_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "config_sha256": self.config.sha256,
            "pair_id": pair.pair_id,
            "source_image_id": pair.source.image_id,
            "target_image_id": pair.target.image_id,
            "source_time_utc": pair.source.time_utc.isoformat(),
            "target_time_utc": pair.target.time_utc.isoformat(),
            "data_file": data_path.name,
        }
        changed = [
            name
            for name, value in expected.items()
            if metadata.get(name) != value
        ]
        if changed:
            raise ValueError(
                f"recovery target identity changed for {pair.pair_id}: {changed}"
            )
        if (
            not data_path.is_file()
            or file_sha256(data_path) != metadata.get("data_sha256")
        ):
            raise ValueError(f"recovery target data failed checksum: {pair.pair_id}")
        if data_path.stat().st_size != metadata.get("data_size_bytes"):
            raise ValueError(f"recovery target data size changed: {pair.pair_id}")
        with np.load(data_path, allow_pickle=False) as values:
            positions = _positions(values["positions_xy_m"])
        if len(positions) != metadata.get("target_count"):
            raise ValueError(f"recovery target count changed: {pair.pair_id}")
        if _positions_sha256(positions) != metadata.get("positions_sha256"):
            raise ValueError(
                f"recovery target positions failed checksum: {pair.pair_id}"
            )
        if metadata.get("trajectory_ids") is not None:
            _trajectory_ids(metadata["trajectory_ids"], len(positions))
        return positions

    def trajectory_ids(self, pair: ImagePair) -> frozenset[str]:
        """Return identities bound to the target array, or reject old unbound data."""
        _data_path, marker_path = self.paths(pair.pair_id)
        metadata = json.loads(marker_path.read_text(encoding="utf-8"))
        return frozenset(
            _trajectory_ids(metadata.get("trajectory_ids"), metadata.get("target_count"))
        )

    def count(self) -> int:
        return sum(1 for _path in self.root.glob("*.json"))

    def paths(self, pair_id: str) -> tuple[Path, Path]:
        identity = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
        base = self.root / identity
        return base.with_suffix(".npz"), base.with_suffix(".json")


def prepare_recovery_targets(
    config: RunConfig,
    primary_composition: str | Path,
    destination: str | Path,
    *,
    request_plan: str | Path | None = None,
) -> dict:
    """Select frozen source positions, optionally from an explicit pair request plan."""
    primary_root = Path(primary_composition)
    primary_manifest_path = primary_root / "composition-manifest-v1.json"
    primary_manifest = json.loads(
        primary_manifest_path.read_text(encoding="utf-8")
    )
    if primary_manifest.get("config_sha256") != config.sha256:
        raise ValueError("primary composition configuration differs from recovery")
    points_path = primary_root / "trajectory-points-v1.parquet"
    if (
        file_sha256(points_path)
        != primary_manifest["products"]["trajectory_points"]["sha256"]
    ):
        raise ValueError("primary trajectory points failed manifest checksum")

    catalogue = load_catalogue(config.catalogue, config.analysis_epsg)
    plan = build_candidate_plan(
        catalogue,
        config.routing,
        grid_spacing_m=config.routing.planning_grid_spacing_m,
        maximum_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
    )
    eligible = recovery_candidates(
        plan.pairs, config.routing.maximum_recovery_elapsed_hours
    )
    requests = None
    request_plan_sha256 = None
    if request_plan is not None:
        request_path = Path(request_plan)
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        expected = {
            "pair_request_schema_version": 1,
            "run_id": config.run_id,
            "config_sha256": config.sha256,
            "primary_composition_manifest_sha256": file_sha256(
                primary_manifest_path
            ),
        }
        if any(request_data.get(key) != value for key, value in expected.items()):
            raise ValueError("recovery request plan refers to a different primary run")
        requests = {}
        eligible_ids = {item.pair.pair_id for item in eligible}
        for record in request_data["pairs"]:
            pair_id = record["pair_id"]
            identities = record["trajectory_ids"]
            if (
                pair_id not in eligible_ids
                or pair_id in requests
                or not identities
                or len(identities) != len(set(identities))
            ):
                raise ValueError(f"invalid recovery pair request: {pair_id}")
            requests[pair_id] = frozenset(identities)
        request_plan_sha256 = file_sha256(request_path)
    images = catalogue.chronological()
    output = Path(destination)
    manifest_path = output / "recovery-targets-manifest-v1.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    store = RecoveryTargetStore(config, output / "pairs")
    counts = {"unscheduled_loss_positions": 0}
    policy = (
        "request_plan" if requests is not None
        else config.routing.recovery_target_policy
    )
    loss_candidates = None
    if policy == "loss_targeted":
        collected: dict[str, tuple[str, ...]] = {}
        _write_recovery_targets(
            store, eligible, images,
            iter_primary_point_batches(points_path, images),
            counts=counts, collect=collected,
        )
        loss_candidates = len(collected)
        elapsed = {item.pair.pair_id: item.pair.elapsed_seconds for item in eligible}
        selected = loss_targeted_pairs(
            (pair_id, elapsed[pair_id], identities)
            for pair_id, identities in collected.items()
        )
        requests = {
            pair_id: frozenset(collected[pair_id]) for pair_id in selected
        }
        del collected
    targeted = _write_recovery_targets(
        store, eligible, images,
        iter_primary_point_batches(points_path, images),
        requests, counts=counts,
    )
    product_set = hashlib.sha256()
    target_count = 0
    records = []
    for item in targeted:
        _data, marker = store.paths(item.pair.pair_id)
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        marker_sha256 = file_sha256(marker)
        product_set.update(item.pair.pair_id.encode("utf-8"))
        product_set.update(marker_sha256.encode("ascii"))
        target_count += metadata["target_count"]
        records.append(
            {
                "pair_id": item.pair.pair_id,
                "source_image_id": item.pair.source.image_id,
                "target_image_id": item.pair.target.image_id,
                "target_count": metadata["target_count"],
                "marker_sha256": marker_sha256,
            }
        )
    manifest = {
        "recovery_target_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_sha256": config.sha256,
        "primary_composition_manifest": str(primary_manifest_path),
        "primary_composition_manifest_sha256": file_sha256(primary_manifest_path),
        "request_plan_sha256": request_plan_sha256,
        "target_policy": policy,
        "loss_candidate_pairs": loss_candidates,
        "unscheduled_loss_positions": counts["unscheduled_loss_positions"],
        "eligible_recovery_pairs": len(eligible),
        "targeted_recovery_pairs": len(targeted),
        "targeted_positions": target_count,
        "pair_target_set_sha256": product_set.hexdigest(),
        "pairs": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(manifest_path)
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return {"manifest": str(manifest_path), **{k: manifest[k] for k in (
        "eligible_recovery_pairs", "targeted_recovery_pairs", "targeted_positions",
        "unscheduled_loss_positions", "target_policy", "loss_candidate_pairs",
    )}}


def iter_primary_point_batches(
    path: str | Path, images: Sequence[ImageRecord]
) -> Iterable[tuple[TrajectoryPoint, ...]]:
    """Read composer output as deterministic per-image primary batches."""
    _pa, pq = _pyarrow()
    parquet = pq.ParquetFile(path)
    image_order = {image.image_id: index for index, image in enumerate(images)}
    grouped: dict[str, list[int]] = {}
    image_column = parquet.schema_arrow.names.index("image_id")
    for row_group in range(parquet.metadata.num_row_groups):
        statistics = (
            parquet.metadata.row_group(row_group).column(image_column).statistics
        )
        image_id = statistics.min if statistics and statistics.has_min_max else None
        if image_id is None or image_id != statistics.max:
            values = parquet.read_row_group(row_group, columns=["image_id"])["image_id"]
            unique = values.unique().to_pylist()
            if len(unique) != 1:
                raise ValueError("primary point row group spans multiple images")
            image_id = unique[0]
        if image_id not in image_order:
            raise ValueError(
                f"primary point image is absent from catalogue: {image_id}"
            )
        grouped.setdefault(image_id, []).append(row_group)
    columns = [
        "trajectory_id", "image_id", "time_utc", "state", "position_basis",
        "x_m", "y_m", "source_pair_id", "selected_matches",
        "support_radius_m", "maximum_residual_m",
    ]
    for image in images:
        rows = []
        for row_group in grouped.get(image.image_id, ()):
            rows.extend(parquet.read_row_group(row_group, columns=columns).to_pylist())
        yield tuple(
            TrajectoryPoint(
                trajectory_id=row["trajectory_id"], image_id=row["image_id"],
                time_utc=row["time_utc"], state=row["state"],
                position_basis=row["position_basis"], x_m=row["x_m"], y_m=row["y_m"],
                source_pair_id=row["source_pair_id"],
                selected_matches=row["selected_matches"],
                support_radius_m=row["support_radius_m"],
                maximum_residual_m=row["maximum_residual_m"],
            )
            for row in rows
        )


def _write_recovery_targets(
    store: RecoveryTargetStore,
    eligible: Sequence[PlannedPair],
    images: Sequence[ImageRecord],
    primary_points_by_image: Iterable[Sequence[TrajectoryPoint]],
    requests: Mapping[str, frozenset[str]] | None = None,
    *,
    counts: dict[str, int] | None = None,
    collect: dict[str, tuple[str, ...]] | None = None,
) -> tuple[PlannedPair, ...]:
    """Write recovery targets for dormant rows and unscheduled losses.

    With ``collect``, nothing is written: each candidate pair's identities are
    recorded instead, for loss-targeted selection.

    An unscheduled loss is an identity whose last measurement is on the pair's
    source image and which has no row at all on the target image, because no
    primary pair from its last image reached that target. Each such identity is
    nominated once, for the earliest eligible target whose footprint contains its
    frozen source position. Nothing is predicted: the recovery field measures it.
    """
    index = {image.image_id: step for step, image in enumerate(images)}
    last_measured: dict[str, int] = {}
    nominated: set[str] = set()
    by_target: dict[int, list[PlannedPair]] = {}
    last_use: dict[int, int] = {}
    source_requests: dict[int, set[str]] = {}
    for item in eligible:
        if requests is not None and item.pair.pair_id not in requests:
            continue
        source = index[item.pair.source.image_id]
        target = index[item.pair.target.image_id]
        by_target.setdefault(target, []).append(item)
        last_use[source] = max(last_use.get(source, target), target)
        if requests is not None:
            source_requests.setdefault(source, set()).update(
                requests[item.pair.pair_id]
            )
    positions: list[dict[str, tuple[float, float]]] = [dict() for _ in images]
    targeted = []
    batches = iter(primary_points_by_image)
    for step, image in enumerate(images):
        try:
            points = tuple(next(batches))
        except StopIteration as error:
            raise ValueError(
                "primary point batches ended before the catalogue"
            ) from error
        if any(point.image_id != image.image_id for point in points):
            raise ValueError("primary point batch does not match image chronology")
        if step in last_use:
            needed = source_requests.get(step)
            positions[step] = {
                point.trajectory_id: (float(point.x_m), float(point.y_m))
                for point in points
                if point.available
                and (needed is None or point.trajectory_id in needed)
            }
        dormant = {
            point.trajectory_id for point in points if point.state == "dormant"
        }
        measured = {
            point.trajectory_id for point in points if point.available
        }
        present = {point.trajectory_id for point in points}
        for item in by_target.get(step, ()):
            source_step = index[item.pair.source.image_id]
            source = positions[source_step]
            if requests is None:
                identities = sorted(dormant & source.keys())
                nominated.update(identities)
                lost = _unscheduled_losses(
                    source, source_step, present, last_measured,
                    nominated, image.footprint,
                )
                nominated.update(lost)
                if counts is not None:
                    counts["unscheduled_loss_positions"] += len(lost)
                identities = sorted(set(identities).union(lost))
            else:
                requested = requests[item.pair.pair_id]
                if requested - source.keys():
                    raise ValueError(
                        f"requested recovery source is not measured: {item.pair.pair_id}"
                    )
                if requested & measured:
                    raise ValueError(
                        f"requested recovery target is already measured: {item.pair.pair_id}"
                    )
                identities = sorted(requested)
            if identities and collect is not None:
                collect[item.pair.pair_id] = tuple(identities)
                targeted.append(item)
            elif identities:
                store.save(
                    item.pair,
                    np.asarray(
                        [source[identity] for identity in identities],
                        dtype=np.float64,
                    ),
                    trajectory_ids=identities,
                )
                targeted.append(item)
        for identity in measured:
            last_measured[identity] = step
        for source_step, target_step in tuple(last_use.items()):
            if target_step == step:
                positions[source_step].clear()
    try:
        next(batches)
    except StopIteration:
        return tuple(targeted)
    raise ValueError("primary point batches exceed the catalogue")


def unscheduled_losses_in_footprint(identities, xy, footprint) -> list[str]:
    """Keep identities whose frozen source position lies in a target footprint."""
    if not identities or footprint is None:
        return list(identities)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    inside = shapely.contains_xy(footprint, xy[:, 0], xy[:, 1])
    return [identity for identity, keep in zip(identities, inside) if keep]


def _unscheduled_losses(
    source: Mapping[str, tuple[float, float]],
    source_step: int,
    present: set[str],
    last_measured: Mapping[str, int],
    nominated: set[str],
    footprint,
) -> list[str]:
    candidates = [
        identity for identity in source
        if identity not in present
        and identity not in nominated
        and last_measured.get(identity) == source_step
    ]
    return unscheduled_losses_in_footprint(
        candidates, [source[identity] for identity in candidates], footprint
    )


def _positions(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("recovery targets must have shape (n, 2)")
    if not np.isfinite(array).all():
        raise ValueError("recovery targets must be finite")
    return array


def _trajectory_ids(values: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError("recovery target trajectory IDs are missing")
    identities = tuple(values)
    if (
        len(identities) != count
        or any(not isinstance(value, str) or not value for value in identities)
        or len(identities) != len(set(identities))
        or identities != tuple(sorted(identities))
    ):
        raise ValueError("recovery target trajectory IDs are invalid")
    return identities


def _positions_sha256(values: np.ndarray) -> str:
    array = _positions(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.writing.{os.getpid()}")


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("recovery targeting requires PyArrow") from error
    return pa, pq
