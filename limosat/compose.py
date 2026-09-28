"""Compose immutable pair products into internal trajectory artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .catalog import load_catalogue
from .config import RunConfig
from .models import FieldEdge
from .pair_products import PAIR_PRODUCT_SCHEMA_VERSION, PairProductStore
from .planning import build_candidate_plan, recovery_candidates
from .qc import QC_PROTOCOL_ID, ExtensionQCConfig, ScoredExtension, score_extensions
from .recovery import (
    RecoveryTargetStore,
    iter_primary_point_batches,
    load_recovery_target_manifest,
)
from .store import file_sha256
from .trajectory import (
    TrajectoryPoint,
    iter_frozen_primary_augmentation_batches,
    iter_global_trajectory_batches,
)


COMPOSITION_SCHEMA_VERSION = 1
TRAJECTORY_POINT_SCHEMA_VERSION = 1
TRAJECTORY_EXTENSION_SCHEMA_VERSION = 1
TRAJECTORY_AUGMENTATION_SCHEMA_VERSION = 1


def compose_primary_parquet(
    config: RunConfig,
    destination: str | Path,
    *,
    qc_config: ExtensionQCConfig | None = None,
) -> dict:
    """Write primary trajectory points and extensions without modifying SQLite."""
    pa, pq = _pyarrow()
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    points_path = output / "trajectory-points-v1.parquet"
    extensions_path = output / "trajectory-extensions-qc-v1.parquet"
    manifest_path = output / "composition-manifest-v1.json"
    for path in (points_path, extensions_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    qc = qc_config or ExtensionQCConfig(
        configured_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
        hard_speed_m_per_day=max(
            60_000.0, config.matcher.maximum_speed_m_per_day
        ),
    )
    catalogue = load_catalogue(config.catalogue, config.analysis_epsg)
    plan = build_candidate_plan(
        catalogue,
        config.routing,
        grid_spacing_m=config.routing.planning_grid_spacing_m,
        maximum_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
    )
    primary = tuple(item for item in plan.pairs if item.selection == "primary")
    products = PairProductStore(config)
    if products.count("primary") != len(primary):
        raise RuntimeError(
            "primary pair-product count differs from the frozen plan: "
            f"{products.count('primary')} != {len(primary)}"
        )

    edges: list[FieldEdge] = []
    producer_hashes: set[str] = set()
    product_set = hashlib.sha256()
    for item in primary:
        product = products.load(
            item.pair,
            "primary",
            False,
            require_current_implementation=False,
        )
        if product is None:
            raise RuntimeError(f"primary pair product is missing: {item.pair.pair_id}")
        producer_hashes.add(product.producer_implementation_sha256)
        product_set.update(item.pair.pair_id.encode("utf-8"))
        product_set.update(product.sha256.encode("ascii"))
        product_set.update(product.content_sha256.encode("ascii"))
        edges.append(FieldEdge(product.result.field))
    if len(producer_hashes) != 1:
        raise ValueError(
            "primary pair products have multiple producer implementations: "
            f"{sorted(producer_hashes)}"
        )

    point_schema = _point_schema(pa, config)
    extension_schema = _extension_schema(pa, config, qc)
    points_temporary = _temporary(points_path)
    extensions_temporary = _temporary(extensions_path)
    point_writer = pq.ParquetWriter(points_temporary, point_schema, compression="zstd")
    extension_writer = pq.ParquetWriter(
        extensions_temporary, extension_schema, compression="zstd"
    )
    point_count = 0
    extension_count = 0
    point_states: Counter[str] = Counter()
    qc_statuses: Counter[str] = Counter()
    try:
        for batch in iter_global_trajectory_batches(
            edges,
            catalogue.chronological(),
            config.field,
            config.trajectories,
        ):
            if batch.points:
                point_writer.write_table(
                    pa.Table.from_pylist(
                        [_point_row(point) for point in batch.points],
                        schema=point_schema,
                    )
                )
                point_count += len(batch.points)
                point_states.update(point.state for point in batch.points)
            scored = score_extensions(batch.extensions, qc)
            if scored:
                extension_writer.write_table(
                    pa.Table.from_pylist(
                        [_extension_row(value) for value in scored],
                        schema=extension_schema,
                    )
                )
                extension_count += len(scored)
                qc_statuses.update(value.qc_status for value in scored)
        point_writer.close()
        point_writer = None
        extension_writer.close()
        extension_writer = None
        os.replace(points_temporary, points_path)
        os.replace(extensions_temporary, extensions_path)
    finally:
        if point_writer is not None:
            point_writer.close()
        if extension_writer is not None:
            extension_writer.close()
        points_temporary.unlink(missing_ok=True)
        extensions_temporary.unlink(missing_ok=True)

    manifest = {
        "composition_schema_version": COMPOSITION_SCHEMA_VERSION,
        "trajectory_point_schema_version": TRAJECTORY_POINT_SCHEMA_VERSION,
        "trajectory_extension_schema_version": TRAJECTORY_EXTENSION_SCHEMA_VERSION,
        "qc_protocol_id": QC_PROTOCOL_ID,
        "run_id": config.run_id,
        "config_sha256": config.sha256,
        "producer_implementation_sha256": next(iter(producer_hashes)),
        "consumer_implementation_note": (
            "consumer code may differ; every pair-product checksum was verified"
        ),
        "pair_product_schema_version": PAIR_PRODUCT_SCHEMA_VERSION,
        "primary_pair_products": len(primary),
        "pair_product_set_sha256": product_set.hexdigest(),
        "qc": asdict(qc),
        "qc_policy": {
            "hard_speed": "reject",
            "configured_speed": "review",
            "nearest_neighbour": "review_only",
            "neighbour_scope": "exact image pair, leave one out",
        },
        "counts": {
            "trajectory_points": point_count,
            "trajectory_point_states": dict(sorted(point_states.items())),
            "trajectory_extensions": extension_count,
            "extension_qc_status": dict(sorted(qc_statuses.items())),
        },
        "products": {
            "trajectory_points": _file_record(points_path),
            "trajectory_extensions": _file_record(extensions_path),
        },
    }
    temporary_manifest = _temporary(manifest_path)
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return {
        "manifest": str(manifest_path),
        "trajectory_points": str(points_path),
        "trajectory_extensions": str(extensions_path),
        "counts": manifest["counts"],
    }


def compose_recovery_parquet(
    config: RunConfig,
    primary_composition: str | Path,
    recovery_targets: str | Path,
    destination: str | Path,
    *,
    qc_config: ExtensionQCConfig | None = None,
) -> dict:
    """Write frozen-primary recovery deltas without modifying primary artifacts."""
    pa, pq = _pyarrow()
    primary_root = Path(primary_composition)
    targets_root = Path(recovery_targets)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    points_path = output / "trajectory-augmentations-v1.parquet"
    extensions_path = output / "trajectory-augmentation-extensions-qc-v1.parquet"
    manifest_path = output / "recovery-composition-manifest-v1.json"
    for path in (points_path, extensions_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    primary_manifest_path = primary_root / "composition-manifest-v1.json"
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    if primary_manifest.get("config_sha256") != config.sha256:
        raise ValueError("primary composition configuration differs from recovery")
    primary_points = primary_root / "trajectory-points-v1.parquet"
    primary_extensions = primary_root / "trajectory-extensions-qc-v1.parquet"
    for name, path in (
        ("trajectory_points", primary_points),
        ("trajectory_extensions", primary_extensions),
    ):
        if file_sha256(path) != primary_manifest["products"][name]["sha256"]:
            raise ValueError(f"primary {name} failed manifest checksum")

    target_manifest_path = targets_root / "recovery-targets-manifest-v1.json"
    target_manifest = load_recovery_target_manifest(config, targets_root)
    if target_manifest.get("primary_composition_manifest_sha256") != file_sha256(
        primary_manifest_path
    ):
        raise ValueError("recovery targets refer to a different primary composition")

    qc = qc_config or ExtensionQCConfig(
        configured_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
        hard_speed_m_per_day=max(60_000.0, config.matcher.maximum_speed_m_per_day),
    )
    catalogue = load_catalogue(config.catalogue, config.analysis_epsg)
    plan = build_candidate_plan(
        catalogue,
        config.routing,
        grid_spacing_m=config.routing.planning_grid_spacing_m,
        maximum_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
    )
    primary = tuple(item for item in plan.pairs if item.selection == "primary")
    eligible = recovery_candidates(
        plan.pairs, config.routing.maximum_recovery_elapsed_hours
    )
    by_pair_id = {item.pair.pair_id: item for item in eligible}
    if target_manifest.get("eligible_recovery_pairs") != len(eligible):
        raise ValueError("recovery target plan differs from current eligible pairs")
    target_pair_ids = [record["pair_id"] for record in target_manifest["pairs"]]
    if len(target_pair_ids) != len(set(target_pair_ids)):
        raise ValueError("recovery target manifest contains duplicate pairs")
    targeted = []
    for pair_id in target_pair_ids:
        item = by_pair_id.get(pair_id)
        if item is None:
            raise ValueError(f"recovery target is not an eligible candidate: {pair_id}")
        targeted.append(item)

    products = PairProductStore(config)
    target_store = RecoveryTargetStore(config, targets_root / "pairs")
    primary_edges = []
    recovery_edges = []
    primary_producers: set[str] = set()
    recovery_producers: set[str] = set()
    recovery_product_set = hashlib.sha256()
    target_set = hashlib.sha256()
    for item in primary:
        product = products.load(
            item.pair, "primary", False, require_current_implementation=False
        )
        if product is None:
            raise RuntimeError(f"primary pair product is missing: {item.pair.pair_id}")
        primary_producers.add(product.producer_implementation_sha256)
        primary_edges.append(FieldEdge(product.result.field))
    target_records = {record["pair_id"]: record for record in target_manifest["pairs"]}
    for item in targeted:
        positions = target_store.load(item.pair)
        if positions is None:
            raise RuntimeError(f"recovery targets are missing: {item.pair.pair_id}")
        record = target_records[item.pair.pair_id]
        _data_path, marker_path = target_store.paths(item.pair.pair_id)
        if (
            record.get("target_count") != len(positions)
            or record.get("marker_sha256") != file_sha256(marker_path)
        ):
            raise ValueError(
                f"recovery target manifest record changed: {item.pair.pair_id}"
            )
        target_set.update(item.pair.pair_id.encode("utf-8"))
        target_set.update(record["marker_sha256"].encode("ascii"))
        product = products.load(
            item.pair,
            "recovery",
            True,
            positions,
            require_current_implementation=False,
        )
        if product is None:
            raise RuntimeError(f"recovery pair product is missing: {item.pair.pair_id}")
        recovery_producers.add(product.producer_implementation_sha256)
        recovery_product_set.update(item.pair.pair_id.encode("utf-8"))
        recovery_product_set.update(product.sha256.encode("ascii"))
        recovery_product_set.update(product.content_sha256.encode("ascii"))
        recovery_edges.append(
            FieldEdge(
                product.result.field,
                pair_kind="recovery",
                skipped_images=item.skipped_images,
                eligible_trajectory_ids=target_store.trajectory_ids(item.pair),
            )
        )
    if target_set.hexdigest() != target_manifest.get("pair_target_set_sha256"):
        raise ValueError("recovery target set failed manifest checksum")
    if len(primary_producers) != 1 or len(recovery_producers) > 1:
        raise ValueError("pair products have inconsistent producer implementations")

    point_schema = _point_schema(pa, config, "trajectory_augmentations_v1")
    extension_schema = _extension_schema(
        pa, config, qc, "trajectory_augmentation_extensions_qc_v1"
    )
    point_temporary = _temporary(points_path)
    extension_temporary = _temporary(extensions_path)
    point_writer = pq.ParquetWriter(point_temporary, point_schema, compression="zstd")
    extension_writer = pq.ParquetWriter(
        extension_temporary, extension_schema, compression="zstd"
    )
    point_count = 0
    extension_count = 0
    point_states: Counter[str] = Counter()
    point_bases: Counter[str] = Counter()
    extension_kinds: Counter[str] = Counter()
    qc_statuses: Counter[str] = Counter()
    try:
        batches = iter_primary_point_batches(
            primary_points, catalogue.chronological()
        )
        for batch in iter_frozen_primary_augmentation_batches(
            batches,
            (*primary_edges, *recovery_edges),
            catalogue.chronological(),
            config.field,
        ):
            if batch.points:
                point_writer.write_table(
                    pa.Table.from_pylist(
                        [_point_row(point) for point in batch.points],
                        schema=point_schema,
                    )
                )
                point_count += len(batch.points)
                point_states.update(point.state for point in batch.points)
                point_bases.update(point.position_basis for point in batch.points)
            scored = score_extensions(batch.extensions, qc)
            if scored:
                extension_writer.write_table(
                    pa.Table.from_pylist(
                        [_extension_row(value) for value in scored],
                        schema=extension_schema,
                    )
                )
                extension_count += len(scored)
                extension_kinds.update(value.extension.pair_kind for value in scored)
                qc_statuses.update(value.qc_status for value in scored)
        point_writer.close()
        point_writer = None
        extension_writer.close()
        extension_writer = None
        os.replace(point_temporary, points_path)
        os.replace(extension_temporary, extensions_path)
    finally:
        if point_writer is not None:
            point_writer.close()
        if extension_writer is not None:
            extension_writer.close()
        point_temporary.unlink(missing_ok=True)
        extension_temporary.unlink(missing_ok=True)

    manifest = {
        "composition_schema_version": COMPOSITION_SCHEMA_VERSION,
        "trajectory_augmentation_schema_version": (
            TRAJECTORY_AUGMENTATION_SCHEMA_VERSION
        ),
        "trajectory_extension_schema_version": TRAJECTORY_EXTENSION_SCHEMA_VERSION,
        "qc_protocol_id": QC_PROTOCOL_ID,
        "run_id": config.run_id,
        "config_sha256": config.sha256,
        "semantics": "frozen primary rows plus recovery and post-reappearance deltas",
        "primary_composition_manifest": str(primary_manifest_path),
        "primary_composition_manifest_sha256": file_sha256(primary_manifest_path),
        "recovery_target_manifest": str(target_manifest_path),
        "recovery_target_manifest_sha256": file_sha256(target_manifest_path),
        "primary_producer_implementation_sha256": next(iter(primary_producers)),
        "recovery_producer_implementation_sha256": (
            next(iter(recovery_producers)) if recovery_producers else None
        ),
        "targeted_recovery_pair_products": len(targeted),
        "recovery_pair_product_set_sha256": recovery_product_set.hexdigest(),
        "qc": asdict(qc),
        "counts": {
            "trajectory_augmentations": point_count,
            "trajectory_augmentation_states": dict(sorted(point_states.items())),
            "trajectory_augmentation_position_bases": dict(sorted(point_bases.items())),
            "trajectory_augmentation_extensions": extension_count,
            "trajectory_augmentation_extension_kinds": dict(
                sorted(extension_kinds.items())
            ),
            "extension_qc_status": dict(sorted(qc_statuses.items())),
        },
        "products": {
            "trajectory_augmentations": _file_record(points_path),
            "trajectory_augmentation_extensions": _file_record(extensions_path),
        },
    }
    temporary_manifest = _temporary(manifest_path)
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return {
        "manifest": str(manifest_path),
        "trajectory_augmentations": str(points_path),
        "trajectory_augmentation_extensions": str(extensions_path),
        "counts": manifest["counts"],
    }


def _point_row(point: TrajectoryPoint) -> dict:
    return {
        "trajectory_id": point.trajectory_id,
        "image_id": point.image_id,
        "time_utc": point.time_utc,
        "state": point.state,
        "position_basis": point.position_basis,
        "x_m": point.x_m,
        "y_m": point.y_m,
        "source_pair_id": point.source_pair_id,
        "selected_matches": point.selected_matches,
        "support_radius_m": point.support_radius_m,
        "maximum_residual_m": point.maximum_residual_m,
    }


def _extension_row(value: ScoredExtension) -> dict:
    extension = value.extension
    return {
        "trajectory_id": extension.trajectory_id,
        "pair_id": extension.pair_id,
        "pair_kind": extension.pair_kind,
        "source_image_id": extension.source_image_id,
        "target_image_id": extension.target_image_id,
        "start_time_utc": extension.start_time_utc,
        "end_time_utc": extension.end_time_utc,
        "elapsed_seconds": value.elapsed_seconds,
        "x0_m": extension.x0_m,
        "y0_m": extension.y0_m,
        "x1_m": extension.x1_m,
        "y1_m": extension.y1_m,
        "dx_m": value.dx_m,
        "dy_m": value.dy_m,
        "distance_m": value.distance_m,
        "speed_m_per_day": value.speed_m_per_day,
        "target_state": extension.target_state,
        "target_position_basis": extension.target_position_basis,
        "selected_matches": extension.selected_matches,
        "support_radius_m": extension.support_radius_m,
        "maximum_residual_m": extension.maximum_residual_m,
        "local_neighbour_count": value.local_neighbour_count,
        "local_support_radius_m": value.local_support_radius_m,
        "local_median_dx_m": value.local_median_dx_m,
        "local_median_dy_m": value.local_median_dy_m,
        "local_residual_m": value.local_residual_m,
        "local_robust_z": value.local_robust_z,
        "exceeds_configured_speed": value.exceeds_configured_speed,
        "hard_speed_reject": value.hard_speed_reject,
        "local_neighbour_review": value.local_neighbour_review,
        "qc_status": value.qc_status,
        "qc_reasons": value.qc_reasons,
        "accepted": not value.hard_speed_reject,
    }


def _point_schema(
    pa, config: RunConfig, name: str = "trajectory_points_v1"
):
    return pa.schema(
        [
            ("trajectory_id", pa.string()),
            ("image_id", pa.string()),
            ("time_utc", pa.timestamp("us", tz="UTC")),
            ("state", pa.string()),
            ("position_basis", pa.string()),
            ("x_m", pa.float64()),
            ("y_m", pa.float64()),
            ("source_pair_id", pa.string()),
            ("selected_matches", pa.float64()),
            ("support_radius_m", pa.float64()),
            ("maximum_residual_m", pa.float64()),
        ],
        metadata=_metadata(
            name,
            config,
            {},
        ),
    )


def _extension_schema(
    pa,
    config: RunConfig,
    qc: ExtensionQCConfig,
    name: str = "trajectory_extensions_qc_v1",
):
    fields = [
        ("trajectory_id", pa.string()),
        ("pair_id", pa.string()),
        ("pair_kind", pa.string()),
        ("source_image_id", pa.string()),
        ("target_image_id", pa.string()),
        ("start_time_utc", pa.timestamp("us", tz="UTC")),
        ("end_time_utc", pa.timestamp("us", tz="UTC")),
        ("elapsed_seconds", pa.float64()),
        ("x0_m", pa.float64()),
        ("y0_m", pa.float64()),
        ("x1_m", pa.float64()),
        ("y1_m", pa.float64()),
        ("dx_m", pa.float64()),
        ("dy_m", pa.float64()),
        ("distance_m", pa.float64()),
        ("speed_m_per_day", pa.float64()),
        ("target_state", pa.string()),
        ("target_position_basis", pa.string()),
        ("selected_matches", pa.float64()),
        ("support_radius_m", pa.float64()),
        ("maximum_residual_m", pa.float64()),
        ("local_neighbour_count", pa.int32()),
        ("local_support_radius_m", pa.float64()),
        ("local_median_dx_m", pa.float64()),
        ("local_median_dy_m", pa.float64()),
        ("local_residual_m", pa.float64()),
        ("local_robust_z", pa.float64()),
        ("exceeds_configured_speed", pa.bool_()),
        ("hard_speed_reject", pa.bool_()),
        ("local_neighbour_review", pa.bool_()),
        ("qc_status", pa.string()),
        ("qc_reasons", pa.string()),
        ("accepted", pa.bool_()),
    ]
    return pa.schema(
        fields,
        metadata=_metadata(
            name,
            config,
            {
                "limosat.qc_protocol": QC_PROTOCOL_ID,
                "limosat.qc_config": json.dumps(
                    asdict(qc), sort_keys=True, separators=(",", ":")
                ),
            },
        ),
    )


def _metadata(
    name: str,
    config: RunConfig,
    extra: dict[str, str],
) -> dict[bytes, bytes]:
    values = {
        "limosat.schema": name,
        "limosat.run_id": config.run_id,
        "limosat.config_sha256": config.sha256,
        "limosat.crs": "EPSG:3413",
        "limosat.coordinate_units": "metres",
        "limosat.time_zone": "UTC",
        **extra,
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.writing.{os.getpid()}")


def _file_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Parquet composition requires PyArrow; install the parquet optional "
            "dependency in the CPU composition environment"
        ) from error
    return pa, pq
