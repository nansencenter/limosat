"""Compose immutable pair products into internal trajectory artifacts."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from .catalog import load_catalogue
from .config import RunConfig
from .field import field_edge_with_fallback
from .imagery import pass_pixel_validity_factory
from .surface import seed_filter_factory, surface_gate, surface_validity_factory
from .pair_products import PAIR_PRODUCT_SCHEMA_VERSION, PairProductStore
from .planning import run_plan
from .qc import QC_PROTOCOL_ID, ExtensionQCConfig, ScoredExtension, score_extensions
from .store import file_sha256
from .trajectory import PairSpec, TrajectoryPoint, _Done, compose_batches, sample_edge


COMPOSITION_SCHEMA_VERSION = 1
TRAJECTORY_POINT_SCHEMA_VERSION = 1
TRAJECTORY_EXTENSION_SCHEMA_VERSION = 1
TARGET_PIXEL_VALIDITY_POLICY = "same_pass_raster_bounds_and_band2_lt2"


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
    merges_path = output / "trajectory-merges-v1.parquet"
    for path in (points_path, extensions_path, manifest_path, merges_path):
        if path.exists():
            raise FileExistsError(path)

    qc = qc_config or ExtensionQCConfig(
        configured_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
        hard_speed_m_per_day=max(
            60_000.0, config.matcher.maximum_speed_m_per_day
        ),
    )
    catalogue = load_catalogue(config.catalogue, config.analysis_epsg)
    plan = run_plan(config, catalogue)
    primary = tuple(item for item in plan.pairs if item.selection == "primary")
    # A shared store may hold other runs' pairs too; every planned pair must load
    # (a missing one raises when it is sampled), extra ones are ignored.
    products = PairProductStore(config)
    images = catalogue.chronological()
    by_pair_id = {item.pair.pair_id: item for item in primary}
    specs = [
        PairSpec(item.pair.pair_id, item.pair.source.image_id, item.pair.target.image_id,
                 item.pair.source.time_utc)
        for item in primary
    ]
    validity_factory = surface_validity_factory(
        pass_pixel_validity_factory(images, config.analysis_epsg, config.matcher.scene_mask_invalid_from),
        surface_gate(config),
    )
    digests: dict[str, tuple[str, str, str]] = {}
    global _COMPOSE
    _COMPOSE = {"config": config, "products": products, "pairs": by_pair_id}
    workers = _compose_workers()
    executor = (
        ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"))
        if workers > 1 else None
    )

    def submit(pair_id, identities, queries, target_step):
        if executor is None:
            return _Done(_sample_pair(pair_id, identities, queries, target_step))
        return executor.submit(_sample_pair, pair_id, identities, queries, target_step)

    def nodes(pair_id):
        return products.field_nodes(by_pair_id[pair_id].pair)

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
    merge_rows: list[dict] = []
    try:
        for batch in compose_batches(
            specs,
            images,
            config.trajectories,
            nodes=nodes,
            submit=_recording(submit, digests),
            target_validity_factory=validity_factory,
            seed_filter_factory=seed_filter_factory(surface_gate(config)),
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
            merge_rows.extend(
                {"trajectory_id": event.trajectory_id, "merged_into": event.merged_into,
                 "image_id": event.image_id, "time_utc": event.time_utc,
                 "separation_m": event.separation_m}
                for event in batch.merges
            )
        point_writer.close()
        point_writer = None
        extension_writer.close()
        extension_writer = None
        os.replace(points_temporary, points_path)
        os.replace(extensions_temporary, extensions_path)
        if config.trajectories.merge_radius_m is not None:
            merge_schema = pa.schema([
                ("trajectory_id", pa.string()), ("merged_into", pa.string()),
                ("image_id", pa.string()), ("time_utc", pa.timestamp("us", tz="UTC")),
                ("separation_m", pa.float64()),
            ])
            merges_temporary = _temporary(merges_path)
            pq.write_table(pa.Table.from_pylist(merge_rows, schema=merge_schema),
                           merges_temporary, compression="zstd")
            os.replace(merges_temporary, merges_path)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        _COMPOSE = {}
        if point_writer is not None:
            point_writer.close()
        if extension_writer is not None:
            extension_writer.close()
        points_temporary.unlink(missing_ok=True)
        extensions_temporary.unlink(missing_ok=True)

    missing = [item.pair.pair_id for item in primary if item.pair.pair_id not in digests]
    if missing:
        raise RuntimeError(f"primary pair products were never sampled: {missing[:3]}")
    producer_hashes = {digests[item.pair.pair_id][2] for item in primary}
    product_set = hashlib.sha256()
    for item in primary:
        sha256, content_sha256, _producer = digests[item.pair.pair_id]
        product_set.update(item.pair.pair_id.encode("utf-8"))
        product_set.update(sha256.encode("ascii"))
        product_set.update(content_sha256.encode("ascii"))
    manifest = {
        "composition_schema_version": COMPOSITION_SCHEMA_VERSION,
        "trajectory_point_schema_version": TRAJECTORY_POINT_SCHEMA_VERSION,
        "trajectory_extension_schema_version": TRAJECTORY_EXTENSION_SCHEMA_VERSION,
        "qc_protocol_id": QC_PROTOCOL_ID,
        "target_pixel_validity": TARGET_PIXEL_VALIDITY_POLICY,
        # Gaps are implied by primary pairs, not stored (see iter_global_trajectory_batches).
        "gap_rows": "derived",
        "run_id": config.run_id,
        "config_sha256": config.sha256,
        "producer_implementation_sha256": (
            next(iter(producer_hashes)) if len(producer_hashes) == 1 else None
        ),
        "producer_implementation_sha256s": sorted(producer_hashes),
        "matching_sha256": products.matching_sha256,
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
            **({"trajectory_merges": len(merge_rows)}
               if config.trajectories.merge_radius_m is not None else {}),
        },
        "products": {
            "trajectory_points": _file_record(points_path),
            "trajectory_extensions": _file_record(extensions_path),
            **({"trajectory_merges": _file_record(merges_path)}
               if config.trajectories.merge_radius_m is not None else {}),
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


# Composition state shared with forked sampling workers (set in compose_primary_parquet).
_COMPOSE: dict = {}


def _compose_workers() -> int:
    """Sampling processes: LIMOSAT_COMPOSE_WORKERS, else the CPUs this process may use."""
    value = os.environ.get("LIMOSAT_COMPOSE_WORKERS")
    if value:
        return max(1, int(value))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - macOS
        return max(1, os.cpu_count() or 1)


def _sample_pair(pair_id, identities, queries, target_step):
    """Load and verify one pair product, sample it at its source positions and return
    the samples with the product's checksums (target validity is applied by the caller)."""
    state = _COMPOSE
    config = state["config"]
    item = state["pairs"][pair_id]
    product = state["products"].load(item.pair)
    if product is None:
        raise RuntimeError(f"primary pair product is missing: {pair_id}")
    edge = field_edge_with_fallback(product.result.field, product.result.matches, config.field)
    samples = sample_edge(edge, identities, queries, config.field)
    return replace(samples, product=(product.sha256, product.content_sha256,
                                     product.producer_implementation_sha256))


def _recording(submit, digests):
    """Wrap ``submit`` so each result's product checksums are kept for the manifest."""
    class _Recorded:
        def __init__(self, future, pair_id):
            self.future, self.pair_id = future, pair_id

        def result(self):
            value = self.future.result()
            digests[self.pair_id] = value.product
            return value

    return lambda pair_id, *args: _Recorded(submit(pair_id, *args), pair_id)


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
        "pair_kind": "primary",
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
