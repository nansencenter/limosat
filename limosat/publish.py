"""Publish the LiMOSAT trajectory dataset (schema 2): one Parquet file of observations per period.

The public dataset is a *point table*: one row per observation of a trajectory, carrying the step that
arrived at it (displacement, speed, interpolation status, correlation, pair statistics and flags), the
Sentinel-1 scene it belongs to, and a ``primary`` flag that marks the rows belonging to the period's own
date window (selecting ``primary`` rows from all periods covers each date once). Rows are sorted by time.

This module only *projects* a frozen internal release. It consumes the internal kept-link Parquet of each
period (``<period>/<period>_qc.parquet`` and its manifest) and the period's image catalogue, and never
re-scores or alters a link. The public schema is defined once, here (``COLUMNS``, ``FLAG_BITS``); the schema
and a generated data dictionary are written into the release so that readers do not need a second copy.

Public outputs carry no local paths and no internal protocol detail. Internal provenance (source paths,
checksums, protocol ids) goes to a sidecar next to, not inside, the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "2.0.0"
CRS = "EPSG:3413"
OWNERSHIP_POLICY = "limosat_release_august_annual_v1"
FIRST_PERIOD_YEAR = 2015  # the first period also owns its first three months (from 1 May 2015)
ROWS_PER_GROUP = 250_000
PERIOD_PATTERN = re.compile(r"(\d{4})05_(\d{4})07")

FLAG_BITS = {"FEW_NEIGHBOURS": 0, "LOW_PAIR_DIRECT_SHARE": 1, "MASK_FALSE_LAND": 2}
FLAG_DESCRIPTIONS = {
    "FEW_NEIGHBOURS": "fewer than 2 other steps of the same scene pair within 30 km",
    "LOW_PAIR_DIRECT_SHARE": "less than 40% of the steps of the scene pair are direct matches "
    "(interpolated steps in such pairs are more often wrong; filter direct_match_share for other thresholds)",
    "MASK_FALSE_LAND": "an end of the step lies in false land of its own scene "
    "(scenes spanning 180 degrees, where the preprocessing land mask had false land strips over sea ice)",
}
KNOWN_FLAG_MASK = sum(1 << bit for bit in FLAG_BITS.values())


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    unit: str
    description: str


COLUMNS = (
    Column("period", "string", False, "",
           "Processing period (YYYYMM_YYYYMM). Trajectory identifiers are unique only within a period."),
    Column("trajectory_id", "int64", False, "",
           "Trajectory identifier, unique within the period. A trajectory is the track of one ice feature "
           "through consecutive Sentinel-1 scenes."),
    Column("point_index", "int32", False, "",
           "Position along the trajectory: 0 for its first observation, then 1, 2, ... The rows of one "
           "trajectory form a continuous chain in point_index order."),
    Column("time", "timestamp_ms_utc", False, "UTC",
           "Acquisition time of the Sentinel-1 scene of this observation."),
    Column("scene", "string", False, "",
           "Sentinel-1 product name of the scene this observation belongs to. For an interpolated position "
           "it is the scene the position was assigned to."),
    Column("x_m", "float64", False, "m", "Position, EPSG:3413 x."),
    Column("y_m", "float64", False, "m", "Position, EPSG:3413 y."),
    Column("dx_m", "float64", True, "m",
           "Displacement in x from the previous observation of the trajectory. Null at point_index 0."),
    Column("dy_m", "float64", True, "m",
           "Displacement in y from the previous observation of the trajectory. Null at point_index 0."),
    Column("speed_m_per_day", "float64", True, "m/day",
           "Distance from the previous observation divided by the elapsed time. Null at point_index 0."),
    Column("interpolated", "bool", True, "",
           "True if this position was interpolated rather than matched directly. Null at point_index 0."),
    Column("corr", "float32", True, "",
           "Template correlation of the match that produced this position (native tracker field, not a "
           "calibrated quality score). Null at point_index 0."),
    Column("direct_match_share", "float32", True, "",
           "Share (0 to 1) of the steps between this scene pair that are direct matches rather than "
           "interpolated. Null at point_index 0."),
    Column("flags", "uint8", True, "",
           "Bit mask describing the step arriving at this point: "
           + "; ".join(f"bit {bit} {name}" for name, bit in FLAG_BITS.items())
           + ". Flags never split trajectories. Null at point_index 0."),
    Column("primary", "bool", False, "",
           "True if the start time of the step arriving at this point (for point_index 0, the point's own "
           "time) lies in the period's primary window. Selecting primary rows from all periods gives each "
           "date exactly once; periods overlap by three months and the overlap rows are not primary in the "
           "later period."),
)
COLUMN_NAMES = tuple(column.name for column in COLUMNS)

# Columns read from the internal kept-link Parquet.
SOURCE_COLUMNS = (
    "qc_trajectory_id", "trajectory_link_index", "source_image_id", "target_image_id",
    "start_time_utc", "end_time_utc", "x0_m", "y0_m", "x1_m", "y1_m", "dx_m", "dy_m",
    "speed_m_per_day", "target_interpolated", "target_corr", "pair_direct_share", "qc_flags",
)


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError(
            "Publishing requires PyArrow. Install it with `conda install -c conda-forge pyarrow` "
            "or `pip install pyarrow`."
        ) from error
    return pa, pc, ds, pq


def _arrow_type(name: str):
    pa, _, _, _ = _pyarrow()
    return {
        "string": pa.string(), "int64": pa.int64(), "int32": pa.int32(), "float64": pa.float64(),
        "float32": pa.float32(), "bool": pa.bool_(), "uint8": pa.uint8(),
        "timestamp_ms_utc": pa.timestamp("ms", tz="UTC"),
    }[name]


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a closed file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------- windows


def ownership_window(period: str) -> tuple[datetime, datetime]:
    """Half-open UTC window [start, stop) of start times that the period owns (its primary window).

    Periods are 15 months (May to July of the following year) and overlap their neighbours by three
    months. Period ``YYYY05_YYYY+107`` owns link start times in [1 Aug YYYY, 1 Aug YYYY+1); the first
    period also owns its first three months. The windows tile May 2015 to July 2025.
    """
    match = PERIOD_PATTERN.fullmatch(period)
    if not match or int(match[2]) != int(match[1]) + 1:
        raise ValueError(f"Not a LiMOSAT processing period: {period!r}")
    year = int(match[1])
    start = datetime(year, 5 if year == FIRST_PERIOD_YEAR else 8, 1, tzinfo=timezone.utc)
    return start, datetime(year + 1, 8, 1, tzinfo=timezone.utc)


def _ms(value: datetime) -> int:
    return int(round(value.timestamp() * 1000))


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ms_to_datetime(value: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=int(value))


def _month_starts(first_ms: int, last_ms: int) -> list[datetime]:
    cursor = _ms_to_datetime(first_ms).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = _ms_to_datetime(last_ms)
    months = []
    while cursor <= last:
        months.append(cursor)
        cursor = datetime(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1, tzinfo=timezone.utc)
    return months


# --------------------------------------------------------------------------------------------- catalogue


@dataclass(frozen=True)
class Catalog:
    path: Path
    sha256: str
    image_ids: np.ndarray
    scenes: Any  # pyarrow string array, aligned with image_ids
    time_ms: np.ndarray


def _epoch_ms(text: str) -> int:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _ms(parsed)


def load_catalog(path: Path) -> Catalog:
    """Read a period's STAC image catalogue: image id, Sentinel-1 scene name and acquisition time."""
    pa, _, _, _ = _pyarrow()
    path = Path(path)
    features = json.loads(path.read_text())["features"]
    rows = sorted(
        (int(f["properties"]["image_id"]), str(f["properties"]["scene_id"]), _epoch_ms(f["properties"]["datetime"]))
        for f in features
    )
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    scenes = [row[1] for row in rows]
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Duplicate image ids in catalogue {path.name}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"Duplicate scene names in catalogue {path.name}")
    return Catalog(path, sha256(path), ids, pa.array(scenes, type=pa.string()),
                   np.array([row[2] for row in rows], dtype=np.int64))


def _lookup(catalog: Catalog, image_ids: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
    """Catalogue positions of image ids; checks that every id exists and its time matches the data."""
    position = np.searchsorted(catalog.image_ids, image_ids)
    safe = np.minimum(position, len(catalog.image_ids) - 1)
    missing = catalog.image_ids[safe] != image_ids
    if missing.any():
        raise ValueError(f"{int(missing.sum())} image ids are not in catalogue {catalog.path.name}")
    mismatch = catalog.time_ms[safe] != times_ms
    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} observations have a time different from their catalogue scene "
            f"({catalog.path.name})"
        )
    return safe


def find_catalog(catalog_dir: Path, period: str) -> Path:
    found = sorted(p for p in Path(catalog_dir).glob(f"{period}_image_catalog_*.json") if not p.name.startswith("._"))
    if len(found) != 1:
        raise ValueError(f"Expected one image catalogue for {period} in {catalog_dir}, found {[p.name for p in found]}")
    return found[0]


# --------------------------------------------------------------------------------------------- source


@dataclass(frozen=True)
class Source:
    period: str
    parquet: Path
    manifest_path: Path
    manifest: dict[str, Any]


def find_sources(source_root: Path, periods: list[str] | None = None) -> list[Source]:
    source_root = Path(source_root)
    names = periods or sorted(
        p.name for p in source_root.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{6}_\d{6}", p.name) and not p.name.startswith("._")
    )
    if not names:
        raise ValueError(f"No period folders under {source_root}")
    return [inspect_source(source_root / name / f"{name}_qc.parquet") for name in names]


def inspect_source(parquet: Path, verify_checksum: bool = False) -> Source:
    """Check an internal kept-link Parquet and its manifest."""
    _, _, _, pq = _pyarrow()
    parquet = Path(parquet)
    period = parquet.parent.name
    ownership_window(period)  # validates the name
    manifest_path = parquet.with_suffix(".manifest.json")
    if not parquet.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Need {parquet} and {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("continuity_violations") != 0:
        raise ValueError(f"{period}: source is not complete with zero continuity violations")
    if manifest.get("period") != period:
        raise ValueError(f"{period}: manifest period {manifest.get('period')!r} differs")
    schema = pq.ParquetFile(parquet).schema_arrow
    missing = set(SOURCE_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"{period}: source lacks columns {sorted(missing)}")
    rows = pq.ParquetFile(parquet).metadata.num_rows
    if manifest.get("kept_links") not in (None, rows):
        raise ValueError(f"{period}: manifest kept_links {manifest.get('kept_links')} != {rows} rows")
    if verify_checksum:
        expected = (manifest.get("outputs") or {}).get(parquet.name)
        if expected and sha256(parquet) != expected:
            raise ValueError(f"{period}: source Parquet checksum differs from its manifest")
    return Source(period, parquet, manifest_path, manifest)


# --------------------------------------------------------------------------------------------- schema


def public_schema(release_id: str | None = None, period: str | None = None):
    pa, _, _, _ = _pyarrow()
    fields = [
        pa.field(c.name, _arrow_type(c.type), nullable=c.nullable,
                 metadata={b"unit": c.unit.encode(), b"description": c.description.encode()})
        for c in COLUMNS
    ]
    metadata = {
        b"limosat.schema_version": SCHEMA_VERSION.encode(),
        b"limosat.crs": CRS.encode(),
        b"limosat.coordinate_units": b"metres",
        b"limosat.time_units": b"UTC timestamps (millisecond); speed in metres per day",
        b"limosat.flag_bits": json.dumps(FLAG_BITS).encode(),
        b"limosat.primary_policy": OWNERSHIP_POLICY.encode(),
    }
    if release_id:
        metadata[b"limosat.release_id"] = release_id.encode()
    if period:
        start, stop = ownership_window(period)
        metadata[b"limosat.period"] = period.encode()
        metadata[b"limosat.primary_window"] = json.dumps(
            {"start": _utc_text(start), "stop_exclusive": _utc_text(stop)}).encode()
    return pa.schema(fields, metadata=metadata)


def columns_spec() -> list[dict[str, Any]]:
    return [dict(name=c.name, type=c.type, nullable=c.nullable, unit=c.unit, description=c.description) for c in COLUMNS]


# --------------------------------------------------------------------------------------------- building


@dataclass
class _Context:
    period: str
    catalog: Catalog
    traj_ids: np.ndarray  # sorted trajectory ids of the source
    first_index: np.ndarray  # first trajectory_link_index of each trajectory
    window_ms: tuple[int, int]
    schema: Any
    used_scenes: np.ndarray


def _column(value, field):
    if value.type != field.type:
        value = value.cast(field.type, safe=False)
    return value


def _arrays(ctx: _Context, table, *, ends: bool):
    """Public columns for the end points (``ends``) or the start points of a table of source links."""
    pa, pc, _, _ = _pyarrow()
    n = table.num_rows
    for name in ("qc_trajectory_id", "trajectory_link_index", "source_image_id", "target_image_id",
                 "start_time_utc", "end_time_utc", "x0_m", "y0_m", "x1_m", "y1_m", "target_interpolated", "qc_flags"):
        if table[name].null_count:
            raise ValueError(f"{ctx.period}: nulls in source column {name}")
    pos = np.searchsorted(ctx.traj_ids, table["qc_trajectory_id"].to_numpy())
    first = ctx.first_index[pos]
    link = table["trajectory_link_index"].to_numpy().astype(np.int64)
    start_ms = table["start_time_utc"].cast(pa.int64()).to_numpy()
    end_ms = table["end_time_utc"].cast(pa.int64()).to_numpy()
    primary = (start_ms >= ctx.window_ms[0]) & (start_ms < ctx.window_ms[1])
    side = "target" if ends else "source"
    images = table[f"{side}_image_id"].to_numpy()
    when = end_ms if ends else start_ms
    scene_pos = _lookup(ctx.catalog, images, when)
    ctx.used_scenes[scene_pos] = True
    schema = ctx.schema
    values = {
        "period": pa.repeat(pa.scalar(ctx.period), n),
        "trajectory_id": table["qc_trajectory_id"],
        "point_index": pa.array((link - first + 1) if ends else np.zeros(n, dtype=np.int64)),
        "time": table["end_time_utc" if ends else "start_time_utc"],
        "scene": ctx.catalog.scenes.take(pa.array(scene_pos)),
        "x_m": table["x1_m" if ends else "x0_m"],
        "y_m": table["y1_m" if ends else "y0_m"],
        "primary": pa.array(primary),
    }
    if ends:
        flags = table["qc_flags"].to_numpy().astype(np.int64)
        if (flags & ~KNOWN_FLAG_MASK).any():
            raise ValueError(f"{ctx.period}: qc_flags carries bits other than {FLAG_BITS}")
        interpolated = table["target_interpolated"].to_numpy()
        if interpolated.size and not np.isin(interpolated, (0, 1)).all():
            raise ValueError(f"{ctx.period}: target_interpolated is not 0/1")
        values.update(
            dx_m=table["dx_m"], dy_m=table["dy_m"], speed_m_per_day=table["speed_m_per_day"],
            interpolated=pa.array(interpolated.astype(bool)), corr=table["target_corr"],
            direct_match_share=table["pair_direct_share"], flags=pa.array(flags.astype(np.uint8)),
        )
    else:
        for name in ("dx_m", "dy_m", "speed_m_per_day", "interpolated", "corr", "direct_match_share", "flags"):
            values[name] = pa.nulls(n, schema.field(name).type)
    columns = [_column(values[name], schema.field(name)) for name in COLUMN_NAMES]
    return pa.Table.from_arrays(columns, schema=pa.schema(list(schema)))


def _time_filter(field: str, lo: datetime, hi: datetime):
    pa, _, ds, _ = _pyarrow()
    kind = pa.timestamp("ms", tz="UTC")
    return (ds.field(field) >= pa.scalar(lo, type=kind)) & (ds.field(field) < pa.scalar(hi, type=kind))


def build_period(
    source: Source,
    catalog_path: Path,
    output: Path,
    release_id: str,
    overwrite: bool = False,
    rows_per_group: int = ROWS_PER_GROUP,
) -> dict[str, Any]:
    """Write one period's public Parquet. Streams month by month; returns the manifest entry."""
    pa, pc, ds, pq = _pyarrow()
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(catalog_path)
    window = ownership_window(source.period)
    window_ms = (_ms(window[0]), _ms(window[1]))

    # Pass 1: per-trajectory first link index (point_index offsets), counts and time bounds.
    head = pq.read_table(source.parquet, columns=["qc_trajectory_id", "trajectory_link_index", "start_time_utc", "end_time_utc"])
    for name in head.column_names:
        if head[name].null_count:
            raise ValueError(f"{source.period}: nulls in source column {name}")
    n_links = head.num_rows
    grouped = head.group_by("qc_trajectory_id").aggregate([
        ("trajectory_link_index", "min"), ("trajectory_link_index", "max"), ("trajectory_link_index", "count")
    ]).sort_by("qc_trajectory_id")
    traj_ids = grouped["qc_trajectory_id"].to_numpy()
    first_index = grouped["trajectory_link_index_min"].to_numpy().astype(np.int64)
    span = grouped["trajectory_link_index_max"].to_numpy().astype(np.int64) - first_index + 1
    if not np.array_equal(span, grouped["trajectory_link_index_count"].to_numpy().astype(np.int64)):
        raise ValueError(f"{source.period}: link indices of a trajectory are not consecutive")
    start_ms = head["start_time_utc"].cast(pa.int64()).to_numpy()
    end_ms = head["end_time_utc"].cast(pa.int64()).to_numpy()
    expected_primary_links = int(((start_ms >= window_ms[0]) & (start_ms < window_ms[1])).sum())
    first_ms, last_ms = int(start_ms.min()), int(end_ms.max())
    n_trajectories = len(traj_ids)
    del head, grouped, start_ms, end_ms

    ctx = _Context(source.period, catalog, traj_ids, first_index, window_ms, public_schema(release_id, source.period),
                   np.zeros(len(catalog.image_ids), dtype=bool))
    dataset = ds.dataset(source.parquet, format="parquet")
    rows = primary_rows = primary_links = end_rows = start_rows = 0
    time_min = time_max = None
    bounds = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    months = _month_starts(first_ms, last_ms)
    try:
        with pq.ParquetWriter(temporary, ctx.schema, compression="zstd") as writer:
            for lo in months:
                hi = datetime(lo.year + lo.month // 12, lo.month % 12 + 1, 1, tzinfo=timezone.utc)
                pieces = []
                ends = dataset.to_table(columns=list(SOURCE_COLUMNS), filter=_time_filter("end_time_utc", lo, hi))
                if ends.num_rows:
                    pieces.append(_arrays(ctx, ends, ends=True))
                    end_rows += ends.num_rows
                starts = dataset.to_table(columns=list(SOURCE_COLUMNS), filter=_time_filter("start_time_utc", lo, hi))
                if starts.num_rows:
                    pos = np.searchsorted(ctx.traj_ids, starts["qc_trajectory_id"].to_numpy())
                    is_first = starts["trajectory_link_index"].to_numpy().astype(np.int64) == ctx.first_index[pos]
                    starts = starts.filter(pa.array(is_first))
                    if starts.num_rows:
                        pieces.append(_arrays(ctx, starts, ends=False))
                        start_rows += starts.num_rows
                if not pieces:
                    continue
                chunk = pa.concat_tables(pieces).sort_by(
                    [("time", "ascending"), ("trajectory_id", "ascending"), ("point_index", "ascending")])
                writer.write_table(chunk, row_group_size=rows_per_group)
                rows += chunk.num_rows
                flags_primary = chunk["primary"].to_numpy(zero_copy_only=False)
                primary_rows += int(flags_primary.sum())
                primary_links += int((flags_primary & (chunk["point_index"].to_numpy() > 0)).sum())
                span_time = pc.min_max(chunk["time"])
                lo_t, hi_t = _ms(span_time["min"].as_py()), _ms(span_time["max"].as_py())
                time_min = lo_t if time_min is None else min(time_min, lo_t)
                time_max = hi_t if time_max is None else max(time_max, hi_t)
                for axis in ("x", "y"):
                    span_axis = pc.min_max(chunk[f"{axis}_m"])
                    low, high = span_axis["min"].as_py(), span_axis["max"].as_py()
                    bounds[f"{axis}_min"] = low if bounds[f"{axis}_min"] is None else min(bounds[f"{axis}_min"], low)
                    bounds[f"{axis}_max"] = high if bounds[f"{axis}_max"] is None else max(bounds[f"{axis}_max"], high)
                del chunk, pieces, ends, starts
        if end_rows != n_links or start_rows != n_trajectories or rows != n_links + n_trajectories:
            raise ValueError(
                f"{source.period}: wrote {rows} rows ({end_rows} steps, {start_rows} starts) but the source has "
                f"{n_links} links and {n_trajectories} trajectories"
            )
        if primary_links != expected_primary_links:
            raise ValueError(f"{source.period}: {primary_links} primary steps, expected {expected_primary_links}")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "period": source.period,
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "rows": rows,
        "trajectories": n_trajectories,
        "steps": n_links,
        "primary_rows": primary_rows,
        "primary_steps": primary_links,
        "scenes": int(ctx.used_scenes.sum()),
        "primary_window": {"start": _utc_text(window[0]), "stop_exclusive": _utc_text(window[1])},
        "time_start": _utc_text(_ms_to_datetime(time_min)),
        "time_end": _utc_text(_ms_to_datetime(time_max)),
        "bounds_epsg3413_m": bounds,
        "_provenance": {
            "source_parquet": str(source.parquet),
            "source_manifest": str(source.manifest_path),
            "source_manifest_sha256": sha256(source.manifest_path),
            "source_parquet_sha256_in_manifest": (source.manifest.get("outputs") or {}).get(source.parquet.name),
            "protocol_id": source.manifest.get("protocol_id"),
            "protocol_sha256": source.manifest.get("protocol_sha256"),
            "links_before_filtering": source.manifest.get("input_links"),
            "catalog": str(catalog.path),
            "catalog_sha256": catalog.sha256,
        },
    }


def data_dictionary() -> str:
    """Markdown data dictionary generated from the schema constants."""
    lines = [
        f"# LiMOSAT trajectory dataset: data dictionary (schema {SCHEMA_VERSION})",
        "",
        "Generated by `limosat.publish` from the schema definition. One Parquet file per processing period.",
        "",
        "## What a row is",
        "",
        "A row is one **observation** of an ice feature in one Sentinel-1 scene. The observations of a trajectory",
        "share `trajectory_id` and are ordered by `point_index`; consecutive rows are linked by the step that",
        "arrived at the later one (`dx_m`, `dy_m`, `speed_m_per_day` and the other step columns describe the step",
        "from the previous row of the same trajectory, and are null on its first row). Rows are sorted by `time`.",
        "Trajectory identifiers are unique only within a period; use (`period`, `trajectory_id`).",
        "",
        f"Coordinates are {CRS} metres. Times are UTC.",
        "",
        "## Columns",
        "",
        "| Column | Type | Unit | Nullable | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in COLUMNS:
        lines.append(f"| `{c.name}` | {c.type} | {c.unit} | {'yes' if c.nullable else 'no'} | {c.description} |")
    lines += ["", "## Flags", "", "| Bit | Name | Meaning |", "| --- | --- | --- |"]
    for name, bit in FLAG_BITS.items():
        lines.append(f"| {bit} | `{name}` | {FLAG_DESCRIPTIONS[name]} |")
    lines += [
        "",
        "## Overlapping periods and `primary`",
        "",
        "Processing periods are 15 months long and overlap their neighbours by three months (May to July). All rows",
        "of every period are published. The rule `" + OWNERSHIP_POLICY + "` assigns each step to one period by its",
        "start time: a period owns start times from 1 August of its first year up to (not including) 1 August of",
        "the next year, and the first period also owns its first three months. `primary` is true on rows that",
        "belong to that window. To use each date once, keep `primary` rows of all periods. Do not combine non-primary",
        "rows with primary rows of the neighbouring period: they are independent processings of the same dates.",
        "",
        "Windows (start inclusive, stop exclusive) are in `manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def build(
    source_root: Path,
    catalog_dir: Path,
    output_dir: Path,
    release_id: str,
    periods: list[str] | None = None,
    overwrite: bool = False,
    verify_source: bool = True,
) -> dict[str, Any]:
    """Publish every (or the selected) period of an internal release into a flat output directory."""
    source_root, catalog_dir, output_dir = Path(source_root), Path(catalog_dir), Path(output_dir)
    sources = find_sources(source_root, periods)
    if verify_source:
        sources = [inspect_source(s.parquet, verify_checksum=True) for s in sources]
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sources:
        output = output_dir / f"limosat_{source.period}.parquet"
        entries.append(build_period(source, find_catalog(catalog_dir, source.period), output, release_id, overwrite))
    provenance = {entry["period"]: entry.pop("_provenance") for entry in entries}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "crs": CRS,
        "primary": {"policy_id": OWNERSHIP_POLICY,
                    "definition": "start time of the arriving step (own time for point_index 0) in the period's "
                    "half-open window"},
        "flag_bits": FLAG_BITS,
        "columns": columns_spec(),
        "files": entries,
        "totals": {key: sum(e[key] for e in entries) for key in ("rows", "trajectories", "steps", "primary_rows", "primary_steps")},
    }
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _atomic_write(output_dir / "DATA_DICTIONARY.md", data_dictionary())
    _atomic_write(output_dir / "checksums.sha256", "".join(f"{e['sha256']}  {e['file']}\n" for e in entries))
    sidecar = output_dir.with_name(output_dir.name + ".provenance.json")
    _atomic_write(sidecar, json.dumps({"release_id": release_id, "source_root": str(source_root), "periods": provenance},
                                      indent=2, sort_keys=True) + "\n")
    return manifest


# --------------------------------------------------------------------------------------------- validation


def _read_sample(path: Path, columns: list[str], key: str, modulus: int):
    """Rows whose trajectory key is divisible by ``modulus``, read row group by row group."""
    pa, _, _, pq = _pyarrow()
    parquet = pq.ParquetFile(path)
    pieces = []
    for index in range(parquet.num_row_groups):
        group = parquet.read_row_group(index, columns=columns)
        mask = (group[key].to_numpy() % modulus) == 0
        if mask.any():
            pieces.append(group.filter(pa.array(mask)))
    return pa.concat_tables(pieces) if pieces else parquet.schema_arrow.empty_table().select(columns)


def validate_period(source: Source, catalog_path: Path, output: Path, entry: dict[str, Any], release_id: str,
                    sample_modulus: int = 50) -> dict[str, Any]:
    """Check a published file against the manifest entry and, on a sample of trajectories, against the source."""
    pa, pc, _, pq = _pyarrow()
    output = Path(output)
    if sha256(output) != entry["sha256"]:
        raise ValueError(f"{output.name}: checksum differs from the manifest")
    parquet = pq.ParquetFile(output)
    expected = public_schema(release_id, source.period)
    if [(f.name, f.type, f.nullable) for f in parquet.schema_arrow] != [(f.name, f.type, f.nullable) for f in expected]:
        raise ValueError(f"{output.name}: columns differ from the schema")
    metadata = parquet.schema_arrow.metadata or {}
    for key, value in (expected.metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"{output.name}: schema metadata {key.decode()} differs")
    if parquet.metadata.num_rows != entry["rows"]:
        raise ValueError(f"{output.name}: {parquet.metadata.num_rows} rows, manifest says {entry['rows']}")
    core = pq.read_table(output, columns=["trajectory_id", "point_index", "time", "primary"])
    time = core["time"].cast(pa.int64()).to_numpy()
    if (np.diff(time) < 0).any():
        raise ValueError(f"{output.name}: rows are not sorted by time")
    per = core.group_by("trajectory_id").aggregate([("point_index", "min"), ("point_index", "max"), ("point_index", "count")])
    if (per["point_index_min"].to_numpy() != 0).any() or \
            (per["point_index_max"].to_numpy().astype(np.int64) + 1 != per["point_index_count"].to_numpy()).any():
        raise ValueError(f"{output.name}: point_index is not 0..n-1 within every trajectory")
    if per.num_rows != entry["trajectories"]:
        raise ValueError(f"{output.name}: {per.num_rows} trajectories, manifest says {entry['trajectories']}")
    primary = core["primary"].to_numpy(zero_copy_only=False)
    if int(primary.sum()) != entry["primary_rows"] or int((primary & (core["point_index"].to_numpy() > 0)).sum()) != entry["primary_steps"]:
        raise ValueError(f"{output.name}: primary counts differ from the manifest")
    del core, per, time

    # Source cross-check on a sample of trajectories: every published step must equal its source link.
    catalog = load_catalog(catalog_path)
    window = ownership_window(source.period)
    w0, w1 = _ms(window[0]), _ms(window[1])
    src = _read_sample(source.parquet, list(SOURCE_COLUMNS), "qc_trajectory_id", sample_modulus)
    pub = _read_sample(output, list(COLUMN_NAMES), "trajectory_id", sample_modulus)

    def col(table, name):
        return table[name].to_numpy(zero_copy_only=False)

    s_order = np.lexsort((col(src, "trajectory_link_index"), col(src, "qc_trajectory_id")))
    p_order = np.lexsort((col(pub, "point_index"), col(pub, "trajectory_id")))
    src, pub = src.take(pa.array(s_order)), pub.take(pa.array(p_order))
    if pub.num_rows != src.num_rows + len(set(col(src, "qc_trajectory_id").tolist())):
        raise ValueError(f"{output.name}: sampled row count is not links + trajectories")
    p_idx = col(pub, "point_index")
    is_step = p_idx > 0
    step = pub.filter(pa.array(is_step))
    first = pub.filter(pa.array(~is_step))
    start_ms = src["start_time_utc"].cast(pa.int64()).to_numpy()
    end_ms = src["end_time_utc"].cast(pa.int64()).to_numpy()

    def same(a, b, name):
        if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind == "f"):
            raise ValueError(f"{output.name}: sampled {name} differ from the source")

    same(col(step, "trajectory_id"), col(src, "qc_trajectory_id"), "trajectory ids")
    same(step["time"].cast(pa.int64()).to_numpy(), end_ms, "end times")
    same(col(step, "x_m"), col(src, "x1_m"), "x"); same(col(step, "y_m"), col(src, "y1_m"), "y")
    same(col(step, "dx_m"), col(src, "dx_m"), "dx"); same(col(step, "dy_m"), col(src, "dy_m"), "dy")
    same(col(step, "speed_m_per_day"), col(src, "speed_m_per_day"), "speed")
    same(col(step, "interpolated"), col(src, "target_interpolated").astype(bool), "interpolated")
    same(col(step, "corr"), col(src, "target_corr").astype(np.float32), "corr")
    same(col(step, "direct_match_share"), col(src, "pair_direct_share").astype(np.float32), "direct_match_share")
    same(col(step, "flags"), col(src, "qc_flags").astype(np.uint8), "flags")
    same(col(step, "primary"), (start_ms >= w0) & (start_ms < w1), "primary flags")
    position = np.searchsorted(catalog.image_ids, col(src, "target_image_id"))
    same(col(step, "scene"), np.asarray(catalog.scenes.take(pa.array(position)).to_pylist(), dtype=object), "scenes")
    # chain: the point before each step is the step's source (position and time)
    previous = pub.take(pa.array(np.flatnonzero(is_step) - 1))
    same(col(previous, "x_m"), col(src, "x0_m"), "chain x"); same(col(previous, "y_m"), col(src, "y0_m"), "chain y")
    same(previous["time"].cast(pa.int64()).to_numpy(), start_ms, "chain times")
    same(col(previous, "trajectory_id"), col(src, "qc_trajectory_id"), "chain trajectory")
    # first rows have no step values
    for name in ("dx_m", "dy_m", "speed_m_per_day", "interpolated", "corr", "direct_match_share", "flags"):
        if first[name].null_count != first.num_rows:
            raise ValueError(f"{output.name}: first observations carry values in {name}")
    return {"period": source.period, "status": "passed", "rows": entry["rows"], "sampled_steps": int(is_step.sum()),
            "sample_modulus": sample_modulus}


def validate(source_root: Path, catalog_dir: Path, output_dir: Path, sample_modulus: int = 50) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    results = []
    for entry in manifest["files"]:
        period = entry["period"]
        source = inspect_source(Path(source_root) / period / f"{period}_qc.parquet")
        results.append(validate_period(source, find_catalog(catalog_dir, period), output_dir / entry["file"],
                                       entry, manifest["release_id"], sample_modulus))
    return results


# --------------------------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--source-root", type=Path, required=True,
                             help="internal release folder holding <period>/<period>_qc.parquet")
        command.add_argument("--catalog-dir", type=Path, required=True, help="folder of <period>_image_catalog_*.json")
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "build":
            command.add_argument("--release-id", required=True)
            command.add_argument("--period", action="append", dest="periods")
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--skip-source-checksum", action="store_true",
                                 help="do not hash the source Parquet files against their manifests")
        else:
            command.add_argument("--sample-modulus", type=int, default=50,
                                 help="check every Nth trajectory against the source (1 checks all)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build(args.source_root, args.catalog_dir, args.output_dir, args.release_id, args.periods,
                       args.overwrite, not args.skip_source_checksum)
    else:
        result = validate(args.source_root, args.catalog_dir, args.output_dir, args.sample_modulus)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
