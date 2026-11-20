"""Build the public LiMOSAT trajectory dataset (schema 2) as CF NetCDF4, one file per processing period.

The internal outputs of each processing period (kept-link Parquet and manifest, removed links, SQLite) are built for our
own analysis and checking. This module turns the internal kept-link Parquet and the period's image catalogue into a format
fit for public use and leaves out what an ordinary data user does not need: internal ids and row numbers, quality-control
bookkeeping, removed links, raw points and local file paths. It never re-scores or alters a link.

The released files are self-describing CF-1.8 *trajectory* files in the contiguous ragged array representation (CF 9.3.3):

* dimensions ``trajectory`` (one entry per trajectory), ``obs`` (one per observation) and ``scene`` (Sentinel-1 scene names);
* ``rowsize(trajectory)`` carries ``sample_dimension = "obs"`` and gives the number of observations of each trajectory, so
  trajectory *i* is the slice ``offset[i]:offset[i] + rowsize[i]`` of every ``obs`` variable;
* trajectories are ordered by start time (``time_start``) and the observations of a trajectory by ``point_index``, so a date
  window is a few contiguous runs and one trajectory is a single slice;
* a row of the underlying point table is one observation of a feature in one scene, carrying the displacement from the
  previous observation (dx, dy, speed, interpolation status, correlation, pair statistics and flags) and an ``is_primary``
  flag that marks the observations belonging to the period's own date window (selecting them from all periods covers each
  date once); every column is its own variable, ``lon`` and ``lat`` (CF requires them) are added next to the EPSG:3413
  ``x_m``/``y_m`` the tracker works in, and the scene names are a lookup table (``scene_name``) indexed by ``scene_index``.

No value is rounded or packed: floating point variables are float64/float32 exactly as in the internal files, times are
float64 seconds, and nulls become the variable's ``_FillValue`` (NaN, or -1 / 255 for the integer flags). Compression is
lossless (zlib with shuffle). The public schema is defined once, here (``COLUMNS``, ``FLAG_BITS``). ``build`` writes the
files and ``validate`` re-reads them and compares every Nth trajectory with the source links, value by value.

Public outputs carry no local paths and no internal protocol detail. Internal provenance (source paths, checksums, protocol
ids) goes to a sidecar next to, not inside, the output directory.

    python -m limosat.publish build --source-root ... --catalog-dir ... --output-dir ... --release-id limosat_v1 --metadata m.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "2.0.0"
CRS = "EPSG:3413"
OWNERSHIP_POLICY = "limosat_release_august_annual_v1"
FIRST_PERIOD_YEAR = 2015  # the first period also owns its first three months (from 1 May 2015)
PERIOD_PATTERN = re.compile(r"(\d{4})05_(\d{4})07")

FLAG_BITS = {"FEW_NEIGHBOURS": 0, "LOW_PAIR_DIRECT_SHARE": 1, "MASK_FALSE_LAND": 2}
FLAG_DESCRIPTIONS = {
    "FEW_NEIGHBOURS": "fewer than 2 other displacements between the same two scenes within 30 km",
    "LOW_PAIR_DIRECT_SHARE": "less than 40% of the displacements between the same two scenes are direct matches "
    "(interpolated displacements in such pairs are more often wrong; use direct_match_share for other thresholds)",
    "MASK_FALSE_LAND": "either end of the displacement lies in false land of its scene's land mask "
    "(scenes crossing 180 degrees longitude, where the mask wrongly marks strips of sea ice as land)",
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
           "Trajectory identifier, unique within the period. A trajectory is the track of one feature "
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
    Column("correlation", "float32", True, "",
           "Correlation of the template match that produced this position (the tracker's own value, not a "
           "calibrated quality score). Null at point_index 0."),
    Column("direct_match_share", "float32", True, "",
           "Share (0 to 1) of the displacements between the two scenes of this displacement that are direct matches rather than "
           "interpolated. Null at point_index 0."),
    Column("quality_flags", "uint8", True, "",
           "Bit mask describing the displacement arriving at this observation: "
           + "; ".join(f"bit {bit} {name}" for name, bit in FLAG_BITS.items())
           + ". Flags never split trajectories. Null at point_index 0."),
    Column("is_primary", "bool", False, "",
           "True if the start time of the displacement arriving at this observation (for point_index 0, its own "
           "time) lies in the period's primary window. Selecting rows with is_primary true from all periods gives each "
           "date exactly once; periods overlap by three months and the overlap rows are not marked is_primary in the "
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


def _epoch_ms_array(column) -> np.ndarray:
    """Epoch milliseconds of a source timestamp column of any unit (the internal QC Parquet is ms or us depending on
    the writer's library versions); fails rather than silently truncating sub-millisecond values."""
    pa, _, _, _ = _pyarrow()
    return column.cast(pa.timestamp("ms", tz="UTC")).cast(pa.int64()).to_numpy()


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



def columns_spec() -> list[dict[str, Any]]:
    return [dict(name=c.name, type=c.type, nullable=c.nullable, unit=c.unit, description=c.description) for c in COLUMNS]


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


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


def _netcdf4():
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError("Writing NetCDF requires netCDF4 (`conda install -c conda-forge netcdf4`).") from error
    return netCDF4


def _pyproj():
    try:
        import pyproj
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError("Computing longitude and latitude requires pyproj (`conda install -c conda-forge pyproj`).") from error
    return pyproj


FORMAT = "NetCDF-4 (HDF5), CF-1.8 trajectory featureType, contiguous ragged array"
CHUNK = 65_536
COMPRESSION_LEVEL = 4
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
DEFAULT_METADATA: dict[str, Any] = {
    "title": "LiMOSAT: Lagrangian sea-ice trajectories from Sentinel-1",
    "summary": ("Trajectories of features on the Arctic sea ice tracked through consecutive Sentinel-1 EW scenes with "
                "the LiMOSAT algorithm. One observation per feature and scene; the displacement from the previous "
                "observation of the trajectory is stored with it."),
    "keywords": "sea ice drift, Lagrangian trajectories, Sentinel-1, SAR, Arctic",
    "institution": "Nansen Environmental and Remote Sensing Centre",
    "source": "Copernicus Sentinel-1 EW GRD scenes, processed with LiMOSAT (https://github.com/nansencenter/limosat)",
}
FLAG_MASKS = np.array([1 << bit for bit in FLAG_BITS.values()], dtype=np.uint8)
SENTINEL_FILL = {"flags": 255, "interpolated": -1}


# --------------------------------------------------------------------------------------------- building


@dataclass
class _Context:
    period: str
    catalog: Catalog
    traj_ids: np.ndarray  # sorted trajectory ids of the source
    first_index: np.ndarray  # first trajectory_link_index of each trajectory
    window_ms: tuple[int, int]
    used_scenes: np.ndarray


def _floats(column, dtype) -> np.ndarray:
    """A float column as numpy with nulls as NaN."""
    if column.null_count:
        column = column.fill_null(float("nan"))
    return column.to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _chunk(ctx: _Context, table, *, ends: bool) -> dict[str, np.ndarray]:
    """Observation arrays for the end points (``ends``) or the start points of a table of source links."""
    n = table.num_rows
    for name in ("qc_trajectory_id", "trajectory_link_index", "source_image_id", "target_image_id",
                 "start_time_utc", "end_time_utc", "x0_m", "y0_m", "x1_m", "y1_m", "target_interpolated", "qc_flags"):
        if table[name].null_count:
            raise ValueError(f"{ctx.period}: nulls in source column {name}")
    pos = np.searchsorted(ctx.traj_ids, table["qc_trajectory_id"].to_numpy())
    first = ctx.first_index[pos]
    link = table["trajectory_link_index"].to_numpy().astype(np.int64)
    start_ms = _epoch_ms_array(table["start_time_utc"])
    end_ms = _epoch_ms_array(table["end_time_utc"])
    primary = (start_ms >= ctx.window_ms[0]) & (start_ms < ctx.window_ms[1])
    side = "target" if ends else "source"
    when = end_ms if ends else start_ms
    scene_pos = _lookup(ctx.catalog, table[f"{side}_image_id"].to_numpy(), when)
    ctx.used_scenes[scene_pos] = True
    out = {
        "tid": table["qc_trajectory_id"].to_numpy().astype(np.int64),
        "pidx": ((link - first + 1) if ends else np.zeros(n, dtype=np.int64)).astype(np.int32),
        "ms": when,
        "x": table["x1_m" if ends else "x0_m"].to_numpy().astype(np.float64),
        "y": table["y1_m" if ends else "y0_m"].to_numpy().astype(np.float64),
        "primary": primary.astype(np.int8),
        "scene": scene_pos.astype(np.int32),
    }
    if ends:
        flags = table["qc_flags"].to_numpy().astype(np.int64)
        if (flags & ~KNOWN_FLAG_MASK).any():
            raise ValueError(f"{ctx.period}: qc_flags carries bits other than {FLAG_BITS}")
        interpolated = table["target_interpolated"].to_numpy()
        if interpolated.size and not np.isin(interpolated, (0, 1)).all():
            raise ValueError(f"{ctx.period}: target_interpolated is not 0/1")
        out.update(dx=_floats(table["dx_m"], np.float64), dy=_floats(table["dy_m"], np.float64),
                   speed=_floats(table["speed_m_per_day"], np.float64), interp=interpolated.astype(np.int8),
                   corr=_floats(table["target_corr"], np.float32), share=_floats(table["pair_direct_share"], np.float32),
                   flags=flags.astype(np.uint8))
    else:
        nan64, nan32 = np.full(n, np.nan), np.full(n, np.nan, dtype=np.float32)
        out.update(dx=nan64, dy=nan64.copy(), speed=nan64.copy(), interp=np.full(n, SENTINEL_FILL["interpolated"], dtype=np.int8),
                   corr=nan32, share=nan32.copy(), flags=np.full(n, SENTINEL_FILL["flags"], dtype=np.uint8))
    return out


def _time_filter(field: str, lo: datetime, hi: datetime):
    pa, _, ds, _ = _pyarrow()
    kind = pa.timestamp("ms", tz="UTC")
    return (ds.field(field) >= pa.scalar(lo, type=kind)) & (ds.field(field) < pa.scalar(hi, type=kind))


def _lonlat(x: np.ndarray, y: np.ndarray, block: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    transformer = _pyproj().Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = np.empty(len(x)), np.empty(len(x))
    for start in range(0, len(x), block):
        stop = start + block
        lon[start:stop], lat[start:stop] = transformer.transform(x[start:stop], y[start:stop])
    return lon, lat


def _trajectory_order(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Row order of the file: trajectories by start time (then id), the observations of a trajectory by point_index."""
    tid, pidx, ms = a["tid"], a["pidx"], a["ms"]
    unique, inverse = np.unique(tid, return_inverse=True)
    first = pidx == 0
    if int(first.sum()) != len(unique):
        raise ValueError("every trajectory needs exactly one observation with point_index 0")
    start = np.empty(len(unique), dtype=np.int64)
    start[inverse[first]] = ms[first]
    order = np.lexsort((pidx, unique[inverse], start[inverse]))
    del inverse, start
    tid_o, pidx_o, ms_o = tid[order], pidx[order], ms[order]
    starts = np.r_[0, np.flatnonzero(np.diff(tid_o)) + 1]
    sizes = np.diff(np.r_[starts, len(tid_o)]).astype(np.int32)
    if len(starts) != len(unique) or np.any(pidx_o != np.arange(len(pidx_o)) - np.repeat(starts, sizes)):
        raise ValueError("point_index is not 0, 1, 2, ... within every trajectory")
    return dict(order=order, tid=tid_o, ms=ms_o, starts=starts, rowsize=sizes)


def _crs_variable(ds) -> None:
    crs = ds.createVariable("crs", "i4")
    crs.grid_mapping_name = "polar_stereographic"
    crs.straight_vertical_longitude_from_pole = -45.0
    crs.latitude_of_projection_origin = 90.0
    crs.standard_parallel = 70.0
    crs.false_easting = 0.0
    crs.false_northing = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563
    crs.long_name = "NSIDC Sea Ice Polar Stereographic North (EPSG:3413)"
    crs.epsg_code = CRS
    crs.crs_wkt = _pyproj().CRS.from_user_input(CRS).to_wkt()


def _write_netcdf(path: Path, a: dict[str, np.ndarray], o: dict[str, np.ndarray], scene_names: list[str], scene_index: np.ndarray,
                  period: str, release_id: str, window: tuple[datetime, datetime], metadata: dict[str, Any],
                  counts: dict[str, int]) -> None:
    netCDF4 = _netcdf4()
    describe = {c.name: c for c in COLUMNS}
    order, n, ntraj = o["order"], len(o["order"]), len(o["starts"])
    kw = dict(zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True, chunksizes=(min(CHUNK, n),))
    kw_traj = dict(zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True, chunksizes=(min(CHUNK, ntraj),))
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("trajectory", ntraj)
        ds.createDimension("obs", n)
        ds.createDimension("scene", len(scene_names))
        # ---- trajectory variables
        v = ds.createVariable("trajectory_id", "i8", ("trajectory",), **kw_traj)
        v.cf_role = "trajectory_id"
        v.long_name = "trajectory identifier, unique within the period"
        v[:] = o["tid"][o["starts"]]
        v = ds.createVariable("rowsize", "i4", ("trajectory",), **kw_traj)
        v.long_name = "number of observations in the trajectory"
        v.sample_dimension = "obs"
        v[:] = o["rowsize"]
        ends = np.r_[o["starts"][1:], n] - 1
        for name, idx, text in (("time_start", o["starts"], "time of the first observation of the trajectory"),
                                ("time_end", ends, "time of the last observation of the trajectory")):
            v = ds.createVariable(name, "f8", ("trajectory",), **kw_traj)
            v.long_name, v.units, v.calendar = text, TIME_UNITS, "standard"
            v[:] = o["ms"][idx] / 1000.0
        # ---- observation coordinates
        coords = "time lat lon"
        time = ds.createVariable("time", "f8", ("obs",), **kw)
        time.standard_name, time.long_name = "time", describe["time"].description
        time.units, time.calendar, time.axis = TIME_UNITS, "standard", "T"
        time[:] = o["ms"] / 1000.0
        x, y = a["x"][order], a["y"][order]
        lon, lat = _lonlat(x, y)
        for name, values, std, units, axis in (("lat", lat, "latitude", "degrees_north", "Y"),
                                               ("lon", lon, "longitude", "degrees_east", "X")):
            v = ds.createVariable(name, "f8", ("obs",), **kw)
            v.standard_name, v.long_name, v.units, v.axis = std, f"{std} of the observation (WGS 84)", units, axis
            v[:] = values
        _crs_variable(ds)
        for name, values, std in (("x_m", x, "projection_x_coordinate"), ("y_m", y, "projection_y_coordinate")):
            v = ds.createVariable(name, "f8", ("obs",), **kw)
            v.standard_name, v.long_name, v.units = std, describe[name].description, "m"
            v.grid_mapping = "crs"
            v.coordinates = coords
            v[:] = values
        # ---- observation data variables
        v = ds.createVariable("point_index", "i4", ("obs",), **kw)
        v.long_name, v.coordinates = describe["point_index"].description, coords
        v[:] = a["pidx"][order]
        for name, key, units in (("dx_m", "dx", "m"), ("dy_m", "dy", "m"), ("speed_m_per_day", "speed", "m day-1")):
            v = ds.createVariable(name, "f8", ("obs",), fill_value=np.nan, **kw)
            v.long_name, v.units, v.coordinates = describe[name].description, units, coords
            v[:] = a[key][order]
        for name, key in (("correlation", "corr"), ("direct_match_share", "share")):
            v = ds.createVariable(name, "f4", ("obs",), fill_value=np.float32(np.nan), **kw)
            v.long_name, v.units, v.coordinates = describe[name].description, "1", coords
            v[:] = a[key][order]
        v = ds.createVariable("interpolated", "i1", ("obs",), fill_value=np.int8(SENTINEL_FILL["interpolated"]), **kw)
        v.long_name = describe["interpolated"].description + " (-1 where null)"
        v.flag_values = np.array([0, 1], dtype=np.int8)
        v.flag_meanings = "matched interpolated"
        v.coordinates = coords
        v[:] = a["interp"][order]
        v = ds.createVariable("quality_flags", "u1", ("obs",), fill_value=np.uint8(SENTINEL_FILL["flags"]), **kw)
        v.long_name = describe["quality_flags"].description
        v.flag_masks = FLAG_MASKS
        v.flag_meanings = " ".join(FLAG_BITS)
        v.comment = "; ".join(f"{name}: {text}" for name, text in FLAG_DESCRIPTIONS.items())
        v.coordinates = coords
        v[:] = a["flags"][order]
        v = ds.createVariable("is_primary", "i1", ("obs",), **kw)
        v.long_name = describe["is_primary"].description
        v.flag_values = np.array([0, 1], dtype=np.int8)
        v.flag_meanings = "not_primary primary"
        v.coordinates = coords
        v[:] = a["primary"][order]
        v = ds.createVariable("scene_index", "i4", ("obs",), **kw)
        v.long_name = "index of the Sentinel-1 scene of the observation into scene_name"
        v.coordinates = coords
        v[:] = scene_index
        v = ds.createVariable("scene_name", str, ("scene",))
        v.long_name = describe["scene"].description
        v[:] = np.array(scene_names, dtype=object)

        # ---- global attributes
        meta = {**DEFAULT_METADATA, **metadata}
        creators = meta.pop("creators", [])
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.featureType = "trajectory"
        ds.title = f"{meta.pop('title')} ({period})"
        ds.summary = meta.pop("summary")
        ds.keywords = meta.pop("keywords")
        ds.institution = meta.pop("institution")
        ds.source = meta.pop("source")
        if creators:
            ds.creator_name = "; ".join(c["name"] for c in creators)
            ds.creator_institution = "; ".join(dict.fromkeys(c.get("affiliation", "") for c in creators if c.get("affiliation")))
            ds.creator_id = "; ".join(f"https://orcid.org/{c['orcid']}" for c in creators if c.get("orcid"))
        for key, value in meta.items():
            ds.setncattr(key, value)
        ds.product_version = release_id
        ds.release_id = release_id
        ds.schema_version = SCHEMA_VERSION
        ds.period = period
        ds.primary_window_start = _utc_text(window[0])
        ds.primary_window_stop_exclusive = _utc_text(window[1])
        ds.primary_rule = OWNERSHIP_POLICY
        ds.time_coverage_start = _utc_text(_ms_to_datetime(int(o["ms"].min())))
        ds.time_coverage_end = _utc_text(_ms_to_datetime(int(o["ms"].max())))
        ds.geospatial_lat_min, ds.geospatial_lat_max = float(lat.min()), float(lat.max())
        ds.geospatial_lon_min, ds.geospatial_lon_max = float(lon.min()), float(lon.max())
        ds.observations, ds.trajectories = np.int64(n), np.int64(ntraj)
        ds.displacements = np.int64(counts["displacements"])
        ds.primary_displacements = np.int64(counts["primary_displacements"])
        ds.history = f"{datetime.now(timezone.utc):%Y-%m-%d}: written by limosat.publish from the internal kept-link table, release {release_id}"
        ds.comment = ("Contiguous ragged array (CF 9.3.3): trajectory i is obs[offset_i : offset_i + rowsize_i] with offset = "
                      "cumsum(rowsize) shifted by one. Trajectories are ordered by time_start. Scene names: "
                      "scene_name[scene_index]. Null displacement values (first observation of a trajectory) are _FillValue.")
    finally:
        ds.close()


def build_period(source: Source, catalog_path: Path, output: Path, release_id: str, metadata: dict[str, Any] | None = None,
                 overwrite: bool = False) -> dict[str, Any]:
    """Write one period's NetCDF file. Reads the source month by month; returns the manifest entry."""
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
    start_ms = _epoch_ms_array(head["start_time_utc"])
    end_ms = _epoch_ms_array(head["end_time_utc"])
    expected_primary_links = int(((start_ms >= window_ms[0]) & (start_ms < window_ms[1])).sum())
    first_ms, last_ms = int(start_ms.min()), int(end_ms.max())
    n_trajectories = len(traj_ids)
    del head, grouped, start_ms, end_ms

    ctx = _Context(source.period, catalog, traj_ids, first_index, window_ms, np.zeros(len(catalog.image_ids), dtype=bool))
    dataset = ds.dataset(source.parquet, format="parquet")
    pieces: list[dict[str, np.ndarray]] = []
    end_rows = start_rows = 0
    for lo in _month_starts(first_ms, last_ms):
        hi = datetime(lo.year + lo.month // 12, lo.month % 12 + 1, 1, tzinfo=timezone.utc)
        ends = dataset.to_table(columns=list(SOURCE_COLUMNS), filter=_time_filter("end_time_utc", lo, hi))
        if ends.num_rows:
            pieces.append(_chunk(ctx, ends, ends=True))
            end_rows += ends.num_rows
        starts = dataset.to_table(columns=list(SOURCE_COLUMNS), filter=_time_filter("start_time_utc", lo, hi))
        if starts.num_rows:
            pos = np.searchsorted(ctx.traj_ids, starts["qc_trajectory_id"].to_numpy())
            is_first = starts["trajectory_link_index"].to_numpy().astype(np.int64) == ctx.first_index[pos]
            starts = starts.filter(pa.array(is_first))
            if starts.num_rows:
                pieces.append(_chunk(ctx, starts, ends=False))
                start_rows += starts.num_rows
        del ends, starts
    arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    del pieces
    rows = len(arrays["tid"])
    if end_rows != n_links or start_rows != n_trajectories or rows != n_links + n_trajectories:
        raise ValueError(
            f"{source.period}: built {rows} rows ({end_rows} displacements, {start_rows} first observations) but the source has "
            f"{n_links} links and {n_trajectories} trajectories"
        )
    primary_rows = int(arrays["primary"].sum())
    primary_links = int((arrays["primary"].astype(bool) & (arrays["pidx"] > 0)).sum())
    if primary_links != expected_primary_links:
        raise ValueError(f"{source.period}: {primary_links} primary displacements, expected {expected_primary_links}")
    bounds = {"x_min": float(arrays["x"].min()), "x_max": float(arrays["x"].max()),
              "y_min": float(arrays["y"].min()), "y_max": float(arrays["y"].max())}
    time_min, time_max = int(arrays["ms"].min()), int(arrays["ms"].max())

    o = _trajectory_order(arrays)
    used = np.flatnonzero(ctx.used_scenes)
    scene_names = catalog.scenes.take(pa.array(used)).to_pylist()
    scene_index = np.searchsorted(used, arrays["scene"][o["order"]]).astype(np.int32)
    try:
        _write_netcdf(temporary, arrays, o, scene_names, scene_index, source.period, release_id, window, metadata or {},
                      {"displacements": n_links, "primary_displacements": primary_links})
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
        "displacements": n_links,
        "primary_rows": primary_rows,
        "primary_displacements": primary_links,
        "scenes": len(scene_names),
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


def build(
    source_root: Path,
    catalog_dir: Path,
    output_dir: Path,
    release_id: str,
    periods: list[str] | None = None,
    overwrite: bool = False,
    verify_source: bool = True,
    metadata: dict[str, Any] | None = None,
    validate_output: bool = True,
    sample_modulus: int = 50,
) -> dict[str, Any]:
    """Publish every (or the selected) period of an internal release into a flat output directory and validate it."""
    source_root, catalog_dir, output_dir = Path(source_root), Path(catalog_dir), Path(output_dir)
    sources = find_sources(source_root, periods)
    if verify_source:
        sources = [inspect_source(s.parquet, verify_checksum=True) for s in sources]
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sources:
        output = output_dir / f"limosat_{source.period}.nc"
        entries.append(build_period(source, find_catalog(catalog_dir, source.period), output, release_id, metadata, overwrite))
        print(f"{source.period}: {entries[-1]['bytes'] / 1e6:.0f} MB", flush=True)
    provenance = {entry["period"]: entry.pop("_provenance") for entry in entries}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": FORMAT,
        "crs": CRS,
        "primary": {"policy_id": OWNERSHIP_POLICY,
                    "definition": "start time of the arriving displacement (own time for point_index 0) in the period's "
                    "half-open window"},
        "flag_bits": FLAG_BITS,
        "columns": columns_spec(),
        "compression": {"codec": "zlib", "level": COMPRESSION_LEVEL, "shuffle": True, "chunk_observations": CHUNK},
        "layout": {"dimensions": {"trajectory": "trajectories", "obs": "observations", "scene": "Sentinel-1 scene names"},
                   "length_variable": "rowsize", "sample_dimension": "obs", "trajectory_order": "time_start ascending"},
        "files": entries,
        "totals": {key: sum(e[key] for e in entries) for key in ("rows", "trajectories", "displacements", "primary_rows", "primary_displacements")},
    }
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sidecar = output_dir.with_name(output_dir.name + ".provenance.json")
    _atomic_write(sidecar, json.dumps({"release_id": release_id, "source_root": str(source_root), "periods": provenance},
                                      indent=2, sort_keys=True) + "\n")
    if validate_output:
        validate(source_root, catalog_dir, output_dir, sample_modulus)
    return manifest


# --------------------------------------------------------------------------------------------- validation


def _same(a, b, what: str, name: str) -> None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
        raise ValueError(f"{name}: sampled {what} differ from the source")


def validate_period(source: Source, catalog_path: Path, output: Path, entry: dict[str, Any], release_id: str,
                    sample_modulus: int = 50) -> dict[str, Any]:
    """Check a published file against the manifest entry and, on a sample of trajectories, against the source."""
    netCDF4 = _netcdf4()
    pa, pc, _, pq = _pyarrow()
    output = Path(output)
    if sha256(output) != entry["sha256"] or output.stat().st_size != entry["bytes"]:
        raise ValueError(f"{output.name}: checksum or size differs from the manifest")
    window = ownership_window(source.period)
    w0, w1 = _ms(window[0]), _ms(window[1])
    d = netCDF4.Dataset(output)
    d.set_auto_maskandscale(False)
    try:
        expected = {"release_id": release_id, "product_version": release_id, "period": source.period, "schema_version": SCHEMA_VERSION,
                    "featureType": "trajectory", "primary_window_start": _utc_text(window[0]),
                    "primary_window_stop_exclusive": _utc_text(window[1]), "primary_rule": OWNERSHIP_POLICY}
        for key, value in expected.items():
            if getattr(d, key, None) != value:
                raise ValueError(f"{output.name}: global attribute {key} is {getattr(d, key, None)!r}, expected {value!r}")
        rowsize = d["rowsize"][:].astype(np.int64)
        ntraj, n = len(rowsize), int(rowsize.sum())
        if n != d.dimensions["obs"].size or n != entry["rows"]:
            raise ValueError(f"{output.name}: {n} observations, manifest says {entry['rows']}")
        if ntraj != entry["trajectories"] or ntraj != d.dimensions["trajectory"].size:
            raise ValueError(f"{output.name}: {ntraj} trajectories, manifest says {entry['trajectories']}")
        ids = d["trajectory_id"][:]
        if len(np.unique(ids)) != ntraj:
            raise ValueError(f"{output.name}: trajectory ids are not unique")
        offset = np.r_[0, np.cumsum(rowsize)]
        pidx = d["point_index"][:]
        if np.any(pidx != np.arange(n) - np.repeat(offset[:-1], rowsize)):
            raise ValueError(f"{output.name}: point_index is not 0..n-1 within every trajectory")
        time = np.rint(d["time"][:] * 1000).astype(np.int64)
        tstart, tend = d["time_start"][:], d["time_end"][:]
        if np.any(np.diff(tstart) < 0) or not np.array_equal(np.rint(tstart * 1000).astype(np.int64), time[offset[:-1]]) \
                or not np.array_equal(np.rint(tend * 1000).astype(np.int64), time[offset[1:] - 1]):
            raise ValueError(f"{output.name}: trajectories are not ordered by start time, or time_start/time_end are wrong")
        primary = d["is_primary"][:].astype(bool)
        if int(primary.sum()) != entry["primary_rows"] or int((primary & (pidx > 0)).sum()) != entry["primary_displacements"]:
            raise ValueError(f"{output.name}: primary counts differ from the manifest")
        if list(d["quality_flags"].flag_masks) != [1 << b for b in FLAG_BITS.values()] or d["quality_flags"].flag_meanings.split() != list(FLAG_BITS):
            raise ValueError(f"{output.name}: flag masks or meanings differ from the schema")

        # Source cross-check on a sample of trajectories: every published displacement must equal its source link.
        catalog = load_catalog(catalog_path)
        keep = np.repeat((ids % sample_modulus) == 0, rowsize)
        sampled = {name: d[name][:][keep] for name in ("x_m", "y_m", "dx_m", "dy_m", "speed_m_per_day", "correlation",
                                                          "direct_match_share", "interpolated", "quality_flags", "is_primary")}
        names = np.array(list(d["scene_name"][:]), dtype=object)
        scene = names[d["scene_index"][:][keep]]
        tid_s, pidx_s, ms_s = np.repeat(ids, rowsize)[keep], pidx[keep], time[keep]
    finally:
        d.close()
    src = _read_sample(source.parquet, list(SOURCE_COLUMNS), "qc_trajectory_id", sample_modulus)

    def col(name):
        return src[name].to_numpy(zero_copy_only=False)

    s_order = np.lexsort((col("trajectory_link_index"), col("qc_trajectory_id")))
    p_order = np.lexsort((pidx_s, tid_s))
    src = src.take(pa.array(s_order))
    tid_s, pidx_s, ms_s, scene = tid_s[p_order], pidx_s[p_order], ms_s[p_order], scene[p_order]
    sampled = {k: v[p_order] for k, v in sampled.items()}
    if len(tid_s) != src.num_rows + len(set(col("qc_trajectory_id").tolist())):
        raise ValueError(f"{output.name}: sampled row count is not links + trajectories")
    is_step = pidx_s > 0
    step = np.flatnonzero(is_step)
    first = np.flatnonzero(~is_step)
    start_ms, end_ms = _epoch_ms_array(src["start_time_utc"]), _epoch_ms_array(src["end_time_utc"])
    name = output.name
    _same(tid_s[step], col("qc_trajectory_id"), "trajectory ids", name)
    _same(ms_s[step], end_ms, "end times", name)
    _same(sampled["x_m"][step], col("x1_m"), "x", name)
    _same(sampled["y_m"][step], col("y1_m"), "y", name)
    _same(sampled["dx_m"][step], col("dx_m"), "dx", name)
    _same(sampled["dy_m"][step], col("dy_m"), "dy", name)
    _same(sampled["speed_m_per_day"][step], col("speed_m_per_day"), "speed", name)
    _same(sampled["interpolated"][step], col("target_interpolated").astype(np.int8), "interpolated", name)
    _same(sampled["correlation"][step], _floats(src["target_corr"], np.float32), "correlation", name)
    _same(sampled["direct_match_share"][step], _floats(src["pair_direct_share"], np.float32), "direct_match_share", name)
    _same(sampled["quality_flags"][step], col("qc_flags").astype(np.uint8), "quality_flags", name)
    _same(sampled["is_primary"][step].astype(bool), (start_ms >= w0) & (start_ms < w1), "is_primary flags", name)
    catalog_names = np.asarray(catalog.scenes.to_pylist(), dtype=object)
    _same(scene[step], catalog_names[np.searchsorted(catalog.image_ids, col("target_image_id"))], "scenes", name)
    # chain: the observation before each displacement is its source (position, time and scene)
    prev = step - 1
    _same(sampled["x_m"][prev], col("x0_m"), "chain x", name)
    _same(sampled["y_m"][prev], col("y0_m"), "chain y", name)
    _same(ms_s[prev], start_ms, "chain times", name)
    _same(tid_s[prev], col("qc_trajectory_id"), "chain trajectory", name)
    _same(scene[prev], catalog_names[np.searchsorted(catalog.image_ids, col("source_image_id"))], "chain scenes", name)
    # first observations have no displacement values
    if not (np.isnan(sampled["dx_m"][first]).all() and np.isnan(sampled["dy_m"][first]).all() and np.isnan(sampled["speed_m_per_day"][first]).all()
            and np.isnan(sampled["correlation"][first]).all() and np.isnan(sampled["direct_match_share"][first]).all()
            and (sampled["interpolated"][first] == SENTINEL_FILL["interpolated"]).all() and (sampled["quality_flags"][first] == SENTINEL_FILL["flags"]).all()):
        raise ValueError(f"{name}: first observations carry displacement values")
    return {"period": source.period, "status": "passed", "rows": entry["rows"], "sampled_displacements": int(is_step.sum()),
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
        print(json.dumps(results[-1]), flush=True)
    return results


# --------------------------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--source-root", type=Path, required=True,
                             help="internal release folder holding <period>/<period>_qc.parquet")
        command.add_argument("--catalog-dir", type=Path, required=True, help="folder of <period>_image_catalog_*.json")
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "build":
            command.add_argument("--release-id", required=True)
            command.add_argument("--metadata", type=Path, help="JSON with title, summary, keywords, creators, license, acknowledgement, ...")
            command.add_argument("--period", action="append", dest="periods")
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--skip-source-checksum", action="store_true",
                                 help="do not hash the source Parquet files against their manifests")
            command.add_argument("--no-validate", action="store_true", help="skip the validation that normally follows the build")
        command.add_argument("--sample-modulus", type=int, default=50,
                             help="check every Nth trajectory against the source (1 checks all)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        result = build(args.source_root, args.catalog_dir, args.output_dir, args.release_id, args.periods, args.overwrite,
                       not args.skip_source_checksum, metadata, not args.no_validate, args.sample_modulus)
        result = result["totals"]
    else:
        result = validate(args.source_root, args.catalog_dir, args.output_dir, args.sample_modulus)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
