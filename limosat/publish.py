"""Build the public LiMOSAT-EfficientLoFTR trajectory dataset (schema 1) as CF NetCDF4.

The primary composition (``limosat compose-parquet``) writes trajectory points and measured trajectory extensions with their
quality-control status for our own analysis and checking. This module turns the extensions table into a format fit for public use
and leaves out what an ordinary data user does not need: internal ids and pair identifiers, quality-control bookkeeping, rejected
extensions, dormant and created-only points and local file paths. It never re-scores or alters an extension.

The released file is a self-describing CF-1.8 *trajectory* file in the contiguous ragged array representation (CF 9.3.3), with the
same layout as the ORB release of LiMOSAT (``limosat.publish`` of the ``olivia`` branch), only the observation variables differ:

* dimensions ``trajectory`` (one entry per trajectory), ``obs`` (one per observation) and ``scene`` (Sentinel-1 scene names);
* ``rowsize(trajectory)`` carries ``sample_dimension = "obs"`` and gives the number of observations of each trajectory, so
  trajectory *i* is the slice ``offset[i]:offset[i] + rowsize[i]`` of every ``obs`` variable;
* trajectories are ordered by start time (``time_start``) and the observations of a trajectory by ``point_index``;
* a row of the observation table is one observation of a feature in one scene, carrying the displacement from the observation it
  starts at (dx, dy, speed; ``source_point_index`` says which row that is) and the tracker's own statistics at the position
  (selected matches, support radius, maximum residual) and the review flags. A trajectory does not have an observation on every
  scene, so the time step between observations varies.

The composition stores a trajectory as a tree, not a chain: when no scene pair leads from a trajectory's latest observation to a new
scene, the trajectory continues from an older observation, and the observation it skipped is left without a continuation (a side
observation). **By default the release publishes the whole tree** (rule ``none``): every accepted displacement, each with the row it
starts at in ``source_point_index`` (``point_index - 1`` in a simple chain, about 80% of the links; smaller otherwise). Whether
reducing every trajectory to one chain helps downstream users is still open; ``--chain-rule latest`` or ``deepest`` publishes the
path from the first observation to the trajectory's latest (most links) one instead (``select_chain``), so that every displacement
starts at the previous row, and drops about 19% of the links of a month, which is counted in the manifest.

No value is rounded or packed: positions, times and displacements are float64 exactly as in the composition, the three match
statistics float32 (a float32 of the composition's float64), and nulls become the variable's ``_FillValue``. Compression is lossless
(zlib with shuffle). The public schema is defined once, here (``COLUMNS``, ``FLAG_BITS``). ``build`` writes the file and ``validate``
re-reads it and compares every Nth trajectory with the source extensions, value by value.

Public outputs carry no local paths and no internal protocol detail. Internal provenance (source paths, checksums, run id,
configuration hash) goes to a sidecar next to, not inside, the output directory.

The extensions table is far larger than memory (about 175 million rows for January to April 2020), so ``build`` works in three
streaming passes: (1) index the trajectories (id, start time, number of links) and fix the order of the file; (2) spill the
observations to bucket files on disk, each holding whole trajectories; (3) sort every bucket and write it into the NetCDF file. Only
the trajectory index (about 100 bytes per trajectory) and one bucket are in memory at a time.

    python -m limosat.publish build --source composition_dir --output-dir out --release-id limosat_eloftr_v1 --metadata m.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0.0"
CRS = "EPSG:3413"
PRODUCT = "LiMOSAT-EfficientLoFTR"
EXTENSIONS_FILE = "trajectory-extensions-qc-v1.parquet"
COMPOSITION_MANIFEST = "composition-manifest-v1.json"

FLAG_BITS = {"EXCEEDS_CONFIGURED_SPEED": 0, "LOCAL_NEIGHBOUR_REVIEW": 1}
FLAG_DESCRIPTIONS = {
    "EXCEEDS_CONFIGURED_SPEED": "the displacement speed is above the speed the matcher is configured for (review flag, the "
    "displacement is kept)",
    "LOCAL_NEIGHBOUR_REVIEW": "the displacement differs strongly from the other displacements between the same two scenes within "
    "the neighbour search radius (review flag, the displacement is kept)",
}
KNOWN_FLAG_MASK = sum(1 << bit for bit in FLAG_BITS.values())
FLAG_MASKS = np.array([1 << bit for bit in FLAG_BITS.values()], dtype=np.uint8)
FLAG_FILL = 255


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    unit: str
    description: str


COLUMNS = (
    Column("trajectory_id", "int64", False, "",
           "Trajectory identifier, the running number of the trajectory in the file (0, 1, 2, ...). A trajectory is the track of one "
           "feature through Sentinel-1 scenes."),
    Column("point_index", "int32", False, "",
           "Position along the trajectory in time order: 0 for its first observation, then 1, 2, ..."),
    Column("source_point_index", "int32", True, "",
           "point_index of the observation of the same trajectory at which the displacement arriving here starts. Usually "
           "point_index - 1; it is smaller where the trajectory continued from an earlier observation because no scene pair "
           "reached this scene from the latest one (then an observation between them has no continuation). Null at point_index 0."),
    Column("time", "timestamp_s_utc", False, "UTC", "Acquisition time of the Sentinel-1 scene of this observation."),
    Column("scene", "string", False, "", "Sentinel-1 product name of the scene this observation belongs to."),
    Column("x_m", "float64", False, "m", "Position, EPSG:3413 x."),
    Column("y_m", "float64", False, "m", "Position, EPSG:3413 y."),
    Column("dx_m", "float64", True, "m",
           "Displacement in x from the observation source_point_index of the trajectory. Null at point_index 0."),
    Column("dy_m", "float64", True, "m",
           "Displacement in y from the observation source_point_index of the trajectory. Null at point_index 0."),
    Column("speed_m_per_day", "float64", True, "m/day",
           "Distance from the source observation divided by the elapsed time since it. Null at point_index 0."),
    Column("selected_matches", "float32", True, "1",
           "Number of feature matches that support the displacement field at this position: the count at each of the three "
           "surrounding field nodes, weighted by the position's barycentric weights (so not an integer). Null at point_index 0."),
    Column("support_radius_m", "float32", True, "m",
           "Radius around the field nodes that holds the selected matches, weighted like selected_matches. Null at point_index 0."),
    Column("maximum_residual_m", "float32", True, "m",
           "Largest disagreement between a selected match and the field node's displacement, weighted like selected_matches "
           "(the tracker's own value, not a calibrated uncertainty). Null at point_index 0."),
    Column("quality_flags", "uint8", True, "",
           "Bit mask describing the displacement arriving at this observation: "
           + "; ".join(f"bit {bit} {name}" for name, bit in FLAG_BITS.items())
           + ". Flags never split trajectories; displacements rejected by the hard speed limit are not published. "
           "Null at point_index 0."),
)
COLUMN_NAMES = tuple(column.name for column in COLUMNS)

# Columns read from the composition's extensions table.
SOURCE_COLUMNS = (
    "trajectory_id", "start_time_utc", "end_time_utc", "elapsed_seconds", "source_image_id", "target_image_id",
    "x0_m", "y0_m", "x1_m", "y1_m", "dx_m", "dy_m", "speed_m_per_day", "selected_matches", "support_radius_m",
    "maximum_residual_m", "exceeds_configured_speed", "local_neighbour_review", "accepted",
)

FORMAT = "NetCDF-4 (HDF5), CF-1.8 trajectory featureType, contiguous ragged array"
CHUNK = 65_536
COMPRESSION_LEVEL = 4
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
BATCH_ROWS = 1_000_000
BUCKET_OBSERVATIONS = 20_000_000
DEFAULT_METADATA: dict[str, Any] = {
    "title": "LiMOSAT-EfficientLoFTR: Lagrangian sea-ice trajectories from Sentinel-1",
    "summary": ("Trajectories of features on the Arctic sea ice tracked through Sentinel-1 EW scenes with the LiMOSAT "
                "algorithm and the EfficientLoFTR matcher. One observation per feature and scene; the displacement from the "
                "observation it starts at is stored with it (see source_point_index). A trajectory continues through any later scene pair "
                "that supports it, so it does not have an observation on every scene."),
    "keywords": "sea ice drift, Lagrangian trajectories, Sentinel-1, SAR, Arctic",
    "institution": "Nansen Environmental and Remote Sensing Centre",
    "source": "Copernicus Sentinel-1 EW GRD scenes, processed with LiMOSAT (https://github.com/nansencenter/limosat)",
}

# Spilled observation record (one row per observation, 77 bytes).
SPILL_DTYPE = np.dtype([
    ("rank", "<i8"), ("ms", "<i8"), ("x", "<f8"), ("y", "<f8"), ("dx", "<f8"), ("dy", "<f8"), ("speed", "<f8"),
    ("el", "<i4"), ("sel", "<f4"), ("sup", "<f4"), ("res", "<f4"), ("flags", "u1"), ("scene", "<i4"),
])
_HEX = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(b"0123456789abcdef"):
    _HEX[_c] = _i
for _i, _c in enumerate(b"ABCDEF"):
    _HEX[_c] = 10 + _i


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError(
            "Publishing requires PyArrow. Install it with `conda install -c conda-forge pyarrow` or `pip install pyarrow`."
        ) from error
    return pa, pc, pq


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
        raise RuntimeError("Computing longitude and latitude requires pyproj.") from error
    return pyproj


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a closed file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _ms_to_datetime(value: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=int(value))


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_ms_array(column) -> np.ndarray:
    """Epoch milliseconds of a timestamp column of any unit; fails rather than silently truncating."""
    pa, _, _ = _pyarrow()
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    return column.cast(pa.timestamp("ms", tz="UTC")).cast(pa.int64()).to_numpy()


def columns_spec() -> list[dict[str, Any]]:
    return [dict(name=c.name, type=c.type, nullable=c.nullable, unit=c.unit, description=c.description) for c in COLUMNS]


def _array(column):
    return column.combine_chunks() if hasattr(column, "combine_chunks") else column


def _floats(column, dtype=np.float64) -> np.ndarray:
    column = _array(column)
    if column.null_count:
        column = column.fill_null(float("nan"))
    return column.to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _flags(table) -> np.ndarray:
    for name in ("exceeds_configured_speed", "local_neighbour_review"):
        if table[name].null_count:
            raise ValueError(f"nulls in source column {name}")
    exceeds = _array(table["exceeds_configured_speed"]).to_numpy(zero_copy_only=False).astype(bool)
    local = _array(table["local_neighbour_review"]).to_numpy(zero_copy_only=False).astype(bool)
    return (exceeds.astype(np.uint8) << FLAG_BITS["EXCEEDS_CONFIGURED_SPEED"]) | (local.astype(np.uint8) << FLAG_BITS["LOCAL_NEIGHBOUR_REVIEW"])


# --------------------------------------------------------------------------------------------- source


@dataclass(frozen=True)
class Source:
    directory: Path
    extensions: Path
    manifest_path: Path
    manifest: dict[str, Any]


def inspect_source(directory: Path, verify_checksum: bool = False) -> Source:
    """Check a primary composition folder: its extensions table, manifest and (optionally) the table's checksum."""
    _, _, pq = _pyarrow()
    directory = Path(directory)
    extensions, manifest_path = directory / EXTENSIONS_FILE, directory / COMPOSITION_MANIFEST
    if not extensions.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Need {EXTENSIONS_FILE} and {COMPOSITION_MANIFEST} in {directory}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("trajectory_extension_schema_version") != 1:
        raise ValueError("unsupported trajectory extension schema version")
    parquet = pq.ParquetFile(extensions)
    missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"source lacks columns {sorted(missing)}")
    counts = manifest.get("counts") or {}
    if counts.get("trajectory_extensions") not in (None, parquet.metadata.num_rows):
        raise ValueError(f"manifest says {counts.get('trajectory_extensions')} extensions, the table has {parquet.metadata.num_rows}")
    if verify_checksum:
        expected = ((manifest.get("products") or {}).get("trajectory_extensions") or {}).get("sha256")
        if expected and sha256(extensions) != expected:
            raise ValueError("extensions table checksum differs from its manifest")
    return Source(directory, extensions, manifest_path, manifest)


# --------------------------------------------------------------------------------------------- trajectory ids


def _id_keys(column) -> tuple[np.ndarray, np.ndarray]:
    """64-bit key (the first 16 hex digits) and 16-bit tail of ids of the form ``trj_`` plus 20 hex digits."""
    pa, _, _ = _pyarrow()
    column = _array(column)
    if column.null_count:
        raise ValueError("nulls in trajectory_id")
    try:
        fixed = column.cast(pa.binary(24))
    except pa.ArrowInvalid as error:
        raise ValueError("trajectory ids are not all of the form trj_ plus 20 hexadecimal digits") from error
    raw = np.frombuffer(fixed.buffers()[1], dtype=np.uint8)[fixed.offset * 24:(fixed.offset + len(fixed)) * 24].reshape(-1, 24)
    if len(raw) and not (raw[:, :4] == np.frombuffer(b"trj_", dtype=np.uint8)).all():
        raise ValueError("trajectory ids do not start with trj_")
    nibble = _HEX[raw[:, 4:]]
    if (nibble == 255).any():
        raise ValueError("trajectory ids are not hexadecimal")
    keys = np.zeros(len(raw), dtype=np.uint64)
    for j in range(16):
        keys = (keys << np.uint64(4)) | nibble[:, j].astype(np.uint64)
    tails = ((nibble[:, 16].astype(np.uint16) << 12) | (nibble[:, 17].astype(np.uint16) << 8)
             | (nibble[:, 18].astype(np.uint16) << 4) | nibble[:, 19].astype(np.uint16))
    return keys, tails


CHAIN_RULES = ("latest", "deepest")
PUBLISH_RULES = ("none",) + CHAIN_RULES   # none: publish the whole tree (default)
_MIX = np.uint64(1000003)


def select_chain(tk: np.ndarray, start_ms: np.ndarray, end_ms: np.ndarray, rule: str = "latest",
                 chunk: int = 20_000_000) -> tuple[np.ndarray, dict[str, int]]:
    """Which links lie on the published chain of their trajectory.

    The composition stores a trajectory as a tree: a link starts at an observation (its trajectory and ``start_ms``) and ends at a new
    one (``end_ms``); a new scene may be reached from an older observation when no pair from the latest one exists, so an observation
    can have no continuation (a side observation) or several. The published chain is the single path from the first observation to the
    observation that ends the trajectory: the latest one (``rule="latest"``, the default: it keeps the branch that continues furthest
    in time) or the one with the most links above it (``"deepest"``). Returns a boolean mask over the links (True = on the chain) and
    counts. Observations are identified by trajectory key and time, which must be unique within a trajectory.
    """
    if rule not in CHAIN_RULES:
        raise ValueError(f"rule must be one of {CHAIN_RULES}")
    n = len(tk)
    if n >= 2**31 - 1:
        raise ValueError("too many links for 32-bit link indices")
    tk = np.asarray(tk, dtype=np.uint64)
    end_u = np.asarray(end_ms).astype(np.uint64)
    key_obs = tk * _MIX + end_u
    order = np.argsort(key_obs, kind="stable")
    sorted_obs = key_obs[order]
    del key_obs
    if n > 1 and (sorted_obs[1:] == sorted_obs[:-1]).any():
        raise ValueError("two links of one trajectory end at the same time")
    order = order.astype(np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    for a in range(0, n, chunk):
        ks = tk[a:a + chunk] * _MIX + np.asarray(start_ms[a:a + chunk]).astype(np.uint64)
        pos = np.minimum(np.searchsorted(sorted_obs, ks), n - 1)
        found = sorted_obs[pos] == ks
        parent[a:a + chunk] = np.where(found, order[pos], -1)
    del sorted_obs, order
    has_child = np.zeros(n, dtype=bool)
    has_child[parent[parent >= 0]] = True
    leaves = np.flatnonzero(~has_child)
    tk_l, end_l = tk[leaves], np.asarray(end_ms)[leaves]
    if rule == "latest":
        o = np.lexsort((end_l, tk_l))
    else:
        anc = np.append(np.where(parent >= 0, parent, n), np.int32(n)).astype(np.int32)
        steps = np.append((parent >= 0).astype(np.int32), np.int32(0))
        while (anc[:n] != n).any():                         # pointer doubling: number of links above every link
            steps = steps + steps[anc]
            anc = anc[anc]
        o = np.lexsort((end_l, steps[leaves], tk_l))
        del anc, steps
    tk_sorted = tk_l[o]
    last = np.r_[tk_sorted[1:] != tk_sorted[:-1], True] if len(o) else np.zeros(0, dtype=bool)
    chosen = leaves[o][last]
    kept = np.zeros(n, dtype=bool)
    current = chosen
    while len(current):
        kept[current] = True
        current = parent[current]
        current = current[current >= 0]
    return kept, {"links": n, "kept": int(kept.sum()), "leaves": int(len(leaves)), "trajectories": int(len(chosen)),
                  "side_observations": int(len(leaves) - len(chosen))}


@dataclass
class Index:
    ukeys: np.ndarray          # sorted unique 64-bit keys
    rank_of_ukey: np.ndarray   # file order of every key
    nlinks: np.ndarray         # by rank
    first_ms: np.ndarray       # by rank
    offset: np.ndarray         # by rank: first observation row of the trajectory
    rowsize: np.ndarray        # by rank: nlinks + 1
    links: int                 # published links (on the chain)
    rejected: int              # extensions rejected by the composition (not published)
    off_chain: int = 0         # accepted extensions off the published chain (not published)
    keep_rows: np.ndarray | None = None  # over the accepted extensions in file order: True = published
    chain_rule: str = "none"


def index_source(extensions: Path, batch_rows: int = BATCH_ROWS, chain_rule: str = "none") -> Index:
    """Pass 1: choose the published chain of every trajectory, then per trajectory the number of links and its first start time;
    fixes the order of the file."""
    pa, pc, pq = _pyarrow()
    keys_list, tails_list, start_list, end_list = [], [], [], []
    rejected = 0
    for batch in pq.ParquetFile(extensions).iter_batches(batch_size=batch_rows, columns=["trajectory_id", "start_time_utc", "end_time_utc", "accepted"]):
        table = pa.Table.from_batches([batch])
        if table["accepted"].null_count:
            raise ValueError("nulls in source column accepted")
        keep = _array(table["accepted"]).to_numpy(zero_copy_only=False).astype(bool)
        rejected += int((~keep).sum())
        keys, tails = _id_keys(table["trajectory_id"])
        keys_list.append(keys[keep]); tails_list.append(tails[keep])
        start_list.append(_epoch_ms_array(table["start_time_utc"])[keep]); end_list.append(_epoch_ms_array(table["end_time_utc"])[keep])
    if not keys_list:
        raise ValueError("the extensions table is empty")
    keys, tails = np.concatenate(keys_list), np.concatenate(tails_list)
    starts, ends = np.concatenate(start_list), np.concatenate(end_list)
    del keys_list, tails_list, start_list, end_list
    if not len(keys):
        raise ValueError("no accepted extensions to publish")
    accepted_links = len(keys)
    if chain_rule == "none":
        keep_rows = np.ones(len(keys), dtype=bool)          # the whole tree
    else:
        keep_rows, _ = select_chain(keys, starts, ends, chain_rule)
    del ends
    keys, tails, starts = keys[keep_rows], tails[keep_rows], starts[keep_rows]
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    bounds = np.r_[0, np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1]
    ukeys = keys_sorted[bounds]
    counts = np.diff(np.r_[bounds, len(keys_sorted)]).astype(np.int64)
    first = np.minimum.reduceat(starts[order], bounds)
    tail_min, tail_max = np.minimum.reduceat(tails[order], bounds), np.maximum.reduceat(tails[order], bounds)
    if (tail_min != tail_max).any():
        raise ValueError("two trajectory ids share their first 16 hexadecimal digits")
    links = len(keys)
    del keys, tails, starts, order, keys_sorted
    by_rank = np.lexsort((ukeys, first))
    rank_of_ukey = np.empty(len(ukeys), dtype=np.int64)
    rank_of_ukey[by_rank] = np.arange(len(ukeys))
    nlinks, first = counts[by_rank], first[by_rank]
    rowsize = nlinks + 1
    offset = np.r_[0, np.cumsum(rowsize)[:-1]]
    return Index(ukeys, rank_of_ukey, nlinks, first, offset, rowsize, links, rejected, accepted_links - links, keep_rows, chain_rule)


# --------------------------------------------------------------------------------------------- spill


@dataclass
class Spill:
    directory: Path
    buckets: list[int]
    scenes: list[str]
    scene_time_ms: np.ndarray
    observations: int


def _bucket_path(directory: Path, bucket: int) -> Path:
    return directory / f"bucket_{bucket:04d}.bin"


def spill_source(extensions: Path, index: Index, directory: Path, bucket_observations: int = BUCKET_OBSERVATIONS,
                 batch_rows: int = BATCH_ROWS) -> Spill:
    """Pass 2: write every observation of every trajectory to the bucket file of its trajectory."""
    pa, pc, pq = _pyarrow()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handles: dict[int, Any] = {}
    scene_id: dict[str, int] = {}
    scene_time: dict[int, int] = {}
    first_emitted = np.zeros(len(index.ukeys), dtype=bool)
    observations = 0
    accepted_seen = 0
    try:
        for batch in pq.ParquetFile(extensions).iter_batches(batch_size=batch_rows, columns=list(SOURCE_COLUMNS)):
            table = pa.Table.from_batches([batch])
            keep = _array(table["accepted"]).to_numpy(zero_copy_only=False).astype(bool)
            if not keep.all():
                table = table.filter(pa.array(keep))
            if not table.num_rows:
                continue
            on_chain = index.keep_rows[accepted_seen:accepted_seen + table.num_rows]
            accepted_seen += table.num_rows
            if not on_chain.all():
                table = table.filter(pa.array(on_chain))
            if not table.num_rows:
                continue
            n = table.num_rows
            for name in ("start_time_utc", "end_time_utc", "elapsed_seconds", "source_image_id", "target_image_id", "x0_m", "y0_m",
                         "x1_m", "y1_m", "dx_m", "dy_m", "speed_m_per_day"):
                if table[name].null_count:
                    raise ValueError(f"nulls in source column {name}")
            keys, _ = _id_keys(table["trajectory_id"])
            position = np.searchsorted(index.ukeys, keys)
            if (position >= len(index.ukeys)).any() or (index.ukeys[position] != keys).any():
                raise ValueError("the extensions table changed between passes")
            rank = index.rank_of_ukey[position]
            start_ms, end_ms = _epoch_ms_array(table["start_time_utc"]), _epoch_ms_array(table["end_time_utc"])
            elapsed = _floats(table["elapsed_seconds"])
            if not np.array_equal(elapsed, np.rint(elapsed)) or elapsed.max() >= 2**31 or elapsed.min() <= 0:
                raise ValueError("elapsed_seconds is not a positive whole number of seconds")
            flags = _flags(table)
            scenes = []
            for name in ("source_image_id", "target_image_id"):
                encoded = _array(table[name]).dictionary_encode()
                mapping = np.empty(len(encoded.dictionary), dtype=np.int32)
                for k, text in enumerate(encoded.dictionary.to_pylist()):
                    mapping[k] = scene_id.setdefault(text, len(scene_id))
                scenes.append(mapping[encoded.indices.to_numpy()])
            source_scene, target_scene = scenes
            links = np.empty(n, dtype=SPILL_DTYPE)
            links["rank"], links["ms"] = rank, end_ms
            links["x"], links["y"] = _floats(table["x1_m"]), _floats(table["y1_m"])
            links["dx"], links["dy"], links["speed"] = _floats(table["dx_m"]), _floats(table["dy_m"]), _floats(table["speed_m_per_day"])
            links["el"] = elapsed.astype(np.int32)
            links["sel"] = _floats(table["selected_matches"], np.float32)
            links["sup"] = _floats(table["support_radius_m"], np.float32)
            links["res"] = _floats(table["maximum_residual_m"], np.float32)
            links["flags"], links["scene"] = flags, target_scene
            # the first observation of a trajectory is the source of its earliest link(s); one record per trajectory
            candidates = np.flatnonzero(start_ms == index.first_ms[rank])
            candidates = candidates[np.unique(rank[candidates], return_index=True)[1]]
            candidates = candidates[~first_emitted[rank[candidates]]]
            first_emitted[rank[candidates]] = True
            firsts = np.empty(len(candidates), dtype=SPILL_DTYPE)
            firsts["rank"], firsts["ms"] = rank[candidates], start_ms[candidates]
            firsts["x"], firsts["y"] = _floats(table["x0_m"])[candidates], _floats(table["y0_m"])[candidates]
            firsts["dx"] = firsts["dy"] = firsts["speed"] = np.nan
            firsts["el"] = 0
            firsts["sel"] = firsts["sup"] = firsts["res"] = np.float32(np.nan)
            firsts["flags"], firsts["scene"] = FLAG_FILL, source_scene[candidates]
            records = np.concatenate([links, firsts])
            for scene_array, when in ((target_scene, end_ms), (source_scene[candidates], start_ms[candidates])):
                for s, t in zip(*np.unique(np.c_[scene_array, when], axis=0).T):
                    if scene_time.setdefault(int(s), int(t)) != int(t):
                        raise ValueError(f"scene {list(scene_id)[int(s)]} has two different acquisition times")
            bucket = (index.offset[records["rank"]] // bucket_observations).astype(np.int64)
            order = np.argsort(bucket, kind="stable")
            records, bucket = records[order], bucket[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1, len(bucket)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                key = int(bucket[a])
                if key not in handles:
                    handles[key] = open(_bucket_path(directory, key), "wb")
                records[a:b].tofile(handles[key])
            observations += len(records)
    finally:
        for handle in handles.values():
            handle.close()
    if accepted_seen != len(index.keep_rows):
        raise ValueError("the extensions table changed between passes")
    if not first_emitted.all():
        raise ValueError(f"{int((~first_emitted).sum())} trajectories have no first observation")
    names = list(scene_id)
    times = np.array([scene_time[i] for i in range(len(names))], dtype=np.int64)
    return Spill(directory, sorted(handles), names, times, observations)


# --------------------------------------------------------------------------------------------- NetCDF


def _lonlat(x: np.ndarray, y: np.ndarray, block: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    transformer = _pyproj().Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = np.empty(len(x)), np.empty(len(x))
    for start in range(0, len(x), block):
        stop = start + block
        lon[start:stop], lat[start:stop] = transformer.transform(x[start:stop], y[start:stop])
    return lon, lat


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


def write_netcdf(path: Path, index: Index, spill: Spill, period: str, release_id: str, metadata: dict[str, Any],
                 qc: dict[str, Any]) -> dict[str, Any]:
    """Pass 3: sort every bucket and write it into the NetCDF file. Returns the manifest entry statistics."""
    netCDF4 = _netcdf4()
    describe = {c.name: c for c in COLUMNS}
    ntraj, n = len(index.rowsize), int(index.rowsize.sum())
    if n != spill.observations:
        raise ValueError(f"{spill.observations} observations were spilled, the index expects {n}")
    # Scenes in the file: by acquisition time, then name; a scene is listed only if an observation uses it.
    order_scene = np.array(sorted(range(len(spill.scenes)), key=lambda i: (int(spill.scene_time_ms[i]), spill.scenes[i])), dtype=np.int64)
    new_of_old = np.empty(len(order_scene), dtype=np.int32)
    new_of_old[order_scene] = np.arange(len(order_scene), dtype=np.int32)
    scene_names = [spill.scenes[i] for i in order_scene]
    kw = dict(zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True, chunksizes=(min(CHUNK, n),))
    kw_traj = dict(zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True, chunksizes=(min(CHUNK, ntraj),))
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    stats = {"x_min": np.inf, "x_max": -np.inf, "y_min": np.inf, "y_max": -np.inf, "lat_min": np.inf, "lat_max": -np.inf,
             "lon_min": np.inf, "lon_max": -np.inf, "ms_min": np.iinfo(np.int64).max, "ms_max": np.iinfo(np.int64).min}
    status_counts = {"unflagged": 0, **{name: 0 for name in FLAG_BITS}}
    try:
        ds.createDimension("trajectory", ntraj)
        ds.createDimension("obs", n)
        ds.createDimension("scene", len(scene_names))
        coords = "time lat lon"
        v_tid = ds.createVariable("trajectory_id", "i8", ("trajectory",), **kw_traj)
        v_tid.cf_role = "trajectory_id"
        v_tid.long_name = "trajectory identifier, the running number of the trajectory in the file"
        v_tid[:] = np.arange(ntraj, dtype=np.int64)
        v_row = ds.createVariable("rowsize", "i4", ("trajectory",), **kw_traj)
        v_row.long_name = "number of observations in the trajectory"
        v_row.sample_dimension = "obs"
        v_row[:] = index.rowsize.astype(np.int32)
        v_t0 = ds.createVariable("time_start", "f8", ("trajectory",), **kw_traj)
        v_t1 = ds.createVariable("time_end", "f8", ("trajectory",), **kw_traj)
        for v, text in ((v_t0, "time of the first observation of the trajectory"), (v_t1, "time of the last observation of the trajectory")):
            v.long_name, v.units, v.calendar = text, TIME_UNITS, "standard"
        v_t0[:] = index.first_ms / 1000.0
        last_ms = np.empty(ntraj, dtype=np.int64)

        time = ds.createVariable("time", "f8", ("obs",), **kw)
        time.standard_name, time.long_name = "time", describe["time"].description
        time.units, time.calendar, time.axis = TIME_UNITS, "standard", "T"
        v_lat = ds.createVariable("lat", "f8", ("obs",), **kw)
        v_lat.standard_name, v_lat.long_name, v_lat.units, v_lat.axis = "latitude", "latitude of the observation (WGS 84)", "degrees_north", "Y"
        v_lon = ds.createVariable("lon", "f8", ("obs",), **kw)
        v_lon.standard_name, v_lon.long_name, v_lon.units, v_lon.axis = "longitude", "longitude of the observation (WGS 84)", "degrees_east", "X"
        _crs_variable(ds)
        v_x = ds.createVariable("x_m", "f8", ("obs",), **kw)
        v_y = ds.createVariable("y_m", "f8", ("obs",), **kw)
        for v, std, name in ((v_x, "projection_x_coordinate", "x_m"), (v_y, "projection_y_coordinate", "y_m")):
            v.standard_name, v.long_name, v.units, v.grid_mapping, v.coordinates = std, describe[name].description, "m", "crs", coords
        v_pidx = ds.createVariable("point_index", "i4", ("obs",), **kw)
        v_pidx.long_name, v_pidx.coordinates = describe["point_index"].description, coords
        v_src = ds.createVariable("source_point_index", "i4", ("obs",), fill_value=np.int32(-1), **kw)
        v_src.long_name, v_src.coordinates = describe["source_point_index"].description + " (-1 where null)", coords
        v_disp = {}
        for name, units in (("dx_m", "m"), ("dy_m", "m"), ("speed_m_per_day", "m day-1")):
            v_disp[name] = ds.createVariable(name, "f8", ("obs",), fill_value=np.nan, **kw)
            v_disp[name].long_name, v_disp[name].units, v_disp[name].coordinates = describe[name].description, units, coords
        v_stat = {}
        for name, units in (("selected_matches", "1"), ("support_radius_m", "m"), ("maximum_residual_m", "m")):
            v_stat[name] = ds.createVariable(name, "f4", ("obs",), fill_value=np.float32(np.nan), **kw)
            v_stat[name].long_name, v_stat[name].units, v_stat[name].coordinates = describe[name].description, units, coords
        v_flag = ds.createVariable("quality_flags", "u1", ("obs",), fill_value=np.uint8(FLAG_FILL), **kw)
        v_flag.long_name = describe["quality_flags"].description
        v_flag.flag_masks = FLAG_MASKS
        v_flag.flag_meanings = " ".join(FLAG_BITS)
        v_flag.comment = "; ".join(f"{name}: {text}" for name, text in FLAG_DESCRIPTIONS.items())
        v_flag.coordinates = coords
        v_scene_index = ds.createVariable("scene_index", "i4", ("obs",), **kw)
        v_scene_index.long_name = "index of the Sentinel-1 scene of the observation into scene_name"
        v_scene_index.coordinates = coords
        width = max(len(name) for name in scene_names)
        ds.createDimension("scene_strlen", width)
        v_names = ds.createVariable("scene_name", "S1", ("scene", "scene_strlen"))  # a char array: valid CF in every NetCDF tool
        v_names.long_name = describe["scene"].description
        v_names._Encoding = "ascii"
        v_names[:] = np.array([name.encode("ascii") for name in scene_names], dtype=f"S{width}").view("S1").reshape(len(scene_names), width)

        used = np.zeros(len(scene_names), dtype=bool)
        written = 0
        for bucket in spill.buckets:
            records = np.fromfile(_bucket_path(spill.directory, bucket), dtype=SPILL_DTYPE)
            records = records[np.lexsort((records["ms"], records["rank"]))]
            rank = records["rank"]
            cuts = np.r_[0, np.flatnonzero(np.diff(rank)) + 1]
            lo, hi = int(rank[0]), int(rank[-1]) + 1
            if len(cuts) != hi - lo or not np.array_equal(rank[cuts], np.arange(lo, hi)):
                raise ValueError(f"bucket {bucket} does not hold consecutive whole trajectories")
            sizes = np.diff(np.r_[cuts, len(rank)])
            if not np.array_equal(sizes, index.rowsize[lo:hi]):
                raise ValueError(f"bucket {bucket}: observation counts differ from the index")
            if int(index.offset[lo]) != written:
                raise ValueError(f"bucket {bucket} starts at row {written}, expected {int(index.offset[lo])}")
            pidx = (np.arange(len(rank)) - np.repeat(cuts, sizes)).astype(np.int32)
            is_first = pidx == 0
            if not np.array_equal(is_first, records["flags"] == FLAG_FILL) or (records["ms"][is_first] != index.first_ms[lo:hi]).any():
                raise ValueError(f"bucket {bucket}: first observations are not the first links' sources")
            step = ~is_first
            ms, x, y = records["ms"], records["x"], records["y"]
            # the observation every displacement starts at: same trajectory, time = end time - elapsed, position = end - displacement
            span = int(ms.max() - ms.min()) + 1
            key_all = (rank - lo) * span + (ms - ms.min())
            if np.any(np.diff(key_all) <= 0):
                raise ValueError(f"bucket {bucket}: two observations of a trajectory have the same time")
            rows = np.flatnonzero(step)
            src_ms = ms[rows] - records["el"][rows].astype(np.int64) * 1000
            key_src = (rank[rows] - lo) * span + (src_ms - ms.min())
            found = np.searchsorted(key_all, key_src)
            found = np.minimum(found, len(key_all) - 1)
            if (key_all[found] != key_src).any() or (found >= rows).any() \
                    or (np.abs(x[found] - (x[rows] - records["dx"][rows])) > 1e-6).any() \
                    or (np.abs(y[found] - (y[rows] - records["dy"][rows])) > 1e-6).any():
                raise ValueError(f"bucket {bucket}: a displacement does not start at an earlier observation of its trajectory")
            source_index = np.full(len(rank), -1, dtype=np.int32)
            source_index[rows] = pidx[found]
            if index.chain_rule != "none" and (source_index[rows] != pidx[rows] - 1).any():
                raise ValueError(f"bucket {bucket}: a chain has a displacement that does not start at the previous observation")
            if (flags_bad := (records["flags"][step] & ~np.uint8(KNOWN_FLAG_MASK)) != 0).any():
                raise ValueError(f"bucket {bucket}: {int(flags_bad.sum())} flag values carry unknown bits")
            sl = slice(written, written + len(rank))
            time[sl] = ms / 1000.0
            lon, lat = _lonlat(x, y)
            v_lat[sl], v_lon[sl], v_x[sl], v_y[sl], v_pidx[sl], v_src[sl] = lat, lon, x, y, pidx, source_index
            for name, key in (("dx_m", "dx"), ("dy_m", "dy"), ("speed_m_per_day", "speed")):
                v_disp[name][sl] = records[key]
            for name, key in (("selected_matches", "sel"), ("support_radius_m", "sup"), ("maximum_residual_m", "res")):
                v_stat[name][sl] = records[key]
            v_flag[sl] = records["flags"]
            scene_new = new_of_old[records["scene"]]
            v_scene_index[sl] = scene_new
            used[scene_new] = True
            last_ms[lo:hi] = ms[np.r_[cuts[1:], len(rank)] - 1]
            stats["x_min"], stats["x_max"] = min(stats["x_min"], float(x.min())), max(stats["x_max"], float(x.max()))
            stats["y_min"], stats["y_max"] = min(stats["y_min"], float(y.min())), max(stats["y_max"], float(y.max()))
            stats["lat_min"], stats["lat_max"] = min(stats["lat_min"], float(lat.min())), max(stats["lat_max"], float(lat.max()))
            stats["lon_min"], stats["lon_max"] = min(stats["lon_min"], float(lon.min())), max(stats["lon_max"], float(lon.max()))
            stats["ms_min"], stats["ms_max"] = min(stats["ms_min"], int(ms.min())), max(stats["ms_max"], int(ms.max()))
            f = records["flags"][step]
            status_counts["unflagged"] += int((f == 0).sum())
            for name, bit in FLAG_BITS.items():
                status_counts[name] += int(((f >> bit) & 1).sum())
            written += len(rank)
            del records
        if written != n:
            raise ValueError(f"{written} observations written, expected {n}")
        if not used.all():
            raise ValueError("a scene in the scene list has no observation")  # cannot happen: scenes come from observations
        v_t1[:] = last_ms / 1000.0

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
        ds.product = PRODUCT
        ds.product_version = release_id
        ds.release_id = release_id
        ds.schema_version = SCHEMA_VERSION
        ds.period = period
        ds.time_coverage_start = _utc_text(_ms_to_datetime(stats["ms_min"]))
        ds.time_coverage_end = _utc_text(_ms_to_datetime(stats["ms_max"]))
        ds.geospatial_lat_min, ds.geospatial_lat_max = stats["lat_min"], stats["lat_max"]
        ds.geospatial_lon_min, ds.geospatial_lon_max = stats["lon_min"], stats["lon_max"]
        ds.observations, ds.trajectories, ds.displacements = np.int64(n), np.int64(ntraj), np.int64(n - ntraj)
        ds.chain_rule = ("none: the whole tree of each trajectory is published; source_point_index gives the observation each displacement starts at"
                         if index.chain_rule == "none" else
                         f"one chain per trajectory: the path from its first observation to its {index.chain_rule} observation; observations and "
                         "displacements off that path are not published")
        ds.qc_policy = ("displacements above the hard speed limit are not published; displacements flagged for review are published "
                        "with quality_flags: " + "; ".join(f"{k}: {v}" for k, v in qc.items()))
        ds.history = f"{datetime.now(timezone.utc):%Y-%m-%d}: written by limosat.publish from the primary composition, release {release_id}"
        ds.comment = ("Contiguous ragged array (CF 9.3.3): trajectory i is obs[offset_i : offset_i + rowsize_i] with offset = "
                      "cumsum(rowsize) shifted by one. Trajectories are ordered by time_start. Scene names: scene_name[scene_index]. "
                      "Null displacement values (first observation of a trajectory) are _FillValue. A trajectory does not have "
                      "an observation on every scene: the time between observations varies.")
    finally:
        ds.close()
    return {
        "rows": n, "trajectories": ntraj, "displacements": n - ntraj, "scenes": len(scene_names),
        "time_start": _utc_text(_ms_to_datetime(stats["ms_min"])), "time_end": _utc_text(_ms_to_datetime(stats["ms_max"])),
        "bounds_epsg3413_m": {k: stats[k] for k in ("x_min", "x_max", "y_min", "y_max")},
        "flag_counts": status_counts,
    }


# --------------------------------------------------------------------------------------------- build


def _qc_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Human-readable QC parameters of the composition, for the file's qc_policy attribute."""
    config = manifest.get("qc") or {}
    out = {}
    if "configured_speed_m_per_day" in config:
        out["configured speed"] = f"{config['configured_speed_m_per_day'] / 1000:g} km/day"
    if "hard_speed_m_per_day" in config:
        out["hard speed limit"] = f"{config['hard_speed_m_per_day'] / 1000:g} km/day"
    if "neighbour_search_radius_m" in config:
        out["neighbour radius"] = f"{config['neighbour_search_radius_m'] / 1000:g} km"
    return out


def default_period(index: Index) -> str:
    first = _ms_to_datetime(int(index.first_ms.min()))
    return f"{first:%Y%m}"


def build(source_dir: Path, output_dir: Path, release_id: str, period: str | None = None, overwrite: bool = False,
          verify_source: bool = True, metadata: dict[str, Any] | None = None, validate_output: bool = True,
          sample_modulus: int = 50, tmp_dir: Path | None = None, bucket_observations: int = BUCKET_OBSERVATIONS,
          batch_rows: int = BATCH_ROWS, chain_rule: str = "none") -> dict[str, Any]:
    """Publish a primary composition into one NetCDF file plus manifest.json in the output directory and validate it."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source = inspect_source(source_dir, verify_checksum=verify_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("pass 1: indexing trajectories", flush=True)
    index = index_source(source.extensions, batch_rows, chain_rule)
    print(f"  chain rule {chain_rule}: {index.links:,} links published, {index.off_chain:,} off the chains (not published)", flush=True)
    last = _ms_to_datetime(int(index.first_ms.max()))
    period = period or f"{default_period(index)}_{last:%Y%m}"
    output = output_dir / f"limosat_eloftr_{period}.nc"
    temporary = output.with_suffix(".nc.tmp")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if temporary.exists():
        raise FileExistsError(temporary)
    spill_dir = Path(tmp_dir) if tmp_dir else output_dir.with_name(output_dir.name + ".spill")
    if spill_dir.exists():
        raise FileExistsError(f"{spill_dir} exists; remove it or choose another --tmp-dir")
    print(f"pass 2: spilling {int(index.rowsize.sum()):,} observations of {len(index.rowsize):,} trajectories", flush=True)
    try:
        spill = spill_source(source.extensions, index, spill_dir, bucket_observations, batch_rows)
        print(f"pass 3: writing {len(spill.buckets)} buckets", flush=True)
        entry = write_netcdf(temporary, index, spill, period, release_id, metadata or {}, _qc_summary(source.manifest))
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)
    entry.update({"period": period, "file": output.name, "bytes": output.stat().st_size, "sha256": sha256(output),
                  "links_rejected_not_published": index.rejected, "links_off_chain_not_published": index.off_chain})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "product": PRODUCT,
        "release_id": release_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": FORMAT,
        "crs": CRS,
        "flag_bits": FLAG_BITS,
        "chain_rule": {"rule": chain_rule, "definition": "none: the whole tree of every trajectory is published, source_point_index gives "
                       "the observation each displacement starts at; latest or deepest: one chain per trajectory, the path from the "
                       "first observation to the latest observation or to the observation with the most links above it"},
        "columns": columns_spec(),
        "compression": {"codec": "zlib", "level": COMPRESSION_LEVEL, "shuffle": True, "chunk_observations": CHUNK},
        "layout": {"dimensions": {"trajectory": "trajectories", "obs": "observations", "scene": "Sentinel-1 scene names"},
                   "length_variable": "rowsize", "sample_dimension": "obs", "trajectory_order": "time_start ascending"},
        "files": [entry],
        "totals": {key: entry[key] for key in ("rows", "trajectories", "displacements")},
    }
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    comp = source.manifest
    sidecar = output_dir.with_name(output_dir.name + ".provenance.json")
    _atomic_write(sidecar, json.dumps({
        "release_id": release_id, "source_directory": str(source_dir), "source_extensions_sha256": (((comp.get("products") or {}).get("trajectory_extensions") or {}).get("sha256")),
        "composition_manifest_sha256": sha256(source.manifest_path), "run_id": comp.get("run_id"), "config_sha256": comp.get("config_sha256"),
        "matching_sha256": comp.get("matching_sha256"), "qc_protocol_id": comp.get("qc_protocol_id"), "qc": comp.get("qc"),
        "extensions_rows": (comp.get("counts") or {}).get("trajectory_extensions"),
    }, indent=2, sort_keys=True) + "\n")
    if validate_output:
        validate(source_dir, output_dir, sample_modulus, batch_rows=batch_rows)
    return manifest


# --------------------------------------------------------------------------------------------- validation


def _same(a, b, what: str, name: str) -> None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
        raise ValueError(f"{name}: sampled {what} differ from the source")


def validate(source_dir: Path, output_dir: Path, sample_modulus: int = 50, batch_rows: int = BATCH_ROWS) -> dict[str, Any]:
    """Check a published file against its manifest and, on a sample of trajectories, against the source extensions."""
    netCDF4 = _netcdf4()
    pa, _, pq = _pyarrow()
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    entry, = manifest["files"]
    release_id = manifest["release_id"]
    source = inspect_source(source_dir)
    output = output_dir / entry["file"]
    if sha256(output) != entry["sha256"] or output.stat().st_size != entry["bytes"]:
        raise ValueError(f"{output.name}: checksum or size differs from the manifest")
    d = netCDF4.Dataset(output)
    d.set_auto_maskandscale(False)
    try:
        expected = {"release_id": release_id, "product_version": release_id, "schema_version": SCHEMA_VERSION, "featureType": "trajectory",
                    "product": PRODUCT, "period": entry["period"]}
        for key, value in expected.items():
            if getattr(d, key, None) != value:
                raise ValueError(f"{output.name}: global attribute {key} is {getattr(d, key, None)!r}, expected {value!r}")
        rowsize = d["rowsize"][:].astype(np.int64)
        ntraj, n = len(rowsize), int(rowsize.sum())
        if n != d.dimensions["obs"].size or n != entry["rows"] or ntraj != entry["trajectories"] or ntraj != d.dimensions["trajectory"].size:
            raise ValueError(f"{output.name}: observation or trajectory counts differ from the manifest")
        if not np.array_equal(d["trajectory_id"][:], np.arange(ntraj)):
            raise ValueError(f"{output.name}: trajectory ids are not 0..n-1")
        offset = np.r_[0, np.cumsum(rowsize)]
        pidx = d["point_index"][:]
        if np.any(pidx != np.arange(n) - np.repeat(offset[:-1], rowsize)):
            raise ValueError(f"{output.name}: point_index is not 0..n-1 within every trajectory")
        sp = d["source_point_index"][:]
        step_rows = pidx > 0
        if np.any(sp[step_rows] < 0) or np.any(sp[step_rows] >= pidx[step_rows]) or not (sp[~step_rows] == -1).all():
            raise ValueError(f"{output.name}: source_point_index is not an earlier point_index of the trajectory (and -1 at the first observation)")
        if manifest["chain_rule"]["rule"] != "none" and np.any(sp[step_rows] != pidx[step_rows] - 1):
            raise ValueError(f"{output.name}: a chain file has a displacement that does not start at the previous observation")
        del step_rows
        time = np.rint(d["time"][:] * 1000).astype(np.int64)
        tstart, tend = d["time_start"][:], d["time_end"][:]
        if np.any(np.diff(tstart) < 0) or not np.array_equal(np.rint(tstart * 1000).astype(np.int64), time[offset[:-1]]) \
                or not np.array_equal(np.rint(tend * 1000).astype(np.int64), time[offset[1:] - 1]):
            raise ValueError(f"{output.name}: trajectories are not ordered by start time, or time_start/time_end are wrong")
        within = np.ones(n, dtype=bool)
        within[offset[:-1]] = False
        if np.any(np.diff(time)[within[1:]] <= 0):
            raise ValueError(f"{output.name}: times do not increase along a trajectory")
        if list(d["quality_flags"].flag_masks) != [1 << b for b in FLAG_BITS.values()] or d["quality_flags"].flag_meanings.split() != list(FLAG_BITS):
            raise ValueError(f"{output.name}: flag masks or meanings differ from the schema")
        first_x, first_y = d["x_m"][:][offset[:-1]], d["y_m"][:][offset[:-1]]
        d.set_auto_chartostring(False)
        chars = np.ascontiguousarray(d["scene_name"][:])
        names_all = np.array([row.rstrip(b"\x00").decode("ascii") for row in chars.view(f"S{chars.shape[1]}").reshape(-1)], dtype=object)

        def get(name, rows):
            """Values of one variable at the given observation rows (reads the variable once, keeps nothing)."""
            return d[name][:][rows]

        # Source cross-check on the trajectories whose id key is divisible by the modulus.
        wanted = list(SOURCE_COLUMNS)
        pieces = []
        for batch in pq.ParquetFile(source.extensions).iter_batches(batch_size=batch_rows, columns=wanted):
            table = pa.Table.from_batches([batch])
            accepted = _array(table["accepted"]).to_numpy(zero_copy_only=False).astype(bool)
            key, _ = _id_keys(table["trajectory_id"])
            mask = accepted & ((key % np.uint64(sample_modulus)) == 0)
            if mask.any():
                sub = table.filter(pa.array(mask))
                pieces.append((_id_keys(sub["trajectory_id"])[0], sub))
        if not pieces:
            raise ValueError("the sample selected no trajectories")
        keys = np.concatenate([k for k, _ in pieces])
        table = pa.concat_tables([t for _, t in pieces])
        order = np.lexsort((_epoch_ms_array(table["end_time_utc"]), keys))
        table, keys = table.take(pa.array(order)), keys[order]
        if manifest["chain_rule"]["rule"] != "none":
            # the chain of every sampled trajectory, chosen again from all its links
            on_chain, _ = select_chain(keys, _epoch_ms_array(table["start_time_utc"]), _epoch_ms_array(table["end_time_utc"]), manifest["chain_rule"]["rule"])
            table, keys = table.filter(pa.array(on_chain)), keys[on_chain]
        src = {"start": _epoch_ms_array(table["start_time_utc"]), "end": _epoch_ms_array(table["end_time_utc"])}
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        sizes = np.diff(np.r_[starts, len(keys)])
        # file trajectory of every sampled source trajectory: the one with the same first observation (time, x, y)
        first_ms = src["start"][starts]
        x0, y0 = _floats(table["x0_m"])[starts], _floats(table["y0_m"])[starts]
        file_first = {}
        wanted_times = set(first_ms.tolist())
        candidates = np.flatnonzero(np.isin(time[offset[:-1]], list(wanted_times)))
        for t in candidates:
            file_first[(int(time[offset[t]]), float(first_x[t]), float(first_y[t]))] = int(t)
        mapped = np.array([file_first.get((int(m), float(a), float(b)), -1) for m, a, b in zip(first_ms, x0, y0)])
        if (mapped < 0).any():
            raise ValueError(f"{output.name}: {int((mapped < 0).sum())} sampled trajectories have no matching first observation in the file")
        if len(set(mapped.tolist())) != len(mapped):
            raise ValueError(f"{output.name}: two sampled trajectories map to the same file trajectory")
        if (rowsize[mapped] != sizes + 1).any():
            raise ValueError(f"{output.name}: sampled trajectories have a different number of observations than in the source")
        # rows of the file: first observation then each link, in the source's end-time order
        link_file_row = np.repeat(offset[mapped] + 1, sizes) + (np.arange(len(keys)) - np.repeat(starts, sizes))
        first_file_row = offset[mapped]
        name = output.name
        scene_names = names_all
        for rows in (link_file_row,):
            _same(time[rows], src["end"], "end times", name)
            _same(get("x_m", rows), _floats(table["x1_m"]), "x", name)
            _same(get("y_m", rows), _floats(table["y1_m"]), "y", name)
            _same(get("dx_m", rows), _floats(table["dx_m"]), "dx", name)
            _same(get("dy_m", rows), _floats(table["dy_m"]), "dy", name)
            _same(get("speed_m_per_day", rows), _floats(table["speed_m_per_day"]), "speed", name)
            _same(get("selected_matches", rows), _floats(table["selected_matches"], np.float32), "selected_matches", name)
            _same(get("support_radius_m", rows), _floats(table["support_radius_m"], np.float32), "support_radius_m", name)
            _same(get("maximum_residual_m", rows), _floats(table["maximum_residual_m"], np.float32), "maximum_residual_m", name)
            _same(get("quality_flags", rows), _flags(table), "quality_flags", name)
            _same(scene_names[get("scene_index", rows)], np.array(_array(table["target_image_id"]).to_pylist(), dtype=object), "scenes", name)
        # the observation a displacement starts at (source_point_index) is the source of the link: position, time, scene
        prev_row = np.repeat(offset[mapped], sizes) + sp[link_file_row]
        _same(get("x_m", prev_row), _floats(table["x0_m"]), "source x", name)
        _same(get("y_m", prev_row), _floats(table["y0_m"]), "source y", name)
        _same(time[prev_row], src["start"], "source times", name)
        _same(scene_names[get("scene_index", prev_row)], np.array(_array(table["source_image_id"]).to_pylist(), dtype=object), "source scenes", name)
        # first observations carry no displacement values
        for key in ("dx_m", "dy_m", "speed_m_per_day", "selected_matches", "support_radius_m", "maximum_residual_m"):
            if not np.isnan(get(key, first_file_row)).all():
                raise ValueError(f"{name}: first observations carry {key}")
        if not (get("quality_flags", first_file_row) == FLAG_FILL).all():
            raise ValueError(f"{name}: first observations carry quality flags")
        result = {"status": "passed", "rows": entry["rows"], "sampled_trajectories": int(len(mapped)), "sampled_displacements": int(len(keys)),
                  "sample_modulus": sample_modulus}
        print(json.dumps(result), flush=True)
    finally:
        d.close()
    return result


# --------------------------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True,
                             help=f"primary composition folder holding {EXTENSIONS_FILE} and {COMPOSITION_MANIFEST}")
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--sample-modulus", type=int, default=50, help="check every Nth trajectory against the source (1 checks all)")
        command.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
        if name == "build":
            command.add_argument("--release-id", required=True)
            command.add_argument("--metadata", type=Path, help="JSON with title, summary, keywords, creators, license, acknowledgement, ...")
            command.add_argument("--period", help="name of the period in the file name and title (default: first and last start month)")
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--skip-source-checksum", action="store_true", help="do not hash the extensions table against its manifest")
            command.add_argument("--no-validate", action="store_true", help="skip the validation that normally follows the build")
            command.add_argument("--chain-rule", choices=PUBLISH_RULES, default="none",
                                 help="none (default): publish the whole tree with source_point_index; latest or deepest: publish one "
                                 "chain per trajectory, ending at its latest observation or at the one with the most links")
            command.add_argument("--tmp-dir", type=Path, help="scratch folder for the spilled observations (about 80 bytes per observation)")
            command.add_argument("--bucket-observations", type=int, default=BUCKET_OBSERVATIONS,
                                 help="observations per scratch bucket and so per in-memory sort (about 150 bytes each)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        result = build(args.source, args.output_dir, args.release_id, args.period, args.overwrite, not args.skip_source_checksum, metadata,
                       not args.no_validate, args.sample_modulus, args.tmp_dir, args.bucket_observations, args.batch_rows, args.chain_rule)["totals"]
    else:
        result = validate(args.source, args.output_dir, args.sample_modulus, args.batch_rows)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
