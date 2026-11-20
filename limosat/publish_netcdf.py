"""Write the public LiMOSAT trajectory dataset as CF NetCDF4 (one file per processing period).

``python -m limosat.publish_netcdf release`` is the entry point: internal kept-link Parquet in, validated NetCDF release out
(see :func:`release`). Internally the public point table of :mod:`limosat.publish` is built first, as a private scratch
Parquet folder that is validated against the source and removed afterwards; ``build``/``validate`` below work on that folder.
This module re-encodes the point table, losslessly, as a self-describing CF-1.8 *trajectory* file, using the contiguous ragged array
representation of the CF discrete sampling geometries (section 9.3.3 of the CF conventions):

* dimension ``trajectory`` (one entry per trajectory) and ``obs`` (one entry per observation);
* ``rowsize(trajectory)`` carries ``sample_dimension = "obs"`` and gives the number of observations of each
  trajectory, so trajectory *i* is the slice ``offset[i]:offset[i] + rowsize[i]`` of every ``obs`` variable;
* trajectories are ordered by start time (``time_start``), and the observations of a trajectory by
  ``point_index``, so a date window is a few contiguous runs and one trajectory is a single slice;
* every column of the Parquet table is its own variable; ``lon`` and ``lat`` (CF requires them) are added next to
  the EPSG:3413 ``x_m``/``y_m`` that the tracker works in; the Sentinel-1 scene names are a lookup table
  (``scene_name``) indexed by ``scene_index``.

No value is rounded or packed: floating point variables are the same float64/float32 numbers as in the Parquet
table, times are float64 seconds, and nulls become the variable's ``_FillValue`` (NaN, or -1 / 255 for the integer
flags). ``validate`` re-reads both files and checks every value for equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .publish import (CRS, FLAG_BITS, FLAG_DESCRIPTIONS, OWNERSHIP_POLICY, SCHEMA_VERSION, _atomic_write, _pyarrow)

FORMAT = "NetCDF-4 (HDF5), CF-1.8 trajectory featureType, contiguous ragged array"
CHUNK = 65_536
COMPRESSION_LEVEL = 4
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
DEFAULT_METADATA: dict[str, Any] = {
    "title": "LiMOSAT: Lagrangian sea-ice trajectories from Sentinel-1",
    "summary": ("Trajectories of features on the Arctic sea ice tracked through consecutive Sentinel-1 EW scenes with "
                "the LiMOSAT algorithm. One row (observation) per feature and scene; the displacement from the previous "
                "observation of the trajectory is stored with it."),
    "keywords": "sea ice drift, Lagrangian trajectories, Sentinel-1, SAR, Arctic",
    "institution": "Nansen Environmental and Remote Sensing Centre",
    "source": "Copernicus Sentinel-1 EW GRD scenes, processed with LiMOSAT (https://github.com/nansencenter/limosat)",
}
FLAG_MASKS = np.array([1 << bit for bit in FLAG_BITS.values()], dtype=np.uint8)


def _netcdf4():
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError("Writing NetCDF requires netCDF4 (`pip install netCDF4`).") from error
    return netCDF4


def _pyproj():
    try:
        import pyproj
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError("Computing longitude and latitude requires pyproj (`pip install pyproj`).") from error
    return pyproj


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_numpy(path: Path, name: str, take: np.ndarray | None = None, fill=None) -> np.ndarray:
    """One Parquet column as a numpy array in the requested row order; nulls become ``fill``."""
    _, _, _, pq = _pyarrow()
    column = pq.read_table(path, columns=[name])[name].combine_chunks()
    if fill is not None and column.null_count:
        column = column.fill_null(fill)
    values = column.to_numpy(zero_copy_only=False)
    return values if take is None else values[take]


def _lonlat(x: np.ndarray, y: np.ndarray, block: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    transformer = _pyproj().Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = np.empty(len(x)), np.empty(len(x))
    for start in range(0, len(x), block):
        stop = start + block
        lon[start:stop], lat[start:stop] = transformer.transform(x[start:stop], y[start:stop])
    return lon, lat


def _order(path: Path) -> dict[str, np.ndarray]:
    """Row order of the NetCDF file: trajectories by start time (then id), observations by point_index."""
    _, _, _, pq = _pyarrow()
    keys = pq.read_table(path, columns=["trajectory_id", "point_index", "time"])
    tid = keys["trajectory_id"].to_numpy()
    pidx = keys["point_index"].to_numpy()
    ms = keys["time"].cast(_pyarrow()[0].int64()).to_numpy()
    unique, inverse = np.unique(tid, return_inverse=True)
    start = np.empty(len(unique), dtype=np.int64)
    first = pidx == 0
    if int(first.sum()) != len(unique):
        raise ValueError(f"{path.name}: every trajectory needs exactly one point_index 0")
    start[inverse[first]] = ms[first]
    order = np.lexsort((pidx, unique[inverse], start[inverse]))
    tid_o, pidx_o, ms_o = tid[order], pidx[order], ms[order]
    starts = np.r_[0, np.flatnonzero(np.diff(tid_o)) + 1]
    sizes = np.diff(np.r_[starts, len(tid_o)]).astype(np.int32)
    if len(starts) != len(unique) or np.any(pidx_o != np.arange(len(pidx_o)) - np.repeat(starts, sizes)):
        raise ValueError(f"{path.name}: point_index is not 0, 1, 2, ... within every trajectory")
    return dict(order=order, tid=tid_o, starts=starts, rowsize=sizes, ms=ms_o, pidx=pidx_o)


def _utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def build_period(parquet: Path, entry: dict[str, Any], output: Path, release_id: str, metadata: dict[str, Any],
                 columns: list[dict[str, Any]], overwrite: bool = False) -> dict[str, Any]:
    """Write one period's NetCDF file from its public Parquet file."""
    netCDF4 = _netcdf4()
    _, _, _, pq = _pyarrow()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists (use --overwrite)")
    describe = {c["name"]: c for c in columns}
    meta = pq.ParquetFile(parquet).schema_arrow.metadata or {}
    period = meta[b"limosat.period"].decode()
    window = json.loads(meta[b"limosat.primary_window"])
    o = _order(parquet)
    n, ntraj, order = len(o["tid"]), len(o["starts"]), o["order"]
    if n != entry["rows"] or ntraj != entry["trajectories"]:
        raise ValueError(f"{period}: counts differ from the manifest")
    temporary = output.with_suffix(".nc.tmp")
    temporary.unlink(missing_ok=True)
    kw = dict(zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True, chunksizes=(min(CHUNK, n),))
    kw_traj = dict(zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True, chunksizes=(min(CHUNK, ntraj),))

    scene = pq.read_table(parquet, columns=["scene"], read_dictionary=["scene"])["scene"].unify_dictionaries().combine_chunks()
    scene_names = scene.dictionary.to_pylist()
    scene_index = scene.indices.to_numpy(zero_copy_only=False)[order].astype(np.int32)

    ds = netCDF4.Dataset(temporary, "w", format="NETCDF4")
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
            v.long_name = text
            v.units = TIME_UNITS
            v.calendar = "standard"
            v[:] = o["ms"][idx] / 1000.0
        # ---- observation coordinates
        coords = "time lat lon"
        time = ds.createVariable("time", "f8", ("obs",), **kw)
        time.standard_name = "time"
        time.long_name = describe["time"]["description"]
        time.units = TIME_UNITS
        time.calendar = "standard"
        time.axis = "T"
        time[:] = o["ms"] / 1000.0
        x = _read_numpy(parquet, "x_m", order)
        y = _read_numpy(parquet, "y_m", order)
        lon, lat = _lonlat(x, y)
        for name, values, std, units, axis in (("lat", lat, "latitude", "degrees_north", "Y"),
                                               ("lon", lon, "longitude", "degrees_east", "X")):
            v = ds.createVariable(name, "f8", ("obs",), **kw)
            v.standard_name, v.long_name, v.units, v.axis = std, f"{std} of the observation (WGS 84)", units, axis
            v[:] = values
        _crs_variable(ds)
        for name, values, std, text in (("x_m", x, "projection_x_coordinate", "x"), ("y_m", y, "projection_y_coordinate", "y")):
            v = ds.createVariable(name, "f8", ("obs",), **kw)
            v.standard_name, v.long_name, v.units = std, describe[name]["description"], "m"
            v.grid_mapping = "crs"
            v.coordinates = coords
            v[:] = values
        # ---- observation data variables
        v = ds.createVariable("point_index", "i4", ("obs",), **kw)
        v.long_name = describe["point_index"]["description"]
        v.coordinates = coords
        v[:] = o["pidx"]
        for name, units in (("dx_m", "m"), ("dy_m", "m"), ("speed_m_per_day", "m day-1")):
            v = ds.createVariable(name, "f8", ("obs",), fill_value=np.nan, **kw)
            v.long_name, v.units, v.coordinates = describe[name]["description"], units, coords
            v[:] = _read_numpy(parquet, name, order, fill=np.nan)
        for name in ("correlation", "direct_match_share"):
            v = ds.createVariable(name, "f4", ("obs",), fill_value=np.float32(np.nan), **kw)
            v.long_name, v.units, v.coordinates = describe[name]["description"], "1", coords
            v[:] = _read_numpy(parquet, name, order, fill=np.float32(np.nan))
        v = ds.createVariable("interpolated", "i1", ("obs",), fill_value=np.int8(-1), **kw)
        v.long_name = describe["interpolated"]["description"] + " (-1 where null)"
        v.flag_values = np.array([0, 1], dtype=np.int8)
        v.flag_meanings = "matched interpolated"
        v.coordinates = coords
        raw = _read_numpy(parquet, "interpolated", order, fill=False)
        isnull = pq.read_table(parquet, columns=["interpolated"])["interpolated"].is_null().to_numpy(zero_copy_only=False)[order]
        v[:] = np.where(isnull, -1, raw.astype(np.int8)).astype(np.int8)
        v = ds.createVariable("quality_flags", "u1", ("obs",), fill_value=np.uint8(255), **kw)
        v.long_name = describe["quality_flags"]["description"]
        v.flag_masks = FLAG_MASKS
        v.flag_meanings = " ".join(FLAG_BITS)
        v.comment = "; ".join(f"{name}: {text}" for name, text in FLAG_DESCRIPTIONS.items())
        v.coordinates = coords
        v[:] = _read_numpy(parquet, "quality_flags", order, fill=255).astype(np.uint8)
        v = ds.createVariable("is_primary", "i1", ("obs",), **kw)
        v.long_name = describe["is_primary"]["description"]
        v.flag_values = np.array([0, 1], dtype=np.int8)
        v.flag_meanings = "not_primary primary"
        v.coordinates = coords
        v[:] = _read_numpy(parquet, "is_primary", order).astype(np.int8)
        v = ds.createVariable("scene_index", "i4", ("obs",), **kw)
        v.long_name = "index of the Sentinel-1 scene of the observation into scene_name"
        v.coordinates = coords
        v[:] = scene_index
        v = ds.createVariable("scene_name", str, ("scene",))
        v.long_name = describe["scene"]["description"]
        v[:] = np.array(scene_names, dtype=object)

        # ---- global attributes
        a = {**DEFAULT_METADATA, **metadata}
        creators = a.pop("creators", [])
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.featureType = "trajectory"
        ds.title = f"{a.pop('title')} ({period})"
        ds.summary = a.pop("summary")
        ds.keywords = a.pop("keywords")
        ds.institution = a.pop("institution")
        ds.source = a.pop("source")
        if creators:
            ds.creator_name = "; ".join(c["name"] for c in creators)
            ds.creator_institution = "; ".join(dict.fromkeys(c.get("affiliation", "") for c in creators if c.get("affiliation")))
            ds.creator_id = "; ".join(f"https://orcid.org/{c['orcid']}" for c in creators if c.get("orcid"))
        for key, value in a.items():
            ds.setncattr(key, value)
        ds.product_version = release_id
        ds.release_id = release_id
        ds.schema_version = SCHEMA_VERSION
        ds.period = period
        ds.primary_window_start = window["start"]
        ds.primary_window_stop_exclusive = window["stop_exclusive"]
        ds.primary_rule = OWNERSHIP_POLICY
        ds.time_coverage_start = _utc(o["ms"].min() / 1000.0)
        ds.time_coverage_end = _utc(o["ms"].max() / 1000.0)
        ds.geospatial_lat_min, ds.geospatial_lat_max = float(lat.min()), float(lat.max())
        ds.geospatial_lon_min, ds.geospatial_lon_max = float(lon.min()), float(lon.max())
        ds.observations, ds.trajectories = np.int64(n), np.int64(ntraj)
        ds.displacements = np.int64(entry["displacements"])
        ds.primary_displacements = np.int64(entry["primary_displacements"])
        ds.history = f"{datetime.now(timezone.utc):%Y-%m-%d}: written by limosat.publish_netcdf from the validated Parquet release {release_id}"
        ds.comment = ("Contiguous ragged array (CF 9.3.3): trajectory i is obs[offset_i : offset_i + rowsize_i] with offset = "
                      "cumsum(rowsize) shifted by one. Trajectories are ordered by time_start. Scene names: "
                      "scene_name[scene_index]. Null displacement values (first observation of a trajectory) are _FillValue.")
    finally:
        ds.close()
    os.replace(temporary, output)
    result = dict(entry)
    result.update(file=output.name, bytes=output.stat().st_size, sha256=_sha256(output))
    return result


def build(parquet_dir: Path, output_dir: Path, metadata: dict[str, Any] | None = None, periods: list[str] | None = None,
          overwrite: bool = False) -> dict[str, Any]:
    """Write every (or the selected) period of the public Parquet release as NetCDF, plus its manifest."""
    parquet_dir, output_dir = Path(parquet_dir), Path(output_dir)
    source = json.loads((parquet_dir / "manifest.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in source["files"]:
        if periods and entry["period"] not in periods:
            continue
        output = output_dir / f"limosat_{entry['period']}.nc"
        entries.append(build_period(parquet_dir / entry["file"], entry, output, source["release_id"], metadata or {},
                                    source["columns"], overwrite))
        print(f"{entry['period']}: {entries[-1]['bytes'] / 1e6:.0f} MB", flush=True)
    manifest = {k: v for k, v in source.items() if k not in ("files", "totals", "created_utc")}
    manifest.update(
        format=FORMAT,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        compression={"codec": "zlib", "level": COMPRESSION_LEVEL, "shuffle": True, "chunk_observations": CHUNK},
        layout={"dimensions": {"trajectory": "trajectories", "obs": "observations", "scene": "Sentinel-1 scene names"},
                "length_variable": "rowsize", "sample_dimension": "obs", "trajectory_order": "time_start ascending"},
        files=entries,
        totals={key: sum(e[key] for e in entries) for key in ("rows", "trajectories", "displacements", "primary_rows", "primary_displacements")},
    )
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def release(source_root: Path, catalog_dir: Path, output_dir: Path, release_id: str, metadata: dict[str, Any] | None = None,
            periods: list[str] | None = None, overwrite: bool = False, verify_source: bool = True, keep_parquet: bool = False,
            sample_modulus: int = 50) -> dict[str, Any]:
    """The whole public release from an internal QC release: one call, NetCDF files out.

    The internal kept-link Parquet is first turned into the public point table (a private Parquet scratch folder next to the
    output, built month by month and validated against the source), which is then written as the CF NetCDF files and
    validated against it. The scratch folder is removed unless ``keep_parquet``. Internal provenance (source paths,
    checksums, protocol ids) goes to ``<output>.provenance.json`` beside the output folder, never into it.
    """
    from . import publish

    source_root, catalog_dir, output_dir = Path(source_root), Path(catalog_dir), Path(output_dir)
    work = output_dir.with_name(f".{output_dir.name}_parquet_work")
    if work.exists():
        shutil.rmtree(work)
    publish.build(source_root, catalog_dir, work, release_id, periods, overwrite=True, verify_source=verify_source)
    publish.validate(source_root, catalog_dir, work, sample_modulus)
    manifest = build(work, output_dir, metadata, periods, overwrite)
    validate(work, output_dir, sample_modulus)
    sidecar = work.with_name(work.name + ".provenance.json")
    os.replace(sidecar, output_dir.with_name(output_dir.name + ".provenance.json"))
    if not keep_parquet:
        shutil.rmtree(work)
    return manifest


# --------------------------------------------------------------------------------------------- validation


def _same(a: np.ndarray, b: np.ndarray, what: str, period: str) -> None:
    if a.dtype.kind == "f":
        ok = np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
    else:
        ok = np.array_equal(a, b)
    if not ok:
        raise AssertionError(f"{period}: {what} differs between the NetCDF and the Parquet file")


def validate_period(parquet: Path, nc: Path, entry: dict[str, Any], sample_modulus: int = 50) -> dict[str, Any]:
    """Compare a NetCDF file with its Parquet source value by value."""
    netCDF4 = _netcdf4()
    pa, _, _, pq = _pyarrow()
    period = entry["period"]
    if _sha256(nc) != entry["sha256"] or nc.stat().st_size != entry["bytes"]:
        raise AssertionError(f"{period}: the NetCDF file differs from its manifest checksum")
    d = netCDF4.Dataset(nc)
    d.set_auto_maskandscale(False)
    try:
        rowsize = d["rowsize"][:].astype(np.int64)
        n, ntraj = int(rowsize.sum()), len(rowsize)
        if n != d.dimensions["obs"].size or n != entry["rows"] or ntraj != entry["trajectories"]:
            raise AssertionError(f"{period}: observation or trajectory counts differ")
        offset = np.r_[0, np.cumsum(rowsize)]
        tid = np.repeat(d["trajectory_id"][:], rowsize)
        pidx = d["point_index"][:]
        if len(np.unique(d["trajectory_id"][:])) != ntraj:
            raise AssertionError(f"{period}: trajectory ids are not unique")
        if np.any(pidx != np.arange(n) - np.repeat(offset[:-1], rowsize)):
            raise AssertionError(f"{period}: point_index is not 0, 1, ... within every trajectory")
        tstart, tend, time = d["time_start"][:], d["time_end"][:], d["time"][:]
        if np.any(np.diff(tstart) < 0) or not np.array_equal(time[offset[:-1]], tstart) or not np.array_equal(time[offset[1:] - 1], tend):
            raise AssertionError(f"{period}: trajectories are not ordered by start time or time_start/time_end are wrong")
        # align the Parquet rows with the NetCDF rows through the (trajectory_id, point_index) key
        keys = pq.read_table(parquet, columns=["trajectory_id", "point_index"])
        ptid, ppidx = keys["trajectory_id"].to_numpy(), keys["point_index"].to_numpy()
        take = np.lexsort((ppidx, ptid))                  # parquet rows in (tid, pidx) order
        nc_take = np.lexsort((pidx, tid))                 # netcdf rows in (tid, pidx) order
        _same(ptid[take], tid[nc_take], "trajectory ids", period)
        _same(ppidx[take], pidx[nc_take], "point_index", period)
        del keys, ptid, ppidx

        def nc_column(name):
            return d[name][:][nc_take]

        pq_ms = pq.read_table(parquet, columns=["time"])["time"].cast(pa.int64()).to_numpy()[take]
        _same(np.rint(time[nc_take] * 1000).astype(np.int64), pq_ms, "time", period)
        for name in ("x_m", "y_m"):
            _same(nc_column(name), _read_numpy(parquet, name, take), name, period)
        for name in ("dx_m", "dy_m", "speed_m_per_day"):
            _same(nc_column(name), _read_numpy(parquet, name, take, fill=np.nan), name, period)
        for name in ("correlation", "direct_match_share"):
            _same(nc_column(name), _read_numpy(parquet, name, take, fill=np.float32(np.nan)), name, period)
        _same(nc_column("quality_flags"), _read_numpy(parquet, "quality_flags", take, fill=255).astype(np.uint8), "quality_flags", period)
        _same(nc_column("is_primary"), _read_numpy(parquet, "is_primary", take).astype(np.int8), "is_primary", period)
        isnull = pq.read_table(parquet, columns=["interpolated"])["interpolated"].is_null().to_numpy(zero_copy_only=False)[take]
        raw = _read_numpy(parquet, "interpolated", take, fill=False).astype(np.int8)
        _same(nc_column("interpolated"), np.where(isnull, -1, raw).astype(np.int8), "interpolated", period)
        # scene names: the pairing of netcdf and parquet scene codes must be one to one and the names equal
        scene = pq.read_table(parquet, columns=["scene"], read_dictionary=["scene"])["scene"].unify_dictionaries().combine_chunks()
        p_codes = scene.indices.to_numpy(zero_copy_only=False)[take].astype(np.int64)
        n_codes = nc_column("scene_index").astype(np.int64)
        p_names, n_names = scene.dictionary.to_pylist(), list(d["scene_name"][:])
        pair = np.unique(n_codes * len(p_names) + p_codes)
        if len(pair) != len(np.unique(n_codes)) or len(pair) != len(np.unique(p_codes)):
            raise AssertionError(f"{period}: scene codes do not pair one to one")
        if any(n_names[int(c // len(p_names))] != p_names[int(c % len(p_names))] for c in pair):
            raise AssertionError(f"{period}: scene names differ")
        # longitude and latitude agree with the projected positions
        sel = np.arange(0, n, max(sample_modulus, 1))
        back = _pyproj().Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        bx, by = back.transform(d["lon"][:][sel], d["lat"][:][sel])
        err = float(np.max(np.hypot(bx - d["x_m"][:][sel], by - d["y_m"][:][sel])))
        if err > 1e-3:
            raise AssertionError(f"{period}: lon/lat disagree with x/y by {err:.2e} m")
    finally:
        d.close()
    return {"period": period, "observations": n, "trajectories": ntraj, "status": "passed", "lonlat_max_error_m": err}


def validate(parquet_dir: Path, output_dir: Path, sample_modulus: int = 50) -> list[dict[str, Any]]:
    parquet_dir, output_dir = Path(parquet_dir), Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    source = {e["period"]: e for e in json.loads((parquet_dir / "manifest.json").read_text())["files"]}
    results = []
    for entry in manifest["files"]:
        results.append(validate_period(parquet_dir / source[entry["period"]]["file"], output_dir / entry["file"], entry, sample_modulus))
        print(json.dumps(results[-1]), flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parquet-dir", type=Path, required=True, help="the validated public Parquet release")
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--metadata", type=Path, help="JSON with title, summary, keywords, creators, license, acknowledgement, ...")
    b.add_argument("--period", action="append", dest="periods")
    b.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("release", help="internal QC release -> validated NetCDF release in one call")
    r.add_argument("--source-root", type=Path, required=True, help="internal release folder holding <period>/<period>_qc.parquet")
    r.add_argument("--catalog-dir", type=Path, required=True, help="folder of <period>_image_catalog_*.json")
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--release-id", required=True)
    r.add_argument("--metadata", type=Path)
    r.add_argument("--period", action="append", dest="periods")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--skip-source-checksum", action="store_true")
    r.add_argument("--keep-parquet", action="store_true", help="keep the private Parquet scratch folder")
    r.add_argument("--sample-modulus", type=int, default=50)
    v = sub.add_parser("validate")
    v.add_argument("--parquet-dir", type=Path, required=True)
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--sample-modulus", type=int, default=50)
    args = parser.parse_args(argv)
    if args.command == "release":
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        result = release(args.source_root, args.catalog_dir, args.output_dir, args.release_id, metadata, args.periods, args.overwrite,
                         not args.skip_source_checksum, args.keep_parquet, args.sample_modulus)
        print(json.dumps(result["totals"], indent=2))
    elif args.command == "build":
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        result = build(args.parquet_dir, args.output_dir, metadata, args.periods, args.overwrite)
        print(json.dumps(result["totals"], indent=2))
    else:
        validate(args.parquet_dir, args.output_dir, args.sample_modulus)


if __name__ == "__main__":
    main()
