"""Trajectory-link QC v3: isolation (R1) and land exclusions plus quality flags on a v2 release.

v3 keeps every v2 decision. It reads the v2 scientific edge Parquet (accepted links only), scores each
link against the other accepted links of the same exact image pair with the frozen v2 local-affine
neighbour model, and then

- excludes isolated direct-match outliers (reason ``R1``) and links with an endpoint on land
  (reason ``LAND``), splitting trajectories at every excluded link as v2 does;
- flags, without splitting, isolated interpolated links, interpolated links far from their neighbours,
  links in incoherent image pairs, links over open water and links in the coastal band.

Thresholds are in ``protocols/trajectory_link_qc_v3.json``. Coordinates are EPSG:3413 metres.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from .core import QCConfig, _local_prediction, validate_frozen_protocol


PROTOCOL_V3_ID = "limosat_trajectory_link_qc_v3_20261024"
PROTOCOL_V3_PATH = Path(__file__).resolve().parent / "protocols" / "trajectory_link_qc_v3.json"

FLAG_BITS = {
    "ISOLATED_INTERP": 0,
    "INTERP_RESIDUAL": 1,
    "PAIR_INCOHERENT": 2,
    "OPEN_WATER": 3,
    "COASTAL": 4,
}
EXCLUSION_CODES = ("R1", "LAND")


@dataclass(frozen=True)
class V3Config:
    """Frozen v3 thresholds (metres, kilometres for land distance, SIC as a fraction)."""

    residual_floor_m: float = 2_000.0
    coherence_min: float = 5.0
    spread_floor_m: float = 150.0
    land_margin_km: float = 2.0
    incoherent_min_links: int = 20
    incoherent_fraction: float = 0.5
    incoherent_residual_m: float = 2_000.0
    open_water_sic: float = 0.15

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not 0 < self.incoherent_fraction <= 1:
            raise ValueError("incoherent_fraction must be in (0, 1]")
        if not 0 < self.open_water_sic < 1:
            raise ValueError("open_water_sic must be in (0, 1)")


def load_v3_protocol() -> dict:
    protocol = json.loads(PROTOCOL_V3_PATH.read_text())
    if protocol.get("protocol_id") != PROTOCOL_V3_ID:
        raise ValueError("Bundled v3 protocol ID does not match the code")
    return protocol


def validate_v3_protocol(config: V3Config) -> None:
    """Fail if v3 code defaults and the bundled protocol have drifted apart."""

    parameters = load_v3_protocol()["parameters"]
    mismatches = {
        field: (value, parameters.get(field))
        for field, value in asdict(config).items()
        if value != parameters.get(field)
    }
    if mismatches:
        raise ValueError(f"v3 configuration does not match frozen protocol: {mismatches}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- scoring


def score_pair_isolation(
    xy: np.ndarray, uv: np.ndarray, config: QCConfig, residual_floor_m: float
) -> dict[str, np.ndarray]:
    """Leave-one-out isolation diagnostics for every link of one image pair.

    ``neighbour_spread_m`` is the median distance of the neighbour displacements from their own
    median; a neighbour corroborates a link when it is closer to the link's displacement than to the
    neighbour median and closer than half the link's coarse residual. The v2 local-affine prediction
    is evaluated wherever its residual could reach ``residual_floor_m`` (coarse residual plus the
    largest neighbour deviation times the prediction-weight bound), so skipped links provably lie
    below the floor.
    """

    n = len(xy)
    out = {key: np.full(n, np.nan) for key in (
        "coarse_residual_m", "neighbour_spread_m", "n_corroborating", "n_neighbours",
        "local_u_m", "local_v_m")}
    out["local_supported"] = np.zeros(n, bool)
    out["evaluable"] = np.zeros(n, bool)
    if n < config.min_neighbors + 1:
        out["local_residual_m"] = np.full(n, np.nan)
        return out
    k = min(config.max_neighbors + 1, n)
    dist, idx = cKDTree(xy).query(xy, k=k, distance_upper_bound=config.search_radius_m)
    valid = np.isfinite(dist) & (idx < n) & (idx != np.arange(n)[:, None])
    nuv = uv[np.where(valid, idx, 0)].astype(float)
    nuv[~valid] = np.nan
    count = valid.sum(1)
    enough = count >= config.min_neighbors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(nuv, axis=1)
        deviation = np.linalg.norm(nuv - median[:, None, :], axis=2)
        spread = np.nanmedian(deviation, axis=1)
        max_deviation = np.nanmax(np.where(valid, deviation, 0.0), axis=1)
        to_self = np.linalg.norm(nuv - uv[:, None, :], axis=2)
    coarse = np.hypot(*(uv - median).T)
    corroborating = (to_self < deviation) & (to_self < 0.5 * coarse[:, None]) & valid
    out["coarse_residual_m"] = np.where(enough, coarse, np.nan)
    out["neighbour_spread_m"] = np.where(enough, spread, np.nan)
    out["n_corroborating"] = np.where(enough, corroborating.sum(1), np.nan)
    out["n_neighbours"] = count.astype(float)
    out["evaluable"] = enough
    bound = coarse + config.max_prediction_weight_l1 * max_deviation
    for i in np.flatnonzero(enough & (bound >= residual_floor_m * (1 - 1e-10))):
        v = valid[i]
        d = _local_prediction(xy[i], xy[idx[i, v]], uv[idx[i, v]], dist[i, v], config)
        out["local_u_m"][i], out["local_v_m"][i] = d["local_u_m"], d["local_v_m"]
        out["local_supported"][i] = d["local_supported"]
    out["local_residual_m"] = np.hypot(uv[:, 0] - out["local_u_m"], uv[:, 1] - out["local_v_m"])
    return out


def score_links(links: pd.DataFrame, qc_config: QCConfig, v3: V3Config) -> pd.DataFrame:
    """Add isolation diagnostics, R1/R1i decisions and the pair-incoherence flag (one pass)."""

    result = links.reset_index(drop=True).copy()
    columns = ("coarse_residual_m", "neighbour_spread_m", "n_corroborating", "n_neighbours",
               "local_u_m", "local_v_m", "local_residual_m", "local_supported", "evaluable")
    buffers = {c: np.full(len(result), np.nan) for c in columns}
    buffers["local_supported"] = np.zeros(len(result), bool)
    buffers["evaluable"] = np.zeros(len(result), bool)
    for _, index in result.groupby(["source_image_id", "target_image_id"], sort=False).indices.items():
        xy = result[["x0_m", "y0_m"]].to_numpy(float)[index]
        uv = result[["dx_m", "dy_m"]].to_numpy(float)[index]
        scored = score_pair_isolation(xy, uv, qc_config, v3.residual_floor_m)
        for column in columns:
            buffers[column][index] = scored[column]
    for column, values in buffers.items():
        result[column] = values
    result["coherence_ratio"] = result.local_residual_m / np.maximum(result.neighbour_spread_m, v3.spread_floor_m)
    direct = (result.source_interpolated == 0) & (result.target_interpolated == 0)
    isolated = (
        result.local_supported
        & (result.local_residual_m > v3.residual_floor_m)
        & (result.coherence_ratio >= v3.coherence_min)
        & (result.n_corroborating == 0)
    )
    result["r1"] = isolated & direct
    result["r1_interpolated"] = isolated & ~direct
    result["interp_residual"] = ~direct & (result.local_residual_m > v3.residual_floor_m)
    far = result.coarse_residual_m > v3.incoherent_residual_m
    pair = result.groupby(["source_image_id", "target_image_id"], sort=False)
    links_in_pair = pair.edge_uid.transform("size")
    far_fraction = far.groupby([result.source_image_id, result.target_image_id], sort=False).transform("mean")
    result["pair_links"] = links_in_pair
    result["pair_far_fraction"] = far_fraction
    result["pair_incoherent"] = (links_in_pair >= v3.incoherent_min_links) & (far_fraction >= v3.incoherent_fraction)
    return result


# --------------------------------------------------------------------------- external inputs


class LandRaster:
    """Inland distance (km) on a regular EPSG:3413 grid, from an ``.npz`` with ``inland_km``,
    ``x0`` (west edge), ``y1`` (north edge) and ``res`` (metres)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        data = np.load(self.path)
        self.inland_km = data["inland_km"]
        self.x0, self.y1, self.res = float(data["x0"]), float(data["y1"]), float(data["res"])
        self.sha256 = sha256(self.path)

    def lookup(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        row = np.floor((self.y1 - np.asarray(y)) / self.res).astype(np.int64)
        ok = (col >= 0) & (col < self.inland_km.shape[1]) & (row >= 0) & (row < self.inland_km.shape[0])
        out = np.full(len(col), np.nan, dtype="float32")
        out[ok] = self.inland_km[row[ok], col[ok]]
        return out


class SicSampler:
    """Nearest-cell daily SIC (fraction) from NOAA/NSIDC CDR v6 north files
    (``<root>/north/daily/<YYYY>/sic_psn25_<YYYYMMDD>_*_v06r00.nc``, EPSG:3411 grid)."""

    def __init__(self, root: Path):
        from pyproj import Transformer

        self.root = Path(root)
        self.transform = Transformer.from_crs("EPSG:3413", "EPSG:3411", always_xy=True).transform
        self.cache: dict[str, tuple] = {}
        self.files: dict[str, str] = {}

    def _grid(self, day: pd.Timestamp):
        key = day.strftime("%Y%m%d")
        if key not in self.cache:
            import xarray as xr

            matches = sorted((self.root / "north" / "daily" / day.strftime("%Y")).glob(f"sic_psn25_{key}_*_v06r00.nc"))
            if not matches:
                self.cache[key] = None
            else:
                with xr.open_dataset(matches[-1]) as data:
                    self.cache[key] = (data["x"].values.astype(float), data["y"].values.astype(float),
                                       data["cdr_seaice_conc"].isel(time=0).values.astype("float32"))
                self.files[key] = matches[-1].name
            if len(self.cache) > 40:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def sample(self, x: np.ndarray, y: np.ndarray, times: pd.Series) -> np.ndarray:
        px, py = self.transform(np.asarray(x, float), np.asarray(y, float))
        px, py = np.asarray(px), np.asarray(py)
        out = np.full(len(px), np.nan, dtype="float32")
        days = pd.to_datetime(times, utc=True).dt.floor("D")
        for day, index in pd.Series(np.arange(len(px))).groupby(days.to_numpy()).groups.items():
            grid = self._grid(pd.Timestamp(day))
            if grid is None:
                continue
            gx, gy, sic = grid
            index = np.asarray(index)
            col = np.rint((px[index] - gx[0]) / (gx[1] - gx[0])).astype(int)
            row = np.rint((py[index] - gy[0]) / (gy[1] - gy[0])).astype(int)
            ok = (col >= 0) & (col < len(gx)) & (row >= 0) & (row < len(gy))
            values = np.full(len(index), np.nan, dtype="float32")
            values[ok] = sic[row[ok], col[ok]]
            values[(values < 0) | (values > 1)] = np.nan  # flags (land, coast, missing) are > 1
            out[index] = values
        return out


# --------------------------------------------------------------------------- decisions


def decide(scored: pd.DataFrame, v3: V3Config, land: LandRaster, sic: SicSampler | None) -> pd.DataFrame:
    """Exclusion reasons and quality flags for scored links."""

    result = scored
    result["inland_start_km"] = land.lookup(result.x0_m.to_numpy(), result.y0_m.to_numpy())
    result["inland_end_km"] = land.lookup(result.x1_m.to_numpy(), result.y1_m.to_numpy())
    inland = np.fmax(result.inland_start_km, result.inland_end_km)
    result["land"] = inland >= v3.land_margin_km
    result["coastal"] = (inland > 0) & ~result.land
    if sic is not None:
        result["sic_start"] = sic.sample(result.x0_m, result.y0_m, result.start_time_utc)
        result["sic_end"] = sic.sample(result.x1_m, result.y1_m, result.end_time_utc)
        result["open_water"] = (result.sic_start < v3.open_water_sic) & (result.sic_end < v3.open_water_sic)
    else:
        result["sic_start"] = np.float32(np.nan)
        result["sic_end"] = np.float32(np.nan)
        result["open_water"] = False

    reason = pd.Series("", index=result.index, dtype="string")
    reason += np.where(result.r1, "R1+", "")
    reason += np.where(result.land, "LAND+", "")
    result["qc3_exclusion"] = reason.str.rstrip("+")
    flags = np.zeros(len(result), dtype=np.uint16)
    names = pd.Series("", index=result.index, dtype="string")
    for column, name in (("r1_interpolated", "ISOLATED_INTERP"), ("interp_residual", "INTERP_RESIDUAL"),
                         ("pair_incoherent", "PAIR_INCOHERENT"), ("open_water", "OPEN_WATER"),
                         ("coastal", "COASTAL")):
        hit = result[column].to_numpy(bool)
        flags |= (hit.astype(np.uint16) << FLAG_BITS[name])
        names += np.where(hit, name + ";", "")
    result["qc3_flags"] = flags
    result["qc3_flag_reasons"] = names.str.rstrip(";")
    return result


def reason_text(row) -> str:
    parts = []
    if "R1" in row.qc3_exclusion:
        parts.append(
            f"R1 isolated direct-match outlier: residual {row.local_residual_m:.0f} m vs neighbour spread "
            f"{max(row.neighbour_spread_m, 150.0):.0f} m (ratio {row.coherence_ratio:.1f}), "
            f"0 corroborating neighbours of {int(row.n_neighbours)}")
    if "LAND" in row.qc3_exclusion:
        parts.append(f"LAND endpoint {max(row.inland_start_km, row.inland_end_km):.1f} km inside land")
    return "; ".join(parts)


def assign_segments(kept: pd.DataFrame, excluded: pd.DataFrame, base_id: int) -> pd.DataFrame:
    """Split v2 trajectories at excluded links (v2 split semantics).

    A kept link's segment index is the number of excluded links of the same v2 trajectory that start
    before it. Segment 0 keeps the v2 ``qc_trajectory_id``; segment ``k`` of trajectory ``t`` gets
    ``base_id`` plus the rank of ``(t, k)`` among all excluded links ordered by trajectory and time,
    which is deterministic and independent of how the input was partitioned.
    """

    order = excluded[["qc_trajectory_id", "start_time_utc"]].sort_values(
        ["qc_trajectory_id", "start_time_utc"], kind="mergesort").reset_index(drop=True)
    order["k"] = order.groupby("qc_trajectory_id").cumcount() + 1
    order["new_id"] = base_id + np.arange(len(order), dtype=np.int64)
    result = kept.copy()
    result["qc3_segment"] = 0
    result["qc3_trajectory_id"] = result["qc_trajectory_id"].astype(np.int64)
    affected = result.qc_trajectory_id.isin(order.qc_trajectory_id)
    if affected.any():
        times = {t: g.start_time_utc.to_numpy() for t, g in order.groupby("qc_trajectory_id")}
        ids = {t: g.new_id.to_numpy() for t, g in order.groupby("qc_trajectory_id")}
        sub = result.loc[affected, ["qc_trajectory_id", "start_time_utc"]]
        seg = np.zeros(len(sub), dtype=np.int64)
        new = sub.qc_trajectory_id.to_numpy().copy()
        for t, index in sub.groupby("qc_trajectory_id").indices.items():
            k = np.searchsorted(times[t], sub.start_time_utc.to_numpy()[index], side="left")
            seg[index] = k
            new[index] = np.where(k > 0, ids[t][np.maximum(k - 1, 0)], t)
        result.loc[affected, "qc3_segment"] = seg
        result.loc[affected, "qc3_trajectory_id"] = new
    return result


def continuity_violations(links: pd.DataFrame) -> int:
    """Links whose source is not the previous link's target within the same v3 trajectory."""

    ordered = links[["qc3_trajectory_id", "start_time_utc", "source_rowid", "target_rowid"]].sort_values(
        ["qc3_trajectory_id", "start_time_utc"], kind="mergesort")
    same = ordered.qc3_trajectory_id.to_numpy()[1:] == ordered.qc3_trajectory_id.to_numpy()[:-1]
    joined = ordered.source_rowid.to_numpy()[1:] == ordered.target_rowid.to_numpy()[:-1]
    return int((same & ~joined).sum())


# --------------------------------------------------------------------------- archive runner

DIAGNOSTICS = ["coarse_residual_m", "neighbour_spread_m", "n_corroborating", "n_neighbours", "local_residual_m",
               "coherence_ratio", "local_supported", "pair_links", "pair_far_fraction", "inland_start_km",
               "inland_end_km", "sic_start", "sic_end"]


def months_in(dataset: ds.Dataset) -> list[tuple[int, int]]:
    table = dataset.to_table(columns=["start_year", "start_month"]).to_pandas().drop_duplicates()
    return sorted((int(y), int(m)) for y, m in table.itertuples(index=False))


def run(input_path: Path, output_dir: Path, land_path: Path, sic_root: Path | None, period: str) -> dict:
    qc_config, v3 = QCConfig(), V3Config()
    qc_config.validate()
    v3.validate()
    validate_frozen_protocol(qc_config)
    validate_v3_protocol(v3)
    land = LandRaster(land_path)
    sic = SicSampler(sic_root) if sic_root else None
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(input_path)
    schema = dataset.schema
    for year, month in months_in(dataset):
        tag = f"{year}-{month:02d}"
        if (work / f"{tag}.done").exists():
            continue
        links = dataset.to_table(filter=(ds.field("start_year") == year) & (ds.field("start_month") == month)).to_pandas()
        decided = decide(score_links(links, qc_config, v3), v3, land, sic)
        decided.to_parquet(work / f"{tag}.scored.parquet", index=False)
        (work / f"{tag}.done").write_text("ok")
        print(json.dumps({"period": period, "month": tag, "links": len(decided),
                          "excluded": int((decided.qc3_exclusion != "").sum())}), flush=True)

    # finalise: exclusion table, segment ids, outputs, checks
    parts = sorted(work.glob("*.scored.parquet"))
    keep_cols = [f.name for f in schema]
    excluded = pd.concat([pd.read_parquet(p, filters=[("qc3_exclusion", "!=", "")]) for p in parts], ignore_index=True)
    excluded["reason_text"] = [reason_text(r) for r in excluded.itertuples(index=False)]
    excluded_out = excluded[["edge_uid", "qc_trajectory_id", "original_trajectory_id", "source_rowid", "target_rowid",
                             "source_image_id", "target_image_id", "start_time_utc", "end_time_utc", "x0_m", "y0_m",
                             "x1_m", "y1_m", "dx_m", "dy_m", "source_interpolated", "target_interpolated",
                             "qc3_exclusion", "reason_text", *DIAGNOSTICS]]
    excluded_out.to_parquet(output_dir / f"{period}_qc3_exclusions.parquet", index=False)
    base_id = int(pc_max(input_path, "qc_trajectory_id")) + 1
    writer, counts, kept_total, input_total = None, {}, 0, 0
    flag_counts = {name: 0 for name in FLAG_BITS}
    violations = 0
    out_path = output_dir / f"{period}_qc3.parquet"
    tmp_path = out_path.with_suffix(".parquet.part")
    for p in parts:
        month = pd.read_parquet(p)
        input_total += len(month)
        kept = month[month.qc3_exclusion == ""]
        kept = assign_segments(kept, excluded, base_id)
        for name, bit in FLAG_BITS.items():
            flag_counts[name] += int(((kept.qc3_flags.to_numpy() >> bit) & 1).sum())
        table_df = kept[keep_cols + ["qc3_trajectory_id", "qc3_segment", "qc3_flags", "qc3_flag_reasons"]]
        table = pa.Table.from_pandas(table_df, preserve_index=False)
        if writer is None:
            meta = dict(table.schema.metadata or {})
            meta[b"limosat.qc3"] = json.dumps({"protocol_id": PROTOCOL_V3_ID, "flag_bits": FLAG_BITS}).encode()
            writer = pq.ParquetWriter(tmp_path, table.schema.with_metadata(meta), compression="zstd")
        writer.write_table(table.cast(writer.schema), row_group_size=250_000)
        kept_total += len(kept)
    writer.close()
    tmp_path.rename(out_path)
    for code in EXCLUSION_CODES:
        counts[code] = int(excluded.qc3_exclusion.str.contains(code).sum())
    out = ds.dataset(out_path).to_table(columns=["edge_uid", "qc3_trajectory_id", "start_time_utc", "source_rowid",
                                                  "target_rowid"]).to_pandas()
    violations = continuity_violations(out)
    unique = out.edge_uid.is_unique and not out.edge_uid.isin(excluded.edge_uid).any()
    manifest = {
        "status": "complete" if (violations == 0 and unique and kept_total + len(excluded) == input_total) else "failed",
        "period": period, "protocol_id": PROTOCOL_V3_ID, "protocol_sha256": sha256(PROTOCOL_V3_PATH),
        "input": str(input_path), "input_sha256": sha256(input_path),
        "land_raster": str(land_path), "land_raster_sha256": land.sha256,
        "sic_root": str(sic_root) if sic_root else None, "sic_files_used": len(sic.files) if sic else 0,
        "input_links": input_total, "kept_links": kept_total, "excluded_links": int(len(excluded)),
        "exclusions_by_reason": counts, "kept_links_by_flag": flag_counts,
        "new_trajectory_id_base": base_id, "continuity_violations": violations, "edge_uid_unique_and_disjoint": bool(unique),
        "outputs": {out_path.name: sha256(out_path),
                    f"{period}_qc3_exclusions.parquet": sha256(output_dir / f"{period}_qc3_exclusions.parquet")},
    }
    (output_dir / f"{period}_qc3.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def pc_max(path: Path, column: str) -> int:
    meta = pq.ParquetFile(path).metadata
    index = pq.ParquetFile(path).schema_arrow.get_field_index(column)
    return max(meta.row_group(i).column(index).statistics.max for i in range(meta.num_row_groups))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply trajectory-link QC v3 to a v2 scientific edge Parquet.")
    parser.add_argument("--input", required=True, type=Path, help="v2 edge Parquet for one period")
    parser.add_argument("--period", required=True, help="period label used in output names")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--land-raster", required=True, type=Path, help=".npz inland-distance raster (EPSG:3413)")
    parser.add_argument("--sic-root", type=Path, help="NOAA/NSIDC CDR v6 root for the open-water flag")
    a = parser.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = run(a.input, a.output_dir, a.land_raster, a.sic_root, a.period)
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit("v3 QC output failed its checks; see the manifest")


if __name__ == "__main__":
    main()
