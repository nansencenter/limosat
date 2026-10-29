"""LiMOSAT QC (protocol v1): one sequential pass of link QC over the regular matching output.

Each link between consecutive keypoints of a trajectory is tested by the rules below, in order. A rule
only sees the links that survived the earlier rules, so the rules we are most confident in act first
and later neighbour tests use cleaner neighbours. Every removed link gets the code of the step that
removed it and splits its trajectory.

0. ``SPEED``: speed over the actual image gap above 60 km/day (physically impossible).
1. ``OPEN_WATER``: sea-ice concentration below 15% at either endpoint (links with no value kept);
   ``LAND``: the deeper endpoint at least 1 km inland on the supplied land raster.
2. ``ISOLATED_UNVERIFIED``: no other link of the image pair within 30 km, and a gap over 54 h or a
   speed over 25 km/day.
3. ``ROGUE_VECTOR``: at least five times the neighbours' own spread from the leave-one-out local-affine
   prediction of its image pair, and either more than 2 km off with no neighbour corroborating it, or
   more than 5 km off with at most 2 corroborating neighbours.
4. ``INTERP_UNSUPPORTED``: an interpolated link without support from the 16 nearest links of the pair
   within 50 km: at least 3 of them, and within 1 km of their median motion, within 2 km with target
   correlation >= 0.40, or within 3 km with target correlation >= 0.50.

Kept links carry ``pair_direct_share`` (share of the image pair's input links that are direct matches)
and flags ``FEW_NEIGHBOURS`` (fewer than 2 other kept links within 30 km) and ``LOW_PAIR_DIRECT_SHARE``
(pair direct share below 0.40). Thresholds are in ``protocols/limosat_qc.json``. EPSG:3413 metres.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .core import QCConfig
from .v3 import score_pair_isolation

PROTOCOL_ID = "limosat_qc_v1_draft_20261028"
PROTOCOL_PATH = Path(__file__).resolve().parent / "protocols" / "limosat_qc.json"
STEPS = ("SPEED", "OPEN_WATER", "LAND", "ISOLATED_UNVERIFIED", "ROGUE_VECTOR", "INTERP_UNSUPPORTED")
FLAG_BITS = {"FEW_NEIGHBOURS": 0, "LOW_PAIR_DIRECT_SHARE": 1}


@dataclass(frozen=True)
class LimosatQCConfig:
    """Thresholds of LiMOSAT QC (metres, km for inland distance, SIC as a fraction)."""

    max_speed_m_per_day: float = 60_000.0
    open_water_sic: float = 0.15
    land_min_inland_km: float = 1.0
    neighbour_radius_m: float = 30_000.0
    isolated_max_hours: float = 54.0
    isolated_max_speed_m_per_day: float = 25_000.0
    rogue_residual_m: float = 2_000.0
    rogue_coherence_min: float = 5.0
    rogue_spread_floor_m: float = 150.0
    rogue_gross_residual_m: float = 5_000.0
    rogue_gross_max_corroborating: int = 2
    support_radius_m: float = 50_000.0
    support_nearest: int = 16
    support_min: int = 3
    support_tolerance_m: float = 1_000.0
    support_mid_tolerance_m: float = 2_000.0
    support_mid_min_corr: float = 0.40
    support_hard_tolerance_m: float = 3_000.0
    support_hard_min_corr: float = 0.50
    few_neighbours_min: int = 2
    low_pair_direct_share: float = 0.40

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not self.support_tolerance_m <= self.support_mid_tolerance_m <= self.support_hard_tolerance_m:
            raise ValueError("support tolerances must increase")
        for field in ("open_water_sic", "support_mid_min_corr", "support_hard_min_corr", "low_pair_direct_share"):
            if not 0 < getattr(self, field) < 1:
                raise ValueError(f"{field} must be in (0, 1)")


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Bundled LiMOSAT QC protocol ID does not match the code")
    return protocol


def validate_protocol(config: LimosatQCConfig) -> None:
    """Fail if the code defaults and the bundled protocol have drifted apart."""

    parameters = load_protocol()["parameters"]
    mismatches = {f: (v, parameters.get(f)) for f, v in asdict(config).items() if v != parameters.get(f)}
    if mismatches:
        raise ValueError(f"LiMOSAT QC configuration does not match the protocol: {mismatches}")


def _by_pair(d: pd.DataFrame, alive: np.ndarray):
    """Row indices of the surviving links of each image pair."""

    sub = np.flatnonzero(alive)
    keys = d[["source_image_id", "target_image_id"]].to_numpy()[sub]
    for index in pd.DataFrame(keys).groupby([0, 1], sort=False).indices.values():
        yield sub[index]


def _support(xy: np.ndarray, uv: np.ndarray, cfg: LimosatQCConfig):
    """Number of the nearest other links within the support radius, and distance of each link's
    displacement from their median."""

    n = len(xy)
    if n < 2:
        return np.zeros(n), np.full(n, np.nan)
    dist, nb = cKDTree(xy).query(xy, k=min(cfg.support_nearest + 1, n), distance_upper_bound=cfg.support_radius_m)
    valid = np.isfinite(dist) & (nb < n) & (nb != np.arange(n)[:, None])
    nuv = uv[np.where(valid, nb, 0)].astype(float)
    nuv[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(nuv, axis=1)
    count = valid.sum(1).astype(float)
    return count, np.where(count > 0, np.hypot(*(uv - median).T), np.nan)


def _neighbour_counts(d: pd.DataFrame, alive: np.ndarray, radius_m: float) -> np.ndarray:
    out = np.full(len(d), np.nan)
    xy = d[["x0_m", "y0_m"]].to_numpy(float)
    for index in _by_pair(d, alive):
        out[index] = cKDTree(xy[index]).query_ball_point(xy[index], radius_m, return_length=True) - 1
    return out


def decide(links: pd.DataFrame, cfg: LimosatQCConfig, land, sic, local: QCConfig | None = None) -> pd.DataFrame:
    """Run the LiMOSAT QC steps on the links of whole image pairs; returns the links with ``qc_exclusion``
    (the step that removed the link, empty when kept), ``qc_flags``, ``qc_flag_reasons`` and diagnostics."""

    local = local or QCConfig()
    d = links.reset_index(drop=True).copy()
    n = len(d)
    direct = ((d.source_interpolated == 0) & (d.target_interpolated == 0)).to_numpy()
    d["direct"] = direct
    pair = d.groupby(["source_image_id", "target_image_id"], sort=False).direct
    d["pair_links"] = pair.transform("size")
    d["pair_direct_share"] = pair.transform("mean").astype("float32")
    reason = np.full(n, "", dtype=object)

    # step 0: impossible speed
    speed = (d.speed_m_per_day > cfg.max_speed_m_per_day).to_numpy()
    reason[speed] = "SPEED"
    alive = ~speed

    # step 1: position (sea-ice concentration, land)
    d["inland_start_km"] = land.lookup(d.x0_m.to_numpy(), d.y0_m.to_numpy())
    d["inland_end_km"] = land.lookup(d.x1_m.to_numpy(), d.y1_m.to_numpy())
    on_land = (np.fmax(d.inland_start_km, d.inland_end_km) >= cfg.land_min_inland_km).to_numpy()
    if sic is not None:
        d["sic_start"] = sic.sample(d.x0_m, d.y0_m, d.start_time_utc)
        d["sic_end"] = sic.sample(d.x1_m, d.y1_m, d.end_time_utc)
        open_water = (np.fmin(d.sic_start, d.sic_end) < cfg.open_water_sic).to_numpy()
    else:
        d["sic_start"] = d["sic_end"] = np.float32(np.nan)
        open_water = np.zeros(n, bool)
    position = alive & (open_water | on_land)
    reason[position] = np.where(open_water[position] & on_land[position], "OPEN_WATER+LAND",
                                np.where(open_water[position], "OPEN_WATER", "LAND"))
    alive &= ~position

    # step 2: isolated links that cannot be verified
    d["neighbours_30km"] = _neighbour_counts(d, alive, cfg.neighbour_radius_m)
    isolated = alive & (d.neighbours_30km == 0).to_numpy() & (
        (d.elapsed_seconds > cfg.isolated_max_hours * 3600) | (d.speed_m_per_day > cfg.isolated_max_speed_m_per_day)).to_numpy()
    reason[isolated] = "ISOLATED_UNVERIFIED"
    alive &= ~isolated

    # step 3: rogue vectors against the leave-one-out local-affine prediction
    xy, uv = d[["x0_m", "y0_m"]].to_numpy(float), d[["dx_m", "dy_m"]].to_numpy(float)
    cols = ("coarse_residual_m", "neighbour_spread_m", "n_corroborating", "n_neighbours", "local_u_m", "local_v_m",
            "local_residual_m")
    buf = {c: np.full(n, np.nan) for c in cols}
    supported = np.zeros(n, bool)
    for index in _by_pair(d, alive):
        s = score_pair_isolation(xy[index], uv[index], local, cfg.rogue_residual_m)
        for c in cols:
            buf[c][index] = s[c]
        supported[index] = s["local_supported"]
    for c in cols:
        d[c] = buf[c]
    d["local_supported"] = supported
    d["coherence_ratio"] = d.local_residual_m / np.maximum(d.neighbour_spread_m, cfg.rogue_spread_floor_m)
    # a single agreeing neighbour protects a moderate outlier (it may be real deformation), but small
    # clusters of grossly wrong vectors that agree with each other do not protect themselves
    unprotected = ((d.local_residual_m > cfg.rogue_residual_m) & (d.n_corroborating == 0)) | (
        (d.local_residual_m > cfg.rogue_gross_residual_m) & (d.n_corroborating <= cfg.rogue_gross_max_corroborating))
    rogue = alive & (supported & (d.coherence_ratio >= cfg.rogue_coherence_min) & unprotected).to_numpy()
    reason[rogue] = "ROGUE_VECTOR"
    alive &= ~rogue

    # step 4: interpolated links without support from their neighbours
    count, residual = np.zeros(n), np.full(n, np.nan)
    for index in _by_pair(d, alive):
        count[index], residual[index] = _support(xy[index], uv[index], cfg)
    d["support_count"], d["support_residual_m"] = count, residual
    corr = d.target_corr.to_numpy(float)
    ok = (count >= cfg.support_min) & (
        (residual <= cfg.support_tolerance_m)
        | ((residual <= cfg.support_mid_tolerance_m) & (corr >= cfg.support_mid_min_corr))
        | ((residual <= cfg.support_hard_tolerance_m) & (corr >= cfg.support_hard_min_corr)))
    unsupported = alive & ~direct & ~ok
    reason[unsupported] = "INTERP_UNSUPPORTED"
    alive &= ~unsupported

    # flags on kept links
    d["qc_exclusion"] = reason
    d["neighbours_kept_30km"] = _neighbour_counts(d, alive, cfg.neighbour_radius_m)
    few = alive & (d.neighbours_kept_30km < cfg.few_neighbours_min).to_numpy()
    low = alive & (d.pair_direct_share < cfg.low_pair_direct_share).to_numpy()
    d["qc_flags"] = ((few.astype(np.uint16) << FLAG_BITS["FEW_NEIGHBOURS"])
                     | (low.astype(np.uint16) << FLAG_BITS["LOW_PAIR_DIRECT_SHARE"]))
    d["qc_flag_reasons"] = np.where(few & low, "FEW_NEIGHBOURS;LOW_PAIR_DIRECT_SHARE",
                                    np.where(few, "FEW_NEIGHBOURS", np.where(low, "LOW_PAIR_DIRECT_SHARE", "")))
    return d


# --------------------------------------------------------------------------- period runner

LINK_COLUMNS = ["edge_uid", "original_trajectory_id", "trajectory_link_index", "source_rowid", "target_rowid", "source_image_id",
                "target_image_id", "start_time_utc", "end_time_utc", "elapsed_seconds", "x0_m", "y0_m", "x1_m", "y1_m",
                "dx_m", "dy_m", "distance_m", "speed_m_per_day", "source_interpolated", "target_interpolated",
                "source_corr", "target_corr", "source_orbit_num", "target_orbit_num", "source_is_last", "target_is_last",
                "source_stopped", "target_stopped", "source_converged_to", "target_converged_to"]
QC_COLUMNS = ["qc_trajectory_id", "qc_segment", "qc_flags", "qc_flag_reasons", "pair_direct_share", "qc_sic_start",
              "qc_sic_end", "qc_neighbours_30km"]
DIAGNOSTICS = ["inland_start_km", "inland_end_km", "sic_start", "sic_end", "neighbours_30km", "coarse_residual_m",
               "neighbour_spread_m", "n_corroborating", "n_neighbours", "local_residual_m", "coherence_ratio",
               "local_supported", "support_count", "support_residual_m", "direct", "pair_links", "pair_direct_share"]
_STATE = ("rowid", "image", "t", "x", "y", "interp", "corr", "last")


def _point_xy(wkt: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    parts = wkt.str.slice(7, -1).str.split(" ", n=1, expand=True)  # "POINT (x y)"
    return parts[0].astype(float).to_numpy(), parts[1].astype(float).to_numpy()


def build_links(compact_database: Path, raw_database: Path | None, links_dir: Path, batch_rows: int = 2_000_000) -> dict:
    """Links between consecutive points of every trajectory, read in time order from a compact point store
    (``compact_points`` from the v2 ``prepare`` stage) and written as one Parquet file per month of the
    source time. Image orbit numbers and the per-point tracker fields ``stopped`` and ``converged_to``
    come from the raw period database when it is given (missing otherwise)."""

    import sqlite3

    import pyarrow as pa
    import pyarrow.parquet as pq

    links_dir.mkdir(parents=True, exist_ok=True)
    orbit, stopped, converged = {}, None, None
    if raw_database is not None:
        # per-point tracker fields that the compact store does not carry, indexed by rowid
        with sqlite3.connect(f"file:{Path(raw_database).resolve()}?mode=ro", uri=True) as raw:
            table = raw.execute("select name from sqlite_master where type = 'table'").fetchone()[0]
            n = raw.execute(f'select max(rowid) from "{table}"').fetchone()[0] + 1
            stopped, converged = np.zeros(n, np.int8), np.full(n, -1, np.int64)
            image_orbit = {}
            cur = raw.execute(f'select rowid, image_id, orbit_num, stopped, converged_to from "{table}"')
            while chunk := cur.fetchmany(2_000_000):
                c = np.array([(r[0], r[1], -1 if r[2] is None else r[2], 1 if r[3] else 0, -1 if r[4] is None else r[4])
                              for r in chunk], dtype=np.int64)
                stopped[c[:, 0]], converged[c[:, 0]] = c[:, 3], c[:, 4]
                image_orbit.update(zip(c[:, 1].tolist(), c[:, 2].tolist()))
            orbit = {k: v for k, v in image_orbit.items() if v >= 0}
    con = sqlite3.connect(f"file:{Path(compact_database).resolve()}?mode=ro", uri=True)
    n_traj = con.execute("select max(trajectory_id) from compact_points").fetchone()[0] + 1
    state = {k: np.full(n_traj, np.nan) for k in _STATE}
    n_links = np.zeros(n_traj, np.int64)
    cursor = con.execute("select source_rowid, image_id, is_last, trajectory_id, geometry, time, corr, interpolated "
                         "from compact_points order by time, image_id, source_rowid")
    writers, counts = {}, {"points": 0, "links": 0}
    while rows := cursor.fetchmany(batch_rows):
        b = pd.DataFrame.from_records(rows, columns=["rowid", "image", "last", "traj", "geometry", "time", "corr", "interp"])
        counts["points"] += len(b)
        b["x"], b["y"] = _point_xy(b.geometry)
        b["t"] = (pd.to_datetime(b.time, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
        b["interp"] = b.interp.fillna(0)
        b = b.assign(seq=np.arange(len(b))).sort_values(["traj", "seq"], kind="mergesort")
        tr = b.traj.to_numpy()
        same = tr[1:] == tr[:-1]
        first = np.r_[True, ~same]
        cur = {k: b[k].to_numpy(float) for k in _STATE}
        prev = {k: np.r_[np.nan, v[:-1]] for k, v in cur.items()}
        for k in _STATE:  # the previous point of a trajectory's first point in this batch is the stored state
            prev[k][first] = state[k][tr[first]]
        ok = ~np.isnan(prev["rowid"])
        if ok.any():
            s, c = {k: v[ok] for k, v in prev.items()}, {k: v[ok] for k, v in cur.items()}
            # position of each link along its trajectory: the order key for splitting (times can tie when a
            # trajectory has two points in the same image)
            tro = tr[ok]
            starts = np.r_[0, np.flatnonzero(tro[1:] != tro[:-1]) + 1]
            within = np.arange(len(tro)) - np.repeat(starts, np.diff(np.r_[starts, len(tro)]))
            link_index = n_links[tro] + within
            np.add.at(n_links, tro, 1)
            elapsed = c["t"] - s["t"]
            dx, dy = c["x"] - s["x"], c["y"] - s["y"]
            dist = np.hypot(dx, dy)
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = np.where(elapsed > 0, dist / (elapsed / 86400.0), np.inf)
            start = pd.to_datetime(s["t"], unit="s", utc=True)
            links = pd.DataFrame({
                "original_trajectory_id": tro.astype(np.int64), "trajectory_link_index": link_index.astype(np.int32),
                "source_rowid": s["rowid"].astype(np.int64), "target_rowid": c["rowid"].astype(np.int64),
                "source_image_id": s["image"].astype(np.int64), "target_image_id": c["image"].astype(np.int64),
                "start_time_utc": start, "end_time_utc": pd.to_datetime(c["t"], unit="s", utc=True),
                "elapsed_seconds": elapsed, "x0_m": s["x"], "y0_m": s["y"], "x1_m": c["x"], "y1_m": c["y"],
                "dx_m": dx, "dy_m": dy, "distance_m": dist, "speed_m_per_day": speed,
                "source_interpolated": s["interp"].astype(np.int8), "target_interpolated": c["interp"].astype(np.int8),
                "source_corr": s["corr"], "target_corr": c["corr"],
                "source_is_last": s["last"].astype(np.int8), "target_is_last": c["last"].astype(np.int8)})
            links["source_orbit_num"] = links.source_image_id.map(orbit).astype("Int64")
            links["target_orbit_num"] = links.target_image_id.map(orbit).astype("Int64")
            for end in ("source", "target"):
                rid = links[f"{end}_rowid"].to_numpy()
                if stopped is None:
                    links[f"{end}_stopped"] = pd.array([pd.NA] * len(links), dtype="Int8")
                    links[f"{end}_converged_to"] = pd.array([pd.NA] * len(links), dtype="Int64")
                else:
                    links[f"{end}_stopped"] = stopped[rid]
                    links[f"{end}_converged_to"] = pd.array(np.where(converged[rid] < 0, 0, converged[rid]), dtype="Int64")
                    links.loc[converged[rid] < 0, f"{end}_converged_to"] = pd.NA
            for ym, part in links.groupby(start.strftime("%Y-%m"), sort=False):
                table = pa.Table.from_pandas(part, preserve_index=False)
                if ym not in writers:
                    writers[ym] = pq.ParquetWriter(links_dir / f"{ym}.links.parquet.part", table.schema, compression="zstd")
                writers[ym].write_table(table.cast(writers[ym].schema))
            counts["links"] += int(ok.sum())
        last = np.r_[~same, True]  # keep the last point of every trajectory seen in this batch
        for k in _STATE:
            state[k][tr[last]] = cur[k][last]
        print(json.dumps({"points": counts["points"], "links": counts["links"]}), flush=True)
    con.close()
    for ym, w in writers.items():
        w.close()
        (links_dir / f"{ym}.links.parquet.part").rename(links_dir / f"{ym}.links.parquet")
    return counts


def _edge_uids(period: str, source: np.ndarray, target: np.ndarray) -> list[str]:
    import hashlib

    return [hashlib.sha256(f"{period}:{a}:{b}".encode()).hexdigest() for a, b in zip(source, target)]


def run(compact_database: Path, raw_database: Path | None, period: str, output_dir: Path, land_path: Path,
        sic, work_dir: Path | None = None, cfg: LimosatQCConfig | None = None) -> dict:
    """LiMOSAT QC for one processing period: build the links, decide them month by month, split
    trajectories at removed links and write ``<period>_qc.parquet`` (kept links),
    ``<period>_qc_removed.parquet`` and ``<period>_qc.manifest.json``. Resumable through ``work_dir``."""

    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from .v3 import LandRaster, assign_segments, edge_uids_unique_and_disjoint, sha256

    cfg = cfg or LimosatQCConfig()
    cfg.validate()
    validate_protocol(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else output_dir / "work"
    links_dir = work / "links"
    if not (work / "links.done").exists():
        counts = build_links(compact_database, raw_database, links_dir)
        (work / "links.done").write_text(json.dumps(counts))
    land = LandRaster(land_path)
    months = sorted(p.name.split(".")[0] for p in links_dir.glob("*.links.parquet") if not p.name.startswith("._"))
    for ym in months:
        if (work / f"{ym}.done").exists():
            continue
        links = pd.read_parquet(links_dir / f"{ym}.links.parquet")
        links.insert(0, "edge_uid", _edge_uids(period, links.source_rowid.to_numpy(), links.target_rowid.to_numpy()))
        decided = decide(links, cfg, land, sic)
        decided.to_parquet(work / f"{ym}.decided.parquet", index=False)
        (work / f"{ym}.done").write_text("ok")
        print(json.dumps({"period": period, "month": ym, "links": len(decided),
                          "removed": int((decided.qc_exclusion != "").sum())}), flush=True)

    parts = [work / f"{ym}.decided.parquet" for ym in months]
    removed = pd.concat([pd.read_parquet(p).query("qc_exclusion != ''") for p in parts], ignore_index=True)
    removed[LINK_COLUMNS + ["qc_exclusion"] + DIAGNOSTICS].to_parquet(output_dir / f"{period}_qc_removed.parquet", index=False)
    base_id = int(max(pd.read_parquet(p, columns=["original_trajectory_id"]).original_trajectory_id.max() for p in parts)) + 1
    # trajectories are split in link order (assign_segments orders by its start_time_utc column, so the link
    # index is passed in that column)
    cuts = removed[["original_trajectory_id", "trajectory_link_index"]].set_axis(["qc_trajectory_id", "start_time_utc"], axis=1)
    out_path = output_dir / f"{period}_qc.parquet"
    tmp = out_path.with_suffix(".parquet.part")
    writer, kept_total, input_total = None, 0, 0
    flag_counts = {k: 0 for k in FLAG_BITS}
    for p in parts:
        month = pd.read_parquet(p)
        input_total += len(month)
        kept = month[month.qc_exclusion == ""]
        seg = assign_segments(pd.DataFrame({"qc_trajectory_id": kept.original_trajectory_id.to_numpy(),
                                            "start_time_utc": kept.trajectory_link_index.to_numpy()}), cuts, base_id)
        kept = kept.assign(qc_trajectory_id=seg.qc3_trajectory_id.to_numpy(), qc_segment=seg.qc3_segment.to_numpy(),
                           qc_sic_start=kept.sic_start.astype("float32"), qc_sic_end=kept.sic_end.astype("float32"),
                           qc_neighbours_30km=kept.neighbours_kept_30km.astype("int16"))
        for name, bit in FLAG_BITS.items():
            flag_counts[name] += int(((kept.qc_flags.to_numpy().astype(int) >> bit) & 1).sum())
        table = pa.Table.from_pandas(kept[LINK_COLUMNS + QC_COLUMNS], preserve_index=False)
        if writer is None:
            meta = dict(table.schema.metadata or {})
            meta[b"limosat.qc"] = json.dumps({"protocol_id": PROTOCOL_ID, "flag_bits": FLAG_BITS}).encode()
            writer = pq.ParquetWriter(tmp, table.schema.with_metadata(meta), compression="zstd")
        writer.write_table(table.cast(writer.schema), row_group_size=250_000)
        kept_total += len(kept)
    writer.close()
    tmp.rename(out_path)
    out = ds.dataset(out_path).to_table(columns=["qc_trajectory_id", "trajectory_link_index", "source_rowid", "target_rowid"]).to_pandas()
    o = out.sort_values(["qc_trajectory_id", "trajectory_link_index"], kind="mergesort")
    same = o.qc_trajectory_id.to_numpy()[1:] == o.qc_trajectory_id.to_numpy()[:-1]
    violations = int((same & (o.source_rowid.to_numpy()[1:] != o.target_rowid.to_numpy()[:-1])).sum())
    del out, o
    unique = edge_uids_unique_and_disjoint(out_path, removed.edge_uid)
    manifest = {
        "status": "complete" if violations == 0 and unique and kept_total + len(removed) == input_total else "failed",
        "period": period, "protocol_id": PROTOCOL_ID, "protocol_sha256": sha256(PROTOCOL_PATH),
        "parameters": asdict(cfg), "compact_database": str(compact_database),
        "raw_database": str(raw_database) if raw_database else None,
        "land_raster": str(land_path), "land_raster_sha256": land.sha256,
        "sic_days_by_product": {k: sorted(v) if len(v) <= 20 else len(v)
                                for k, v in getattr(sic, "days_by_product", {}).items()},
        "links": json.loads((work / "links.done").read_text()),
        "input_links": input_total, "kept_links": kept_total, "removed_links": int(len(removed)),
        "removed_by_step": {k: int(v) for k, v in removed.qc_exclusion.value_counts().items()},
        "kept_links_by_flag": flag_counts, "new_trajectory_id_base": base_id,
        "continuity_violations": violations, "edge_uid_unique_and_disjoint": bool(unique),
        "outputs": {out_path.name: sha256(out_path),
                    f"{period}_qc_removed.parquet": sha256(output_dir / f"{period}_qc_removed.parquet")},
    }
    (output_dir / f"{period}_qc.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .v3 import AsiSampler, FallbackSicSampler, SicSampler

    p = argparse.ArgumentParser(description="LiMOSAT QC: one sequential pass of link QC over one processing period.")
    p.add_argument("--compact-database", required=True, type=Path, help="compact_points store of the period")
    p.add_argument("--raw-database", type=Path, help="raw period SQLite (image orbit numbers)")
    p.add_argument("--period", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--land-raster", required=True, type=Path)
    p.add_argument("--sic-root", required=True, type=Path, help="AMSR2 ASI 3.125 km root")
    p.add_argument("--sic-fallback-asi6250", type=Path)
    p.add_argument("--sic-fallback-nsidc", type=Path)
    a = p.parse_args(argv)
    chain = [("asi", AsiSampler(a.sic_root))]
    if a.sic_fallback_asi6250:
        chain.append(("asi_n6250", AsiSampler(a.sic_fallback_asi6250, grid="n6250")))
    if a.sic_fallback_nsidc:
        chain.append(("nsidc", SicSampler(a.sic_fallback_nsidc)))
    manifest = run(a.compact_database, a.raw_database, a.period, a.output_dir, a.land_raster,
                   FallbackSicSampler(chain), a.work_dir)
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit("LiMOSAT QC output failed its checks; see the manifest")


if __name__ == "__main__":
    main()
