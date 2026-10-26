"""Trajectory-link QC v3.1: exclusions and quality flags applied on top of a v2 release.

v3.1 keeps every v2 decision. It reads the v2 scientific edge Parquet (accepted links only), scores each
link against the other accepted links of the same exact image pair with the frozen v2 local-affine
neighbour model (single pass), and excludes, splitting trajectories at every excluded link as v2 does:

- ``R1``: isolated outliers (direct or interpolated endpoints);
- ``LAND``: an endpoint inside land (any land cell of the inland-distance raster);
- ``OPEN_WATER``: sea-ice concentration below 15% at either endpoint (links without any value are kept);
- ``INTERP_UNSUPPORTED``: an interpolated endpoint without enough agreeing neighbours;
- ``ISOLATED_UNVERIFIED``: no other link within 30 km and an interval over 54 h or a speed over 25 km/day.

Incoherent image pairs and links with fewer than two neighbours are flagged, not excluded. Thresholds are
in ``protocols/trajectory_link_qc_v3_1.json``. Coordinates are EPSG:3413 metres.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from .core import QCConfig, _local_prediction, validate_frozen_protocol


PROTOCOL_V3_ID = "limosat_trajectory_link_qc_v3_1_20261026"
PROTOCOL_V3_PATH = Path(__file__).resolve().parent / "protocols" / "trajectory_link_qc_v3_1.json"

FLAG_BITS = {
    "PAIR_INCOHERENT": 0,
    "FEW_NEIGHBOURS": 1,
}
EXCLUSION_CODES = ("R1", "LAND", "OPEN_WATER", "INTERP_UNSUPPORTED", "ISOLATED_UNVERIFIED")


@dataclass(frozen=True)
class V3Config:
    """Frozen v3 thresholds (metres, kilometres for land distance, SIC as a fraction)."""

    residual_floor_m: float = 2_000.0
    coherence_min: float = 5.0
    spread_floor_m: float = 150.0
    land_min_inland_km: float = 1.0
    incoherent_min_links: int = 20
    incoherent_fraction: float = 0.5
    incoherent_residual_m: float = 2_000.0
    open_water_sic: float = 0.15
    support_radius_m: float = 50_000.0
    support_nearest: int = 16
    support_min: int = 3
    support_tolerance_m: float = 1_000.0
    few_neighbours_radius_m: float = 30_000.0
    few_neighbours_min: int = 2
    isolated_max_hours: float = 54.0
    isolated_max_speed_m_per_day: float = 25_000.0

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


def validate_v3_protocol(config: V3Config, overrides: frozenset = frozenset()) -> None:
    """Fail if v3 code defaults and the bundled protocol have drifted apart (``overrides`` are declared
    run-time departures, recorded in the manifest)."""

    parameters = load_v3_protocol()["parameters"]
    mismatches = {
        field: (value, parameters.get(field))
        for field, value in asdict(config).items()
        if value != parameters.get(field) and field not in overrides
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
    result["direct"] = direct
    result["r1"] = isolated
    result["r1_interpolated"] = isolated & ~direct
    support = np.zeros(len(result))
    support_direct = np.zeros(len(result))
    support_residual = np.full(len(result), np.nan)
    neighbours = np.zeros(len(result))
    for _, index in result.groupby(["source_image_id", "target_image_id"], sort=False).indices.items():
        n, nd, r, few = pair_support(result[["x0_m", "y0_m"]].to_numpy(float)[index],
                                     result[["dx_m", "dy_m"]].to_numpy(float)[index], direct.to_numpy()[index], v3)
        support[index], support_direct[index], support_residual[index], neighbours[index] = n, nd, r, few
    result["support_count"] = support
    result["support_direct_count"] = support_direct
    result["support_residual_m"] = support_residual
    result["neighbours_30km"] = neighbours
    supported = (support >= v3.support_min) & (support_residual <= v3.support_tolerance_m)
    result["interp_unsupported"] = ~direct.to_numpy() & ~supported
    far = result.coarse_residual_m > v3.incoherent_residual_m
    pair = result.groupby(["source_image_id", "target_image_id"], sort=False)
    links_in_pair = pair.edge_uid.transform("size")
    far_fraction = far.groupby([result.source_image_id, result.target_image_id], sort=False).transform("mean")
    result["pair_links"] = links_in_pair
    result["pair_far_fraction"] = far_fraction
    result["pair_incoherent"] = (links_in_pair >= v3.incoherent_min_links) & (far_fraction >= v3.incoherent_fraction)
    return result


def pair_support(xy: np.ndarray, uv: np.ndarray, direct: np.ndarray, v3: "V3Config"):
    """Support for each link of one pair from its nearest other links (direct or interpolated).

    Returns the number of the ``support_nearest`` nearest other links within ``support_radius_m``, how
    many of them are direct, the distance of the link displacement from their median, and the count
    of other links within ``few_neighbours_radius_m``.
    """
    n = len(xy)
    count, count_direct, residual = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    if n < 2:
        return count, count_direct, residual, np.zeros(n)
    tree = cKDTree(xy)
    few = tree.query_ball_point(xy, v3.few_neighbours_radius_m, return_length=True) - 1
    k = min(v3.support_nearest + 1, n)
    dist, nb = tree.query(xy, k=k, distance_upper_bound=v3.support_radius_m)
    valid = np.isfinite(dist) & (nb < n) & (nb != np.arange(n)[:, None])
    safe = np.where(valid, nb, 0)
    nuv = uv[safe].astype(float)
    nuv[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(nuv, axis=1)
    count = valid.sum(1).astype(float)
    count_direct = (valid & direct[safe]).sum(1).astype(float)
    residual = np.where(count > 0, np.hypot(*(uv - median).T), np.nan)
    return count, count_direct, residual, few.astype(float)


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


class AsiSampler:
    """Nearest-cell daily SIC (fraction) from University of Bremen AMSR2 ASI GeoTIFFs
    (``<root>/<YYYY>/<mon>/asi-AMSR2-<grid>-<YYYYMMDD>-v5.4.tif``, EPSG:3411, percent; values above 100
    are land or missing). ``grid`` is ``n3125`` (3.125 km) or ``n6250`` (6.25 km)."""

    def __init__(self, root: Path, grid: str = "n3125"):
        from pyproj import Transformer

        self.root = Path(root)
        self.grid = grid
        self.transform = Transformer.from_crs("EPSG:3413", "EPSG:3411", always_xy=True).transform
        self.cache: dict[str, tuple | None] = {}
        self.files: dict[str, str] = {}

    def _grid(self, day: pd.Timestamp):
        key = day.strftime("%Y%m%d")
        if key not in self.cache:
            import rasterio

            path = self.root / day.strftime("%Y") / day.strftime("%b").lower() / f"asi-AMSR2-{self.grid}-{key}-v5.4.tif"
            if not path.exists():
                self.cache[key] = None
            else:
                with rasterio.open(path) as src:
                    self.cache[key] = (src.transform, src.read(1))
                self.files[key] = path.name
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
            transform, sic = grid
            index = np.asarray(index)
            col = np.floor((px[index] - transform.c) / transform.a).astype(int)
            row = np.floor((py[index] - transform.f) / transform.e).astype(int)
            ok = (col >= 0) & (col < sic.shape[1]) & (row >= 0) & (row < sic.shape[0])
            values = np.full(len(index), np.nan, dtype="float32")
            values[ok] = sic[row[ok], col[ok]]
            values[values > 100] = np.nan
            out[index] = values / 100.0
        return out


class FallbackSicSampler:
    """Use the first sampler that has a file for each day (e.g. ASI 3.125 km, then ASI 6.25 km, then
    NOAA/NSIDC CDR); ``days_by_product`` records which product served each day."""

    def __init__(self, samplers: list[tuple[str, object]]):
        self.samplers = samplers
        self.days_by_product: dict[str, set] = {name: set() for name, _ in samplers}

    @property
    def files(self) -> dict:
        return {f"{name}:{k}": v for name, s in self.samplers for k, v in s.files.items()}

    def sample(self, x: np.ndarray, y: np.ndarray, times: pd.Series) -> np.ndarray:
        out = np.full(len(x), np.nan, dtype="float32")
        days = pd.to_datetime(times, utc=True).dt.floor("D")
        for day, index in pd.Series(np.arange(len(x))).groupby(days.to_numpy()).groups.items():
            index = np.asarray(index)
            for name, sampler in self.samplers:
                if sampler._grid(pd.Timestamp(day)) is not None:
                    out[index] = sampler.sample(np.asarray(x)[index], np.asarray(y)[index], times.iloc[index])
                    self.days_by_product[name].add(pd.Timestamp(day).strftime("%Y-%m-%d"))
                    break
        return out


# --------------------------------------------------------------------------- decisions


def decide(scored: pd.DataFrame, v3: V3Config, land: LandRaster, sic: "SicSampler | AsiSampler | None") -> pd.DataFrame:
    """Exclusion reasons and quality flags for scored links."""

    result = scored
    result["inland_start_km"] = land.lookup(result.x0_m.to_numpy(), result.y0_m.to_numpy())
    result["inland_end_km"] = land.lookup(result.x1_m.to_numpy(), result.y1_m.to_numpy())
    inland = np.fmax(result.inland_start_km, result.inland_end_km)
    result["land"] = inland >= v3.land_min_inland_km
    if sic is not None:
        result["sic_start"] = sic.sample(result.x0_m, result.y0_m, result.start_time_utc)
        result["sic_end"] = sic.sample(result.x1_m, result.y1_m, result.end_time_utc)
        # either endpoint below the threshold (a missing value at one endpoint does not protect the other)
        result["open_water"] = np.fmin(result.sic_start, result.sic_end) < v3.open_water_sic
    else:
        result["sic_start"] = np.float32(np.nan)
        result["sic_end"] = np.float32(np.nan)
        result["open_water"] = False

    return apply_exclusions(result, v3)


def apply_exclusions(result: pd.DataFrame, v3: V3Config) -> pd.DataFrame:
    """Exclusion reasons and flags from stored diagnostics, so a finished scoring can be re-decided
    with a different support tolerance without rescoring."""

    direct = result.direct.to_numpy(bool)
    supported = (result.support_count >= v3.support_min) & (result.support_residual_m <= v3.support_tolerance_m)
    result["interp_unsupported"] = ~direct & ~supported.to_numpy()
    result["few_neighbours"] = result.neighbours_30km < v3.few_neighbours_min
    result["isolated_unverified"] = (result.neighbours_30km == 0) & (
        (result.elapsed_seconds > v3.isolated_max_hours * 3600)
        | (result.speed_m_per_day > v3.isolated_max_speed_m_per_day))
    reason = pd.Series("", index=result.index, dtype="string")
    for column, code in (("r1", "R1"), ("land", "LAND"), ("open_water", "OPEN_WATER"),
                         ("interp_unsupported", "INTERP_UNSUPPORTED"), ("isolated_unverified", "ISOLATED_UNVERIFIED")):
        reason += np.where(result[column], code + "+", "")
    result["qc3_exclusion"] = reason.str.rstrip("+")
    flags = np.zeros(len(result), dtype=np.uint16)
    names = pd.Series("", index=result.index, dtype="string")
    for column, name in (("pair_incoherent", "PAIR_INCOHERENT"), ("few_neighbours", "FEW_NEIGHBOURS")):
        hit = result[column].to_numpy(bool)
        flags |= (hit.astype(np.uint16) << FLAG_BITS[name])
        names += np.where(hit, name + ";", "")
    result["qc3_flags"] = flags
    result["qc3_flag_reasons"] = names.str.rstrip(";")
    return result


def reason_text(row) -> str:
    parts = []
    codes = row.qc3_exclusion.split("+")
    if "R1" in codes:
        kind = "direct-match" if row.direct else "interpolated"
        parts.append(
            f"R1 isolated {kind} outlier: residual {row.local_residual_m:.0f} m vs neighbour spread "
            f"{max(row.neighbour_spread_m, 150.0):.0f} m (ratio {row.coherence_ratio:.1f}), "
            f"0 corroborating neighbours of {int(row.n_neighbours)}")
    if "LAND" in codes:
        parts.append(f"LAND endpoint {max(row.inland_start_km, row.inland_end_km):.1f} km inside land")
    if "OPEN_WATER" in codes:
        parts.append(f"OPEN_WATER SIC {row.sic_start:.0%} at start, {row.sic_end:.0%} at end")
    if "ISOLATED_UNVERIFIED" in codes:
        parts.append(f"ISOLATED_UNVERIFIED no other link within 30 km, {row.elapsed_seconds / 3600:.0f} h interval, "
                     f"{row.speed_m_per_day / 1000:.1f} km/day")
    if "INTERP_UNSUPPORTED" in codes:
        parts.append(f"INTERP_UNSUPPORTED interpolated link {row.support_residual_m:.0f} m from the median of "
                     f"{int(row.support_count)} neighbours within 50 km")
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
    affected = np.flatnonzero(result.qc_trajectory_id.isin(order.qc_trajectory_id).to_numpy())
    if len(affected):
        # the last excluded link of the same trajectory that starts strictly before each kept link
        sub = pd.DataFrame({"row": affected, "qc_trajectory_id": result.qc_trajectory_id.to_numpy()[affected],
                            "t": result.start_time_utc.to_numpy()[affected]}).sort_values("t", kind="mergesort")
        ref = order.rename(columns={"start_time_utc": "t"}).sort_values("t", kind="mergesort")
        hit = pd.merge_asof(sub, ref, on="t", by="qc_trajectory_id", direction="backward", allow_exact_matches=False)
        seg = hit.k.fillna(0).to_numpy(np.int64)
        new = np.where(seg > 0, hit.new_id.fillna(0).to_numpy(np.int64), hit.qc_trajectory_id.to_numpy(np.int64))
        rows = hit.row.to_numpy()
        segment = result.qc3_segment.to_numpy().copy()
        ids = result.qc3_trajectory_id.to_numpy().copy()
        segment[rows], ids[rows] = seg, new
        result["qc3_segment"], result["qc3_trajectory_id"] = segment, ids
    return result


def continuity_violations(links: pd.DataFrame) -> int:
    """Links whose source is not the previous link's target within the same v3 trajectory."""

    ordered = links[["qc3_trajectory_id", "start_time_utc", "source_rowid", "target_rowid"]].sort_values(
        ["qc3_trajectory_id", "start_time_utc"], kind="mergesort")
    same = ordered.qc3_trajectory_id.to_numpy()[1:] == ordered.qc3_trajectory_id.to_numpy()[:-1]
    joined = ordered.source_rowid.to_numpy()[1:] == ordered.target_rowid.to_numpy()[:-1]
    return int((same & ~joined).sum())


# --------------------------------------------------------------------------- archive runner

DIAGNOSTICS = ["elapsed_seconds", "speed_m_per_day", "coarse_residual_m", "neighbour_spread_m", "n_corroborating", "n_neighbours", "local_residual_m",
               "coherence_ratio", "local_supported", "pair_links", "pair_far_fraction", "inland_start_km",
               "inland_end_km", "sic_start", "sic_end", "direct", "support_count", "support_direct_count",
               "support_residual_m", "neighbours_30km"]


def months_in(dataset: ds.Dataset) -> list[tuple[int, int]]:
    table = dataset.to_table(columns=["start_year", "start_month"]).to_pandas().drop_duplicates()
    return sorted((int(y), int(m)) for y, m in table.itertuples(index=False))


def run(input_path: Path, output_dir: Path, land_path: Path, sic_root: Path | None, period: str,
        sic_product: str = "asi", sic_fallback_roots: list[tuple[str, Path]] | None = None,
        support_tolerance_m: float | None = None, work_dir: Path | None = None) -> dict:
    qc_config, v3 = QCConfig(), V3Config()
    overrides = {}
    if support_tolerance_m is not None and support_tolerance_m != v3.support_tolerance_m:
        v3 = replace(v3, support_tolerance_m=float(support_tolerance_m))
        overrides["support_tolerance_m"] = float(support_tolerance_m)
    qc_config.validate()
    v3.validate()
    validate_frozen_protocol(qc_config)
    validate_v3_protocol(v3, frozenset(overrides))
    land = LandRaster(land_path)
    sic = None
    if sic_root:
        primary = AsiSampler(sic_root) if sic_product == "asi" else SicSampler(sic_root)
        chain = [(sic_product, primary)]
        for name, root in sic_fallback_roots or []:
            chain.append((name, AsiSampler(root, grid="n6250") if name == "asi_n6250" else SicSampler(root)))
        sic = FallbackSicSampler(chain)
    work = Path(work_dir) if work_dir else output_dir / "work"
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
    # skip AppleDouble "._" companions that macOS writes on non-native volumes
    parts = sorted(p for p in work.glob("*.scored.parquet") if not p.name.startswith("._"))
    keep_cols = [f.name for f in schema]
    def decided_month(p: Path) -> pd.DataFrame:
        return apply_exclusions(pd.read_parquet(p), v3)

    excluded = pd.concat([m[m.qc3_exclusion != ""] for m in map(decided_month, parts)], ignore_index=True)
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
        month = decided_month(p)
        input_total += len(month)
        kept = month[month.qc3_exclusion == ""]
        kept = assign_segments(kept, excluded, base_id)
        for name, bit in FLAG_BITS.items():
            flag_counts[name] += int(((kept.qc3_flags.to_numpy() >> bit) & 1).sum())
        kept = kept.assign(qc3_neighbours_30km=kept.neighbours_30km.astype("int16"),
                           qc3_sic_start=kept.sic_start.astype("float32"), qc3_sic_end=kept.sic_end.astype("float32"))
        table_df = kept[keep_cols + ["qc3_trajectory_id", "qc3_segment", "qc3_flags", "qc3_flag_reasons",
                                     "qc3_neighbours_30km", "qc3_sic_start", "qc3_sic_end"]]
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
    out = ds.dataset(out_path).to_table(columns=["qc3_trajectory_id", "start_time_utc", "source_rowid",
                                                  "target_rowid"]).to_pandas()
    violations = continuity_violations(out)
    del out
    unique = edge_uids_unique_and_disjoint(out_path, excluded.edge_uid)
    manifest = {
        "status": "complete" if (violations == 0 and unique and kept_total + len(excluded) == input_total) else "failed",
        "period": period, "protocol_id": PROTOCOL_V3_ID, "protocol_sha256": sha256(PROTOCOL_V3_PATH),
        "parameter_overrides": overrides, "work_dir": str(work),
        "input": str(input_path), "input_sha256": sha256(input_path),
        "land_raster": str(land_path), "land_raster_sha256": land.sha256,
        "sic_root": str(sic_root) if sic_root else None, "sic_product": sic_product if sic_root else None,
        "sic_files_used": len(sic.files) if sic else 0,
        "sic_days_by_product": {k: sorted(v) if len(v) <= 20 else len(v) for k, v in sic.days_by_product.items()} if sic else None,
        "input_links": input_total, "kept_links": kept_total, "excluded_links": int(len(excluded)),
        "exclusions_by_reason": counts, "kept_links_by_flag": flag_counts,
        "new_trajectory_id_base": base_id, "continuity_violations": violations, "edge_uid_unique_and_disjoint": bool(unique),
        "outputs": {out_path.name: sha256(out_path),
                    f"{period}_qc3_exclusions.parquet": sha256(output_dir / f"{period}_qc3_exclusions.parquet")},
    }
    (output_dir / f"{period}_qc3.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def edge_uids_unique_and_disjoint(path: Path, excluded_uids: pd.Series) -> bool:
    """Hash-based check (64-bit) that output edge_uids are unique and none was excluded."""

    hashes = [pd.util.hash_array(b.column("edge_uid").to_numpy(zero_copy_only=False))
              for b in pq.ParquetFile(path).iter_batches(columns=["edge_uid"], batch_size=1_000_000)]
    kept = np.concatenate(hashes) if hashes else np.empty(0, np.uint64)
    return bool(len(np.unique(kept)) == len(kept)
                and not np.isin(pd.util.hash_array(excluded_uids.to_numpy()), kept).any())


def pc_max(path: Path, column: str) -> int:
    meta = pq.ParquetFile(path).metadata
    index = pq.ParquetFile(path).schema_arrow.get_field_index(column)
    return max(meta.row_group(i).column(index).statistics.max for i in range(meta.num_row_groups))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply trajectory-link QC v3.1 to a v2 scientific edge Parquet.")
    parser.add_argument("--input", required=True, type=Path, help="v2 edge Parquet for one period")
    parser.add_argument("--period", required=True, help="period label used in output names")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--land-raster", required=True, type=Path, help=".npz inland-distance raster (EPSG:3413)")
    parser.add_argument("--sic-root", type=Path, help="SIC root for the open-water exclusion")
    parser.add_argument("--sic-product", choices=["asi", "nsidc"], default="asi",
                        help="asi: Bremen AMSR2 ASI 3.125 km GeoTIFF (default); nsidc: NOAA/NSIDC CDR v6 25 km")
    parser.add_argument("--sic-fallback-asi6250", type=Path, help="Bremen AMSR2 ASI 6.25 km root for days missing at 3.125 km")
    parser.add_argument("--sic-fallback-nsidc", type=Path, help="NOAA/NSIDC CDR v6 root for days missing in the ASI products")
    parser.add_argument("--support-tolerance-m", type=float,
                        help="override the interpolated-support tolerance (recorded in the manifest)")
    parser.add_argument("--work-dir", type=Path, help="per-month scoring directory (reuse one scoring for several tolerances)")
    a = parser.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = ([("asi_n6250", a.sic_fallback_asi6250)] if a.sic_fallback_asi6250 else []) + \
        ([("nsidc", a.sic_fallback_nsidc)] if a.sic_fallback_nsidc else [])
    manifest = run(a.input, a.output_dir, a.land_raster, a.sic_root, a.period, a.sic_product, fallbacks,
                   a.support_tolerance_m, a.work_dir)
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit("v3 QC output failed its checks; see the manifest")


if __name__ == "__main__":
    main()
