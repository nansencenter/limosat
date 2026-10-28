"""LiMOSAT QC v1: one sequential pass of link QC over the regular matching output.

Each link between consecutive keypoints of a trajectory is tested by the rules below, in order. A rule
only sees the links that survived the earlier rules, so the rules we are most confident in act first
and later neighbour tests use cleaner neighbours. Every removed link gets the code of the step that
removed it and splits its trajectory.

0. ``SPEED``: speed over the actual image gap above 60 km/day (physically impossible).
1. ``OPEN_WATER``: sea-ice concentration below 15% at either endpoint (links with no value kept);
   ``LAND``: the deeper endpoint at least 1 km inland on the supplied land raster.
2. ``ISOLATED_UNVERIFIED``: no other link of the image pair within 30 km, and a gap over 54 h or a
   speed over 25 km/day.
3. ``ROGUE_VECTOR``: more than 2 km from the leave-one-out local-affine prediction of its image pair,
   at least five times the neighbours' own spread, and no neighbour corroborates it.
4. ``INTERP_UNSUPPORTED``: an interpolated link without support from the 16 nearest links of the pair
   within 50 km: at least 3 of them, and within 1 km of their median motion, within 2 km with target
   correlation >= 0.40, or within 3 km with target correlation >= 0.50.

Kept links carry ``pair_direct_share`` (share of the image pair's input links that are direct matches)
and flags ``FEW_NEIGHBOURS`` (fewer than 2 other kept links within 30 km) and ``LOW_PAIR_DIRECT_SHARE``
(pair direct share below 0.40). Thresholds are in ``protocols/limosat_qc_v1.json``. EPSG:3413 metres.
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
PROTOCOL_PATH = Path(__file__).resolve().parent / "protocols" / "limosat_qc_v1.json"
STEPS = ("SPEED", "OPEN_WATER", "LAND", "ISOLATED_UNVERIFIED", "ROGUE_VECTOR", "INTERP_UNSUPPORTED")
FLAG_BITS = {"FEW_NEIGHBOURS": 0, "LOW_PAIR_DIRECT_SHARE": 1}


@dataclass(frozen=True)
class QCv1Config:
    """Thresholds of LiMOSAT QC v1 (metres, km for inland distance, SIC as a fraction)."""

    max_speed_m_per_day: float = 60_000.0
    open_water_sic: float = 0.15
    land_min_inland_km: float = 1.0
    neighbour_radius_m: float = 30_000.0
    isolated_max_hours: float = 54.0
    isolated_max_speed_m_per_day: float = 25_000.0
    rogue_residual_m: float = 2_000.0
    rogue_coherence_min: float = 5.0
    rogue_spread_floor_m: float = 150.0
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
        raise ValueError("Bundled QC v1 protocol ID does not match the code")
    return protocol


def validate_protocol(config: QCv1Config) -> None:
    """Fail if the code defaults and the bundled protocol have drifted apart."""

    parameters = load_protocol()["parameters"]
    mismatches = {f: (v, parameters.get(f)) for f, v in asdict(config).items() if v != parameters.get(f)}
    if mismatches:
        raise ValueError(f"QC v1 configuration does not match the protocol: {mismatches}")


def _by_pair(d: pd.DataFrame, alive: np.ndarray):
    """Row indices of the surviving links of each image pair."""

    sub = np.flatnonzero(alive)
    keys = d[["source_image_id", "target_image_id"]].to_numpy()[sub]
    for index in pd.DataFrame(keys).groupby([0, 1], sort=False).indices.values():
        yield sub[index]


def _support(xy: np.ndarray, uv: np.ndarray, cfg: QCv1Config):
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


def decide(links: pd.DataFrame, cfg: QCv1Config, land, sic, local: QCConfig | None = None) -> pd.DataFrame:
    """Run the QC v1 steps on the links of whole image pairs; returns the links with ``qc_exclusion``
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
    rogue = alive & (supported & (d.local_residual_m > cfg.rogue_residual_m)
                     & (d.coherence_ratio >= cfg.rogue_coherence_min) & (d.n_corroborating == 0)).to_numpy()
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
