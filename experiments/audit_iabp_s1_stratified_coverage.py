#!/usr/bin/env python3
"""Audit independent IABP/Sentinel-1 coverage and propose targeted acquisitions.

This experiment separates four quantities that raw spatial-join row counts mix:
exact SAR-time buoy positions, independent buoys, Sentinel-1 scenes/passes, and
200 km spatial blocks.  It also applies provisional on-ice QC using daily
NOAA/NSIDC CDR v6 sea-ice concentration and local buoy-track continuity.

The output is an acquisition *manifest*, not a downloader.  Marginal-ice-zone
sequences remain blocked until IABP platform metadata (and, where possible,
surface temperature or higher-resolution SIC) confirms that the platform was
on ice.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
import xarray as xr
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT_DIR = ROOT / "results/arctic_sequence_extension_inventory"
DEFAULT_IABP_MATCHES = Path(
    "/Users/seachu/data/shared/iapb_buoy_satellite_matches_2020_01_04_new.geojson"
)
DEFAULT_IMAGE_CATALOG = Path(
    "/Users/seachu/data/shared/s1_2020_01_04_sie_filtered_new.geojson"
)
DEFAULT_SIC_ROOT = Path(
    "/Volumes/KINGSTON/arktalas/external/sea_ice_concentration/"
    "noaa_nsidc_cdr_v6"
)
DEFAULT_FROZEN_FIXTURES = ROOT / "results/arctic_fixture_ledger/q2q98_clahe25"
DEFAULT_OUTPUT = ROOT / "results/iabp_s1_stratified_coverage"

ANALYSIS_CRS = "EPSG:3413"
SIC_CRS = "EPSG:3411"


@dataclass(frozen=True)
class AuditThresholds:
    spatial_block_m: float = 200_000.0
    maximum_track_bracket_hours: float = 6.0
    gross_track_speed_m_per_day: float = 100_000.0
    descriptor_border_pixels: int = 128
    processing_pixel_size_m: float = 82.0
    maximum_sequence_gap_hours: float = 72.0
    frames_per_sequence: int = 4

    @property
    def descriptor_border_m(self) -> float:
        return self.descriptor_border_pixels * self.processing_pixel_size_m


@dataclass(frozen=True)
class SicGrid:
    x: np.ndarray
    y: np.ndarray
    sic: np.ndarray
    stdev: np.ndarray
    qa: np.ndarray
    spatial_interp: np.ndarray
    temporal_interp: np.ndarray


class SicGridCache:
    def __init__(self, root: Path, max_days: int = 16) -> None:
        self.root = root
        self.max_days = max_days
        self._cache: OrderedDict[str, SicGrid] = OrderedDict()

    def get(self, day: pd.Timestamp) -> SicGrid:
        key = day.strftime("%Y-%m-%d")
        if key in self._cache:
            grid = self._cache.pop(key)
            self._cache[key] = grid
            return grid
        path = (
            self.root
            / "north"
            / "daily"
            / f"{day:%Y}"
            / f"sic_psn25_{day:%Y%m%d}_F17_v06r00.nc"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            grid = SicGrid(
                x=ds["x"].values.astype("float64"),
                y=ds["y"].values.astype("float64"),
                sic=ds["cdr_seaice_conc"].isel(time=0).values.astype("float32"),
                stdev=ds["cdr_seaice_conc_stdev"].isel(time=0).values.astype(
                    "float32"
                ),
                qa=ds["cdr_seaice_conc_qa_flag"].isel(time=0).values.astype(
                    "uint8"
                ),
                spatial_interp=ds["cdr_seaice_conc_interp_spatial_flag"]
                .isel(time=0)
                .values.astype("uint8"),
                temporal_interp=ds["cdr_seaice_conc_interp_temporal_flag"]
                .isel(time=0)
                .values.astype("uint8"),
            )
        self._cache[key] = grid
        while len(self._cache) > self.max_days:
            self._cache.popitem(last=False)
        return grid


def clean_buoy_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def sic_regime(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    if float(value) < 0.15:
        return "open_water_lt15"
    if float(value) < 0.80:
        return "marginal_ice_15_80"
    return "pack_ice_ge80"


def cadence_band(hours: float | int | None) -> str:
    if hours is None or not np.isfinite(hours) or float(hours) <= 0:
        return "not_a_transition"
    value = float(hours)
    if value < 6:
        return "under_6h"
    if value < 12:
        return "6_to_12h"
    if value < 24:
        return "12_to_24h"
    if value < 48:
        return "24_to_48h"
    if value < 96:
        return "48_to_96h"
    return "over_96h"


def spatial_block_id(x: float, y: float, block_m: float) -> str:
    if not np.isfinite(x) or not np.isfinite(y):
        return "missing"
    return f"x{math.floor(x / block_m):+04d}_y{math.floor(y / block_m):+04d}"


def load_exact_positions(exact_dir: Path) -> pd.DataFrame:
    paths = sorted(exact_dir.glob("potential_exact_time_coincidences_2020_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No exact coincidence CSVs under {exact_dir}")
    frames = [pd.read_csv(path, dtype={"buoy_id": "string"}) for path in paths]
    out = pd.concat(frames, ignore_index=True)
    out["buoy_id"] = clean_buoy_id(out["buoy_id"])
    out["image_time"] = pd.to_datetime(out["image_time"], utc=True)
    out["image_id"] = pd.to_numeric(out["image_id"], errors="raise").astype("int64")
    out = out.drop_duplicates(["buoy_id", "image_id"]).copy()
    return out.sort_values(["buoy_id", "image_time", "image_id"]).reset_index(
        drop=True
    )


def load_buoy_tracks(path: Path) -> pd.DataFrame:
    rows = pyogrio.read_dataframe(
        path,
        columns=["BuoyID", "timestamp", "x", "y"],
        read_geometry=False,
    ).rename(columns={"BuoyID": "buoy_id"})
    rows["buoy_id"] = clean_buoy_id(rows["buoy_id"])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows[["x", "y"]] = rows[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return (
        rows.dropna(subset=["buoy_id", "timestamp", "x", "y"])
        .groupby(["buoy_id", "timestamp"], as_index=False)[["x", "y"]]
        .mean()
        .sort_values(["buoy_id", "timestamp"])
        .reset_index(drop=True)
    )


def bracket_track_quality(
    exact: pd.DataFrame,
    tracks: pd.DataFrame,
    maximum_gap_hours: float,
    gross_speed_m_per_day: float,
) -> pd.DataFrame:
    """Measure the observed track segment bracketing each SAR acquisition."""

    by_buoy = {key: group for key, group in tracks.groupby("buoy_id", sort=False)}
    records: list[dict[str, float | bool]] = []
    for row in exact[["buoy_id", "image_time"]].itertuples(index=False):
        track = by_buoy.get(row.buoy_id)
        if track is None or len(track) < 2:
            records.append(
                {
                    "track_bracketed": False,
                    "track_bracket_gap_hours": np.nan,
                    "track_bracket_speed_m_per_day": np.nan,
                    "track_interpolation_fraction": np.nan,
                    "track_qc_status": "unbracketed_track_context",
                    "track_qc_pass": False,
                }
            )
            continue
        # Normalize explicitly because pandas may preserve microsecond input
        # resolution while Timestamp.value is always nanoseconds.
        times = pd.DatetimeIndex(track["timestamp"]).as_unit("ns").asi8
        target = pd.Timestamp(row.image_time).value
        right = int(np.searchsorted(times, target, side="right"))
        left = right - 1
        if right >= len(track) and left >= 0 and times[left] == target:
            right = left
            left -= 1
        if left < 0 or right >= len(track):
            records.append(
                {
                    "track_bracketed": False,
                    "track_bracket_gap_hours": np.nan,
                    "track_bracket_speed_m_per_day": np.nan,
                    "track_interpolation_fraction": np.nan,
                    "track_qc_status": "unbracketed_track_context",
                    "track_qc_pass": False,
                }
            )
            continue
        gap_hours = (times[right] - times[left]) / 3.6e12
        distance_m = float(
            np.hypot(
                track.iloc[right]["x"] - track.iloc[left]["x"],
                track.iloc[right]["y"] - track.iloc[left]["y"],
            )
        )
        speed = distance_m / (gap_hours / 24.0) if gap_hours > 0 else np.inf
        fraction = (
            (target - times[left]) / (times[right] - times[left])
            if times[right] > times[left]
            else np.nan
        )
        passed = (
            gap_hours > 0
            and gap_hours <= maximum_gap_hours
            and np.isfinite(speed)
            and speed <= gross_speed_m_per_day
        )
        if gap_hours <= 0:
            qc_status = "nonpositive_track_time_exclude"
        elif speed > gross_speed_m_per_day:
            qc_status = "gross_track_speed_exclude"
        elif gap_hours > maximum_gap_hours:
            # The local source is a SAR-coincidence table, not the complete
            # Level-1 track. A long bracket is missing context, not evidence
            # that the reported position itself is erroneous.
            qc_status = "track_context_gap_unverified"
        else:
            qc_status = "pass"
        records.append(
            {
                "track_bracketed": True,
                "track_bracket_gap_hours": float(gap_hours),
                "track_bracket_speed_m_per_day": speed,
                "track_interpolation_fraction": float(fraction),
                "track_qc_status": qc_status,
                "track_qc_pass": bool(passed),
            }
        )
    return pd.DataFrame.from_records(records, index=exact.index)


def sample_sic(exact: pd.DataFrame, sic_root: Path) -> pd.DataFrame:
    transformer = Transformer.from_crs(ANALYSIS_CRS, SIC_CRS, always_xy=True)
    sx, sy = transformer.transform(
        exact["x"].to_numpy(dtype=float), exact["y"].to_numpy(dtype=float)
    )
    dates = exact["image_time"].dt.floor("D").dt.tz_localize(None)
    n = len(exact)
    result = {
        "sic_fraction": np.full(n, np.nan, dtype="float32"),
        "sic_stdev": np.full(n, np.nan, dtype="float32"),
        "sic_qa_flag": np.full(n, 255, dtype="uint8"),
        "sic_spatial_interp_flag": np.full(n, 255, dtype="uint8"),
        "sic_temporal_interp_flag": np.full(n, 255, dtype="uint8"),
        "sic_grid_distance_m": np.full(n, np.nan, dtype="float32"),
    }
    cache = SicGridCache(sic_root)
    date_values = dates.to_numpy()
    for day in sorted(pd.unique(dates.dropna())):
        positions = np.flatnonzero(date_values == np.datetime64(day))
        grid = cache.get(pd.Timestamp(day))
        x_step = float(grid.x[1] - grid.x[0])
        y_step = float(grid.y[1] - grid.y[0])
        ix = np.rint((sx[positions] - grid.x[0]) / x_step).astype("int32")
        iy = np.rint((sy[positions] - grid.y[0]) / y_step).astype("int32")
        inside = (ix >= 0) & (ix < len(grid.x)) & (iy >= 0) & (iy < len(grid.y))
        target = positions[inside]
        rows = iy[inside]
        cols = ix[inside]
        result["sic_fraction"][target] = grid.sic[rows, cols]
        result["sic_stdev"][target] = grid.stdev[rows, cols]
        result["sic_qa_flag"][target] = grid.qa[rows, cols]
        result["sic_spatial_interp_flag"][target] = grid.spatial_interp[rows, cols]
        result["sic_temporal_interp_flag"][target] = grid.temporal_interp[rows, cols]
        result["sic_grid_distance_m"][target] = np.hypot(
            sx[target] - grid.x[cols], sy[target] - grid.y[rows]
        )
    out = pd.DataFrame(result, index=exact.index)
    out.insert(0, "sic_date", dates.dt.strftime("%Y-%m-%d"))
    return out


def load_catalog(path: Path) -> gpd.GeoDataFrame:
    catalog = gpd.read_file(path).copy()
    catalog["image_id"] = pd.to_numeric(catalog["image_id"], errors="raise").astype(
        "int64"
    )
    catalog["image_filename"] = catalog["filename"].astype(str).map(
        lambda value: Path(value).name
    )
    catalog["orbit_num"] = catalog["orbit_num"].astype("string")
    catalog["platform"] = catalog["image_filename"].str.slice(0, 3)
    catalog["acquisition_pass_id"] = (
        catalog["platform"] + "_orbit_" + catalog["orbit_num"].str.zfill(6)
    )
    return catalog[
        [
            "image_id",
            "image_filename",
            "orbit_num",
            "platform",
            "acquisition_pass_id",
            "filepath",
            "geometry",
        ]
    ].drop_duplicates("image_id")


def scene_geometry_quality(
    exact: pd.DataFrame, catalog_geometry: pd.Series, safe_border_m: float
) -> pd.DataFrame:
    scenes = np.asarray(catalog_geometry, dtype=object)
    points = shapely.points(
        exact["x"].to_numpy(dtype=float), exact["y"].to_numpy(dtype=float)
    )
    missing = np.fromiter((item is None for item in scenes), dtype=bool, count=len(scenes))
    contains = np.zeros(len(exact), dtype=bool)
    signed_distance = np.full(len(exact), np.nan, dtype=float)
    valid = ~missing
    if valid.any():
        contains[valid] = shapely.covers(scenes[valid], points[valid])
        boundary_distance = shapely.distance(
            shapely.boundary(scenes[valid]), points[valid]
        )
        outside_distance = shapely.distance(scenes[valid], points[valid])
        signed_distance[valid] = np.where(
            contains[valid], boundary_distance, -outside_distance
        )
    return pd.DataFrame(
        {
            "exact_position_inside_scene": contains,
            "signed_scene_boundary_distance_m": signed_distance,
            "descriptor_border_safe": contains & (signed_distance >= safe_border_m),
        },
        index=exact.index,
    )


def frozen_pixel_filenames(fixture_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(fixture_dir.glob("coincidences_2020_*.csv")):
        frame = pd.read_csv(path, usecols=["image_filename"])
        names.update(frame["image_filename"].astype(str))
    return names


def audit_frozen_pixel_scales(fixture_dir: Path) -> pd.DataFrame:
    """Measure GCP scale in metres/pixel for every frozen standard-VAE raster."""

    paths: set[Path] = set()
    for fixture_path in sorted(fixture_dir.glob("coincidences_2020_*.csv")):
        frame = pd.read_csv(fixture_path, usecols=["image_filepath"])
        paths.update(Path(value) for value in frame["image_filepath"].astype(str))
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        if not path.exists():
            continue
        metadata = json.loads(
            subprocess.check_output(["gdalinfo", "-json", str(path)], text=True)
        )
        gcps = pd.DataFrame(metadata["gcps"]["gcpList"])
        scales: list[float] = []
        for _, line in gcps.groupby("line", sort=False):
            line = line.sort_values("pixel")
            dp = line["pixel"].diff().to_numpy(dtype=float)[1:]
            dx = line["x"].diff().to_numpy(dtype=float)[1:]
            dy = line["y"].diff().to_numpy(dtype=float)[1:]
            valid = dp > 0
            scales.extend((np.hypot(dx[valid], dy[valid]) / dp[valid]).tolist())
        for _, column in gcps.groupby("pixel", sort=False):
            column = column.sort_values("line")
            dp = column["line"].diff().to_numpy(dtype=float)[1:]
            dx = column["x"].diff().to_numpy(dtype=float)[1:]
            dy = column["y"].diff().to_numpy(dtype=float)[1:]
            valid = dp > 0
            scales.extend((np.hypot(dx[valid], dy[valid]) / dp[valid]).tolist())
        if not scales:
            raise ValueError(f"No usable GCP pairs in {path}")
        records.append(
            {
                "image_filename": path.name,
                "image_filepath": str(path),
                "width_pixels": int(metadata["size"][0]),
                "height_pixels": int(metadata["size"][1]),
                "median_gcp_scale_m_per_pixel": float(np.median(scales)),
                "minimum_gcp_scale_m_per_pixel": float(np.min(scales)),
                "maximum_gcp_scale_m_per_pixel": float(np.max(scales)),
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No local frozen standard-VAE rasters found under {fixture_dir}"
        )
    return pd.DataFrame.from_records(records)


def assign_buoy_ice_status(frame: pd.DataFrame) -> pd.Series:
    status = np.full(len(frame), "missing_sic_exclude", dtype=object)
    track_status = frame["track_qc_status"].astype(str).to_numpy()
    track_unverified = np.isin(
        track_status, ["unbracketed_track_context", "track_context_gap_unverified"]
    )
    track_bad = np.isin(
        track_status,
        ["gross_track_speed_exclude", "nonpositive_track_time_exclude"],
    )
    sic = frame["sic_fraction"].to_numpy(dtype=float)
    finite = np.isfinite(sic)
    status[track_unverified] = "track_context_unverified"
    status[track_bad] = "gross_track_speed_exclude"
    usable = ~track_unverified & ~track_bad & finite
    status[usable & (sic < 0.15)] = "open_water_exclude"
    status[usable & (sic >= 0.15) & (sic < 0.80)] = (
        "provisional_miz_needs_platform_qc"
    )
    status[usable & (sic >= 0.80)] = "on_ice_high_confidence_from_sic_track"
    return pd.Series(status, index=frame.index, dtype="string")


def add_transition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["buoy_id", "image_time", "image_id"]).copy()
    previous_time = out.groupby("buoy_id", sort=False)["image_time"].shift()
    previous_x = out.groupby("buoy_id", sort=False)["x"].shift()
    previous_y = out.groupby("buoy_id", sort=False)["y"].shift()
    out["previous_gap_hours"] = (
        out["image_time"] - previous_time
    ).dt.total_seconds() / 3600.0
    out["previous_displacement_m"] = np.hypot(
        out["x"] - previous_x, out["y"] - previous_y
    )
    out["previous_speed_m_per_day"] = out["previous_displacement_m"] / (
        out["previous_gap_hours"] / 24.0
    )
    out["cadence_band"] = out["previous_gap_hours"].map(cadence_band)
    return out.sort_index()


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (month, status), group in frame.groupby(
        ["month", "buoy_ice_qc_status"], dropna=False, sort=True
    ):
        transitions = group["previous_gap_hours"].where(
            group["previous_gap_hours"] > 0
        )
        rows.append(
            {
                "month": month,
                "buoy_ice_qc_status": status,
                "exact_positions": len(group),
                "independent_buoys": group["buoy_id"].nunique(),
                "sentinel1_scenes": group["image_id"].nunique(),
                "sentinel1_passes": group["acquisition_pass_id"].nunique(),
                "spatial_blocks_200km": group["spatial_block"].nunique(),
                "descriptor_border_safe_positions": int(
                    group["descriptor_border_safe"].sum()
                ),
                "local_standard_vae_positions": int(
                    group["standard_vae_pixels_local"].sum()
                ),
                "positive_gap_transitions": int(transitions.notna().sum()),
                "median_gap_hours": float(transitions.median()),
            }
        )
    return pd.DataFrame.from_records(rows)


def spatial_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["spatial_block", "month", "sic_regime"], dropna=False)
        .agg(
            exact_positions=("image_id", "size"),
            independent_buoys=("buoy_id", "nunique"),
            sentinel1_scenes=("image_id", "nunique"),
            sentinel1_passes=("acquisition_pass_id", "nunique"),
            local_standard_vae_positions=("standard_vae_pixels_local", "sum"),
            centroid_x_m=("x", "mean"),
            centroid_y_m=("y", "mean"),
            centroid_latitude=("latitude", "mean"),
            centroid_longitude=("longitude", "mean"),
        )
        .reset_index()
        .sort_values(["month", "sic_regime", "spatial_block"])
    )


def motion_limit_sensitivity(
    frame: pd.DataFrame,
    limits_km_per_day: tuple[float, ...] = (20, 30, 40, 50, 75, 100),
    maximum_gap_hours: float = 72.0,
) -> pd.DataFrame:
    """Count buoy-truth transitions excluded by candidate speed limits."""

    collapsed = (
        frame.sort_values(
            [
                "buoy_id",
                "month",
                "acquisition_pass_id",
                "signed_scene_boundary_distance_m",
            ],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["buoy_id", "month", "acquisition_pass_id"])
        .sort_values(["buoy_id", "month", "image_time"])
        .copy()
    )
    groups = collapsed.groupby(["buoy_id", "month"], sort=False)
    previous_time = groups["image_time"].shift()
    previous_x = groups["x"].shift()
    previous_y = groups["y"].shift()
    previous_status = groups["buoy_ice_qc_status"].shift()
    collapsed["transition_gap_hours"] = (
        collapsed["image_time"] - previous_time
    ).dt.total_seconds() / 3600.0
    collapsed["transition_speed_km_per_day"] = (
        np.hypot(collapsed["x"] - previous_x, collapsed["y"] - previous_y)
        / 1000.0
        / (collapsed["transition_gap_hours"] / 24.0)
    )
    eligible_status = {
        "on_ice_high_confidence_from_sic_track",
        "provisional_miz_needs_platform_qc",
    }
    eligible = collapsed[
        collapsed["buoy_ice_qc_status"].isin(eligible_status)
        & previous_status.isin(eligible_status)
        & collapsed["transition_gap_hours"].gt(0)
        & collapsed["transition_gap_hours"].le(maximum_gap_hours)
    ]
    rows: list[dict[str, object]] = []
    for month, group in eligible.groupby("month", sort=True):
        for limit in limits_km_per_day:
            excluded = group["transition_speed_km_per_day"].gt(limit)
            rows.append(
                {
                    "month": month,
                    "maximum_candidate_speed_km_per_day": limit,
                    "eligible_truth_transitions": len(group),
                    "truth_transitions_above_limit": int(excluded.sum()),
                    "truth_transitions_above_limit_percent": float(
                        100.0 * excluded.mean()
                    ),
                    "maximum_observed_truth_speed_km_per_day": float(
                        group["transition_speed_km_per_day"].max()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def candidate_windows(
    frame: pd.DataFrame, thresholds: AuditThresholds
) -> pd.DataFrame:
    eligible = frame[
        frame["buoy_ice_qc_status"].isin(
            [
                "on_ice_high_confidence_from_sic_track",
                "provisional_miz_needs_platform_qc",
            ]
        )
        & frame["descriptor_border_safe"]
    ].copy()
    eligible = eligible.sort_values(
        [
            "buoy_id",
            "month",
            "acquisition_pass_id",
            "signed_scene_boundary_distance_m",
        ],
        ascending=[True, True, True, False],
    ).drop_duplicates(["buoy_id", "month", "acquisition_pass_id"])
    eligible = eligible.sort_values(["buoy_id", "month", "image_time", "image_id"])
    records: list[dict[str, object]] = []
    n_frames = thresholds.frames_per_sequence
    for (buoy_id, month), group in eligible.groupby(
        ["buoy_id", "month"], sort=True
    ):
        group = group.reset_index(drop=True)
        for start in range(0, len(group) - n_frames + 1):
            window = group.iloc[start : start + n_frames]
            gaps = window["image_time"].diff().dt.total_seconds().div(3600).dropna()
            if (gaps <= 0).any() or (
                gaps > thresholds.maximum_sequence_gap_hours
            ).any():
                continue
            if window["buoy_ice_qc_status"].eq(
                "provisional_miz_needs_platform_qc"
            ).any():
                regime = "miz_requires_platform_qc"
            else:
                regime = "pack_ice"
            blocks = window["spatial_block"].value_counts()
            records.append(
                {
                    "buoy_id": buoy_id,
                    "month": month,
                    "sequence_regime": regime,
                    "primary_spatial_block": blocks.index[0],
                    "first_time": window["image_time"].iloc[0],
                    "last_time": window["image_time"].iloc[-1],
                    "duration_hours": (
                        window["image_time"].iloc[-1]
                        - window["image_time"].iloc[0]
                    ).total_seconds()
                    / 3600.0,
                    "median_gap_hours": float(gaps.median()),
                    "minimum_sic_fraction": float(window["sic_fraction"].min()),
                    "median_scene_boundary_distance_m": float(
                        window["signed_scene_boundary_distance_m"].median()
                    ),
                    "local_frames": int(window["standard_vae_pixels_local"].sum()),
                    "image_ids": tuple(window["image_id"].astype(int)),
                }
            )
    return pd.DataFrame.from_records(records)


def select_diverse_sequences(
    windows: pd.DataFrame,
    frame: pd.DataFrame,
    pack_per_month: int,
    miz_per_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if windows.empty:
        return windows.copy(), pd.DataFrame()
    selected: list[pd.Series] = []
    for (month, regime), pool in windows.groupby(
        ["month", "sequence_regime"], sort=True
    ):
        target = pack_per_month if regime == "pack_ice" else miz_per_month
        pool = pool.copy()
        used_buoys: set[str] = set()
        used_blocks: set[str] = set()
        for _ in range(target):
            available = pool[~pool["buoy_id"].isin(used_buoys)].copy()
            if available.empty:
                break
            available["diversity_score"] = (
                (~available["primary_spatial_block"].isin(used_blocks)).astype(int)
                * 1000
                + (available["local_frames"] == 0).astype(int) * 100
                + np.minimum(
                    available["median_scene_boundary_distance_m"] / 1000.0, 99
                )
            )
            chosen = available.sort_values(
                [
                    "diversity_score",
                    "median_scene_boundary_distance_m",
                    "duration_hours",
                ],
                ascending=[False, False, True],
            ).iloc[0]
            selected.append(chosen)
            used_buoys.add(str(chosen["buoy_id"]))
            used_blocks.add(str(chosen["primary_spatial_block"]))
            chosen_images = tuple(chosen["image_ids"])
            pool = pool[
                ~pool["image_ids"].map(lambda value: tuple(value) == chosen_images)
            ]
    selected_frame = pd.DataFrame(selected).reset_index(drop=True)
    if selected_frame.empty:
        return selected_frame, pd.DataFrame()
    selected_frame = selected_frame.sort_values(
        ["month", "sequence_regime", "primary_spatial_block", "first_time"]
    ).reset_index(drop=True)
    counters: dict[tuple[str, str], int] = {}
    ids: list[str] = []
    for row in selected_frame.itertuples(index=False):
        key = (row.month, row.sequence_regime)
        counters[key] = counters.get(key, 0) + 1
        readable_regime = "pack" if row.sequence_regime == "pack_ice" else "miz_qc"
        ids.append(
            f"{row.month.replace('-', '_')}_{readable_regime}_"
            f"{counters[key]:02d}_buoy_{row.buoy_id}"
        )
    selected_frame.insert(0, "sequence_id", ids)

    frame_by_id = frame.set_index("image_id", drop=False)
    frame_records: list[dict[str, object]] = []
    for sequence in selected_frame.itertuples(index=False):
        for order, image_id in enumerate(sequence.image_ids, start=1):
            item = frame_by_id.loc[int(image_id)]
            if isinstance(item, pd.DataFrame):
                item = item.iloc[0]
            frame_records.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "frame_order": order,
                    "sequence_regime": sequence.sequence_regime,
                    "buoy_id": sequence.buoy_id,
                    "spatial_block": sequence.primary_spatial_block,
                    "image_id": int(image_id),
                    "image_time": item["image_time"],
                    "image_filename": item["image_filename"],
                    "sentinel1_product_name": Path(item["image_filename"]).stem,
                    "acquisition_pass_id": item["acquisition_pass_id"],
                    "orbit_num": item["orbit_num"],
                    "sic_fraction": item["sic_fraction"],
                    "scene_boundary_distance_m": item[
                        "signed_scene_boundary_distance_m"
                    ],
                    "standard_vae_pixels_local": item[
                        "standard_vae_pixels_local"
                    ],
                }
            )
    return selected_frame, pd.DataFrame.from_records(frame_records)


def acquisition_manifest(sequence_frames: pd.DataFrame) -> pd.DataFrame:
    if sequence_frames.empty:
        return sequence_frames.copy()
    rows: list[dict[str, object]] = []
    for image_id, group in sequence_frames.groupby("image_id", sort=False):
        first = group.iloc[0]
        regimes = set(group["sequence_regime"])
        pack = "pack_ice" in regimes
        month = pd.Timestamp(first["image_time"]).strftime("%Y-%m")
        rows.append(
            {
                "priority_tier": 1 if pack and month in {"2020-01", "2020-04"} else 2,
                "download_decision": (
                    "ready_for_restore_or_download"
                    if regimes == {"pack_ice"}
                    else "wait_for_iabp_platform_qc"
                ),
                "image_id": int(image_id),
                "image_time": first["image_time"],
                "image_filename": first["image_filename"],
                "sentinel1_product_name": first["sentinel1_product_name"],
                "acquisition_pass_id": first["acquisition_pass_id"],
                "orbit_num": first["orbit_num"],
                "sequence_ids": ";".join(sorted(group["sequence_id"].unique())),
                "buoy_ids": ";".join(sorted(group["buoy_id"].astype(str).unique())),
                "standard_vae_pixels_local": bool(
                    group["standard_vae_pixels_local"].any()
                ),
                "required_preprocessing": "standard_vae",
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["download_decision", "priority_tier", "image_time", "image_id"]
    )


def platform_qc_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for buoy_id, group in selected.groupby("buoy_id", sort=True):
        regimes = set(group["sequence_regime"])
        needs_miz_qc = "miz_requires_platform_qc" in regimes
        rows.append(
            {
                "buoy_id": buoy_id,
                "selected_sequence_ids": ";".join(sorted(group["sequence_id"])),
                "months": ";".join(sorted(group["month"].unique())),
                "sequence_regimes": ";".join(sorted(regimes)),
                "qc_priority": "required_before_sar_download" if needs_miz_qc else "verify_before_split_freeze",
                "iabp_level1_download_url": (
                    "https://iabp.apl.uw.edu/downloadL1?bid="
                    f"{buoy_id}&requesttype=bybuoy&option=download"
                ),
                "required_checks": (
                    "platform_type_not_ocean_drifter;level1_position_qc;"
                    "track_continuity;surface_temperature_if_available"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.1f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *[
                "| " + " | ".join(map(str, row)) + " |"
                for row in view.to_numpy()
            ],
        ]
    )


def write_report(
    path: Path,
    frame: pd.DataFrame,
    coverage: pd.DataFrame,
    selected: pd.DataFrame,
    acquisitions: pd.DataFrame,
    thresholds: AuditThresholds,
    pixel_scales: pd.DataFrame,
    motion_limits: pd.DataFrame,
) -> None:
    local = frame[frame["standard_vae_pixels_local"]]
    on_ice = frame[
        frame["buoy_ice_qc_status"].eq(
            "on_ice_high_confidence_from_sic_track"
        )
    ]
    miz = frame[
        frame["buoy_ice_qc_status"].eq(
            "provisional_miz_needs_platform_qc"
        )
    ]
    track_context = frame[
        frame["buoy_ice_qc_status"].eq("track_context_unverified")
    ]
    summary_columns = [
        "month",
        "buoy_ice_qc_status",
        "exact_positions",
        "independent_buoys",
        "sentinel1_scenes",
        "sentinel1_passes",
        "spatial_blocks_200km",
        "local_standard_vae_positions",
    ]
    selected_by_regime = (
        selected.groupby(["month", "sequence_regime"]).size().reset_index(name="sequences")
        if not selected.empty
        else pd.DataFrame(columns=["month", "sequence_regime", "sequences"])
    )
    at_50 = motion_limits[
        motion_limits["maximum_candidate_speed_km_per_day"].eq(50)
    ]
    total_motion_transitions = int(at_50["eligible_truth_transitions"].sum())
    over_50 = int(at_50["truth_transitions_above_limit"].sum())
    text = f"""# Stratified IABP / Sentinel-1 coverage audit

## Decision

The January-April catalogue is large enough to design a pack-ice experiment, but the local pixel-ready subset is not a stratified descriptor-training set. The full audit contains {len(frame):,} exact SAR-time positions from {frame.buoy_id.nunique():,} buoys, {frame.image_id.nunique():,} scenes, {frame.acquisition_pass_id.nunique():,} acquisition passes, and {frame.spatial_block.nunique():,} 200 km blocks. The standard-VAE subset contains only {len(local):,} positions, {local.buoy_id.nunique():,} buoys, {local.image_id.nunique():,} scenes, and {local.spatial_block.nunique():,} blocks.

After track-continuity and daily SIC checks, {len(on_ice):,} positions are high-confidence on-ice candidates from SIC plus track behaviour. The {len(miz):,} marginal-ice positions remain provisional: SIC alone cannot prove that an IABP platform is attached to ice. A further {len(track_context):,} positions are labelled `track_context_unverified`, not rejected: the local coincidence table leaves a >{thresholds.maximum_track_bracket_hours:g} h gap around them, so complete Level-1 tracks are needed.

Exact-time interpolation moves {int((~frame.exact_position_inside_scene).sum()):,} positions just outside their recorded scene footprint. In total, {int((~frame.descriptor_border_safe).sum()):,} positions fail the explicit {thresholds.descriptor_border_pixels}-pixel scene-edge safety check and are not proposed for patch extraction.

## Independence-aware coverage

{markdown_table(coverage, summary_columns)}

## Acquisition proposal

{markdown_table(selected_by_regime, ["month", "sequence_regime", "sequences"])}

The selected set uses readable sequence IDs, four acquisition passes per sequence, distinct buoys within each month/regime, and spatial-block diversity. The {len(selected):,} sequences contain {selected.buoy_id.nunique() if len(selected) else 0:,} unique buoys and {selected.primary_spatial_block.nunique() if len(selected) else 0:,} primary blocks. Repeated buoys across months are retained as useful longitudinal cases but must remain in one final split. The proposal requires {len(acquisitions):,} unique Sentinel-1 scenes; {int(acquisitions.standard_vae_pixels_local.sum()) if len(acquisitions) else 0:,} are already local.

Pack-ice rows marked `ready_for_restore_or_download` can proceed. MIZ rows marked `wait_for_iabp_platform_qc` must first be joined to official IABP Level-1 platform metadata; exclude ocean drifters and require either an ice-specific buoy type, credible sub-freezing surface temperature, or agreement with a higher-resolution SIC/ice-edge product.

## QC contract

- Positions are linearly interpolated in EPSG:3413 to exact SAR acquisition time; temporal extrapolation is forbidden upstream.
- The observed buoy segment bracketing each acquisition must span no more than {thresholds.maximum_track_bracket_hours:g} h and move no faster than {thresholds.gross_track_speed_m_per_day / 1000:g} km/day. This is a gross position-QC ceiling, not LiMOSAT's 50 km/day candidate-search setting.
- Daily NOAA/NSIDC CDR v6 SIC is sampled at the exact position/date with SIC, standard deviation, QA, and interpolation flags retained. <15% is excluded; 15-80% is provisional; >=80% is the primary training pool.
- The exact buoy position must lie inside the scene and at least {thresholds.descriptor_border_m / 1000:g} km from its footprint boundary. The {thresholds.processing_pixel_size_m:g} m/pixel value is a conservative ceiling derived from {len(pixel_scales)} local standard-VAE rasters (median GCP scale {pixel_scales.median_gcp_scale_m_per_pixel.median():.2f} m/pixel; observed maximum {pixel_scales.maximum_gcp_scale_m_per_pixel.max():.2f}), so it safely represents the current {thresholds.descriptor_border_pixels}-pixel whole-image border. Verify restored rasters against the same contract.
- NOAA/NSIDC is 25 km SIC. MIZ decisions require a second, finer-resolution product; the connected AMSR2 archive currently does not cover January-April 2020.

## Split contract

Do not split observations at random. Assign complete buoy IDs and complete acquisition passes to one role only. Report performance by month, SIC regime, cadence, 200 km block, and scene/pass—not just pooled transitions. Keep April 2020 as a high-cadence temporal stress set and N-ICE2015 as the materially different external holdout. A final train/validation assignment should be frozen only after the selected pixels and IABP platform metadata have passed QC.

The current 50 km/day physics gate excludes {over_50} of {total_motion_transitions:,} eligible <=72 h buoy-truth transitions ({100.0 * over_50 / total_motion_transitions:.3f}%). This is a small real tail, not zero. Keep 50 km/day as the primary arm, but report 40/50/75/100 km/day sensitivity so descriptor failures are not silently relabelled as physics failures.

## Next concrete action

Fetch Level-1 metadata only for the selected buoy IDs, resolve or restore the tier-1 January/April pack-ice scenes, verify their raster pixel scale/footprints, run the unchanged standard VAE preprocessing, and then rerun the descriptor/update baseline on these named sequences. Do not bulk-download the annual IABP file or all 12,080 Sentinel-1 catalogue scenes first.
"""
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", type=Path, default=DEFAULT_EXACT_DIR)
    parser.add_argument("--iabp-matches", type=Path, default=DEFAULT_IABP_MATCHES)
    parser.add_argument("--image-catalog", type=Path, default=DEFAULT_IMAGE_CATALOG)
    parser.add_argument("--sic-root", type=Path, default=DEFAULT_SIC_ROOT)
    parser.add_argument("--frozen-fixtures", type=Path, default=DEFAULT_FROZEN_FIXTURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pack-sequences-per-month", type=int, default=4)
    parser.add_argument("--miz-sequences-per-month", type=int, default=2)
    parser.add_argument("--spatial-block-km", type=float, default=200.0)
    parser.add_argument("--track-bracket-hours", type=float, default=6.0)
    parser.add_argument("--gross-track-speed-km-day", type=float, default=100.0)
    parser.add_argument("--descriptor-border-pixels", type=int, default=128)
    parser.add_argument(
        "--processing-pixel-size-m",
        type=float,
        default=None,
        help="Override empirical conservative scale derived from frozen VAE rasters",
    )
    parser.add_argument("--maximum-sequence-gap-hours", type=float, default=72.0)
    parser.add_argument("--frames-per-sequence", type=int, default=4)
    args = parser.parse_args()
    pixel_scales = audit_frozen_pixel_scales(args.frozen_fixtures)
    empirical_safe_pixel_size_m = float(
        math.ceil(pixel_scales["maximum_gcp_scale_m_per_pixel"].max())
    )
    processing_pixel_size_m = (
        args.processing_pixel_size_m
        if args.processing_pixel_size_m is not None
        else empirical_safe_pixel_size_m
    )
    thresholds = AuditThresholds(
        spatial_block_m=args.spatial_block_km * 1000.0,
        maximum_track_bracket_hours=args.track_bracket_hours,
        gross_track_speed_m_per_day=args.gross_track_speed_km_day * 1000.0,
        descriptor_border_pixels=args.descriptor_border_pixels,
        processing_pixel_size_m=processing_pixel_size_m,
        maximum_sequence_gap_hours=args.maximum_sequence_gap_hours,
        frames_per_sequence=args.frames_per_sequence,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    exact = load_exact_positions(args.exact_dir)
    tracks = load_buoy_tracks(args.iabp_matches)
    exact = pd.concat(
        [
            exact,
            bracket_track_quality(
                exact,
                tracks,
                thresholds.maximum_track_bracket_hours,
                thresholds.gross_track_speed_m_per_day,
            ),
            sample_sic(exact, args.sic_root),
        ],
        axis=1,
    )
    catalog = load_catalog(args.image_catalog)
    exact = exact.merge(catalog, how="left", on="image_id", suffixes=("", "_catalog"))
    if exact["image_filename_catalog"].notna().any():
        filename_mismatch = (
            exact["image_filename_catalog"].notna()
            & exact["image_filename"].ne(exact["image_filename_catalog"])
        )
        if filename_mismatch.any():
            raise ValueError(f"{int(filename_mismatch.sum())} image filename mismatches")
    exact = pd.concat(
        [
            exact.drop(columns=["geometry"]),
            scene_geometry_quality(
                exact, exact["geometry"], thresholds.descriptor_border_m
            ),
        ],
        axis=1,
    )
    lon, lat = Transformer.from_crs(
        ANALYSIS_CRS, "EPSG:4326", always_xy=True
    ).transform(exact["x"].to_numpy(dtype=float), exact["y"].to_numpy(dtype=float))
    exact["longitude"] = lon
    exact["latitude"] = lat
    exact["month"] = exact["image_time"].dt.strftime("%Y-%m")
    exact["spatial_block"] = [
        spatial_block_id(x, y, thresholds.spatial_block_m)
        for x, y in exact[["x", "y"]].itertuples(index=False)
    ]
    exact["sic_regime"] = exact["sic_fraction"].map(sic_regime)
    exact["buoy_ice_qc_status"] = assign_buoy_ice_status(exact)
    exact["sic_no_interpolation"] = (
        exact["sic_spatial_interp_flag"].eq(0)
        & exact["sic_temporal_interp_flag"].eq(0)
    )
    local_names = frozen_pixel_filenames(args.frozen_fixtures)
    exact["standard_vae_pixels_local"] = exact["image_filename"].isin(local_names)
    exact = add_transition_columns(exact)

    coverage = coverage_summary(exact)
    spatial = spatial_summary(exact)
    motion_limits = motion_limit_sensitivity(
        exact, maximum_gap_hours=thresholds.maximum_sequence_gap_hours
    )
    windows = candidate_windows(exact, thresholds)
    selected, sequence_frames = select_diverse_sequences(
        windows,
        exact,
        pack_per_month=args.pack_sequences_per_month,
        miz_per_month=args.miz_sequences_per_month,
    )
    acquisitions = acquisition_manifest(sequence_frames)
    platform_qc = platform_qc_manifest(selected)

    exact.drop(columns=["filepath"], errors="ignore").to_csv(
        args.out_dir / "exact_coincidence_qc.csv", index=False
    )
    coverage.to_csv(args.out_dir / "coverage_by_month_qc_status.csv", index=False)
    spatial.to_csv(args.out_dir / "coverage_by_spatial_block.csv", index=False)
    motion_limits.to_csv(
        args.out_dir / "true_motion_limit_sensitivity.csv", index=False
    )
    windows.to_csv(args.out_dir / "candidate_sequence_windows.csv", index=False)
    selected.to_csv(args.out_dir / "selected_sequences.csv", index=False)
    sequence_frames.to_csv(args.out_dir / "selected_sequence_frames.csv", index=False)
    acquisitions.to_csv(args.out_dir / "sentinel1_acquisition_manifest.csv", index=False)
    platform_qc.to_csv(args.out_dir / "iabp_platform_qc_manifest.csv", index=False)
    pixel_scales.to_csv(args.out_dir / "local_standard_vae_pixel_scales.csv", index=False)
    write_report(
        args.out_dir / "report.md",
        exact,
        coverage,
        selected,
        acquisitions,
        thresholds,
        pixel_scales,
        motion_limits,
    )
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": {
            "exact_dir": str(args.exact_dir),
            "iabp_matches": str(args.iabp_matches),
            "image_catalog": str(args.image_catalog),
            "sic_root": str(args.sic_root),
            "frozen_fixtures": str(args.frozen_fixtures),
        },
        "analysis_crs": ANALYSIS_CRS,
        "sic_crs": SIC_CRS,
        "thresholds": thresholds.__dict__,
        "pixel_scale_audit": {
            "rasters": len(pixel_scales),
            "median_of_raster_median_m_per_pixel": float(
                pixel_scales["median_gcp_scale_m_per_pixel"].median()
            ),
            "maximum_observed_m_per_pixel": float(
                pixel_scales["maximum_gcp_scale_m_per_pixel"].max()
            ),
            "conservative_selected_m_per_pixel": processing_pixel_size_m,
        },
        "selection": {
            "pack_sequences_per_month": args.pack_sequences_per_month,
            "miz_sequences_per_month": args.miz_sequences_per_month,
        },
        "counts": {
            "exact_positions": len(exact),
            "buoys": int(exact["buoy_id"].nunique()),
            "scenes": int(exact["image_id"].nunique()),
            "passes": int(exact["acquisition_pass_id"].nunique()),
            "spatial_blocks": int(exact["spatial_block"].nunique()),
            "selected_sequences": len(selected),
            "selected_unique_scenes": len(acquisitions),
        },
        "split_frozen": False,
        "reason_split_not_frozen": (
            "selected pixels and IABP platform metadata have not yet passed final QC"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(coverage.to_string(index=False))
    print(f"\nSelected sequences: {len(selected)}")
    print(f"Unique scenes in acquisition manifest: {len(acquisitions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
