#!/usr/bin/env python3
"""Pilot ATL07/ATL10 validation of ORB and ALIKED deformation fields."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import find_peaks
from scipy.spatial import Delaunay
from scipy.stats import spearmanr
import shapely

if __package__:
    from experiments.multisensor_event_ledger import (
        DeformationFieldIdentity,
        dataframe_sha256,
        deterministic_checkpoints,
        exact_common_support,
        file_sha256,
        write_event_ledger,
    )
else:
    from multisensor_event_ledger import (
        DeformationFieldIdentity,
        dataframe_sha256,
        deterministic_checkpoints,
        exact_common_support,
        file_sha256,
        write_event_ledger,
    )


ATLAS_EPOCH_UTC = pd.Timestamp("2018-01-01T00:00:00Z")


def _values(variable, fill_value=np.nan) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[:], fill_value))


def atlas_utc(delta_time_seconds: np.ndarray) -> pd.DatetimeIndex:
    """Convert ATL10 delta_time to UTC using the ATLAS SDP epoch."""
    return ATLAS_EPOCH_UTC + pd.to_timedelta(delta_time_seconds, unit="s")


def load_atl10(path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with Dataset(path) as dataset:
        for beam in sorted(name for name in dataset.groups if name.startswith("gt")):
            group = dataset.groups[beam]
            shared = {
                "beam": beam,
                "beam_type": group.getncattr("atlas_beam_type"),
                "spot_number": int(group.getncattr("atlas_spot_number")),
            }
            freeboard = group.groups["freeboard_segment"].variables
            delta_time = _values(freeboard["delta_time"])
            rows.append(
                pd.DataFrame(
                    {
                        **shared,
                        "observation_type": "freeboard",
                        "time_utc": atlas_utc(delta_time),
                        "latitude": _values(freeboard["latitude"]),
                        "longitude": _values(freeboard["longitude"]),
                        "along_track_m": _values(freeboard["seg_dist_x"]),
                        "freeboard_m": _values(freeboard["beam_fb_height"]),
                        "freeboard_uncertainty_m": _values(freeboard["beam_fb_unc"]),
                        "freeboard_quality": _values(
                            freeboard["beam_fb_quality_flag"], fill_value=-1
                        ).astype(int),
                        "freeboard_confidence": _values(
                            freeboard["beam_fb_confidence"]
                        ),
                        "lead_length_m": np.nan,
                        "lead_height_m": np.nan,
                        "lead_sigma_m": np.nan,
                    }
                )
            )
            leads = group.groups["leads"].variables
            delta_time = _values(leads["delta_time"])
            rows.append(
                pd.DataFrame(
                    {
                        **shared,
                        "observation_type": "lead",
                        "time_utc": atlas_utc(delta_time),
                        "latitude": _values(leads["latitude"]),
                        "longitude": _values(leads["longitude"]),
                        "along_track_m": _values(leads["lead_dist_x"]),
                        "freeboard_m": np.nan,
                        "freeboard_uncertainty_m": np.nan,
                        "freeboard_quality": -1,
                        "freeboard_confidence": np.nan,
                        "lead_length_m": _values(leads["lead_length"]),
                        "lead_height_m": _values(leads["lead_height"]),
                        "lead_sigma_m": _values(leads["lead_sigma"]),
                    }
                )
            )
    result = pd.concat(rows, ignore_index=True)
    finite = np.isfinite(result["latitude"]) & np.isfinite(result["longitude"])
    result = result.loc[finite].reset_index(drop=True)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    result["laser_x"], result["laser_y"] = project.transform(
        result["longitude"].to_numpy(), result["latitude"].to_numpy()
    )
    return result


def load_atl07(path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with Dataset(path) as dataset:
        for beam in sorted(name for name in dataset.groups if name.startswith("gt")):
            group = dataset.groups[beam]
            segments = group.groups["sea_ice_segments"]
            heights = segments.groups["heights"].variables
            stats = segments.groups["stats"].variables
            delta_time = _values(segments.variables["delta_time"])
            rows.append(
                pd.DataFrame(
                    {
                        "beam": beam,
                        "beam_type": group.getncattr("atlas_beam_type"),
                        "spot_number": int(group.getncattr("atlas_spot_number")),
                        "time_utc": atlas_utc(delta_time),
                        "latitude": _values(segments.variables["latitude"]),
                        "longitude": _values(segments.variables["longitude"]),
                        "along_track_m": _values(segments.variables["seg_dist_x"]),
                        "surface_height_m": _values(heights["height_segment_height"]),
                        "segment_length_m": _values(heights["height_segment_length_seg"]),
                        "surface_error_m": _values(
                            heights["height_segment_surface_error_est"]
                        ),
                        "gaussian_width_m": _values(
                            heights["height_segment_w_gaussian"]
                        ),
                        "fit_quality": _values(
                            heights["height_segment_fit_quality_flag"], fill_value=-1
                        ).astype(int),
                        "height_quality": _values(
                            heights["height_segment_quality"], fill_value=0
                        ).astype(int),
                        "ssh_flag": _values(
                            heights["height_segment_ssh_flag"], fill_value=-1
                        ).astype(int),
                        "surface_type": _values(
                            heights["height_segment_type"], fill_value=-1
                        ).astype(int),
                        "ice_concentration": _values(stats["ice_conc_amsr2"]),
                    }
                )
            )
    result = pd.concat(rows, ignore_index=True)
    finite = np.isfinite(result["latitude"]) & np.isfinite(result["longitude"])
    result = result.loc[finite].reset_index(drop=True)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    result["laser_x"], result["laser_y"] = project.transform(
        result["longitude"].to_numpy(), result["latitude"].to_numpy()
    )
    return add_atl07_relative_topography(result)


def add_atl07_relative_topography(
    observations: pd.DataFrame,
    baseline_length_m: float = 5000.0,
    ridge_threshold_m: float = 0.6,
) -> pd.DataFrame:
    """Detrend ATL07 and identify literature-prior ridge peaks without SAR tuning."""
    result = observations.copy()
    result["topography_valid"] = False
    result["baseline_height_m"] = np.nan
    result["relative_height_m"] = np.nan
    result["ridge_event"] = False
    for _, index in result.groupby("beam").groups.items():
        beam = result.loc[index].sort_values("along_track_m")
        good = (
            beam["beam_type"].eq("strong")
            & beam["height_quality"].eq(1)
            & beam["fit_quality"].isin([1, 2])
            & beam["ssh_flag"].eq(0)
            & np.isfinite(beam["surface_height_m"])
        )
        selected = beam.loc[good]
        if len(selected) < 5:
            continue
        spacing = float(np.nanmedian(np.diff(selected["along_track_m"])))
        window = max(5, int(round(baseline_length_m / max(spacing, 1.0))))
        if window % 2 == 0:
            window += 1
        baseline = selected["surface_height_m"].rolling(
            window=window,
            center=True,
            min_periods=max(3, window // 3),
        ).median()
        relative = selected["surface_height_m"] - baseline
        peak_distance = max(1, int(round(20.0 / max(spacing, 1.0))))
        peak_rows, _ = find_peaks(
            relative.fillna(-np.inf).to_numpy(),
            height=ridge_threshold_m,
            distance=peak_distance,
        )
        selected_index = selected.index.to_numpy()
        valid_index = selected.index[baseline.notna()]
        result.loc[valid_index, "topography_valid"] = True
        result.loc[selected.index, "baseline_height_m"] = baseline.to_numpy()
        result.loc[selected.index, "relative_height_m"] = relative.to_numpy()
        result.loc[selected_index[peak_rows], "ridge_event"] = True
    return result


def load_orb_vectors(
    database: Path, table: str, source_image_id: int, target_image_id: int
) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        rows = pd.read_sql_query(
            f'SELECT image_id, trajectory_id, geometry, interpolated, corr '
            f'FROM "{table}" WHERE image_id IN (?, ?)',
            connection,
            params=(source_image_id, target_image_id),
        )
    geometry = shapely.from_wkt(rows["geometry"].to_numpy())
    rows["x"] = shapely.get_x(geometry)
    rows["y"] = shapely.get_y(geometry)
    source = rows.loc[
        rows.image_id.eq(source_image_id), ["trajectory_id", "x", "y"]
    ].rename(columns={"x": "source_x", "y": "source_y"})
    target = rows.loc[
        rows.image_id.eq(target_image_id),
        ["trajectory_id", "x", "y", "interpolated", "corr"],
    ].rename(columns={"x": "target_x", "y": "target_y"})
    paired = source.merge(target, on="trajectory_id", validate="one_to_one")
    paired["dx_m"] = paired["target_x"] - paired["source_x"]
    paired["dy_m"] = paired["target_y"] - paired["source_y"]
    return paired


def load_aliked_vectors(path: Path) -> pd.DataFrame:
    field = pd.read_csv(path)
    available = field["available"].fillna(False).astype(bool)
    result = field.loc[available, ["source_x", "source_y", "proposal_dx_m", "proposal_dy_m"]].copy()
    return result.rename(columns={"proposal_dx_m": "dx_m", "proposal_dy_m": "dy_m"})


@dataclass
class TriangleDisplacementField:
    source: np.ndarray
    displacement: np.ndarray
    triangulation: Delaunay
    valid_triangle: np.ndarray
    gradient: np.ndarray

    @classmethod
    def build(
        cls,
        vectors: pd.DataFrame,
        maximum_edge_m: float,
        minimum_quality: float = 0.0,
    ) -> "TriangleDisplacementField":
        vectors = vectors.dropna(subset=["source_x", "source_y", "dx_m", "dy_m"])
        vectors = vectors.drop_duplicates(["source_x", "source_y"], keep="first")
        source = vectors[["source_x", "source_y"]].to_numpy(dtype=float)
        displacement = vectors[["dx_m", "dy_m"]].to_numpy(dtype=float)
        triangulation = Delaunay(source)
        triangles = triangulation.simplices
        source_triangles = source[triangles]
        target_triangles = source_triangles + displacement[triangles]
        source_edges = np.stack(
            (
                source_triangles[:, 1] - source_triangles[:, 0],
                source_triangles[:, 2] - source_triangles[:, 0],
            ),
            axis=2,
        )
        target_edges = np.stack(
            (
                target_triangles[:, 1] - target_triangles[:, 0],
                target_triangles[:, 2] - target_triangles[:, 0],
            ),
            axis=2,
        )
        edge_lengths = np.stack(
            (
                np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 0], axis=1),
                np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 1], axis=1),
                np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        source_cross = np.linalg.det(source_edges)
        target_cross = np.linalg.det(target_edges)
        finite = np.isfinite(source_cross) & (np.abs(source_cross) > 1.0)
        quality = 2.0 * np.sqrt(3.0) * np.abs(source_cross) / np.maximum(
            np.square(edge_lengths).sum(axis=1), 1.0
        )
        valid = (
            finite
            & (edge_lengths.max(axis=1) <= maximum_edge_m)
            & (quality >= minimum_quality)
            & (source_cross * target_cross > 0.0)
        )
        gradient = np.full((len(triangles), 2, 2), np.nan)
        gradient[finite] = (
            target_edges[finite] @ np.linalg.inv(source_edges[finite]) - np.eye(2)
        )
        return cls(source, displacement, triangulation, valid, gradient)

    def sample_displacement(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        simplex = self.triangulation.find_simplex(points)
        available = simplex >= 0
        available[available] &= self.valid_triangle[simplex[available]]
        result = np.full((len(points), 2), np.nan)
        if available.any():
            selected_simplex = simplex[available]
            transform = self.triangulation.transform[selected_simplex]
            first = np.einsum(
                "nij,nj->ni", transform[:, :2], points[available] - transform[:, 2]
            )
            weights = np.column_stack((first, 1.0 - first.sum(axis=1)))
            vertices = self.triangulation.simplices[selected_simplex]
            result[available] = np.einsum(
                "ni,nij->nj", weights, self.displacement[vertices]
            )
        return result, available

    def sample_deformation(
        self, points: np.ndarray, elapsed_days: float
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        simplex = self.triangulation.find_simplex(points)
        available = simplex >= 0
        available[available] &= self.valid_triangle[simplex[available]]
        gradient = np.full((len(points), 2, 2), np.nan)
        gradient[available] = self.gradient[simplex[available]] / elapsed_days
        divergence = np.trace(gradient, axis1=1, axis2=2)
        shear = np.hypot(
            gradient[:, 0, 0] - gradient[:, 1, 1],
            gradient[:, 0, 1] + gradient[:, 1, 0],
        )
        strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
        principal_strain = np.linalg.eigvalsh(strain)
        maximum_compression = np.maximum(-principal_strain[:, 0], 0.0)
        maximum_extension = np.maximum(principal_strain[:, 1], 0.0)
        return {
            "divergence_per_day": divergence,
            "shear_per_day": shear,
            "total_per_day": np.hypot(divergence, shear),
            "maximum_compression_per_day": maximum_compression,
            "maximum_extension_per_day": maximum_extension,
        }, available


def invert_to_source_time(
    field: TriangleDisplacementField,
    laser_xy: np.ndarray,
    interval_fraction: np.ndarray,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median_displacement = np.median(field.displacement, axis=0)
    source = laser_xy - interval_fraction[:, None] * median_displacement
    available = np.ones(len(laser_xy), dtype=bool)
    for _ in range(iterations):
        displacement, valid = field.sample_displacement(source)
        available &= valid
        update = laser_xy - interval_fraction[:, None] * displacement
        source[valid] = update[valid]
    displacement, valid = field.sample_displacement(source)
    residual = np.linalg.norm(
        source + interval_fraction[:, None] * displacement - laser_xy, axis=1
    )
    available &= valid & np.isfinite(residual) & (residual <= 1.0)
    return source, available, residual


def colocate_method(
    observations: pd.DataFrame,
    field: TriangleDisplacementField,
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
    prefix: str,
    endpoint_error_p90_m: float | None = None,
) -> pd.DataFrame:
    """Map laser observations to material points and retain a static control.

    The motion model is the pairwise linear trajectory
    ``x(t) = x0 + alpha * u(x0)``. The inverse is solved at every individual
    laser timestamp. Sampling the deformation directly at ``laser_xy`` is
    retained only as a no-advection control.
    """
    if endpoint_error_p90_m is not None and endpoint_error_p90_m < 0:
        raise ValueError("endpoint_error_p90_m must be non-negative")
    elapsed_days = (pair_end - pair_start).total_seconds() / 86400.0
    fraction = (
        (observations["time_utc"] - pair_start).dt.total_seconds()
        / (pair_end - pair_start).total_seconds()
    ).to_numpy()
    laser_xy = observations[["laser_x", "laser_y"]].to_numpy(dtype=float)
    source, invert_available, residual = invert_to_source_time(
        field, laser_xy, fraction
    )
    displacement, displacement_available = field.sample_displacement(source)
    deformation, deformation_available = field.sample_deformation(source, elapsed_days)
    inside_interval = (fraction >= 0) & (fraction <= 1)
    available = (
        invert_available
        & displacement_available
        & deformation_available
        & inside_interval
    )
    target = source + displacement
    drift_to_laser = laser_xy - source
    pair_displacement_m = np.linalg.norm(displacement, axis=1)
    drift_correction_m = np.linalg.norm(drift_to_laser, axis=1)
    remaining_drift_m = np.linalg.norm(target - laser_xy, axis=1)
    for values in (
        displacement,
        target,
        drift_to_laser,
    ):
        values[~available] = np.nan
    pair_displacement_m[~available] = np.nan
    drift_correction_m[~available] = np.nan
    remaining_drift_m[~available] = np.nan
    result = pd.DataFrame(
        {
            f"{prefix}_available": available,
            f"{prefix}_interval_fraction": fraction,
            f"{prefix}_source_x": source[:, 0],
            f"{prefix}_source_y": source[:, 1],
            f"{prefix}_target_x": target[:, 0],
            f"{prefix}_target_y": target[:, 1],
            f"{prefix}_pair_dx_m": displacement[:, 0],
            f"{prefix}_pair_dy_m": displacement[:, 1],
            f"{prefix}_pair_displacement_m": pair_displacement_m,
            f"{prefix}_drift_to_laser_dx_m": drift_to_laser[:, 0],
            f"{prefix}_drift_to_laser_dy_m": drift_to_laser[:, 1],
            f"{prefix}_drift_correction_m": drift_correction_m,
            f"{prefix}_remaining_drift_to_pair_end_m": remaining_drift_m,
            f"{prefix}_inversion_residual_m": residual,
        }
    )
    if endpoint_error_p90_m is None:
        result[f"{prefix}_position_error_p90_m"] = np.nan
    else:
        result[f"{prefix}_position_error_p90_m"] = (
            fraction * endpoint_error_p90_m
        )
        result.loc[~available, f"{prefix}_position_error_p90_m"] = np.nan
    for name, values in deformation.items():
        values = values.copy()
        values[~available] = np.nan
        result[f"{prefix}_{name}"] = values
    result[f"{prefix}_cumulative_opening"] = np.maximum(
        result[f"{prefix}_divergence_per_day"], 0.0
    ) * elapsed_days * fraction
    result[f"{prefix}_cumulative_convergence"] = np.maximum(
        -result[f"{prefix}_divergence_per_day"], 0.0
    ) * elapsed_days * fraction
    result[f"{prefix}_cumulative_maximum_compression"] = (
        result[f"{prefix}_maximum_compression_per_day"] * elapsed_days * fraction
    )
    result[f"{prefix}_cumulative_maximum_extension"] = (
        result[f"{prefix}_maximum_extension_per_day"] * elapsed_days * fraction
    )

    static_deformation, static_available = field.sample_deformation(
        laser_xy, elapsed_days
    )
    static_available &= inside_interval
    result[f"{prefix}_static_available"] = static_available
    for name, values in static_deformation.items():
        values = values.copy()
        values[~static_available] = np.nan
        result[f"{prefix}_static_{name}"] = values
    result[f"{prefix}_static_cumulative_opening"] = np.maximum(
        result[f"{prefix}_static_divergence_per_day"], 0.0
    ) * elapsed_days * fraction
    result[f"{prefix}_static_cumulative_convergence"] = np.maximum(
        -result[f"{prefix}_static_divergence_per_day"], 0.0
    ) * elapsed_days * fraction
    result[f"{prefix}_static_cumulative_maximum_compression"] = (
        result[f"{prefix}_static_maximum_compression_per_day"]
        * elapsed_days
        * fraction
    )
    result[f"{prefix}_static_cumulative_maximum_extension"] = (
        result[f"{prefix}_static_maximum_extension_per_day"]
        * elapsed_days
        * fraction
    )
    return result


def aggregate_track_bins(
    observations: pd.DataFrame, prefix: str, bin_size_m: float
) -> pd.DataFrame:
    data = observations.copy()
    data["track_bin"] = np.floor(data["along_track_m"] / bin_size_m).astype(int)
    freeboard = data[
        data["observation_type"].eq("freeboard")
        & data["beam_type"].eq("strong")
        & data["freeboard_quality"].isin([1, 2])
        & data[f"{prefix}_available"]
    ]
    leads = data[
        data["observation_type"].eq("lead")
        & data["beam_type"].eq("strong")
        & data[f"{prefix}_available"]
        & data["lead_length_m"].gt(0)
    ]
    value_columns = [
        f"{prefix}_divergence_per_day",
        f"{prefix}_shear_per_day",
        f"{prefix}_total_per_day",
        f"{prefix}_maximum_compression_per_day",
        f"{prefix}_maximum_extension_per_day",
        f"{prefix}_cumulative_opening",
        f"{prefix}_cumulative_convergence",
        f"{prefix}_cumulative_maximum_compression",
        f"{prefix}_cumulative_maximum_extension",
    ]
    bins = (
        freeboard.groupby(["beam", "track_bin"], as_index=False)
        .agg(
            along_track_m=("along_track_m", "median"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            freeboard_m=("freeboard_m", "median"),
            freeboard_segments=("freeboard_m", "size"),
            **{column: (column, "median") for column in value_columns},
        )
    )
    lead_bins = (
        leads.groupby(["beam", "track_bin"], as_index=False)
        .agg(lead_length_m=("lead_length_m", "sum"), lead_events=("lead_length_m", "size"))
    )
    bins = bins.merge(lead_bins, on=["beam", "track_bin"], how="left")
    bins[["lead_length_m", "lead_events"]] = bins[
        ["lead_length_m", "lead_events"]
    ].fillna(0)
    bins["lead_fraction"] = np.minimum(bins["lead_length_m"] / bin_size_m, 1.0)
    bins["method"] = prefix
    return bins


def aggregate_atl07_bins(
    observations: pd.DataFrame, prefix: str, bin_size_m: float
) -> pd.DataFrame:
    data = observations[
        observations["beam_type"].eq("strong")
        & observations["topography_valid"]
        & observations[f"{prefix}_available"]
    ].copy()
    data["ridge_relative_height_m"] = data["relative_height_m"].where(
        data["ridge_event"]
    )
    data["track_bin"] = np.floor(data["along_track_m"] / bin_size_m).astype(int)
    bins = (
        data.groupby(["beam", "track_bin"], as_index=False)
        .agg(
            along_track_m=("along_track_m", "median"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            observed_length_m=("segment_length_m", "sum"),
            segments=("surface_height_m", "size"),
            ridge_events=("ridge_event", "sum"),
            ridge_mean_height_m=("ridge_relative_height_m", "mean"),
            relative_height_p90_m=(
                "relative_height_m",
                lambda values: float(np.nanquantile(values, 0.9)),
            ),
            relative_height_p10_m=(
                "relative_height_m",
                lambda values: float(np.nanquantile(values, 0.1)),
            ),
            gaussian_width_m=("gaussian_width_m", "median"),
            relative_height_std_m=("relative_height_m", "std"),
            **{
                column: (column, "median")
                for column in (
                    f"{prefix}_divergence_per_day",
                    f"{prefix}_shear_per_day",
                    f"{prefix}_total_per_day",
                    f"{prefix}_maximum_compression_per_day",
                    f"{prefix}_maximum_extension_per_day",
                    f"{prefix}_cumulative_opening",
                    f"{prefix}_cumulative_convergence",
                    f"{prefix}_cumulative_maximum_compression",
                    f"{prefix}_cumulative_maximum_extension",
                )
            },
        )
    )
    bins["relative_roughness_m"] = (
        bins["relative_height_p90_m"] - bins["relative_height_p10_m"]
    )
    bins["ridge_density_per_km"] = bins["ridge_events"] / (
        bins["observed_length_m"] / 1000.0
    )
    bins["ridging_intensity_m_per_km"] = (
        bins["ridge_density_per_km"] * bins["ridge_mean_height_m"].fillna(0.0)
    )
    bins["method"] = prefix
    return bins


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 3:
        return None
    if np.unique(first[finite]).size < 2 or np.unique(second[finite]).size < 2:
        return None
    return float(spearmanr(first[finite], second[finite]).statistic)


def top_fraction_mask(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Select exactly the highest finite fraction, including when values tie."""
    finite = np.flatnonzero(np.isfinite(values))
    selected = np.zeros(len(values), dtype=bool)
    if finite.size == 0:
        return selected
    count = max(1, int(np.ceil(fraction * finite.size)))
    order = finite[np.argsort(values[finite], kind="stable")]
    selected[order[-count:]] = True
    return selected


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summarize_bins(bins: pd.DataFrame, prefix: str) -> dict:
    divergence = bins[f"{prefix}_divergence_per_day"].to_numpy(dtype=float)
    lead_fraction = bins["lead_fraction"].to_numpy(dtype=float)
    correlation = safe_spearman(divergence, lead_fraction)
    opening = bins[f"{prefix}_cumulative_opening"].to_numpy(dtype=float)
    high = top_fraction_mask(opening)
    high_mean = float(np.nanmean(lead_fraction[high])) if high.any() else np.nan
    other_mean = float(np.nanmean(lead_fraction[~high])) if (~high).any() else np.nan
    enrichment = high_mean / other_mean if other_mean > 0 else None
    lead_bins = int((lead_fraction > 0).sum())
    return {
        "bins": int(len(bins)),
        "bins_with_leads": lead_bins,
        "lead_length_m": float(bins["lead_length_m"].sum()),
        "spearman_divergence_vs_lead_fraction": correlation,
        "top_opening_quintile_lead_fraction": high_mean,
        "other_bins_lead_fraction": other_mean,
        "top_opening_quintile_enrichment": enrichment,
        "lead_inference_sufficient": lead_bins >= 20,
    }


def summarize_atl07_bins(bins: pd.DataFrame, prefix: str) -> dict:
    convergence = bins[f"{prefix}_cumulative_convergence"].to_numpy(dtype=float)
    compression = bins[
        f"{prefix}_cumulative_maximum_compression"
    ].to_numpy(dtype=float)
    shear = bins[f"{prefix}_shear_per_day"].to_numpy(dtype=float)
    ridge_density = bins["ridge_density_per_km"].to_numpy(dtype=float)
    ridging_intensity = bins["ridging_intensity_m_per_km"].to_numpy(dtype=float)
    roughness = bins["relative_roughness_m"].to_numpy(dtype=float)
    roughness_std = bins["relative_height_std_m"].to_numpy(dtype=float)
    high = top_fraction_mask(convergence)
    high_density = float(np.nanmean(ridge_density[high])) if high.any() else np.nan
    other_density = float(np.nanmean(ridge_density[~high])) if (~high).any() else np.nan
    enrichment = high_density / other_density if other_density > 0 else None
    ridge_events = int(bins["ridge_events"].sum())
    return {
        "bins": int(len(bins)),
        "ridge_events": ridge_events,
        "observed_length_km": float(bins["observed_length_m"].sum() / 1000.0),
        "spearman_convergence_vs_ridge_density": safe_spearman(
            convergence, ridge_density
        ),
        "spearman_maximum_compression_vs_ridging_intensity": safe_spearman(
            compression, ridging_intensity
        ),
        "spearman_shear_vs_relative_roughness": safe_spearman(shear, roughness),
        "spearman_shear_vs_relative_height_std": safe_spearman(
            shear, roughness_std
        ),
        "top_convergence_quintile_ridge_density_per_km": high_density,
        "other_bins_ridge_density_per_km": other_density,
        "top_convergence_quintile_ridge_enrichment": enrichment,
        "ridge_inference_sufficient": ridge_events >= 20,
    }


def circular_shift_test(
    bins: pd.DataFrame,
    predictor: str,
    response: str,
    repetitions: int = 999,
    minimum_shift_bins: int | None = None,
    bin_size_m: float = 4000.0,
    minimum_shift_m: float = 20_000.0,
    seed: int = 20260819,
) -> dict:
    if bin_size_m <= 0 or minimum_shift_m <= 0:
        raise ValueError("bin_size_m and minimum_shift_m must be positive")
    if minimum_shift_bins is None:
        minimum_shift_bins = max(1, int(np.ceil(minimum_shift_m / bin_size_m)))
    ordered = bins.sort_values(["beam", "track_bin"]).reset_index(drop=True)
    eligible_beams = [
        beam
        for beam, group in ordered.groupby("beam")
        if len(group) >= 2 * minimum_shift_bins
    ]
    ordered = ordered.loc[ordered["beam"].isin(eligible_beams)].reset_index(
        drop=True
    )
    observed = safe_spearman(
        ordered[predictor].to_numpy(dtype=float),
        ordered[response].to_numpy(dtype=float),
    )
    if observed is None:
        return {
            "observed": None,
            "repetitions": 0,
            "one_sided_p": None,
            "reason": "insufficient bins for a within-beam minimum shift",
        }
    rng = np.random.default_rng(seed)
    predictor_values = ordered[predictor].to_numpy(dtype=float)
    response_values = ordered[response].to_numpy(dtype=float)
    beam_indices = [indices.to_numpy() for _, indices in ordered.groupby("beam").groups.items()]
    null = []
    for _ in range(repetitions):
        shifted = predictor_values.copy()
        for indices in beam_indices:
            count = len(indices)
            allowed = np.arange(minimum_shift_bins, count - minimum_shift_bins + 1)
            if allowed.size == 0:
                continue
            shifted[indices] = np.roll(
                predictor_values[indices], int(rng.choice(allowed))
            )
        statistic = safe_spearman(shifted, response_values)
        if statistic is not None:
            null.append(statistic)
    null_values = np.asarray(null, dtype=float)
    return {
        "observed": observed,
        "repetitions": int(len(null_values)),
        "one_sided_p": float((1 + (null_values >= observed).sum()) / (len(null_values) + 1)),
        "null_p025": float(np.quantile(null_values, 0.025)),
        "null_median": float(np.median(null_values)),
        "null_p975": float(np.quantile(null_values, 0.975)),
        "minimum_shift_km": minimum_shift_bins * bin_size_m / 1000.0,
    }


def paired_common_support_comparison(
    orb: pd.DataFrame,
    aliked: pd.DataFrame,
    repetitions: int = 999,
    block_bins: int | None = None,
    bin_size_m: float = 4000.0,
    block_length_m: float = 20_000.0,
    seed: int = 20260819,
) -> dict:
    if bin_size_m <= 0 or block_length_m <= 0:
        raise ValueError("bin_size_m and block_length_m must be positive")
    if block_bins is None:
        block_bins = max(1, int(np.ceil(block_length_m / bin_size_m)))
    paired = orb.merge(
        aliked,
        on=["beam", "track_bin"],
        suffixes=("_orb", "_aliked"),
        validate="one_to_one",
    ).sort_values(["beam", "track_bin"])
    pairs = {
        "convergence_vs_ridge_density": (
            "orb_cumulative_convergence",
            "aliked_cumulative_convergence",
            "ridge_density_per_km_orb",
        ),
        "shear_vs_relative_roughness": (
            "orb_shear_per_day",
            "aliked_shear_per_day",
            "relative_roughness_m_orb",
        ),
        "maximum_compression_vs_ridging_intensity": (
            "orb_cumulative_maximum_compression",
            "aliked_cumulative_maximum_compression",
            "ridging_intensity_m_per_km_orb",
        ),
    }
    block_labels = []
    for beam, group in paired.groupby("beam", sort=False):
        for number, start in enumerate(range(0, len(group), block_bins)):
            block_labels.extend([(beam, number)] * min(block_bins, len(group) - start))
    paired = paired.assign(_block=block_labels)
    blocks = [group.index.to_numpy() for _, group in paired.groupby("_block", sort=False)]
    rng = np.random.default_rng(seed)
    result = {}
    for label, (orb_column, aliked_column, response_column) in pairs.items():
        response = paired[response_column].to_numpy(dtype=float)
        orb_values = paired[orb_column].to_numpy(dtype=float)
        aliked_values = paired[aliked_column].to_numpy(dtype=float)
        orb_stat = safe_spearman(orb_values, response)
        aliked_stat = safe_spearman(aliked_values, response)
        observed_difference = (
            aliked_stat - orb_stat
            if aliked_stat is not None and orb_stat is not None
            else None
        )
        if len(paired) < 20 or len(blocks) < 3:
            result[label] = {
                "orb": orb_stat,
                "aliked": aliked_stat,
                "aliked_minus_orb": observed_difference,
                "block_bootstrap_repetitions": 0,
                "difference_ci025": None,
                "difference_ci975": None,
                "block_length_km": block_bins * bin_size_m / 1000.0,
                "reason": (
                    "fewer than 20 common bins or three common-support "
                    "spatial blocks"
                ),
            }
            continue
        differences = []
        for _ in range(repetitions):
            selected = rng.integers(0, len(blocks), size=len(blocks))
            indices = np.concatenate([blocks[value] for value in selected])
            orb_sample = safe_spearman(orb_values[indices], response[indices])
            aliked_sample = safe_spearman(aliked_values[indices], response[indices])
            if orb_sample is not None and aliked_sample is not None:
                differences.append(aliked_sample - orb_sample)
        values = np.asarray(differences, dtype=float)
        result[label] = {
            "orb": orb_stat,
            "aliked": aliked_stat,
            "aliked_minus_orb": observed_difference,
            "block_bootstrap_repetitions": int(len(values)),
            "difference_ci025": float(np.quantile(values, 0.025)),
            "difference_ci975": float(np.quantile(values, 0.975)),
            "block_length_km": block_bins * bin_size_m / 1000.0,
        }
    return result


def drift_awareness_summary(
    observations: pd.DataFrame,
    prefix: str,
    product_name: str,
    bin_size_m: float,
) -> dict:
    """Quantify how material-point registration differs from a static overlay."""
    dynamic = observations[f"{prefix}_available"].fillna(False).to_numpy(bool)
    static = observations[f"{prefix}_static_available"].fillna(False).to_numpy(bool)
    common = dynamic & static
    correction = observations.loc[dynamic, f"{prefix}_drift_correction_m"]
    position_error = observations.loc[dynamic, f"{prefix}_position_error_p90_m"]
    result = {
        "dynamic_points": int(dynamic.sum()),
        "static_points": int(static.sum()),
        "common_points": int(common.sum()),
        "dynamic_only_points": int((dynamic & ~static).sum()),
        "static_only_points": int((static & ~dynamic).sum()),
        "drift_correction_m_p05_median_p95": (
            [float(value) for value in correction.quantile([0.05, 0.5, 0.95])]
            if len(correction)
            else []
        ),
        "approximate_position_error_p90_m_p05_median_p95": (
            [float(value) for value in position_error.quantile([0.05, 0.5, 0.95])]
            if position_error.notna().any()
            else []
        ),
        "interpretation": (
            "The static arm is a no-advection control and must not be used to "
            "choose tracker parameters or altimetry thresholds."
        ),
    }
    if not common.any():
        result["common_bin_comparison"] = None
        return result

    aggregate = aggregate_track_bins if product_name == "atl10" else aggregate_atl07_bins
    summarize = summarize_bins if product_name == "atl10" else summarize_atl07_bins
    common_observations = observations.loc[common].copy()
    dynamic_bins = aggregate(common_observations, prefix, bin_size_m)
    static_prefix = f"{prefix}_static"
    static_bins = aggregate(common_observations, static_prefix, bin_size_m)
    paired = dynamic_bins.merge(
        static_bins,
        on=["beam", "track_bin"],
        suffixes=("_dynamic", "_static"),
        validate="one_to_one",
    )
    comparisons = {}
    for name in (
        "divergence_per_day",
        "shear_per_day",
        "maximum_compression_per_day",
    ):
        dynamic_values = paired[f"{prefix}_{name}"].to_numpy(dtype=float)
        static_values = paired[f"{static_prefix}_{name}"].to_numpy(dtype=float)
        comparisons[name] = {
            "spearman": safe_spearman(dynamic_values, static_values),
            "median_absolute_difference": float(
                np.nanmedian(np.abs(dynamic_values - static_values))
            ),
        }
    result["common_bin_comparison"] = {
        "bins": int(len(paired)),
        "dynamic": summarize(dynamic_bins, prefix),
        "static_no_advection_control": summarize(static_bins, static_prefix),
        "dynamic_vs_static_deformation": comparisons,
    }
    return result


def compare_common_bins(orb: pd.DataFrame, aliked: pd.DataFrame) -> dict:
    paired = orb.merge(
        aliked,
        on=["beam", "track_bin"],
        suffixes=("_orb", "_aliked"),
        validate="one_to_one",
    )
    orb_divergence = paired["orb_divergence_per_day"].to_numpy(dtype=float)
    aliked_divergence = paired["aliked_divergence_per_day"].to_numpy(dtype=float)
    if paired.empty:
        return {
            "bins": 0,
            "divergence_spearman": None,
            "median_absolute_divergence_difference_per_day": None,
        }
    return {
        "bins": int(len(paired)),
        "divergence_spearman": safe_spearman(orb_divergence, aliked_divergence),
        "median_absolute_divergence_difference_per_day": float(
            np.nanmedian(np.abs(orb_divergence - aliked_divergence))
        ),
    }


def write_icesat2_event_ledger(
    args,
    summary: dict,
    product_name: str,
    product_path: Path,
    candidate_observations: pd.DataFrame,
    colocated_observations: pd.DataFrame,
    common_bin_count: int,
    orb_vectors: pd.DataFrame,
    aliked_vectors: pd.DataFrame,
) -> None:
    """Write shared ledger files without changing the scientific analysis."""
    if product_name == "atl07":
        product_qc = (
            colocated_observations["beam_type"].eq("strong")
            & colocated_observations["topography_valid"].fillna(False)
        )
    else:
        product_qc = colocated_observations["beam_type"].eq("strong") & (
            (
                colocated_observations["observation_type"].eq("freeboard")
                & colocated_observations["freeboard_quality"].isin([1, 2])
            )
            | (
                colocated_observations["observation_type"].eq("lead")
                & colocated_observations["lead_length_m"].gt(0)
            )
        )
    union = product_qc & (
        colocated_observations["orb_available"].fillna(False)
        | colocated_observations["aliked_available"].fillna(False)
    )
    common = exact_common_support(
        colocated_observations,
        ["orb_available", "aliked_available"],
        product_qc,
    )
    pair_start = pd.Timestamp(summary["pair_start_utc"])
    pair_end = pd.Timestamp(summary["pair_end_utc"])
    source_image_id = args.sar_source_product_id or args.orb_source_image_id
    target_image_id = args.sar_target_product_id or args.orb_target_image_id
    orb_hash = dataframe_sha256(
        orb_vectors, ["source_x", "source_y", "dx_m", "dy_m"]
    )
    aliked_hash = file_sha256(args.aliked_field)
    fields = [
        DeformationFieldIdentity(
            method="orb",
            field_id=f"orb:{source_image_id}:{target_image_id}:{orb_hash[:12]}",
            path=(
                f"{args.orb_database}::{args.orb_table}"
                f"[{args.orb_source_image_id},{args.orb_target_image_id}]"
            ),
            sha256=orb_hash,
            vector_count=len(orb_vectors),
            source_image_id=str(source_image_id),
            target_image_id=str(target_image_id),
            source_time_utc=pair_start.isoformat(),
            target_time_utc=pair_end.isoformat(),
            interpolation=(
                "linear barycentric interpolation on source-time Delaunay triangles; "
                "maximum edge 20000 m; minimum triangle quality 0.05; folded "
                "triangles rejected"
            ),
            boundary_rule=(
                "convex-hull exterior and invalid triangles excluded; valid triangle "
                "edges included by scipy Delaunay.find_simplex; no extrapolation"
            ),
        ),
        DeformationFieldIdentity(
            method="aliked",
            field_id=f"aliked:{source_image_id}:{target_image_id}:{aliked_hash[:12]}",
            path=str(args.aliked_field),
            sha256=aliked_hash,
            vector_count=len(aliked_vectors),
            source_image_id=str(source_image_id),
            target_image_id=str(target_image_id),
            source_time_utc=pair_start.isoformat(),
            target_time_utc=pair_end.isoformat(),
            interpolation=(
                "linear barycentric interpolation on source-time Delaunay triangles; "
                "maximum edge 6400 m; folded triangles rejected"
            ),
            boundary_rule=(
                "convex-hull exterior and invalid triangles excluded; valid triangle "
                "edges included by scipy Delaunay.find_simplex; no extrapolation"
            ),
        ),
    ]
    track_column = "beam"
    checkpoints = deterministic_checkpoints(
        colocated_observations, track_column, common
    )
    event_id = (
        f"{product_name}_{product_path.stem}_pair_{source_image_id}_{target_image_id}_"
        f"{int(args.bin_size_m)}m"
    )
    point_path = args.output_dir / f"{product_name}_colocated_points.csv"
    bin_path = args.output_dir / f"{product_name}_exact_common_bins.csv"
    analysis_role = (
        "insufficient_support_control"
        if summary["status"].startswith("insufficient")
        else args.analysis_role
    )
    write_event_ledger(
        args.output_dir,
        event_id=event_id,
        sensor="ICESat-2",
        product_identifiers=[product_path.name],
        product_time_start_utc=candidate_observations["time_utc"].min(),
        product_time_end_utc=candidate_observations["time_utc"].max(),
        pair_start_utc=pair_start,
        pair_end_utc=pair_end,
        source_image_id=str(source_image_id),
        target_image_id=str(target_image_id),
        inclusion_reason=args.candidate_inclusion_reason,
        analysis_role=analysis_role,
        result_status=summary["status"],
        selection_counts={
            "candidate_observations": len(candidate_observations),
            "temporally_eligible_observations": len(colocated_observations),
            "product_qc_survivors": int(product_qc.sum()),
            "spatially_supported_observations": int(union.sum()),
            "common_method_observations": int(common.sum()),
            "final_bins": common_bin_count,
        },
        reporting_resolution_m=args.bin_size_m,
        minimum_observations_per_bin=1,
        deformation_fields=fields,
        point_ledger_path=point_path,
        bin_ledger_path=bin_path,
        checkpoints=checkpoints,
        missing_support_reasons={
            "outside_sar_interval": len(candidate_observations)
            - len(colocated_observations),
            "failed_product_qc_or_morphology": len(colocated_observations)
            - int(product_qc.sum()),
            "no_deformation_method_support": int(product_qc.sum()) - int(union.sum()),
            "not_on_exact_common_method_support": int(union.sum())
            - int(common.sum()),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    product = parser.add_mutually_exclusive_group(required=True)
    product.add_argument("--atl10", type=Path)
    product.add_argument("--atl07", type=Path)
    parser.add_argument("--orb-database", type=Path, required=True)
    parser.add_argument("--orb-table", required=True)
    parser.add_argument("--orb-source-image-id", type=int, required=True)
    parser.add_argument("--orb-target-image-id", type=int, required=True)
    parser.add_argument("--aliked-field", type=Path, required=True)
    parser.add_argument("--pair-start", required=True)
    parser.add_argument("--pair-end", required=True)
    parser.add_argument("--bin-size-m", type=float, default=4000.0)
    parser.add_argument(
        "--orb-endpoint-error-p90-m",
        type=float,
        help="Frozen buoy-derived ORB endpoint-error P90 for uncertainty scaling.",
    )
    parser.add_argument(
        "--aliked-endpoint-error-p90-m",
        type=float,
        help="Frozen buoy-derived ALIKED endpoint-error P90 for uncertainty scaling.",
    )
    parser.add_argument(
        "--minimum-spatial-shift-m",
        type=float,
        default=20_000.0,
        help="Physical shift/block length used by spatial nulls and bootstrap.",
    )
    parser.add_argument("--sar-source-product-id")
    parser.add_argument("--sar-target-product-id")
    parser.add_argument(
        "--candidate-inclusion-reason",
        default=(
            "Frozen before outcome inspection from SAR-interval timing, granule "
            "geometry, and expected spatial support."
        ),
    )
    parser.add_argument(
        "--analysis-role",
        choices=("development", "confirmation", "independent_evaluation"),
        default="development",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pair_start = pd.Timestamp(args.pair_start)
    pair_end = pd.Timestamp(args.pair_end)
    if pair_start.tzinfo is None or pair_end.tzinfo is None:
        raise ValueError("Pair times must be timezone-aware")
    product_name = "atl10" if args.atl10 is not None else "atl07"
    product_path = args.atl10 if args.atl10 is not None else args.atl07
    candidate_observations = (
        load_atl10(product_path) if product_name == "atl10" else load_atl07(product_path)
    )
    inside_time = candidate_observations["time_utc"].between(pair_start, pair_end)
    observations = candidate_observations.loc[inside_time].reset_index(drop=True)
    observations_in_interval = int(len(observations))

    orb_vectors = load_orb_vectors(
        args.orb_database,
        args.orb_table,
        args.orb_source_image_id,
        args.orb_target_image_id,
    )
    aliked_vectors = load_aliked_vectors(args.aliked_field)
    fields = {
        "orb": TriangleDisplacementField.build(
            orb_vectors, maximum_edge_m=20_000.0, minimum_quality=0.05
        ),
        "aliked": TriangleDisplacementField.build(
            aliked_vectors, maximum_edge_m=6_400.0
        ),
    }
    endpoint_error_p90_m = {
        "orb": args.orb_endpoint_error_p90_m,
        "aliked": args.aliked_endpoint_error_p90_m,
    }
    for prefix, field in fields.items():
        observations = pd.concat(
            [
                observations,
                colocate_method(
                    observations,
                    field,
                    pair_start,
                    pair_end,
                    prefix,
                    endpoint_error_p90_m=endpoint_error_p90_m[prefix],
                ),
            ],
            axis=1,
        )
    drift_awareness = {
        prefix: drift_awareness_summary(
            observations, prefix, product_name, args.bin_size_m
        )
        for prefix in fields
    }
    colocated_observations = observations.copy()
    union_available = observations["orb_available"] | observations["aliked_available"]
    observations = observations.loc[union_available].reset_index(drop=True)
    common_observations = observations[
        observations["orb_available"] & observations["aliked_available"]
    ].copy()
    aggregate = aggregate_track_bins if product_name == "atl10" else aggregate_atl07_bins
    summarize = summarize_bins if product_name == "atl10" else summarize_atl07_bins
    bins = pd.concat(
        [aggregate(observations, prefix, args.bin_size_m) for prefix in fields],
        ignore_index=True,
    )
    common_bins = {
        prefix: aggregate(common_observations, prefix, args.bin_size_m)
        for prefix in fields
    }
    exact_common_bins = pd.concat(common_bins.values(), ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(
        args.output_dir / f"{product_name}_colocated_points.csv", index=False
    )
    bins.to_csv(args.output_dir / f"{product_name}_track_bins.csv", index=False)
    exact_common_bins.to_csv(
        args.output_dir / f"{product_name}_exact_common_bins.csv", index=False
    )
    if observations.empty:
        summary = {
            "status": "insufficient_method_support",
            "reason": (
                "No in-interval laser observations fall inside a valid ORB or "
                "ALIKED deformation triangle."
            ),
            "pair_start_utc": pair_start.isoformat(),
            "pair_end_utc": pair_end.isoformat(),
            "product": product_name.upper(),
            "product_path": str(product_path),
            "bin_size_m": args.bin_size_m,
            "orb_vectors": int(len(orb_vectors)),
            "aliked_vectors": int(len(aliked_vectors)),
            "observations_in_sar_interval": observations_in_interval,
            "observations_in_method_union": {
                "total": 0,
                "strong_beam": 0,
                "common": 0,
                "interval_fraction_p05_median_p95": [],
            },
            "methods": {
                prefix: {
                    "point_coverage_fraction_of_method_union": None,
                    "common_support": None,
                }
                for prefix in fields
            },
            "common_support_comparison": None,
            "drift_aware_colocation": {
                "reference_time": "SAR pair start",
                "trajectory_model": "x(t) = x0 + alpha * u(x0)",
                "laser_timestamp_handling": "per observation",
                "methods": drift_awareness,
            },
            "interpretation": (
                "This is a spatial-support result, not evidence about either "
                "deformation field's accuracy."
            ),
        }
        encoded = json.dumps(summary, indent=2, allow_nan=False)
        (args.output_dir / "summary.json").write_text(encoded + "\n")
        write_icesat2_event_ledger(
            args,
            summary,
            product_name,
            product_path,
            candidate_observations,
            colocated_observations,
            len(common_bins["orb"]),
            orb_vectors,
            aliked_vectors,
        )
        print(encoded)
        return 0
    observation_summary = {
        "total": int(len(observations)),
        "strong_beam": int(observations["beam_type"].eq("strong").sum()),
        "common": int(
            (observations["orb_available"] & observations["aliked_available"]).sum()
        ),
    }
    interval_fraction = (
        (observations["time_utc"] - pair_start).dt.total_seconds()
        / (pair_end - pair_start).total_seconds()
    ).to_numpy(dtype=float)
    observation_summary["interval_fraction_p05_median_p95"] = [
        float(value) for value in np.quantile(interval_fraction, [0.05, 0.5, 0.95])
    ]
    if product_name == "atl10":
        observation_summary.update(
            freeboard=int(observations["observation_type"].eq("freeboard").sum()),
            leads=int(observations["observation_type"].eq("lead").sum()),
        )
    else:
        observation_summary.update(
            topography_valid=int(observations["topography_valid"].sum()),
            ridge_events=int(observations["ridge_event"].sum()),
        )
    summary = {
        "status": (
            "complete"
            if len(common_bins["orb"])
            else "insufficient_topography_support"
        ),
        "pair_start_utc": pair_start.isoformat(),
        "pair_end_utc": pair_end.isoformat(),
        "product": product_name.upper(),
        "product_path": str(product_path),
        "bin_size_m": args.bin_size_m,
        "orb_vectors": int(len(orb_vectors)),
        "aliked_vectors": int(len(aliked_vectors)),
        "observations_in_method_union": observation_summary,
        "methods": {
            prefix: {
                "point_coverage_fraction_of_method_union": float(
                    observations[f"{prefix}_available"].mean()
                ),
                "inversion_residual_p99_m": float(
                    observations.loc[
                        observations[f"{prefix}_available"],
                        f"{prefix}_inversion_residual_m",
                    ].quantile(0.99)
                ),
                **summarize(bins[bins["method"].eq(prefix)], prefix),
                "common_support": summarize(common_bins[prefix], prefix),
            }
            for prefix in fields
        },
        "common_support_comparison": compare_common_bins(
            common_bins["orb"], common_bins["aliked"]
        ),
        "drift_aware_colocation": {
            "reference_time": "SAR pair start",
            "trajectory_model": "x(t) = x0 + alpha * u(x0)",
            "laser_timestamp_handling": "per observation",
            "uncertainty_model": (
                "approximate alpha times a frozen empirical endpoint-error P90; "
                "does not include unresolved within-pair acceleration"
            ),
            "endpoint_error_p90_m": endpoint_error_p90_m,
            "methods": drift_awareness,
        },
        "interpretation": (
            f"{product_name.upper()} association is structural validation, not direct "
            "displacement truth. The SAR field is advected to each laser time; "
            + (
                "ridge and height-quality thresholds are literature priors fixed before "
                "reviewing these results."
                if product_name == "atl07"
                else "lead and freeboard-quality thresholds are fixed before reviewing "
                "these results."
            )
        ),
    }
    if product_name == "atl07":
        summary["atl07_fixed_priors"] = {
            "beam_selection": "strong",
            "height_quality": 1,
            "fit_quality": [1, 2],
            "ssh_flag": 0,
            "baseline_length_m": 5000.0,
            "ridge_peak_threshold_m": 0.6,
            "reporting_bin_m": args.bin_size_m,
        }
        summary["common_support_spatial_nulls"] = {
            prefix: {
                "convergence_vs_ridge_density": circular_shift_test(
                    common_bins[prefix],
                    f"{prefix}_cumulative_convergence",
                    "ridge_density_per_km",
                    bin_size_m=args.bin_size_m,
                    minimum_shift_m=args.minimum_spatial_shift_m,
                    seed=20260819,
                ),
                "maximum_compression_vs_ridging_intensity": circular_shift_test(
                    common_bins[prefix],
                    f"{prefix}_cumulative_maximum_compression",
                    "ridging_intensity_m_per_km",
                    bin_size_m=args.bin_size_m,
                    minimum_shift_m=args.minimum_spatial_shift_m,
                    seed=20260821,
                ),
                "shear_vs_relative_roughness": circular_shift_test(
                    common_bins[prefix],
                    f"{prefix}_shear_per_day",
                    "relative_roughness_m",
                    bin_size_m=args.bin_size_m,
                    minimum_shift_m=args.minimum_spatial_shift_m,
                    seed=20260820,
                ),
            }
            for prefix in fields
        }
        summary["paired_common_support_association"] = paired_common_support_comparison(
            common_bins["orb"],
            common_bins["aliked"],
            bin_size_m=args.bin_size_m,
            block_length_m=args.minimum_spatial_shift_m,
        )
    encoded = json.dumps(json_safe(summary), indent=2, allow_nan=False)
    (args.output_dir / "summary.json").write_text(encoded + "\n")
    write_icesat2_event_ledger(
        args,
        summary,
        product_name,
        product_path,
        candidate_observations,
        colocated_observations,
        len(common_bins["orb"]),
        orb_vectors,
        aliked_vectors,
    )
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
