#!/usr/bin/env python3
"""Drift-aware CryoSat-2 roughness validation of ORB and ALIKED deformation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pyproj import Transformer

if __package__:
    from experiments.validate_icesat2_deformation import (
        TriangleDisplacementField,
        circular_shift_test,
        colocate_method,
        json_safe,
        load_aliked_vectors,
        load_orb_vectors,
        safe_spearman,
        top_fraction_mask,
    )
else:
    from validate_icesat2_deformation import (
        TriangleDisplacementField,
        circular_shift_test,
        colocate_method,
        json_safe,
        load_aliked_vectors,
        load_orb_vectors,
        safe_spearman,
        top_fraction_mask,
    )

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


CRYOSAT_EPOCH_UTC = pd.Timestamp("2000-01-01T00:00:00Z")
DEFORMATION_NAMES = (
    "divergence_per_day",
    "shear_per_day",
    "total_per_day",
    "maximum_compression_per_day",
    "maximum_extension_per_day",
)


def _values(variable, fill_value=np.nan) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[:], fill_value))


def cryosat_utc(day: np.ndarray, seconds: np.ndarray) -> pd.DatetimeIndex:
    """Convert RDWES1B day/second fields to timezone-aware UTC."""
    return (
        CRYOSAT_EPOCH_UTC
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(seconds, unit="s")
    )


def load_rdwes1b(paths: list[Path]) -> pd.DataFrame:
    """Load CryoSat-2 RDWES1B footprint roughness in EPSG:3413."""
    project = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    rows: list[pd.DataFrame] = []
    for path in sorted(paths):
        if path.name.startswith("._"):
            continue
        with Dataset(path) as dataset:
            latitude = _values(dataset.variables["lat"])
            longitude = _values(dataset.variables["lon"])
            longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
            laser_x, laser_y = project.transform(longitude, latitude)
            step = np.hypot(np.diff(laser_x), np.diff(laser_y))
            along_track_m = np.r_[0.0, np.cumsum(step)]
            rows.append(
                pd.DataFrame(
                    {
                        "track_id": path.stem,
                        "footprint_index": _values(dataset.variables["i"], -1).astype(int),
                        "time_utc": cryosat_utc(
                            _values(dataset.variables["day"]),
                            _values(dataset.variables["sec"]),
                        ),
                        "latitude": latitude,
                        "longitude": longitude,
                        "laser_x": laser_x,
                        "laser_y": laser_y,
                        "along_track_m": along_track_m,
                        "surface_elevation_m": (
                            _values(dataset.variables["elev"])
                            + _values(dataset.variables["retrack_elev"])
                        ),
                        "roughness_m": _values(dataset.variables["roughness"]),
                        "norm_res": _values(dataset.variables["norm_res"]),
                        "peakiness": _values(dataset.variables["peakiness"]),
                        "stack_sd": _values(dataset.variables["stack_sd"]),
                    }
                )
            )
    if not rows:
        raise ValueError("No readable RDWES1B NetCDF files were supplied")
    result = pd.concat(rows, ignore_index=True)
    finite = np.isfinite(
        result[["latitude", "longitude", "laser_x", "laser_y"]]
    ).all(axis=1)
    return result.loc[finite].reset_index(drop=True)


def cryosat_quality_mask(
    observations: pd.DataFrame, exclude_one_metre_fit_bound: bool = False
) -> pd.Series:
    """Apply the published fit-residual rule and basic finite-value checks."""
    valid = (
        np.isfinite(observations["roughness_m"])
        & np.isfinite(observations["norm_res"])
        & observations["roughness_m"].gt(0.0)
        & observations["norm_res"].le(0.5)
    )
    if exclude_one_metre_fit_bound:
        valid &= ~np.isclose(observations["roughness_m"], 1.0)
    return valid


def add_surface_class(observations: pd.DataFrame) -> pd.DataFrame:
    """Apply Kurtz et al. (2014) SAR-mode lead/floe waveform thresholds."""
    result = observations.copy()
    lead = result["peakiness"].gt(0.18) & result["stack_sd"].lt(4.0)
    floe = result["peakiness"].lt(0.09) & result["stack_sd"].gt(4.0)
    result["surface_class"] = "ambiguous"
    result.loc[lead, "surface_class"] = "lead"
    result.loc[floe, "surface_class"] = "floe"
    result["lead_return"] = lead.astype(float)
    result["floe_return"] = floe.astype(float)
    result["floe_roughness_m"] = result["roughness_m"].where(floe)
    return result


def aggregate_track_bins(
    observations: pd.DataFrame,
    prefix: str,
    bin_size_m: float,
    minimum_footprints: int = 3,
) -> pd.DataFrame:
    """Aggregate one deformation field and CryoSat roughness along track."""
    if bin_size_m <= 0 or minimum_footprints <= 0:
        raise ValueError("bin_size_m and minimum_footprints must be positive")
    available_column = f"{prefix}_available"
    selected = observations.loc[observations[available_column].fillna(False)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["beam", "track_bin"])
    selected["beam"] = selected["track_id"]
    selected["track_bin"] = np.floor(
        selected["along_track_m"] / bin_size_m
    ).astype(int)
    aggregation = {
        "footprints": ("roughness_m", "size"),
        "roughness_m": ("roughness_m", "median"),
        "roughness_std_m": ("roughness_m", "std"),
        "lead_fraction": ("lead_return", "mean"),
        "floe_fraction": ("floe_return", "mean"),
        "floe_footprints": ("floe_return", "sum"),
        "floe_roughness_m": ("floe_roughness_m", "median"),
        "norm_res": ("norm_res", "median"),
        "peakiness": ("peakiness", "median"),
        "stack_sd": ("stack_sd", "median"),
        "x": ("laser_x", "median"),
        "y": ("laser_y", "median"),
        "longitude": ("longitude", "median"),
        "latitude": ("latitude", "median"),
        "time_utc": ("time_utc", "median"),
        "along_track_start_m": ("along_track_m", "min"),
        "along_track_end_m": ("along_track_m", "max"),
        **{
            f"{prefix}_{name}": (f"{prefix}_{name}", "median")
            for name in DEFORMATION_NAMES
        },
    }
    bins = (
        selected.groupby(["beam", "track_bin"], as_index=False)
        .agg(**aggregation)
        .sort_values(["beam", "track_bin"])
        .reset_index(drop=True)
    )
    bins["observed_length_m"] = (
        bins["along_track_end_m"] - bins["along_track_start_m"]
    )
    bins = bins.loc[bins["footprints"].ge(minimum_footprints)].reset_index(
        drop=True
    )
    bins["method"] = prefix
    return bins


def summarize_roughness_relationship(
    bins: pd.DataFrame, prefix: str, bin_size_m: float, repetitions: int
) -> dict:
    if bins.empty:
        return {"bins": 0, "tracks": 0}
    shear = bins[f"{prefix}_shear_per_day"].to_numpy(float)
    compression = bins[f"{prefix}_maximum_compression_per_day"].to_numpy(float)
    mixed_roughness = bins["roughness_m"].to_numpy(float)
    lead_fraction = bins["lead_fraction"].to_numpy(float)
    high_shear = top_fraction_mask(shear)
    high_lead_fraction = float(np.nanmean(lead_fraction[high_shear]))
    other_lead_fraction = float(np.nanmean(lead_fraction[~high_shear]))
    lead_shift = circular_shift_test(
        bins,
        f"{prefix}_shear_per_day",
        "lead_fraction",
        repetitions=repetitions,
        bin_size_m=bin_size_m,
        minimum_shift_m=20_000.0,
    )
    floe_bins = bins.loc[bins["floe_footprints"].ge(3)].copy()
    floe_shear = floe_bins[f"{prefix}_shear_per_day"].to_numpy(float)
    floe_roughness = floe_bins["floe_roughness_m"].to_numpy(float)
    floe_shift = circular_shift_test(
        floe_bins,
        f"{prefix}_shear_per_day",
        "floe_roughness_m",
        repetitions=repetitions,
        bin_size_m=bin_size_m,
        minimum_shift_m=20_000.0,
    )
    per_track = {
        beam: {
            "bins": int(len(group)),
            "spearman_shear_vs_lead_fraction": safe_spearman(
                group[f"{prefix}_shear_per_day"].to_numpy(float),
                group["lead_fraction"].to_numpy(float),
            ),
        }
        for beam, group in bins.groupby("beam")
    }
    leave_one_track_out = {
        beam: safe_spearman(
            bins.loc[bins["beam"].ne(beam), f"{prefix}_shear_per_day"].to_numpy(
                float
            ),
            bins.loc[bins["beam"].ne(beam), "lead_fraction"].to_numpy(float),
        )
        for beam in bins["beam"].unique()
    }
    return {
        "bins": int(len(bins)),
        "tracks": int(bins["beam"].nunique()),
        "footprints": int(bins["footprints"].sum()),
        "observed_length_km": float(bins["observed_length_m"].sum() / 1000.0),
        "primary_lead_fraction": {
            "spearman_shear_vs_lead_fraction": safe_spearman(
                shear, lead_fraction
            ),
            "spearman_maximum_compression_vs_lead_fraction": safe_spearman(
                compression, lead_fraction
            ),
            "high_shear_quintile_mean_lead_fraction": high_lead_fraction,
            "other_bins_mean_lead_fraction": other_lead_fraction,
            "high_shear_minus_other_lead_fraction": (
                high_lead_fraction - other_lead_fraction
            ),
            "within_track_20km_shift_null": lead_shift,
            "per_track": per_track,
            "leave_one_track_out": leave_one_track_out,
        },
        "secondary_floe_roughness": {
            "bins_with_at_least_three_floe_returns": int(len(floe_bins)),
            "spearman_shear_vs_floe_roughness": safe_spearman(
                floe_shear, floe_roughness
            ),
            "spearman_maximum_compression_vs_floe_roughness": safe_spearman(
                floe_bins[f"{prefix}_maximum_compression_per_day"].to_numpy(float),
                floe_roughness,
            ),
            "within_track_20km_shift_null": floe_shift,
        },
        "diagnostic_mixed_lead_and_floe_roughness": {
            "spearman_shear_vs_mixed_roughness": safe_spearman(
                shear, mixed_roughness
            ),
            "warning": (
                "Lead fits use a different roughness bound, so this is not a "
                "physical floe-roughness test."
            ),
        },
    }


def compare_methods(orb: pd.DataFrame, aliked: pd.DataFrame) -> dict:
    paired = orb.merge(
        aliked,
        on=["beam", "track_bin"],
        suffixes=("_orb", "_aliked"),
        validate="one_to_one",
    )
    if paired.empty:
        return {"bins": 0}
    result = {"bins": int(len(paired))}
    lead_fraction = paired["lead_fraction_orb"].to_numpy(float)
    floe_roughness = paired["floe_roughness_m_orb"].to_numpy(float)
    for name in ("shear_per_day", "maximum_compression_per_day"):
        orb_values = paired[f"orb_{name}"].to_numpy(float)
        aliked_values = paired[f"aliked_{name}"].to_numpy(float)
        result[name] = {
            "orb_vs_aliked_spearman": safe_spearman(orb_values, aliked_values),
            "median_absolute_difference": float(
                np.nanmedian(np.abs(orb_values - aliked_values))
            ),
            "orb_vs_lead_fraction": safe_spearman(orb_values, lead_fraction),
            "aliked_vs_lead_fraction": safe_spearman(
                aliked_values, lead_fraction
            ),
            "orb_vs_floe_roughness": safe_spearman(
                orb_values, floe_roughness
            ),
            "aliked_vs_floe_roughness": safe_spearman(
                aliked_values, floe_roughness
            ),
        }
    return result


def run_scale(
    observations: pd.DataFrame,
    bin_size_m: float,
    repetitions: int,
    exclude_one_metre_fit_bound: bool = False,
) -> tuple[dict, pd.DataFrame]:
    quality = cryosat_quality_mask(
        observations, exclude_one_metre_fit_bound=exclude_one_metre_fit_bound
    )
    common = quality & observations["orb_available"] & observations["aliked_available"]
    common_observations = observations.loc[common].copy()
    bins = {
        prefix: aggregate_track_bins(common_observations, prefix, bin_size_m)
        for prefix in ("orb", "aliked")
    }
    combined = pd.concat(bins.values(), ignore_index=True)
    summary = {
        "bin_size_m": bin_size_m,
        "quality_controlled_common_footprints": int(len(common_observations)),
        "common_tracks": int(common_observations["track_id"].nunique()),
        "exclude_one_metre_fit_bound": exclude_one_metre_fit_bound,
        "methods": {
            prefix: summarize_roughness_relationship(
                method_bins, prefix, bin_size_m, repetitions
            )
            for prefix, method_bins in bins.items()
        },
        "method_comparison": compare_methods(bins["orb"], bins["aliked"]),
    }
    return summary, combined


def drift_control_summary(
    observations: pd.DataFrame, prefix: str, bin_size_m: float, repetitions: int
) -> dict:
    quality = cryosat_quality_mask(observations)
    common = (
        quality
        & observations[f"{prefix}_available"]
        & observations[f"{prefix}_static_available"]
    )
    selected = observations.loc[common].copy()
    dynamic = aggregate_track_bins(selected, prefix, bin_size_m)
    static_prefix = f"{prefix}_static"
    static = aggregate_track_bins(selected, static_prefix, bin_size_m)
    correction = selected[f"{prefix}_drift_correction_m"]
    return {
        "common_footprints": int(len(selected)),
        "drift_correction_m_p05_median_p95": [
            float(value) for value in correction.quantile([0.05, 0.5, 0.95])
        ],
        "drift_aware": summarize_roughness_relationship(
            dynamic, prefix, bin_size_m, repetitions
        ),
        "static_no_advection": summarize_roughness_relationship(
            static, static_prefix, bin_size_m, repetitions
        ),
    }


def write_cryosat2_event_ledger(
    args,
    summary: dict,
    paths: list[Path],
    candidate_observations: pd.DataFrame,
    observations: pd.DataFrame,
    orb_vectors: pd.DataFrame,
    aliked_vectors: pd.DataFrame,
) -> None:
    """Write the same alignment and selection contract as ICESat-2."""
    quality = cryosat_quality_mask(observations)
    union = quality & (
        observations["orb_available"].fillna(False)
        | observations["aliked_available"].fillna(False)
    )
    common = exact_common_support(
        observations, ["orb_available", "aliked_available"], quality
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
    checkpoints = deterministic_checkpoints(
        observations, "track_id", common, per_track=3
    )
    event_id = f"cryosat2_rdwes1b_pair_{source_image_id}_{target_image_id}_4000m"
    final_bins = summary["primary_4km_common_support"]["methods"]["orb"]["bins"]
    write_event_ledger(
        args.output_dir,
        event_id=event_id,
        sensor="CryoSat-2",
        product_identifiers=[path.name for path in sorted(paths)],
        product_time_start_utc=candidate_observations["time_utc"].min(),
        product_time_end_utc=candidate_observations["time_utc"].max(),
        pair_start_utc=pair_start,
        pair_end_utc=pair_end,
        source_image_id=str(source_image_id),
        target_image_id=str(target_image_id),
        inclusion_reason=args.candidate_inclusion_reason,
        analysis_role=args.analysis_role,
        result_status=summary["status"],
        selection_counts={
            "candidate_observations": len(candidate_observations),
            "temporally_eligible_observations": len(observations),
            "product_qc_survivors": int(quality.sum()),
            "spatially_supported_observations": int(union.sum()),
            "common_method_observations": int(common.sum()),
            "final_bins": final_bins,
        },
        reporting_resolution_m=4000.0,
        minimum_observations_per_bin=3,
        deformation_fields=fields,
        point_ledger_path=args.output_dir / "cryosat2_colocated_points.csv",
        bin_ledger_path=args.output_dir / "cryosat2_track_bins_4km.csv",
        checkpoints=checkpoints,
        missing_support_reasons={
            "outside_sar_interval": len(candidate_observations) - len(observations),
            "failed_product_qc": len(observations) - int(quality.sum()),
            "no_deformation_method_support": int(quality.sum()) - int(union.sum()),
            "not_on_exact_common_method_support": int(union.sum())
            - int(common.sum()),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cryosat-dir", type=Path, required=True)
    parser.add_argument("--orb-database", type=Path, required=True)
    parser.add_argument("--orb-table", required=True)
    parser.add_argument("--orb-source-image-id", type=int, required=True)
    parser.add_argument("--orb-target-image-id", type=int, required=True)
    parser.add_argument("--aliked-field", type=Path, required=True)
    parser.add_argument("--pair-start", required=True)
    parser.add_argument("--pair-end", required=True)
    parser.add_argument("--orb-endpoint-error-p90-m", type=float)
    parser.add_argument("--aliked-endpoint-error-p90-m", type=float)
    parser.add_argument("--null-repetitions", type=int, default=999)
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

    paths = [path for path in args.cryosat_dir.glob("*.nc") if not path.name.startswith("._")]
    candidate_observations = add_surface_class(load_rdwes1b(paths))
    observations = candidate_observations.loc[
        candidate_observations["time_utc"].between(pair_start, pair_end)
    ].reset_index(drop=True)
    inputs_in_interval = int(len(observations))

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
    endpoint_errors = {
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
                    endpoint_error_p90_m=endpoint_errors[prefix],
                ),
            ],
            axis=1,
        )

    quality = cryosat_quality_mask(observations)
    primary, primary_bins = run_scale(
        observations, 4000.0, args.null_repetitions
    )
    sensitivity_1km, sensitivity_bins = run_scale(
        observations, 1000.0, args.null_repetitions
    )
    fit_bound, _ = run_scale(
        observations,
        4000.0,
        args.null_repetitions,
        exclude_one_metre_fit_bound=True,
    )
    summary = {
        "status": "complete",
        "hypothesis": (
            "Higher drift-aware SAR shear is associated with more lead-type "
            "CryoSat-2 waveforms and, separately, greater floe-only surface roughness."
        ),
        "pair_start_utc": pair_start.isoformat(),
        "pair_end_utc": pair_end.isoformat(),
        "crs": "EPSG:3413",
        "cryosat_product": "RDWES1B v1",
        "quality_control": {
            "primary": "finite positive roughness and norm_res <= 0.5",
            "surface_classification": (
                "Kurtz et al. (2014) SAR thresholds: lead peakiness > 0.18 and "
                "stack_sd < 4; floe peakiness < 0.09 and stack_sd > 4"
            ),
            "reason_for_separation": (
                "Lead waveform fits constrain roughness to 0-0.1 m, so lead and "
                "floe roughness cannot be pooled as one physical response."
            ),
            "one_metre_roughness_values": (
                "retained in the primary analysis; excluded only in a named sensitivity"
            ),
        },
        "inputs": {
            "granules": int(len(paths)),
            "cryosat_directory": str(args.cryosat_dir),
            "granule_names": [path.name for path in sorted(paths)],
            "footprints_in_sar_interval": inputs_in_interval,
            "quality_controlled_footprints": int(quality.sum()),
            "orb_database": str(args.orb_database),
            "orb_table": args.orb_table,
            "orb_image_ids": [
                args.orb_source_image_id,
                args.orb_target_image_id,
            ],
            "aliked_field": str(args.aliked_field),
            "orb_vectors": int(len(orb_vectors)),
            "aliked_vectors": int(len(aliked_vectors)),
        },
        "method_support": {
            prefix: {
                "quality_controlled_footprints": int(
                    (quality & observations[f"{prefix}_available"]).sum()
                ),
                "tracks": int(
                    observations.loc[
                        quality & observations[f"{prefix}_available"], "track_id"
                    ].nunique()
                ),
            }
            for prefix in fields
        },
        "primary_4km_common_support": primary,
        "sensitivity_1km_common_support": sensitivity_1km,
        "sensitivity_4km_excluding_one_metre_fit_bound": fit_bound,
        "drift_awareness_control_4km": {
            prefix: drift_control_summary(
                observations, prefix, 4000.0, args.null_repetitions
            )
            for prefix in fields
        },
        "interpretation_guardrail": (
            "This tests spatial association with an independent roughness observable, "
            "not pointwise displacement error or causation by the 23-hour SAR interval."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "cryosat2_colocated_points.csv", index=False)
    primary_bins.to_csv(args.output_dir / "cryosat2_track_bins_4km.csv", index=False)
    sensitivity_bins.to_csv(args.output_dir / "cryosat2_track_bins_1km.csv", index=False)
    encoded = json.dumps(json_safe(summary), indent=2, allow_nan=False)
    (args.output_dir / "summary.json").write_text(encoded + "\n")
    write_cryosat2_event_ledger(
        args,
        summary,
        paths,
        candidate_observations,
        observations,
        orb_vectors,
        aliked_vectors,
    )
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
