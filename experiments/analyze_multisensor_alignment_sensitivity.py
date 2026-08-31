#!/usr/bin/env python3
"""Run frozen symmetric registration sensitivities for Arctic altimetry events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from experiments.multisensor_event_ledger import file_sha256
    from experiments.validate_cryosat2_deformation import (
        add_surface_class,
        aggregate_track_bins as aggregate_cryosat_bins,
        cryosat_quality_mask,
        load_rdwes1b,
    )
    from experiments.validate_icesat2_deformation import (
        TriangleDisplacementField,
        aggregate_atl07_bins,
        invert_to_source_time,
        load_aliked_vectors,
        load_atl07,
        load_orb_vectors,
        safe_spearman,
    )
else:
    from multisensor_event_ledger import file_sha256
    from validate_cryosat2_deformation import (
        add_surface_class,
        aggregate_track_bins as aggregate_cryosat_bins,
        cryosat_quality_mask,
        load_rdwes1b,
    )
    from validate_icesat2_deformation import (
        TriangleDisplacementField,
        aggregate_atl07_bins,
        invert_to_source_time,
        load_aliked_vectors,
        load_atl07,
        load_orb_vectors,
        safe_spearman,
    )


DEFORMATION_NAMES = (
    "divergence_per_day",
    "shear_per_day",
    "total_per_day",
    "maximum_compression_per_day",
    "maximum_extension_per_day",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_event_observations(event: dict, pair: dict) -> pd.DataFrame:
    start = pd.Timestamp(pair["pair_start_utc"])
    end = pd.Timestamp(pair["pair_end_utc"])
    if event["product"] == "ATL07":
        observations = load_atl07(Path(event["product_path"]))
        temporal = observations["time_utc"].between(start, end)
        quality = observations["beam_type"].eq("strong") & observations[
            "topography_valid"
        ]
    else:
        paths = [
            path
            for path in Path(event["product_path"]).glob("*.nc")
            if not path.name.startswith("._")
        ]
        observations = add_surface_class(load_rdwes1b(paths))
        temporal = observations["time_utc"].between(start, end)
        quality = cryosat_quality_mask(observations)
    return observations.loc[temporal & quality].reset_index(drop=True)


def build_fields(manifest: dict) -> dict[str, dict[str, TriangleDisplacementField]]:
    database = Path(manifest["orb_database"])
    table = manifest["orb_table"]
    fields = {}
    for pair_id, pair in manifest["pair_fields"].items():
        orb_vectors = load_orb_vectors(
            database,
            table,
            int(pair["source_database_id"]),
            int(pair["target_database_id"]),
        )
        aliked_vectors = load_aliked_vectors(Path(pair["aliked_field"]))
        fields[pair_id] = {
            "orb": TriangleDisplacementField.build(
                orb_vectors, maximum_edge_m=20_000.0, minimum_quality=0.05
            ),
            "aliked": TriangleDisplacementField.build(
                aliked_vectors, maximum_edge_m=6_400.0
            ),
        }
    return fields


def track_unit_vectors(
    observations: pd.DataFrame, track_column: str
) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.full((len(observations), 2), np.nan)
    for _, group in observations.groupby(track_column, sort=False):
        ordered = group.sort_values("along_track_m")
        xy = ordered[["laser_x", "laser_y"]].to_numpy(float)
        if len(xy) == 1:
            direction = np.array([[1.0, 0.0]])
        else:
            direction = np.column_stack(
                (np.gradient(xy[:, 0]), np.gradient(xy[:, 1]))
            )
            norm = np.linalg.norm(direction, axis=1)
            invalid = ~np.isfinite(norm) | (norm == 0)
            direction[~invalid] /= norm[~invalid, None]
            direction[invalid] = [1.0, 0.0]
        tangent[ordered.index] = direction
    cross = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return tangent, cross


def apply_track_offset(
    observations: pd.DataFrame,
    track_column: str,
    along_m: float,
    cross_m: float,
) -> pd.DataFrame:
    result = observations.copy()
    tangent, normal = track_unit_vectors(result, track_column)
    shift = along_m * tangent + cross_m * normal
    result["laser_x"] = result["laser_x"].to_numpy(float) + shift[:, 0]
    result["laser_y"] = result["laser_y"].to_numpy(float) + shift[:, 1]
    return result


def add_deformation_columns(
    result: pd.DataFrame,
    method: str,
    deformation: dict[str, np.ndarray],
    available: np.ndarray,
    elapsed_days: float,
    fraction: np.ndarray,
) -> None:
    result[f"{method}_available"] = available
    for name, values in deformation.items():
        values = values.copy()
        values[~available] = np.nan
        result[f"{method}_{name}"] = values
    result[f"{method}_cumulative_opening"] = np.maximum(
        result[f"{method}_divergence_per_day"], 0.0
    ) * elapsed_days * fraction
    result[f"{method}_cumulative_convergence"] = np.maximum(
        -result[f"{method}_divergence_per_day"], 0.0
    ) * elapsed_days * fraction
    result[f"{method}_cumulative_maximum_compression"] = (
        result[f"{method}_maximum_compression_per_day"] * elapsed_days * fraction
    )
    result[f"{method}_cumulative_maximum_extension"] = (
        result[f"{method}_maximum_extension_per_day"] * elapsed_days * fraction
    )


def direct_registration(
    observations: pd.DataFrame,
    fields: dict[str, TriangleDisplacementField],
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
) -> pd.DataFrame:
    result = observations.copy()
    elapsed_seconds = (pair_end - pair_start).total_seconds()
    elapsed_days = elapsed_seconds / 86400.0
    fraction = (
        (result["time_utc"] - pair_start).dt.total_seconds() / elapsed_seconds
    ).to_numpy(float)
    observed_xy = result[["laser_x", "laser_y"]].to_numpy(float)
    for method, field in fields.items():
        source, invert_available, _ = invert_to_source_time(
            field, observed_xy, fraction
        )
        deformation, deformation_available = field.sample_deformation(
            source, elapsed_days
        )
        available = invert_available & deformation_available
        add_deformation_columns(
            result, method, deformation, available, elapsed_days, fraction
        )
    return result


def static_registration(
    observations: pd.DataFrame,
    fields: dict[str, TriangleDisplacementField],
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
) -> pd.DataFrame:
    result = observations.copy()
    elapsed_seconds = (pair_end - pair_start).total_seconds()
    elapsed_days = elapsed_seconds / 86400.0
    fraction = (
        (result["time_utc"] - pair_start).dt.total_seconds() / elapsed_seconds
    ).to_numpy(float)
    observed_xy = result[["laser_x", "laser_y"]].to_numpy(float)
    for method, field in fields.items():
        deformation, available = field.sample_deformation(observed_xy, elapsed_days)
        add_deformation_columns(
            result, method, deformation, available, elapsed_days, fraction
        )
    return result


def piecewise_source_coordinates(
    observations: pd.DataFrame,
    first_field: TriangleDisplacementField,
    second_field: TriangleDisplacementField,
    pair_start: pd.Timestamp,
    middle_time: pd.Timestamp,
    pair_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    observed = observations[["laser_x", "laser_y"]].to_numpy(float)
    times = observations["time_utc"]
    early = times.le(middle_time).to_numpy()
    source = np.full_like(observed, np.nan)
    available = np.zeros(len(observations), dtype=bool)
    if early.any():
        fraction = (
            (times.loc[early] - pair_start).dt.total_seconds()
            / (middle_time - pair_start).total_seconds()
        ).to_numpy(float)
        early_source, early_available, _ = invert_to_source_time(
            first_field, observed[early], fraction
        )
        source[early] = early_source
        available[early] = early_available
    late = ~early
    if late.any():
        second_fraction = (
            (times.loc[late] - middle_time).dt.total_seconds()
            / (pair_end - middle_time).total_seconds()
        ).to_numpy(float)
        middle_xy, second_available, _ = invert_to_source_time(
            second_field, observed[late], second_fraction
        )
        start_xy, first_available, _ = invert_to_source_time(
            first_field, middle_xy, np.ones(len(middle_xy))
        )
        source[late] = start_xy
        available[late] = second_available & first_available
    return source, available


def piecewise_registration(
    observations: pd.DataFrame,
    full_fields: dict[str, TriangleDisplacementField],
    first_fields: dict[str, TriangleDisplacementField],
    second_fields: dict[str, TriangleDisplacementField],
    pair_start: pd.Timestamp,
    middle_time: pd.Timestamp,
    pair_end: pd.Timestamp,
) -> pd.DataFrame:
    result = observations.copy()
    elapsed_seconds = (pair_end - pair_start).total_seconds()
    elapsed_days = elapsed_seconds / 86400.0
    fraction = (
        (result["time_utc"] - pair_start).dt.total_seconds() / elapsed_seconds
    ).to_numpy(float)
    for method, full_field in full_fields.items():
        source, motion_available = piecewise_source_coordinates(
            result,
            first_fields[method],
            second_fields[method],
            pair_start,
            middle_time,
            pair_end,
        )
        deformation, deformation_available = full_field.sample_deformation(
            source, elapsed_days
        )
        available = motion_available & deformation_available
        add_deformation_columns(
            result, method, deformation, available, elapsed_days, fraction
        )
    return result


def relationship_columns(product: str, method: str) -> tuple[str, str, str]:
    if product == "ATL07":
        return (
            f"{method}_shear_per_day",
            "relative_roughness_m",
            "spearman_shear_vs_relative_roughness",
        )
    return (
        f"{method}_shear_per_day",
        "lead_fraction",
        "spearman_shear_vs_lead_fraction",
    )


def summarize_bins(
    bins: pd.DataFrame, product: str, method: str
) -> dict[str, Any]:
    if bins.empty:
        return {
            "metric": (
                "spearman_shear_vs_relative_roughness"
                if product == "ATL07"
                else "spearman_shear_vs_lead_fraction"
            ),
            "rho": None,
            "secondary_rho": None,
            "tracks": 0,
            "per_track_json": "{}",
            "leave_one_track_out_json": "{}",
            "leave_one_track_out_min": None,
            "leave_one_track_out_max": None,
        }
    predictor, response, metric = relationship_columns(product, method)
    rho = safe_spearman(
        bins[predictor].to_numpy(float), bins[response].to_numpy(float)
    )
    per_track = {
        str(track): safe_spearman(
            group[predictor].to_numpy(float), group[response].to_numpy(float)
        )
        for track, group in bins.groupby("beam")
    }
    leave_one = {
        str(track): safe_spearman(
            bins.loc[bins["beam"].ne(track), predictor].to_numpy(float),
            bins.loc[bins["beam"].ne(track), response].to_numpy(float),
        )
        for track in bins["beam"].unique()
    }
    finite_leave_one = [value for value in leave_one.values() if value is not None]
    if product == "ATL07":
        secondary = safe_spearman(
            bins[f"{method}_cumulative_maximum_compression"].to_numpy(float),
            bins["ridging_intensity_m_per_km"].to_numpy(float),
        )
    else:
        floe = bins.loc[bins["floe_footprints"].ge(3)]
        secondary = safe_spearman(
            floe[f"{method}_shear_per_day"].to_numpy(float),
            floe["floe_roughness_m"].to_numpy(float),
        )
    return {
        "metric": metric,
        "rho": rho,
        "secondary_rho": secondary,
        "tracks": int(bins["beam"].nunique()) if len(bins) else 0,
        "per_track_json": json.dumps(per_track, sort_keys=True),
        "leave_one_track_out_json": json.dumps(leave_one, sort_keys=True),
        "leave_one_track_out_min": min(finite_leave_one)
        if finite_leave_one
        else None,
        "leave_one_track_out_max": max(finite_leave_one)
        if finite_leave_one
        else None,
    }


def summarize_registration(
    registered: pd.DataFrame,
    event: dict,
    scenario: str,
    registration: str,
    along_offset_m: float,
    cross_offset_m: float,
    bin_sizes_m: list[float],
    minimum_bins: int,
) -> list[dict[str, Any]]:
    rows = []
    common = registered["orb_available"] & registered["aliked_available"]
    aggregate = (
        aggregate_atl07_bins
        if event["product"] == "ATL07"
        else aggregate_cryosat_bins
    )
    for bin_size_m in bin_sizes_m:
        for support_mode in ("exact_common", "method_specific"):
            for method in ("orb", "aliked"):
                selected = common if support_mode == "exact_common" else registered[
                    f"{method}_available"
                ]
                observations = registered.loc[selected].copy()
                bins = aggregate(observations, method, bin_size_m)
                summary = summarize_bins(bins, event["product"], method)
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "sensor": event["sensor"],
                        "product": event["product"],
                        "pair_id": event["pair_id"],
                        "scenario": scenario,
                        "registration": registration,
                        "along_offset_m": along_offset_m,
                        "cross_offset_m": cross_offset_m,
                        "bin_size_m": bin_size_m,
                        "scale_role": {
                            1000.0: "morphology",
                            4000.0: "tracker_comparison",
                        }.get(bin_size_m, "reporting_sensitivity"),
                        "support_mode": support_mode,
                        "method": method,
                        "supported_observations": int(selected.sum()),
                        "bins": len(bins),
                        "interpretation_status": (
                            "interpretable"
                            if len(bins) >= minimum_bins
                            else "insufficient_bins"
                        ),
                        **summary,
                    }
                )
    return rows


def alignment_envelopes(results: pd.DataFrame, magnitude: float) -> pd.DataFrame:
    direct = results.loc[
        results["registration"].eq("direct")
        & results["support_mode"].eq("exact_common")
    ]
    rows = []
    keys = ["event_id", "sensor", "pair_id", "method", "bin_size_m"]
    for values, group in direct.groupby(keys, dropna=False):
        rho = group["rho"].dropna()
        minimum = float(rho.min()) if len(rho) else np.nan
        maximum = float(rho.max()) if len(rho) else np.nan
        sign_change = bool(minimum < 0 < maximum) if len(rho) else False
        span = maximum - minimum if len(rho) else np.nan
        rows.append(
            {
                **dict(zip(keys, values)),
                "rho_min": minimum,
                "rho_max": maximum,
                "rho_span": span,
                "minimum_bins_across_scenarios": int(group["bins"].min()),
                "all_scenarios_interpretable": bool(
                    group["interpretation_status"].eq("interpretable").all()
                ),
                "changes_sign": sign_change,
                "material_magnitude_change": bool(span >= magnitude)
                if np.isfinite(span)
                else False,
                "alignment_sensitive": sign_change
                or (bool(span >= magnitude) if np.isfinite(span) else False),
            }
        )
    return pd.DataFrame(rows)


def leave_one_pair_out(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.loc[
        results["scenario"].eq("direct_zero")
        & results["support_mode"].eq("exact_common")
        & results["bin_size_m"].eq(4000.0)
        & results["interpretation_status"].eq("interpretable")
    ]
    rows = []
    for (sensor, method), group in selected.groupby(["sensor", "method"]):
        for pair_id in group["pair_id"].unique():
            remaining = group.loc[group["pair_id"].ne(pair_id), "rho"].dropna()
            rows.append(
                {
                    "sensor": sensor,
                    "method": method,
                    "excluded_pair_id": pair_id,
                    "remaining_event_count": len(remaining),
                    "equal_event_weight_median_rho": float(remaining.median())
                    if len(remaining)
                    else None,
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame, path: Path) -> None:
    selected = results.loc[
        results["bin_size_m"].eq(4000.0)
        & results["support_mode"].eq("exact_common")
        & results["registration"].eq("direct")
    ].copy()
    selected["row"] = selected["event_id"] + " | " + selected["method"].str.upper()
    table = selected.pivot(index="row", columns="scenario", values="rho")
    order = [
        "direct_zero",
        "along_negative",
        "along_positive",
        "cross_negative",
        "cross_positive",
    ]
    table = table.reindex(columns=order)
    figure_height = max(6, 0.42 * len(table))
    figure, axis = plt.subplots(figsize=(10, figure_height), constrained_layout=True)
    image = axis.imshow(table.to_numpy(float), cmap="coolwarm", vmin=-0.6, vmax=0.6)
    axis.set_xticks(np.arange(len(table.columns)), table.columns, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(table.index)), table.index)
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            value = table.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    axis.set_title("4 km exact-common alignment sensitivity (Spearman rho)")
    figure.colorbar(image, ax=axis, label="rho")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_resolution(results: pd.DataFrame, path: Path) -> None:
    selected = results.loc[
        results["scenario"].eq("direct_zero")
        & results["support_mode"].eq("exact_common")
    ]
    sensors = list(selected["sensor"].unique())
    figure, axes = plt.subplots(
        1, len(sensors), figsize=(7 * len(sensors), 6), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, sensor in zip(axes, sensors):
        subset = selected.loc[selected["sensor"].eq(sensor)]
        for (event_id, method), group in subset.groupby(["event_id", "method"]):
            group = group.sort_values("bin_size_m")
            axis.plot(
                group["bin_size_m"] / 1000.0,
                group["rho"],
                marker="o",
                linewidth=1,
                alpha=0.75,
                label=f"{event_id} | {method.upper()}",
            )
        axis.axhline(0, color="0.4", linewidth=0.8)
        axis.set_title(sensor)
        axis.set_xlabel("reporting scale (km)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("primary structural-association rho")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    figure.suptitle("Resolution sensitivity on direct, exact-common support")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    manifest_hash: str,
    results: pd.DataFrame,
    envelopes: pd.DataFrame,
) -> None:
    four_km = envelopes.loc[
        envelopes["bin_size_m"].eq(4000.0)
        & envelopes["all_scenarios_interpretable"]
    ]
    sensitive = four_km.loc[four_km["alignment_sensitive"]]
    piecewise = results.loc[
        results["registration"].eq("piecewise")
        & results["support_mode"].eq("exact_common")
        & results["bin_size_m"].eq(4000.0)
        & results["method"].eq("orb")
    ]
    lines = [
        "# Frozen multisensor alignment sensitivity",
        "",
        f"Manifest SHA-256: `{manifest_hash}`.",
        "",
        "All positive/negative offsets use the frozen 1,310 m independent buoy-error scale. No offset was selected from an association result.",
        "",
        f"At 4 km, {len(sensitive)} of {len(four_km)} event-method envelopes met the predeclared sign-change or 0.20-rho-span alignment-sensitivity rule.",
        "",
        f"Piecewise support produced {int(piecewise['bins'].sum()) if len(piecewise) else 0} unique exact-common 4 km event bins across retained events; zero-bin cases remain explicit.",
        "",
        "CryoSat-2 lead fraction and floe-only roughness remain separate columns. ICESat-2 roughness and ridging-intensity relationships remain separate, and 1 km morphology rows are labelled separately from 4 km tracker-comparison rows.",
        "",
        "Whole-track leave-one-out values are stored as JSON per result row. `leave_one_pair_out.csv` gives equal-event-weight summaries and does not treat footprints or bins as independent replicates.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate envelopes, report, and figures from an existing results CSV.",
    )
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / "frozen_sensitivity_manifest.json"
    shutil.copy2(args.manifest, frozen_path)
    if args.postprocess_only:
        results = pd.read_csv(args.output_dir / "sensitivity_results.csv")
        envelopes = alignment_envelopes(results, magnitude=0.20)
        envelopes.to_csv(args.output_dir / "alignment_envelopes.csv", index=False)
        leave_one_pair_out(results).to_csv(
            args.output_dir / "leave_one_pair_out.csv", index=False
        )
        plot_heatmap(results, args.output_dir / "alignment_sensitivity_4km.png")
        plot_resolution(results, args.output_dir / "resolution_sensitivity.png")
        write_report(
            args.output_dir, file_sha256(frozen_path), results, envelopes
        )
        print(envelopes.to_string(index=False))
        return 0
    fields = build_fields(manifest)
    bin_sizes = [float(value) for value in manifest["bin_sizes_m"]]
    minimum_bins = int(manifest["minimum_interpretable_bins"])
    distance = float(manifest["offset_distance_m"])
    scenarios = (
        ("direct_zero", 0.0, 0.0),
        ("along_negative", -distance, 0.0),
        ("along_positive", distance, 0.0),
        ("cross_negative", 0.0, -distance),
        ("cross_positive", 0.0, distance),
    )
    rows = []
    for event in manifest["events"]:
        pair = manifest["pair_fields"][event["pair_id"]]
        pair_start = pd.Timestamp(pair["pair_start_utc"])
        pair_end = pd.Timestamp(pair["pair_end_utc"])
        observations = load_event_observations(event, pair)
        track_column = "beam" if event["product"] == "ATL07" else "track_id"
        for scenario, along_m, cross_m in scenarios:
            shifted = apply_track_offset(
                observations, track_column, along_m, cross_m
            )
            registered = direct_registration(
                shifted, fields[event["pair_id"]], pair_start, pair_end
            )
            rows.extend(
                summarize_registration(
                    registered,
                    event,
                    scenario,
                    "direct",
                    along_m,
                    cross_m,
                    bin_sizes,
                    minimum_bins,
                )
            )
        static = static_registration(
            observations, fields[event["pair_id"]], pair_start, pair_end
        )
        rows.extend(
            summarize_registration(
                static,
                event,
                "static_zero",
                "static",
                0.0,
                0.0,
                bin_sizes,
                minimum_bins,
            )
        )
        piecewise_ids = pair.get("piecewise_pair_ids")
        if piecewise_ids:
            first_pair = manifest["pair_fields"][piecewise_ids[0]]
            middle_time = pd.Timestamp(first_pair["pair_end_utc"])
            piecewise = piecewise_registration(
                observations,
                fields[event["pair_id"]],
                fields[piecewise_ids[0]],
                fields[piecewise_ids[1]],
                pair_start,
                middle_time,
                pair_end,
            )
            rows.extend(
                summarize_registration(
                    piecewise,
                    event,
                    "piecewise_zero",
                    "piecewise",
                    0.0,
                    0.0,
                    bin_sizes,
                    minimum_bins,
                )
            )
        print(f"completed {event['event_id']}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "sensitivity_results.csv", index=False)
    envelopes = alignment_envelopes(results, magnitude=0.20)
    envelopes.to_csv(args.output_dir / "alignment_envelopes.csv", index=False)
    leave_one_pair_out(results).to_csv(
        args.output_dir / "leave_one_pair_out.csv", index=False
    )
    plot_heatmap(results, args.output_dir / "alignment_sensitivity_4km.png")
    plot_resolution(results, args.output_dir / "resolution_sensitivity.png")
    manifest_hash = file_sha256(frozen_path)
    write_report(args.output_dir, manifest_hash, results, envelopes)
    print(envelopes.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
