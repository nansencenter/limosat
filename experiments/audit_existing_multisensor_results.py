#!/usr/bin/env python3
"""Reconstruct ledgers and visual audits for frozen Arctic altimetry events."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from experiments.multisensor_event_ledger import (
        DeformationFieldIdentity,
        dataframe_sha256,
        deterministic_checkpoints,
        exact_common_support,
        file_sha256,
        interval_fraction,
        write_event_ledger,
    )
    from experiments.validate_cryosat2_deformation import (
        add_surface_class,
        aggregate_track_bins as aggregate_cryosat_bins,
        cryosat_quality_mask,
        load_rdwes1b,
    )
    from experiments.validate_icesat2_deformation import (
        TriangleDisplacementField,
        aggregate_atl07_bins,
        aggregate_track_bins as aggregate_atl10_bins,
        load_aliked_vectors,
        load_atl07,
        load_atl10,
        load_orb_vectors,
    )
else:
    from multisensor_event_ledger import (
        DeformationFieldIdentity,
        dataframe_sha256,
        deterministic_checkpoints,
        exact_common_support,
        file_sha256,
        interval_fraction,
        write_event_ledger,
    )
    from validate_cryosat2_deformation import (
        add_surface_class,
        aggregate_track_bins as aggregate_cryosat_bins,
        cryosat_quality_mask,
        load_rdwes1b,
    )
    from validate_icesat2_deformation import (
        TriangleDisplacementField,
        aggregate_atl07_bins,
        aggregate_track_bins as aggregate_atl10_bins,
        load_aliked_vectors,
        load_atl07,
        load_atl10,
        load_orb_vectors,
    )


INCLUSION_REASON = (
    "Existing March 2020 event selected from SAR-interval timing, granule geometry, "
    "and expected support before inspecting deformation association outcomes."
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(), parse_constant=lambda _: None)


def result_files(result_dir: Path, summary: dict) -> tuple[Path, Path]:
    product = str(summary.get("product", "cryosat2")).lower()
    point_candidates = sorted(
        path
        for path in result_dir.glob("*colocated_points.csv")
        if not path.name.startswith("._")
    )
    bin_candidates = sorted(
        path
        for path in result_dir.glob("*track_bins_4km.csv")
        if not path.name.startswith("._")
    )
    if not bin_candidates:
        bin_candidates = sorted(
            path
            for path in result_dir.glob("*track_bins.csv")
            if not path.name.startswith("._")
        )
    if not point_candidates or not bin_candidates:
        raise FileNotFoundError(f"Missing point/bin output in {result_dir}")
    point_path = point_candidates[0]
    bin_path = bin_candidates[0]
    if product in {"atl07", "atl10"} and product not in point_path.name:
        raise ValueError(f"Unexpected point product in {result_dir}")
    return point_path, bin_path


@lru_cache(maxsize=1)
def load_icesat_product(product: str, path: str) -> pd.DataFrame:
    return load_atl07(Path(path)) if product == "ATL07" else load_atl10(Path(path))


@lru_cache(maxsize=1)
def load_cryosat_product(directory: str) -> pd.DataFrame:
    paths = [
        path
        for path in Path(directory).glob("*.nc")
        if not path.name.startswith("._")
    ]
    return add_surface_class(load_rdwes1b(paths))


def load_candidate_observations(summary: dict) -> tuple[pd.DataFrame, list[str]]:
    if summary.get("product") in {"ATL07", "ATL10"}:
        path = str(summary["product_path"])
        return load_icesat_product(summary["product"], path), [Path(path).name]
    directory = str(summary["inputs"]["cryosat_directory"])
    observations = load_cryosat_product(directory)
    identifiers = list(summary["inputs"]["granule_names"])
    return observations, identifiers


def product_quality_mask(observations: pd.DataFrame, product: str) -> pd.Series:
    if product == "ATL07":
        return observations["beam_type"].eq("strong") & observations[
            "topography_valid"
        ].fillna(False)
    if product == "ATL10":
        return observations["beam_type"].eq("strong") & (
            (
                observations["observation_type"].eq("freeboard")
                & observations["freeboard_quality"].isin([1, 2])
            )
            | (
                observations["observation_type"].eq("lead")
                & observations["lead_length_m"].gt(0)
            )
        )
    return cryosat_quality_mask(observations)


def used_aliked_path(pair: dict, summary: dict) -> Path:
    selected = Path(pair["aliked_selected_field"])
    expected = int(
        summary.get("aliked_vectors", summary.get("inputs", {}).get("aliked_vectors", 0))
    )
    used = pair.get("aliked_field_used_by_existing_v2")
    if used is not None:
        candidate = Path(used)
        if len(load_aliked_vectors(candidate)) == expected:
            return candidate
    if len(load_aliked_vectors(selected)) != expected:
        raise ValueError(
            f"No configured ALIKED field has the recorded {expected} vectors"
        )
    return selected


def field_contracts(
    manifest: dict,
    pair: dict,
    summary: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    TriangleDisplacementField,
    TriangleDisplacementField,
    list[DeformationFieldIdentity],
    Path,
]:
    database = Path(manifest["orb_database"])
    table = manifest["orb_table"]
    orb_vectors = load_orb_vectors(
        database,
        table,
        int(pair["source_database_id"]),
        int(pair["target_database_id"]),
    )
    aliked_path = used_aliked_path(pair, summary)
    aliked_vectors = load_aliked_vectors(aliked_path)
    pair_start = pd.Timestamp(summary["pair_start_utc"])
    pair_end = pd.Timestamp(summary["pair_end_utc"])
    orb_hash = dataframe_sha256(
        orb_vectors, ["source_x", "source_y", "dx_m", "dy_m"]
    )
    aliked_hash = file_sha256(aliked_path)
    source_id = str(pair["source_catalog_id"])
    target_id = str(pair["target_catalog_id"])
    boundary = (
        "convex-hull exterior and invalid triangles excluded; valid triangle edges "
        "included by scipy Delaunay.find_simplex; no extrapolation"
    )
    identities = [
        DeformationFieldIdentity(
            method="orb",
            field_id=f"orb:{source_id}:{target_id}:{orb_hash[:12]}",
            path=(
                f"{database}::{table}[{pair['source_database_id']},"
                f"{pair['target_database_id']}]"
            ),
            sha256=orb_hash,
            vector_count=len(orb_vectors),
            source_image_id=source_id,
            target_image_id=target_id,
            source_time_utc=pair_start.isoformat(),
            target_time_utc=pair_end.isoformat(),
            interpolation=(
                "linear barycentric source-time Delaunay; maximum edge 20000 m; "
                "minimum triangle quality 0.05; folds rejected"
            ),
            boundary_rule=boundary,
        ),
        DeformationFieldIdentity(
            method="aliked",
            field_id=f"aliked:{source_id}:{target_id}:{aliked_hash[:12]}",
            path=str(aliked_path),
            sha256=aliked_hash,
            vector_count=len(aliked_vectors),
            source_image_id=source_id,
            target_image_id=target_id,
            source_time_utc=pair_start.isoformat(),
            target_time_utc=pair_end.isoformat(),
            interpolation=(
                "linear barycentric source-time Delaunay; maximum edge 6400 m; "
                "folds rejected when the used field is the selected fold-rejected file"
            ),
            boundary_rule=boundary,
        ),
    ]
    fields = (
        TriangleDisplacementField.build(
            orb_vectors, maximum_edge_m=20_000.0, minimum_quality=0.05
        ),
        TriangleDisplacementField.build(aliked_vectors, maximum_edge_m=6_400.0),
    )
    return orb_vectors, aliked_vectors, fields[0], fields[1], identities, aliked_path


def read_points(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, low_memory=False)
    if "time_utc" in result:
        result["time_utc"] = pd.to_datetime(result["time_utc"], utc=True)
    for column in ("orb_available", "aliked_available"):
        if column in result:
            result[column] = result[column].fillna(False).astype(bool)
    return result


def reconstruct_common_bins(
    points: pd.DataFrame, product: str, bin_size_m: float
) -> tuple[pd.DataFrame, pd.Series]:
    if points.empty:
        return pd.DataFrame(), pd.Series(False, index=points.index, dtype=bool)
    quality = product_quality_mask(points, product)
    common = exact_common_support(
        points, ["orb_available", "aliked_available"], quality
    )
    selected = points.loc[common].copy()
    # Early v1 outputs predate the added principal-strain columns. Preserve
    # those historical results and make the unavailable measures explicit.
    for method in ("orb", "aliked"):
        for suffix in (
            "maximum_compression_per_day",
            "maximum_extension_per_day",
            "cumulative_maximum_compression",
            "cumulative_maximum_extension",
        ):
            column = f"{method}_{suffix}"
            if column not in selected:
                selected[column] = np.nan
    if product == "ATL07":
        aggregate = aggregate_atl07_bins
    elif product == "ATL10":
        aggregate = aggregate_atl10_bins
    else:
        aggregate = aggregate_cryosat_bins
    bins = pd.concat(
        [aggregate(selected, method, bin_size_m) for method in ("orb", "aliked")],
        ignore_index=True,
    )
    if len(bins):
        key_columns = ["beam", "track_bin"]
        orb_keys = bins.loc[bins["method"].eq("orb"), key_columns].reset_index(
            drop=True
        )
        aliked_keys = bins.loc[
            bins["method"].eq("aliked"), key_columns
        ].reset_index(drop=True)
        if not orb_keys.equals(aliked_keys):
            raise AssertionError("ORB and ALIKED exact-common bin identities differ")
    return bins, common


def verify_points(
    points: pd.DataFrame,
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
    fields: dict[str, TriangleDisplacementField],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "point_rows": int(len(points)),
        "time_outside_sar_interval": 0,
    }
    if points.empty:
        return result
    outside = ~points["time_utc"].between(pair_start, pair_end)
    result["time_outside_sar_interval"] = int(outside.sum())
    expected_fraction = interval_fraction(points["time_utc"], pair_start, pair_end)
    for method, field in fields.items():
        available = points[f"{method}_available"].fillna(False).to_numpy(bool)
        method_result: dict[str, Any] = {"available_points": int(available.sum())}
        if not available.any():
            result[method] = method_result
            continue
        selected = points.loc[available]
        sample_count = min(10_000, len(selected))
        positions = np.linspace(0, len(selected) - 1, sample_count).astype(int)
        sample = selected.iloc[positions]
        sample_fraction = expected_fraction[available][positions]
        sampled_displacement, sampled_available = field.sample_displacement(
            sample[[f"{method}_source_x", f"{method}_source_y"]]
            .to_numpy(float)
        )
        sampled_source = sample[
            [f"{method}_source_x", f"{method}_source_y"]
        ].to_numpy(float)
        observed = sample[["laser_x", "laser_y"]].to_numpy(float)
        forward_error = np.linalg.norm(
            sampled_source + sample_fraction[:, None] * sampled_displacement - observed,
            axis=1,
        )
        method_result.update(
            forward_mapping_sample_max_error_m=float(np.nanmax(forward_error)),
            recorded_inversion_residual_max_m=float(
                selected[f"{method}_inversion_residual_m"].max()
            ),
            field_reconstruction_sample_points=sample_count,
            field_reconstruction_unavailable=int((~sampled_available).sum()),
        )
        fraction_column = f"{method}_interval_fraction"
        if fraction_column in selected:
            fraction_error = np.abs(
                selected[fraction_column].to_numpy(float)
                - expected_fraction[available]
            )
            method_result["interval_fraction_max_abs_error"] = float(
                fraction_error.max()
            )
        stored_columns = [f"{method}_pair_dx_m", f"{method}_pair_dy_m"]
        if all(column in sample for column in stored_columns):
            stored = sample[stored_columns].to_numpy(float)
            displacement_error = np.linalg.norm(
                sampled_displacement - stored, axis=1
            )
            method_result["field_reconstruction_max_error_m"] = float(
                np.nanmax(displacement_error)
            )
        target_columns = [f"{method}_target_x", f"{method}_target_y"]
        if all(column in selected for column in target_columns + stored_columns):
            target_error = np.linalg.norm(
                selected[target_columns].to_numpy(float)
                - selected[
                    [f"{method}_source_x", f"{method}_source_y"]
                ].to_numpy(float)
                - selected[stored_columns].to_numpy(float),
                axis=1,
            )
            method_result["target_identity_max_error_m"] = float(
                target_error.max()
            )
        result[method] = method_result
    return result


def enrich_checkpoints(
    checkpoints: pd.DataFrame,
    fields: dict[str, TriangleDisplacementField],
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
) -> pd.DataFrame:
    """Backfill the complete numerical coordinate audit for legacy outputs."""
    result = checkpoints.copy()
    if result.empty:
        return result
    fractions = interval_fraction(result["time_utc"], pair_start, pair_end)
    observed = result[["laser_x", "laser_y"]].to_numpy(float)
    for method, field in fields.items():
        source_columns = [f"{method}_source_x", f"{method}_source_y"]
        if not all(column in result for column in source_columns):
            continue
        source = result[source_columns].to_numpy(float)
        displacement, available = field.sample_displacement(source)
        displacement[~available] = np.nan
        target = source + displacement
        drift = observed - source
        residual = np.linalg.norm(
            source + fractions[:, None] * displacement - observed, axis=1
        )
        result[f"{method}_interval_fraction"] = fractions
        result[f"{method}_pair_dx_m"] = displacement[:, 0]
        result[f"{method}_pair_dy_m"] = displacement[:, 1]
        result[f"{method}_target_x"] = target[:, 0]
        result[f"{method}_target_y"] = target[:, 1]
        result[f"{method}_drift_to_laser_dx_m"] = drift[:, 0]
        result[f"{method}_drift_to_laser_dy_m"] = drift[:, 1]
        result[f"{method}_drift_correction_m"] = np.linalg.norm(drift, axis=1)
        result[f"{method}_inversion_residual_m"] = residual
    return result


def triangle_shear(
    field: TriangleDisplacementField, elapsed_days: float
) -> tuple[np.ndarray, np.ndarray]:
    triangles = field.triangulation.simplices[field.valid_triangle]
    centroids = field.source[triangles].mean(axis=1)
    gradient = field.gradient[field.valid_triangle] / elapsed_days
    shear = np.hypot(
        gradient[:, 0, 0] - gradient[:, 1, 1],
        gradient[:, 0, 1] + gradient[:, 1, 0],
    )
    return centroids, shear


def plot_event_audit(
    path: Path,
    event_id: str,
    points: pd.DataFrame,
    observed_fallback: pd.DataFrame,
    track_column: str,
    fields: dict[str, TriangleDisplacementField],
    common: pd.Series,
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
    checkpoints: pd.DataFrame,
) -> None:
    elapsed_days = (pair_end - pair_start).total_seconds() / 86400.0
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    observed = points if len(points) else observed_fallback
    for axis, (method, field) in zip(axes, fields.items()):
        centroids, shear = triangle_shear(field, elapsed_days)
        limit = np.nanquantile(shear, 0.98) if len(shear) else 1.0
        image = axis.scatter(
            centroids[:, 0] / 1000.0,
            centroids[:, 1] / 1000.0,
            c=shear,
            s=3,
            cmap="magma",
            vmin=0,
            vmax=limit,
            rasterized=True,
            label="valid SAR deformation support",
        )
        if not observed.empty:
            for _, track in observed.groupby(track_column, sort=False):
                order = (
                    track.sort_values("along_track_m")
                    if "along_track_m" in track
                    else track.sort_values("time_utc")
                )
                stride = max(1, len(order) // 3000)
                axis.plot(
                    order["laser_x"].iloc[::stride] / 1000.0,
                    order["laser_y"].iloc[::stride] / 1000.0,
                    color="0.65",
                    linewidth=0.7,
                    alpha=0.8,
                )
        if len(points):
            available = points[f"{method}_available"].fillna(False)
            supported = points.loc[available]
            stride = max(1, len(supported) // 8000)
            axis.scatter(
                supported[f"{method}_source_x"].iloc[::stride] / 1000.0,
                supported[f"{method}_source_y"].iloc[::stride] / 1000.0,
                s=1,
                color="#2ca02c",
                alpha=0.45,
                label="material track on method support",
            )
            shared = points.loc[common]
            stride = max(1, len(shared) // 5000)
            axis.scatter(
                shared[f"{method}_source_x"].iloc[::stride] / 1000.0,
                shared[f"{method}_source_y"].iloc[::stride] / 1000.0,
                s=2,
                color="#00ffff",
                alpha=0.65,
                label="exact common support",
            )
        for _, checkpoint in checkpoints.iterrows():
            x_column = f"{method}_source_x"
            y_column = f"{method}_source_y"
            if x_column not in checkpoint or not np.isfinite(checkpoint[x_column]):
                continue
            x = checkpoint[x_column] / 1000.0
            y = checkpoint[y_column] / 1000.0
            axis.scatter([x], [y], s=20, color="white", edgecolor="black", zorder=5)
            axis.annotate(
                str(checkpoint["checkpoint_id"]).split(":")[-1],
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="white",
                path_effects=[],
            )
        axis.set_title(method.upper())
        axis.set_xlabel("EPSG:3413 x (km)")
        axis.set_ylabel("EPSG:3413 y (km)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.15)
        figure.colorbar(image, ax=axis, shrink=0.75, label="shear (day$^{-1}$)")
    axes[0].legend(loc="best", fontsize=7)
    title = textwrap.fill(event_id, width=55).replace("_", " ")
    figure.suptitle(
        f"{title}\nObserved (grey), material (green), exact common support (cyan)",
        fontsize=13,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def audit_event(
    manifest: dict,
    event: dict,
    output_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result_dir = Path(event["result_dir"])
    summary = read_json(result_dir / "summary.json")
    product = summary.get("product", "CryoSat-2")
    pair = manifest["pair_fields"][event["pair_id"]]
    point_path, _ = result_files(result_dir, summary)
    points = read_points(point_path)
    pair_start = pd.Timestamp(summary["pair_start_utc"])
    pair_end = pd.Timestamp(summary["pair_end_utc"])
    candidate, product_identifiers = load_candidate_observations(summary)
    temporal = candidate.loc[candidate["time_utc"].between(pair_start, pair_end)].copy()
    candidate_qc = product_quality_mask(temporal, product)

    (
        orb_vectors,
        aliked_vectors,
        orb_field,
        aliked_field,
        identities,
        aliked_path,
    ) = field_contracts(manifest, pair, summary)
    fields = {"orb": orb_field, "aliked": aliked_field}
    bins, common = reconstruct_common_bins(
        points, product, float(summary.get("bin_size_m", 4000.0))
    )
    point_qc = product_quality_mask(points, product) if len(points) else pd.Series(dtype=bool)
    union = (
        point_qc
        & (points["orb_available"].fillna(False) | points["aliked_available"].fillna(False))
        if len(points)
        else pd.Series(False, index=points.index, dtype=bool)
    )
    final_bins = int((bins["method"] == "orb").sum()) if len(bins) else 0
    event_output = output_root / "events" / event["event_id"]
    event_output.mkdir(parents=True, exist_ok=True)
    exact_bin_path = event_output / "exact_common_bins.csv"
    bins.to_csv(exact_bin_path, index=False)
    track_column = "track_id" if product == "CryoSat-2" else "beam"
    checkpoints = deterministic_checkpoints(points, track_column, common)
    checkpoints = enrich_checkpoints(checkpoints, fields, pair_start, pair_end)

    status = summary.get("status")
    if not status:
        status = "complete" if final_bins else "insufficient_method_support"
    if product == "ATL10":
        lead_sufficient = all(
            (method.get("common_support") or {}).get(
                "lead_inference_sufficient", False
            )
            for method in summary.get("methods", {}).values()
        )
        if not lead_sufficient:
            status = "insufficient_lead_support"
    selection_counts = {
        "candidate_observations": len(candidate),
        "temporally_eligible_observations": len(temporal),
        "product_qc_survivors": int(candidate_qc.sum()),
        "spatially_supported_observations": int(union.sum()),
        "common_method_observations": int(common.sum()),
        "final_bins": final_bins,
    }
    analysis_role = event["analysis_role"]
    if status.startswith("insufficient"):
        analysis_role = "insufficient_support_control"
    write_event_ledger(
        event_output,
        event_id=event["event_id"],
        sensor=event["sensor"],
        product_identifiers=product_identifiers,
        product_time_start_utc=candidate["time_utc"].min(),
        product_time_end_utc=candidate["time_utc"].max(),
        pair_start_utc=pair_start,
        pair_end_utc=pair_end,
        source_image_id=str(pair["source_catalog_id"]),
        target_image_id=str(pair["target_catalog_id"]),
        inclusion_reason=INCLUSION_REASON,
        analysis_role=analysis_role,
        result_status=status,
        selection_counts=selection_counts,
        reporting_resolution_m=float(summary.get("bin_size_m", 4000.0)),
        minimum_observations_per_bin=3 if product == "CryoSat-2" else 1,
        deformation_fields=identities,
        point_ledger_path=point_path,
        bin_ledger_path=exact_bin_path,
        checkpoints=checkpoints,
        missing_support_reasons={
            "outside_sar_interval": len(candidate) - len(temporal),
            "failed_product_qc_or_morphology": len(temporal)
            - int(candidate_qc.sum()),
            "no_deformation_method_support": int(candidate_qc.sum()) - int(union.sum()),
            "not_on_exact_common_method_support": int(union.sum())
            - int(common.sum()),
        },
    )
    checks = verify_points(points, pair_start, pair_end, fields)
    selected_aliked = Path(pair["aliked_selected_field"])
    selected_count = len(load_aliked_vectors(selected_aliked))
    used_count = len(aliked_vectors)
    if event.get("visual_audit", False):
        fallback = temporal.loc[candidate_qc].copy()
        plot_event_audit(
            event_output / "visual_audit.png",
            event["event_id"],
            points,
            fallback,
            track_column,
            fields,
            common,
            pair_start,
            pair_end,
            checkpoints,
        )
    row = {
        "event_id": event["event_id"],
        "sensor": event["sensor"],
        "product": product,
        "pair_id": event["pair_id"],
        "status": status,
        **selection_counts,
        "recorded_orb_vectors": int(
            summary.get("orb_vectors", summary.get("inputs", {}).get("orb_vectors", 0))
        ),
        "reconstructed_orb_vectors": len(orb_vectors),
        "recorded_aliked_vectors": int(
            summary.get(
                "aliked_vectors", summary.get("inputs", {}).get("aliked_vectors", 0)
            )
        ),
        "used_aliked_vectors": used_count,
        "selected_fold_rejected_aliked_vectors": selected_count,
        "used_aliked_field": str(aliked_path),
        "selected_aliked_field": str(selected_aliked),
        "used_field_is_selected_field": aliked_path == selected_aliked,
        "time_outside_sar_interval": checks["time_outside_sar_interval"],
    }
    for method in ("orb", "aliked"):
        for name, value in checks.get(method, {}).items():
            row[f"{method}_{name}"] = value
    flow = pd.read_csv(event_output / "selection_flow.csv")
    return row, flow


def write_report(output: Path, rows: pd.DataFrame, manifest_hash: str) -> None:
    insufficient = rows.loc[rows["status"].str.startswith("insufficient")]
    mismatch = rows.loc[~rows["used_field_is_selected_field"]]
    lines = [
        "# Existing multisensor alignment audit",
        "",
        f"Frozen event manifest SHA-256: `{manifest_hash}`.",
        "",
        f"Audited {len(rows)} result variants across {rows['event_id'].nunique()} named events.",
        f"Retained {len(insufficient)} insufficient-support controls.",
        "",
        "## Provenance findings",
        "",
    ]
    if len(mismatch):
        lines.append(
            f"{len(mismatch)} result variants used the pre-final `refined_field.csv` "
            "with 8,523 vectors rather than the selected 8,520-vector fixed-point "
            "fold-rejected field. These are preserved as existing-result audits and "
            "must be rerun with the selected field before final synthesis."
        )
    else:
        lines.append("All result variants used their selected fold-rejected ALIKED field.")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "`event_audit.csv` records timestamp, advection-identity, field-reconstruction, "
            "support, and bin counts. Each event directory contains the machine-readable "
            "ledger, selection flow, exact-common bins, numerical checkpoints, and (for "
            "non-duplicate reporting scales) a compact visual audit.",
            "",
            "Insufficient-support events were not converted into scientific associations.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.manifest, args.output_dir / "frozen_event_manifest.json")
    rows = []
    flows = []
    # Sorting by product path improves reuse of the one-product raw-data cache.
    events_with_summary = []
    for event in manifest["events"]:
        summary = read_json(Path(event["result_dir"]) / "summary.json")
        product_key = summary.get(
            "product_path", summary.get("inputs", {}).get("cryosat_directory", "")
        )
        events_with_summary.append((product_key, event))
    for _, event in sorted(events_with_summary, key=lambda item: item[0]):
        row, flow = audit_event(manifest, event, args.output_dir)
        rows.append(row)
        flows.append(flow)
        print(f"audited {event['event_id']}", flush=True)
    audit = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    audit.to_csv(args.output_dir / "event_audit.csv", index=False)
    pd.concat(flows, ignore_index=True).to_csv(
        args.output_dir / "selection_flow.csv", index=False
    )
    manifest_hash = file_sha256(args.output_dir / "frozen_event_manifest.json")
    write_report(args.output_dir, audit, manifest_hash)
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
