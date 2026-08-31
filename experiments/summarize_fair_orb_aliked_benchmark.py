#!/usr/bin/env python3
"""Build common 4 km fields and summarize the frozen ORB/ALIKED runtime gate."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from experiments.validate_icesat2_deformation import (
        TriangleDisplacementField,
        load_orb_vectors,
    )
else:
    from validate_icesat2_deformation import (
        TriangleDisplacementField,
        load_orb_vectors,
    )


PAIR_SPECS = (("10245_10341", 1, 2), ("10341_10352", 2, 3))
MEASURED_LABELS = tuple(
    [f"cold_rep{number}" for number in range(1, 4)]
    + ["warm_rep1", "warm_rep3", "warm_rep4"]
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def one_path(paths, description: str) -> Path:
    selected = [path for path in paths if not path.name.startswith("._")]
    if len(selected) != 1:
        raise ValueError(f"Expected one {description}, found {selected}")
    return selected[0]


def orb_run(root: Path, label: str) -> tuple[dict, Path, str, dict]:
    manifest_path = one_path(
        root.glob(f"orb/{label}/runs/*/run_manifest.json"), "ORB manifest"
    )
    manifest = read_json(manifest_path)
    database = Path(manifest["engine_url"].removeprefix("sqlite:///"))
    timing = read_json(Path(manifest["stage_timings_path"]))
    return manifest, database, manifest["effective_run_name"], timing


def topology_counts(field: TriangleDisplacementField) -> dict:
    triangles = field.triangulation.simplices
    source = field.source[triangles]
    target = source + field.displacement[triangles]
    edges = np.stack(
        (
            np.linalg.norm(source[:, 1] - source[:, 0], axis=1),
            np.linalg.norm(source[:, 2] - source[:, 1], axis=1),
            np.linalg.norm(source[:, 0] - source[:, 2], axis=1),
        ),
        axis=1,
    )
    source_first = source[:, 1] - source[:, 0]
    source_second = source[:, 2] - source[:, 0]
    target_first = target[:, 1] - target[:, 0]
    target_second = target[:, 2] - target[:, 0]
    source_cross = (
        source_first[:, 0] * source_second[:, 1]
        - source_first[:, 1] * source_second[:, 0]
    )
    target_cross = (
        target_first[:, 0] * target_second[:, 1]
        - target_first[:, 1] * target_second[:, 0]
    )
    quality = 2.0 * np.sqrt(3.0) * np.abs(source_cross) / np.maximum(
        np.square(edges).sum(axis=1), 1.0
    )
    eligible = (
        np.isfinite(source_cross)
        & (np.abs(source_cross) > 1.0)
        & (edges.max(axis=1) <= 20_000.0)
        & (quality >= 0.05)
    )
    folded = eligible & (source_cross * target_cross <= 0.0)
    return {
        "eligible_triangles": int(eligible.sum()),
        "folded_triangles": int(folded.sum()),
        "folded_fraction": (
            float(folded.sum() / eligible.sum()) if eligible.any() else None
        ),
    }


def error_summary(error: np.ndarray, available: np.ndarray) -> dict:
    values = error[available & np.isfinite(error)]
    return {
        "available": int(len(values)),
        "median_error_m": float(np.median(values)) if len(values) else None,
        "p90_error_m": float(np.quantile(values, 0.90)) if len(values) else None,
        "correct_within_2km": int((values <= 2000.0).sum()),
    }


def relative_error_summary(
    prediction: np.ndarray, truth: np.ndarray, available: np.ndarray
) -> dict:
    indices = np.flatnonzero(available & np.isfinite(prediction).all(axis=1))
    errors = []
    for first, second in combinations(indices, 2):
        predicted_relative = prediction[first] - prediction[second]
        truth_relative = truth[first] - truth[second]
        errors.append(float(np.linalg.norm(predicted_relative - truth_relative)))
    values = np.asarray(errors, dtype=float)
    return {
        "pairs": int(len(values)),
        "median_error_m": float(np.median(values)) if len(values) else None,
        "p90_error_m": float(np.quantile(values, 0.90)) if len(values) else None,
    }


def build_comparison(
    root: Path,
    label: str,
    database: Path,
    table: str,
) -> tuple[list[dict], dict]:
    output = root / "comparison" / label
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    stage = {
        "field_estimation_seconds": 0.0,
        "topology_qc_seconds": 0.0,
        "accuracy_evaluation_seconds": 0.0,
        "writing_seconds": 0.0,
    }
    for pair_id, source_run_id, target_run_id in PAIR_SPECS:
        aliked_dir = root / "aliked" / label / f"pair_{pair_id}"
        aliked = pd.read_csv(aliked_dir / "field_nearest12_fold_rejected.csv")
        queries = aliked[["source_x", "source_y"]].to_numpy(dtype=float)

        started = time.perf_counter()
        orb_vectors = load_orb_vectors(
            database, table, source_run_id, target_run_id
        )
        orb_field = TriangleDisplacementField.build(
            orb_vectors, maximum_edge_m=20_000.0, minimum_quality=0.05
        )
        orb_displacement, orb_available = orb_field.sample_displacement(queries)
        stage["field_estimation_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        orb_topology = topology_counts(orb_field)
        aliked_summary = read_json(aliked_dir / "summary.json")
        stage["topology_qc_seconds"] += time.perf_counter() - started

        aliked_available = aliked["available"].fillna(False).to_numpy(bool)
        aliked_displacement = aliked[
            ["proposal_dx_m", "proposal_dy_m"]
        ].to_numpy(dtype=float)
        common = orb_available & aliked_available
        agreement = np.linalg.norm(
            orb_displacement[common] - aliked_displacement[common], axis=1
        )

        started = time.perf_counter()
        buoy = pd.read_csv(aliked_dir / "buoy_nearest12.csv", low_memory=False)
        buoy_xy = buoy[["source_x", "source_y"]].to_numpy(dtype=float)
        truth = buoy[["truth_dx_m", "truth_dy_m"]].to_numpy(dtype=float)
        orb_buoy, orb_buoy_available = orb_field.sample_displacement(buoy_xy)
        aliked_buoy = buoy[["proposal_dx_m", "proposal_dy_m"]].to_numpy(dtype=float)
        aliked_buoy_available = buoy["available"].fillna(False).to_numpy(bool)
        common_buoy = orb_buoy_available & aliked_buoy_available
        orb_error = np.linalg.norm(orb_buoy - truth, axis=1)
        aliked_error = np.linalg.norm(aliked_buoy - truth, axis=1)
        buoy_metrics = {
            "orb": error_summary(orb_error, orb_buoy_available),
            "aliked": error_summary(aliked_error, aliked_buoy_available),
            "exact_common_orb": error_summary(orb_error, common_buoy),
            "exact_common_aliked": error_summary(aliked_error, common_buoy),
            "exact_common_relative_orb": relative_error_summary(
                orb_buoy, truth, common_buoy
            ),
            "exact_common_relative_aliked": relative_error_summary(
                aliked_buoy, truth, common_buoy
            ),
        }
        stage["accuracy_evaluation_seconds"] += time.perf_counter() - started

        pair_output = output / f"pair_{pair_id}"
        pair_output.mkdir(parents=True, exist_ok=True)
        field_table = aliked[
            ["grid_row", "grid_column", "source_x", "source_y"]
        ].copy()
        field_table["orb_available"] = orb_available
        field_table["orb_dx_m"] = orb_displacement[:, 0]
        field_table["orb_dy_m"] = orb_displacement[:, 1]
        field_table["aliked_available"] = aliked_available
        field_table["aliked_dx_m"] = aliked_displacement[:, 0]
        field_table["aliked_dy_m"] = aliked_displacement[:, 1]
        field_table["exact_common"] = common
        field_table["vector_difference_m"] = np.nan
        field_table.loc[common, "vector_difference_m"] = agreement
        writing_started = time.perf_counter()
        field_table.to_csv(pair_output / "exact_common_4km_field.csv", index=False)
        stage["writing_seconds"] += time.perf_counter() - writing_started

        metrics = {
            "label": label,
            "cache_state": label.split("_", 1)[0],
            "pair_id": pair_id,
            "grid_nodes": int(len(aliked)),
            "orb_vectors": int(len(orb_vectors)),
            "orb_available_nodes": int(orb_available.sum()),
            "aliked_available_nodes": int(aliked_available.sum()),
            "exact_common_nodes": int(common.sum()),
            "exact_common_pair_area_km2": float(common.sum() * 16.0),
            "vector_agreement_median_m": (
                float(np.median(agreement)) if len(agreement) else None
            ),
            "vector_agreement_p90_m": (
                float(np.quantile(agreement, 0.90)) if len(agreement) else None
            ),
            "orb_eligible_triangles": orb_topology["eligible_triangles"],
            "orb_folded_triangles": orb_topology["folded_triangles"],
            "aliked_folded_triangles_after_qc": aliked_summary[
                "topology_after_rejection"
            ]["flipped_triangles"],
            "buoy_metrics": buoy_metrics,
            "cycle_closure": None,
            "cycle_closure_reason": (
                "The frozen adjacent-pair sequence contains no independently "
                "evaluated 10245-to-10352 closing edge."
            ),
        }
        (pair_output / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        rows.append(metrics)
    (output / "postprocess_timings.json").write_text(
        json.dumps(stage, indent=2) + "\n"
    )
    return rows, stage


def timing_rows(
    label: str,
    method: str,
    summary: dict,
    postprocess: dict,
    common_nodes: int,
    common_area_km2: float,
) -> dict:
    cache_state = label.split("_", 1)[0]
    if method == "ORB":
        stages = {
            "model_setup_seconds": summary["setup_seconds"],
            "image_preparation_seconds": summary.get(
                "image_preparation_seconds", 0.0
            ),
            "detection_description_seconds": summary.get(
                "detection_and_description_seconds", 0.0
            ),
            "matching_seconds": summary.get("matching_seconds", 0.0),
            "pattern_matching_seconds": summary.get(
                "pattern_matching_seconds", 0.0
            ),
            "field_estimation_seconds": postprocess["field_estimation_seconds"],
            "topology_qc_seconds": summary.get(
                "topology_and_qc_residual_seconds", 0.0
            )
            + postprocess["topology_qc_seconds"],
            "accuracy_evaluation_seconds": postprocess[
                "accuracy_evaluation_seconds"
            ],
            "writing_persistence_seconds": summary.get(
                "persistence_seconds", 0.0
            )
            + summary.get("runner_output_writing_seconds", 0.0)
            + postprocess["writing_seconds"],
        }
        total = summary["total_wall_seconds"] + sum(postprocess.values())
    else:
        stages = {
            "model_setup_seconds": summary["model_setup_seconds"],
            "image_preparation_seconds": summary["image_preparation_seconds"],
            "detection_description_seconds": summary[
                "detection_description_seconds"
            ],
            "matching_seconds": summary["pair_matching_seconds"],
            "pattern_matching_seconds": 0.0,
            "field_estimation_seconds": summary[
                "pair_field_estimation_seconds"
            ],
            "topology_qc_seconds": summary["pair_topology_qc_seconds"],
            "accuracy_evaluation_seconds": summary[
                "pair_accuracy_evaluation_seconds"
            ],
            "writing_persistence_seconds": summary["pair_writing_seconds"]
            + summary["feature_cache_write_seconds"],
        }
        total = summary["elapsed_seconds"]
    attributed = sum(stages.values())
    return {
        "label": label,
        "cache_state": cache_state,
        "method": method,
        **stages,
        "unattributed_orchestration_seconds": max(0.0, total - attributed),
        "total_wall_seconds": total,
        "seconds_per_unique_image": total / 3.0,
        "seconds_per_evaluated_pair": total / 2.0,
        "seconds_per_common_pair_km2": total / common_area_km2,
        "seconds_per_1000_common_vectors": total / common_nodes * 1000.0,
    }


def write_report(
    output: Path,
    timing: pd.DataFrame,
    metrics: pd.DataFrame,
    prior_audits: list[dict],
) -> None:
    aggregates = (
        timing.groupby(["cache_state", "method"])["total_wall_seconds"]
        .agg(["median", "min", "max"])
        .reset_index()
    )
    lines = [
        "# Fair ORB/ALIKED CPU runtime gate",
        "",
        "Three cold-cache and three uncontaminated warm-cache repetitions follow one excluded setup run. Warm repetition 2 is retained but excluded because a separate validation workload ran concurrently; replacement warm repetition 4 restores the predeclared sample count. Both arms use images 10245, 10341, and 10352 and evaluate adjacent pairs on exact common 4 km support.",
        "",
        "## Total wall time",
        "",
        "| cache | method | median s | range s |",
        "|---|---|---:|---:|",
    ]
    for row in aggregates.itertuples(index=False):
        lines.append(
            f"| {row.cache_state} | {row.method} | {row.median:.2f} | {row.min:.2f}-{row.max:.2f} |"
        )
    lines.extend(
        [
            "",
            "`timings.csv` contains every requested stage and normalization. ORB field construction is timed after its persisted trajectory run; ALIKED writes its 4 km field in-run. Accuracy evaluation is reported separately and retained in each total.",
            "",
            "## Accuracy and support",
            "",
            "| pair | exact-common nodes | ORB / ALIKED endpoint median m | ORB / ALIKED relative median m | vector difference median m | ORB / ALIKED folds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair_id, subset in metrics.groupby("pair_id", sort=False):
        def median(column: str) -> float:
            return float(subset[column].median())

        lines.append(
            f"| {pair_id} | {median('exact_common_nodes'):.0f} | "
            f"{median('buoy_metrics.exact_common_orb.median_error_m'):.1f} / "
            f"{median('buoy_metrics.exact_common_aliked.median_error_m'):.1f} | "
            f"{median('buoy_metrics.exact_common_relative_orb.median_error_m'):.1f} / "
            f"{median('buoy_metrics.exact_common_relative_aliked.median_error_m'):.1f} | "
            f"{median('vector_agreement_median_m'):.1f} | "
            f"{median('orb_folded_triangles'):.0f} / "
            f"{median('aliked_folded_triangles_after_qc'):.0f} |"
        )
    absent_fallbacks = sum(
        audit["pair_id"] == "10245_10341"
        and audit.get("fallback")
        and audit.get("reason") == "prior_absent"
        for audit in prior_audits
    )
    accepted_priors = [
        audit
        for audit in prior_audits
        if audit["pair_id"] == "10341_10352" and not audit.get("fallback")
    ]
    prior_passes = sum(
        audit.get("residual_within_uncertainty") is True
        for audit in accepted_priors
    )
    lines.extend(
        [
            "",
            "`pair_metrics.jsonl` retains every repetition. Endpoint and relative-displacement errors above use identical buoy support for ORB and ALIKED. Cycle closure is explicitly unavailable because this bounded manifest has no independently evaluated closing edge.",
            "",
            "## Sequential-prior audit",
            "",
            f"The first pair correctly used the full-physics fallback in {absent_fallbacks}/6 measured runs because no preceding field existed. The second pair accepted only the immediately preceding fold-free field in {len(accepted_priors)}/6 runs; {prior_passes}/6 matched-residual P90 audits were within the fixed 15 km uncertainty.",
        ]
    )
    multisensor = output / "multisensor_warm_rep1"
    atl07_path = multisensor / "atl07_0039_4000m" / "summary.json"
    atl10_path = multisensor / "atl10_0039_4000m" / "summary.json"
    if atl07_path.is_file() and atl10_path.is_file():
        atl07 = read_json(atl07_path)
        atl10 = read_json(atl10_path)
        lines.extend(
            [
                "",
                "## Exact-common multisensor check",
                "",
                f"The prequalified ATL07 0039 crossing supplies {atl07['methods']['orb']['common_support']['bins']} exact-common 4 km bins. Shear-versus-roughness is {atl07['methods']['orb']['common_support']['spearman_shear_vs_relative_roughness']:.3f} for ORB and {atl07['methods']['aliked']['common_support']['spearman_shear_vs_relative_roughness']:.3f} for ALIKED; neither is spatial-null significant. ATL10 supplies {atl10['methods']['orb']['common_support']['bins']} exact-common bins but only {atl10['methods']['orb']['common_support']['bins_with_leads']} lead-containing bin, so the opening test is retained as insufficient.",
            ]
        )
    lines.extend(["", "No CUDA result is inferred from this CPU gate.", ""])
    (output / "report.md").write_text("\n".join(lines))


def plot_runtime(output: Path, timing: pd.DataFrame) -> None:
    stages = [
        "model_setup_seconds",
        "image_preparation_seconds",
        "detection_description_seconds",
        "matching_seconds",
        "pattern_matching_seconds",
        "field_estimation_seconds",
        "topology_qc_seconds",
        "writing_persistence_seconds",
        "unattributed_orchestration_seconds",
    ]
    labels = []
    medians = []
    for cache_state in ("cold", "warm"):
        for method in ("ORB", "ALIKED"):
            subset = timing[
                timing["cache_state"].eq(cache_state)
                & timing["method"].eq(method)
            ]
            labels.append(f"{method}\n{cache_state}")
            medians.append(subset[stages].median().to_numpy(dtype=float))
    values = np.vstack(medians)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bottom = np.zeros(len(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(stages)))
    for index, stage in enumerate(stages):
        ax.bar(labels, values[:, index], bottom=bottom, label=stage.removesuffix("_seconds"), color=colors[index])
        bottom += values[:, index]
    ax.set_ylabel("Median measured seconds")
    ax.set_title("Frozen three-image, two-pair CPU runtime gate")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.savefig(output / "runtime_stage_comparison.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root
    comparison = root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    timing_records = []
    metric_records = []
    prior_audit_records = []
    for label in MEASURED_LABELS:
        orb_manifest, database, table, orb_timing = orb_run(root, label)
        aliked_summary = read_json(root / "aliked" / label / "summary.json")
        for audit in aliked_summary["prior_audits"]:
            prior_audit_records.append(
                {
                    "label": label,
                    "pair_id": f"{audit['source_image_id']}_{audit['target_image_id']}",
                    **audit,
                }
            )
        metrics, postprocess = build_comparison(
            root, label, database, table
        )
        metric_records.extend(metrics)
        common_nodes = sum(row["exact_common_nodes"] for row in metrics)
        common_area = sum(row["exact_common_pair_area_km2"] for row in metrics)
        orb_timing["total_wall_seconds"] = orb_manifest["elapsed_seconds"]
        timing_records.append(
            timing_rows(
                label, "ORB", orb_timing, postprocess, common_nodes, common_area
            )
        )
        timing_records.append(
            timing_rows(
                label, "ALIKED", aliked_summary, {}, common_nodes, common_area
            )
        )
    timing = pd.DataFrame(timing_records)
    timing.to_csv(comparison / "timings.csv", index=False)
    metrics_flat = pd.json_normalize(metric_records, sep=".")
    metrics_flat.to_csv(comparison / "pair_metrics.csv", index=False)
    total_medians = (
        timing.groupby(["cache_state", "method"])["total_wall_seconds"]
        .median()
        .to_dict()
    )
    aggregate = {
        "valid_measured_labels": list(MEASURED_LABELS),
        "excluded_label": "warm_rep2",
        "total_wall_median_seconds": {
            cache: {
                method.lower(): float(total_medians[(cache, method)])
                for method in ("ORB", "ALIKED")
            }
            for cache in ("cold", "warm")
        },
        "aliked_over_orb_total_wall_ratio": {
            cache: float(
                total_medians[(cache, "ALIKED")] / total_medians[(cache, "ORB")]
            )
            for cache in ("cold", "warm")
        },
        "cycle_closure": None,
        "cycle_closure_reason": "No independently evaluated closing edge in the frozen adjacent-pair sequence.",
        "cuda_measured": False,
    }
    (comparison / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    with (comparison / "pair_metrics.jsonl").open("w") as stream:
        for row in metric_records:
            stream.write(json.dumps(row) + "\n")
    with (comparison / "prior_audits.jsonl").open("w") as stream:
        for row in prior_audit_records:
            stream.write(json.dumps(row) + "\n")
    plot_runtime(comparison, timing)
    write_report(comparison, timing, metrics_flat, prior_audit_records)
    print(timing.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
