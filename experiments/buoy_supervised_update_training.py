#!/usr/bin/env python3
"""Train and evaluate interpretable ORB descriptor-memory update rules with buoys.

Buoy truth is used for policy selection and evaluation only. The graph matcher sees
truth at the initial seed, matching the existing experimental graph contract.
February 2020 is the training/validation sequence. N-ICE2015 is read only after one
final policy has been selected. The production-safe 128-pixel candidate border is
retained throughout this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from orb_multiframe_graph import (  # noqa: E402
    GraphSearchConfig,
    precompute_layers,
    trajectory_rows,
)
from buoy_descriptor_benchmark import (  # noqa: E402
    DescriptorVariant,
    build_extractor,
    exact_descriptor,
    read_scene,
)


DEFAULT_GRAPH_ROOT = ROOT / "results/orb_multiframe_graph/final_arctic_matrix"
DEFAULT_PATCH_ROOT = ROOT / "results/buoy_patch_evolution/q2q98_clahe25"
DEFAULT_OUT_DIR = ROOT / "results/buoy_supervised_update_training/q2q98_clahe25"

PLAIN_LANGUAGE_GLOSSARY = (
    {
        "report_name": "first view reference",
        "meaning": "Descriptor from the first SAR observation of a tracked ice location.",
        "legacy_internal_name": "anchor",
    },
    {
        "report_name": "latest confirmed reference",
        "meaning": "Newest descriptor accepted into persistent appearance memory.",
        "legacy_internal_name": "rolling descriptor in confidence_rolling",
    },
    {
        "report_name": "previous selected reference",
        "meaning": "Descriptor at the immediately previous selected location, even if it was not confirmed.",
        "legacy_internal_name": "provisional previous",
    },
    {
        "report_name": "best-match lead",
        "meaning": "Normalized descriptor-cost gap between the best and second-best candidates; larger is clearer.",
        "legacy_internal_name": "update_min_margin",
    },
    {
        "report_name": "maximum descriptor difference",
        "meaning": "Largest normalized Hamming difference allowed before a selected descriptor may enter memory.",
        "legacy_internal_name": "update_max_cost",
    },
    {
        "report_name": "safe to remember",
        "meaning": "Evaluation label: selected location is within 2 km of the exact-time buoy.",
        "legacy_internal_name": "not previously named",
    },
)


def graph_arguments(manifest: dict) -> SimpleNamespace:
    keys = (
        "analysis_epsg",
        "max_speed_m_per_day",
        "grid_stride",
        "grid_border",
        "orb_nfeatures",
        "orb_scale_factor",
        "orb_nlevels",
        "orb_edge_threshold",
        "orb_patch_size",
        "keypoint_size",
        "octave",
        "angle_mode",
        "descriptor_norm",
    )
    return SimpleNamespace(**{key: manifest[key] for key in keys})


def policy_name(best_match_lead: float, maximum_descriptor_difference: float) -> str:
    lead = int(round(best_match_lead * 1000.0))
    limit = int(round(maximum_descriptor_difference * 1000.0))
    return f"update_when_match_clear__lead_{lead:03d}__difference_{limit:03d}"


def threshold_policies(
    best_match_leads: tuple[float, ...],
    maximum_descriptor_differences: tuple[float, ...],
) -> tuple[GraphSearchConfig, ...]:
    return tuple(
        GraphSearchConfig(
            policy_name(lead, limit),
            "confidence_rolling",
            beam_width=32,
            branching=8,
            update_min_margin=lead,
            update_max_cost=limit,
        )
        for lead in best_match_leads
        for limit in maximum_descriptor_differences
    )


def comparison_policies(selected: GraphSearchConfig) -> tuple[GraphSearchConfig, ...]:
    policies = [
        GraphSearchConfig("keep_first_view_only", "anchor", 32, 8),
        GraphSearchConfig("replace_memory_after_every_match", "rolling", 32, 8),
        GraphSearchConfig(
            "current_hand_set_clear_match_rule",
            "confidence_rolling",
            32,
            8,
            update_min_margin=0.032,
            update_max_cost=0.35,
        ),
        selected,
    ]
    unique = {}
    for policy in policies:
        unique[policy.name] = policy
    return tuple(unique.values())


def eligible_transitions(coincidences: pd.DataFrame, buoy_ids=None) -> int:
    data = coincidences
    if buoy_ids is not None:
        data = data[data.buoy_id.astype(str).isin(set(map(str, buoy_ids)))]
    counts = data.groupby(data.buoy_id.astype(str)).size()
    return int(np.maximum(counts.to_numpy(dtype=int) - 1, 0).sum())


def summarize_policies(
    records: pd.DataFrame,
    coincidences: pd.DataFrame,
    policies: tuple[GraphSearchConfig, ...],
    buoy_ids=None,
) -> pd.DataFrame:
    policy_lookup = {policy.name: policy for policy in policies}
    data = records.copy()
    coincidence_data = coincidences.copy()
    if buoy_ids is not None:
        selected_buoys = set(map(str, buoy_ids))
        data = data[data.buoy_id.astype(str).isin(selected_buoys)]
        coincidence_data = coincidence_data[
            coincidence_data.buoy_id.astype(str).isin(selected_buoys)
        ]
    denominator = eligible_transitions(coincidence_data)
    rows = []
    for name, policy in policy_lookup.items():
        group = data[data.config == name].copy()
        observation_index = pd.to_numeric(group.get("observation_index"), errors="coerce")
        tracked = group[(group.status == "ok") & (observation_index > 0)].copy()
        errors = tracked.endpoint_error_m.to_numpy(dtype=float)
        safe = errors <= 2000.0
        updated = tracked.descriptor_updated.fillna(False).astype(bool).to_numpy()
        rows.append(
            {
                "policy": name,
                "best_match_lead": policy.update_min_margin,
                "maximum_descriptor_difference": policy.update_max_cost,
                "eligible_transitions": denominator,
                "tracked_transitions": len(tracked),
                "tracking_fraction_all": len(tracked) / max(denominator, 1),
                "within_2km_count": int(safe.sum()),
                "within_2km_fraction_all": float(safe.sum() / max(denominator, 1)),
                "catastrophic_50km_count": int((errors > 50000.0).sum()),
                "catastrophic_50km_fraction_all": float(
                    (errors > 50000.0).sum() / max(denominator, 1)
                ),
                "median_error_tracked_m": (
                    float(np.median(errors)) if len(errors) else math.nan
                ),
                "memory_updates": int(updated.sum()),
                "safe_memory_updates": int((updated & safe).sum()),
                "false_memory_updates": int((updated & ~safe).sum()),
                "safe_update_precision": float(
                    (updated & safe).sum() / max(updated.sum(), 1)
                ),
                "safe_match_update_recall": float(
                    (updated & safe).sum() / max(safe.sum(), 1)
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def select_policy(summary: pd.DataFrame, minimum_safe_update_precision: float = 0.95) -> str:
    """Require safe memory first, then select path accuracy and stability."""
    eligible = summary[
        (summary.safe_update_precision >= minimum_safe_update_precision)
        & (summary.memory_updates > 0)
    ]
    if eligible.empty:
        raise ValueError(
            "No update policy meets the minimum safe-update precision "
            f"{minimum_safe_update_precision:.3f}."
        )
    ordered = eligible.sort_values(
        [
            "within_2km_fraction_all",
            "catastrophic_50km_fraction_all",
            "tracking_fraction_all",
            "false_memory_updates",
            "safe_memory_updates",
            "median_error_tracked_m",
            "best_match_lead",
            "maximum_descriptor_difference",
        ],
        ascending=[False, True, False, True, False, True, False, True],
        kind="stable",
    )
    return str(ordered.iloc[0].policy)


def grouped_policy_selection(
    records: pd.DataFrame,
    coincidences: pd.DataFrame,
    policies: tuple[GraphSearchConfig, ...],
    folds: int,
    random_seed: int,
    minimum_safe_update_precision: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_counts = coincidences.groupby(coincidences.buoy_id.astype(str)).size()
    buoy_ids = np.asarray(sorted(path_counts[path_counts >= 2].index.astype(str)))
    splitter = KFold(n_splits=min(folds, len(buoy_ids)), shuffle=True, random_state=random_seed)
    fold_rows = []
    selected_records = []
    for fold, (train_index, test_index) in enumerate(splitter.split(buoy_ids), start=1):
        train_buoys = buoy_ids[train_index]
        test_buoys = buoy_ids[test_index]
        training = summarize_policies(records, coincidences, policies, train_buoys)
        selected_name = select_policy(training, minimum_safe_update_precision)
        testing = summarize_policies(
            records,
            coincidences,
            tuple(policy for policy in policies if policy.name == selected_name),
            test_buoys,
        ).iloc[0]
        fold_rows.append(
            {
                "fold": fold,
                "training_buoys": len(train_buoys),
                "held_out_buoys": len(test_buoys),
                "selected_policy": selected_name,
                **{
                    f"held_out_{column}": testing[column]
                    for column in (
                        "eligible_transitions",
                        "tracking_fraction_all",
                        "within_2km_fraction_all",
                        "catastrophic_50km_fraction_all",
                        "memory_updates",
                        "safe_memory_updates",
                        "false_memory_updates",
                    )
                },
            }
        )
        selected = records[
            records.buoy_id.astype(str).isin(set(test_buoys))
            & (records.config == selected_name)
        ].copy()
        selected["config"] = "cross_validated_buoy_selected_rule"
        selected_records.append(selected)
    return pd.DataFrame.from_records(fold_rows), pd.concat(selected_records, ignore_index=True)


def hamming_distance(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return (
        np.unpackbits(np.bitwise_xor(reference, candidates), axis=-1).sum(axis=-1)
        / float(candidates.shape[-1] * 8)
    )


def cosine_distance(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(candidates, axis=1) * np.linalg.norm(reference)
    return 1.0 - np.divide(
        candidates @ reference,
        denominator,
        out=np.full(len(candidates), np.nan, dtype=float),
        where=denominator > 1.0e-12,
    )


def load_or_extract_brisk(
    observations: pd.DataFrame,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    observation_ids = observations.observation_id.astype(str).to_numpy(dtype="U")
    if cache_path.exists():
        with np.load(cache_path) as cached:
            if np.array_equal(cached["observation_id"].astype(str), observation_ids):
                return cached["descriptor"].astype(np.uint8), cached["available"].astype(bool)
    extractor = build_extractor("brisk")
    variant = DescriptorVariant(
        name="brisk_exact_geographic_hamming",
        extractor_key="brisk",
        norm="hamming",
        angle_mode="geographic",
        keypoint_size=31.0,
        octave=0,
    )
    descriptors = np.zeros((len(observations), 64), dtype=np.uint8)
    available = np.zeros(len(observations), dtype=bool)
    for index, observation in enumerate(observations.itertuples(index=False)):
        image, _ = read_scene(observation.image_filepath)
        descriptor = exact_descriptor(
            extractor,
            image,
            np.array([observation.col, observation.row], dtype=float),
            variant,
            float(observation.image_angle_deg),
        )
        if descriptor is not None and descriptor.shape == (64,):
            descriptors[index] = descriptor.astype(np.uint8)
            available[index] = True
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        observation_id=observation_ids,
        descriptor=descriptors,
        available=available,
    )
    return descriptors, available


def descriptor_separability(
    patch_root: Path,
    output_root: Path,
    sequence: str,
    max_speed_m_per_day: float,
    error_threshold_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_dir = patch_root / sequence
    observations = pd.read_csv(sequence_dir / "observations.csv", dtype={"buoy_id": str})
    transitions = pd.read_csv(sequence_dir / "transitions.csv", dtype={"buoy_id": str})
    archive = np.load(sequence_dir / "descriptors.npz")
    archive_ids = archive["observation_id"].astype(str)
    indexes = {observation_id: index for index, observation_id in enumerate(archive_ids)}
    observation_lookup = observations.set_index("observation_id")
    first_ids = (
        observations.sort_values("image_time")
        .groupby("buoy_id", sort=False)
        .first()
        .observation_id.to_dict()
    )
    brisk_descriptors, brisk_available = load_or_extract_brisk(
        observations,
        output_root / f"{sequence}_brisk_observation_descriptors.npz",
    )
    descriptor_specs = (
        ("ORB", archive["orb"], archive["orb_available"], hamming_distance),
        ("BRISK", brisk_descriptors, brisk_available, hamming_distance),
        (
            "sparse XFeat nearest feature",
            archive["xfeat"],
            archive["xfeat_within_limit"],
            cosine_distance,
        ),
    )
    pair_rows = []
    for descriptor_name, descriptor_values, availability, distance_function in descriptor_specs:
        for reference_name in ("first view reference", "previous true view reference"):
            for transition in transitions.itertuples(index=False):
                reference_id = (
                    first_ids[str(transition.buoy_id)]
                    if reference_name == "first view reference"
                    else transition.source_observation_id
                )
                reference_index = indexes[reference_id]
                target_index = indexes[transition.target_observation_id]
                if not availability[reference_index] or not availability[target_index]:
                    continue
                source = observation_lookup.loc[transition.source_observation_id]
                target = observation_lookup.loc[transition.target_observation_id]
                radius_m = max_speed_m_per_day * transition.dt_hours / 24.0
                distractors = observations[
                    (observations.image_id == transition.target_image_id)
                    & (observations.buoy_id != str(transition.buoy_id))
                ].copy()
                distractors = distractors[
                    (np.hypot(distractors.x - source.x, distractors.y - source.y) <= radius_m)
                    & (
                        np.hypot(distractors.x - target.x, distractors.y - target.y)
                        > error_threshold_m
                    )
                ]
                distractor_indexes = [
                    indexes[observation_id]
                    for observation_id in distractors.observation_id
                    if availability[indexes[observation_id]]
                ]
                if not distractor_indexes:
                    continue
                positive_distance = float(
                    distance_function(
                        descriptor_values[reference_index],
                        descriptor_values[target_index : target_index + 1],
                    )[0]
                )
                negative_distances = distance_function(
                    descriptor_values[reference_index], descriptor_values[distractor_indexes]
                )
                rank = 1 + int(np.sum(negative_distances < positive_distance))
                pair_rows.append(
                    {
                        "sequence": sequence,
                        "buoy_id": str(transition.buoy_id),
                        "target_observation_id": transition.target_observation_id,
                        "descriptor": descriptor_name,
                        "reference_memory": reference_name,
                        "same_buoy_distance": positive_distance,
                        "same_buoy_rank": rank,
                        "distractor_count": len(negative_distances),
                        "distractor_distances": negative_distances,
                    }
                )
    pairs = pd.DataFrame.from_records(pair_rows)
    summaries = []
    for (descriptor_name, reference_name), group in pairs.groupby(
        ["descriptor", "reference_memory"], sort=False
    ):
        positive = group.same_buoy_distance.to_numpy(dtype=float)
        negative = np.concatenate(group.distractor_distances.to_list())
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        scores = -np.r_[positive, negative]
        summaries.append(
            {
                "sequence": sequence,
                "descriptor": descriptor_name,
                "reference_memory": reference_name,
                "eligible_transitions": len(group),
                "unique_buoys": group.buoy_id.nunique(),
                "distractors": len(negative),
                "same_buoy_top1_fraction": float((group.same_buoy_rank == 1).mean()),
                "same_buoy_top3_fraction": float((group.same_buoy_rank <= 3).mean()),
                "same_vs_distractor_auc": float(roc_auc_score(labels, scores)),
                "median_same_buoy_distance": float(np.median(positive)),
                "median_distractor_distance": float(np.median(negative)),
            }
        )
    serializable_pairs = pairs.drop(columns="distractor_distances")
    return serializable_pairs, pd.DataFrame.from_records(summaries)


def run_graph_policies(
    graph_root: Path,
    sequence: str,
    policies: tuple[GraphSearchConfig, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleNamespace, float]:
    sequence_dir = graph_root / sequence
    manifest = json.loads((sequence_dir / "run_manifest.json").read_text())
    args = graph_arguments(manifest)
    if args.grid_border != 128:
        raise ValueError(
            f"Expected the production-safe 128-pixel border, found {args.grid_border}."
        )
    coincidences = pd.read_csv(sequence_dir / "coincidences.csv", dtype={"buoy_id": str})
    coincidences["image_time"] = pd.to_datetime(coincidences.image_time, utc=True)
    layers, precompute_seconds = precompute_layers(coincidences, args)
    rows = []
    for policy in policies:
        rows.extend(trajectory_rows(coincidences, layers, policy, args))
    return pd.DataFrame.from_records(rows), coincidences, args, precompute_seconds


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    view = data[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
        ]
    )


def write_report(
    path: Path,
    descriptor_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    cross_validated_summary: pd.DataFrame,
    final_policy: GraphSearchConfig,
    holdout_summary: pd.DataFrame,
    folds: pd.DataFrame,
    minimum_safe_update_precision: float,
) -> None:
    descriptor_columns = [
        "sequence",
        "descriptor",
        "reference_memory",
        "eligible_transitions",
        "same_buoy_top1_fraction",
        "same_buoy_top3_fraction",
        "same_vs_distractor_auc",
    ]
    policy_columns = [
        "policy",
        "eligible_transitions",
        "tracking_fraction_all",
        "within_2km_fraction_all",
        "catastrophic_50km_fraction_all",
        "memory_updates",
        "safe_memory_updates",
        "false_memory_updates",
    ]
    selected_validation = validation_summary[
        validation_summary.policy == final_policy.name
    ]
    policy_results = pd.concat(
        [
            selected_validation.assign(evaluation="February full-data selection"),
            cross_validated_summary.assign(evaluation="February grouped cross-validation"),
            holdout_summary.assign(evaluation="N-ICE2015 holdout"),
        ],
        ignore_index=True,
    )
    policy_result_columns = ["evaluation", *policy_columns]
    glossary = pd.DataFrame(PLAIN_LANGUAGE_GLOSSARY)
    fold_counts = folds.selected_policy.value_counts().rename_axis("policy").reset_index(name="folds_selected")
    descriptor_lookup = descriptor_summary.set_index(
        ["sequence", "descriptor", "reference_memory"]
    )
    validation_orb_previous = descriptor_lookup.loc[
        ("2020_02", "ORB", "previous true view reference"),
        "same_buoy_top1_fraction",
    ]
    validation_xfeat_previous = descriptor_lookup.loc[
        ("2020_02", "sparse XFeat nearest feature", "previous true view reference"),
        "same_buoy_top1_fraction",
    ]
    validation_brisk_previous = descriptor_lookup.loc[
        ("2020_02", "BRISK", "previous true view reference"),
        "same_buoy_top1_fraction",
    ]
    holdout_orb_previous = descriptor_lookup.loc[
        ("2015_full15", "ORB", "previous true view reference"),
        "same_buoy_top1_fraction",
    ]
    holdout_xfeat_previous = descriptor_lookup.loc[
        (
            "2015_full15",
            "sparse XFeat nearest feature",
            "previous true view reference",
        ),
        "same_buoy_top1_fraction",
    ]
    holdout_brisk_previous = descriptor_lookup.loc[
        ("2015_full15", "BRISK", "previous true view reference"),
        "same_buoy_top1_fraction",
    ]
    path.write_text(
        "# Buoy-supervised descriptor memory training\n\n"
        "The standard VAE band and the production-safe 128-pixel raster border are "
        "unchanged. February 2020 selects the update rule using whole-buoy grouped "
        "cross-validation. N-ICE2015 is not used until one final rule is selected.\n\n"
        "## Plain-language names\n\n"
        + markdown_table(glossary, ["report_name", "meaning", "legacy_internal_name"])
        + "\n\n## Which descriptor separates the buoy path from distractors?\n\n"
        "A positive is the same buoy in the next SAR image. Distractors are other buoy "
        "locations in that image that satisfy the 50 km/day motion bound and are more "
        "than 2 km from the positive. Sparse XFeat uses its nearest feature within 5 km, "
        "so its localization contract differs from exact-location ORB.\n\n"
        + markdown_table(descriptor_summary, descriptor_columns)
        + f"\n\nUsing the previous true view, ORB ranks the same buoy first in "
        f"{validation_orb_previous:.1%} of eligible February cases and "
        f"{holdout_orb_previous:.1%} of N-ICE cases. BRISK reaches "
        f"{validation_brisk_previous:.1%} and {holdout_brisk_previous:.1%}; sparse "
        f"XFeat reaches {validation_xfeat_previous:.1%} and "
        f"{holdout_xfeat_previous:.1%}. ORB is therefore the supported descriptor "
        "for the next update experiment. BRISK remains a useful binary control, "
        "while the sparse XFeat setup is not competitive in this role.\n"
        + "\n\n## Direct update-rule training\n\n"
        f"A rule is selectable only if at least {minimum_safe_update_precision:.1%} "
        "of its February memory updates are within 2 km of the buoy. Within that "
        "safety constraint, complete-path accuracy selects the rule. "
        f"The selected rule requires a best-match lead of {final_policy.update_min_margin:.3f} "
        f"and a maximum descriptor difference of {final_policy.update_max_cost:.3f}. "
        "Buoy truth is used only to score entire paths and label whether a memory update "
        "was safe; it is never used by candidate ranking.\n\n"
        + markdown_table(policy_results, policy_result_columns)
        + "\n\nSelection stability across February folds:\n\n"
        + markdown_table(fold_counts, ["policy", "folds_selected"])
        + "\n\nThe grouped cross-validation row combines held-out buoy paths, each evaluated "
        "with a rule selected without that buoy. Fractions retain untracked transitions "
        "in the denominator. The N-ICE table also includes the two predefined memory "
        "baselines and the previous hand-set rule.\n\n"
        "## Interpretation\n\n"
        "The buoy-selected 0.032/0.40 rule and the existing hand-set 0.032/0.35 "
        "rule produce identical N-ICE paths, updates, and errors. The selected "
        "thresholds also vary across February folds. This is evidence to retain the "
        "existing conservative rule, not to change its threshold. Previous-view ORB "
        "is valuable as proposal evidence, but should enter persistent memory only "
        "after independent confirmation.\n\n"
        "This milestone trains thresholds, not a SAR neural descriptor. A learned "
        "safe-update classifier was tested separately but did not preserve its false-"
        "update rate on N-ICE; it is not selected. The next update model should use "
        "relative, image-normalized evidence and must be rerun inside the graph so each "
        "memory decision changes later matching.\n"
    )


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--validation-sequence", default="2020_02")
    parser.add_argument("--holdout-sequence", default="2015_full15")
    parser.add_argument("--development-sequence", default="2020_03")
    parser.add_argument("--best-match-leads", default="0,0.008,0.016,0.032,0.064,0.096")
    parser.add_argument(
        "--maximum-descriptor-differences",
        default="0.20,0.25,0.30,0.35,0.40,0.45",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260815)
    parser.add_argument("--error-threshold-m", type=float, default=2000.0)
    parser.add_argument("--minimum-safe-update-precision", type=float, default=0.95)
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="Reuse the existing full validation grid and rebuild selection/holdout outputs.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    leads = parse_float_tuple(args.best_match_leads)
    limits = parse_float_tuple(args.maximum_descriptor_differences)
    training_policies = threshold_policies(leads, limits)
    validation_manifest = json.loads(
        (args.graph_root / args.validation_sequence / "run_manifest.json").read_text()
    )
    graph_args = graph_arguments(validation_manifest)
    if args.reuse_validation:
        validation_records = pd.read_csv(
            args.out_dir / "validation_policy_paths.csv", dtype={"buoy_id": str}
        )
        validation_summary = pd.read_csv(
            args.out_dir / "validation_policy_summary.csv"
        )
        validation_coincidences = pd.read_csv(
            args.graph_root / args.validation_sequence / "coincidences.csv",
            dtype={"buoy_id": str},
        )
        validation_coincidences["image_time"] = pd.to_datetime(
            validation_coincidences.image_time, utc=True
        )
        validation_precompute = 0.0
    else:
        validation_records, validation_coincidences, graph_args, validation_precompute = run_graph_policies(
            args.graph_root, args.validation_sequence, training_policies
        )
        validation_summary = summarize_policies(
            validation_records, validation_coincidences, training_policies
        )
    fold_results, cross_validated_records = grouped_policy_selection(
        validation_records,
        validation_coincidences,
        training_policies,
        args.folds,
        args.random_seed,
        args.minimum_safe_update_precision,
    )
    cross_validated_policy = GraphSearchConfig(
        "cross_validated_buoy_selected_rule",
        "confidence_rolling",
        32,
        8,
        update_min_margin=math.nan,
        update_max_cost=math.nan,
    )
    cross_validated_summary = summarize_policies(
        cross_validated_records,
        validation_coincidences,
        (cross_validated_policy,),
    )
    selected_name = select_policy(
        validation_summary, args.minimum_safe_update_precision
    )
    final_policy = next(policy for policy in training_policies if policy.name == selected_name)

    holdout_policies = comparison_policies(final_policy)
    holdout_records, holdout_coincidences, holdout_args, holdout_precompute = run_graph_policies(
        args.graph_root, args.holdout_sequence, holdout_policies
    )
    holdout_summary = summarize_policies(
        holdout_records, holdout_coincidences, holdout_policies
    )

    descriptor_pair_frames = []
    descriptor_summary_frames = []
    for sequence in (args.validation_sequence, args.holdout_sequence):
        pairs, summary = descriptor_separability(
            args.patch_root,
            args.out_dir,
            sequence,
            graph_args.max_speed_m_per_day,
            args.error_threshold_m,
        )
        descriptor_pair_frames.append(pairs)
        descriptor_summary_frames.append(summary)
    descriptor_pairs = pd.concat(descriptor_pair_frames, ignore_index=True)
    descriptor_summary = pd.concat(descriptor_summary_frames, ignore_index=True)

    validation_records.to_csv(args.out_dir / "validation_policy_paths.csv", index=False)
    validation_summary.to_csv(args.out_dir / "validation_policy_summary.csv", index=False)
    fold_results.to_csv(args.out_dir / "validation_grouped_folds.csv", index=False)
    cross_validated_summary.to_csv(
        args.out_dir / "validation_cross_validated_summary.csv", index=False
    )
    holdout_records.to_csv(args.out_dir / "holdout_policy_paths.csv", index=False)
    holdout_summary.to_csv(args.out_dir / "holdout_policy_summary.csv", index=False)
    descriptor_pairs.to_csv(args.out_dir / "descriptor_pair_ranks.csv", index=False)
    descriptor_summary.to_csv(args.out_dir / "descriptor_summary.csv", index=False)
    pd.DataFrame(PLAIN_LANGUAGE_GLOSSARY).to_csv(
        args.out_dir / "plain_language_glossary.csv", index=False
    )
    write_report(
        args.out_dir / "report.md",
        descriptor_summary,
        validation_summary,
        cross_validated_summary,
        final_policy,
        holdout_summary,
        fold_results,
        args.minimum_safe_update_precision,
    )
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "validation_sequence": args.validation_sequence,
        "holdout_sequence": args.holdout_sequence,
        "development_sequence": args.development_sequence,
        "split_unit": "whole buoy path",
        "grid_border_px": graph_args.grid_border,
        "analysis_epsg": graph_args.analysis_epsg,
        "distance_units": "metres",
        "error_threshold_m": args.error_threshold_m,
        "minimum_safe_update_precision": args.minimum_safe_update_precision,
        "training_policies": [asdict(policy) for policy in training_policies],
        "selected_policy": asdict(final_policy),
        "production_change_recommended": False,
        "validation_precompute_seconds": validation_precompute,
        "holdout_precompute_seconds": holdout_precompute,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SELECTED POLICY")
    print(json.dumps(asdict(final_policy), indent=2))
    print("\nDESCRIPTORS")
    print(descriptor_summary.to_string(index=False))
    print("\nVALIDATION CROSS-VALIDATED")
    print(cross_validated_summary.to_string(index=False))
    print("\nHOLDOUT")
    print(holdout_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
