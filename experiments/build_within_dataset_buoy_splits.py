#!/usr/bin/env python3
"""Build deterministic whole-buoy splits within the full-70 2020 dataset.

The primary split deliberately permits images to be shared between folds because
many buoys are observed in the same Sentinel-1 acquisition. Requiring both buoy
and image exclusivity collapses most observations into one connected component.
The split therefore tests generalisation to unseen buoy paths under the observed
2020 image distribution. Image/pass overlap is measured and reported explicitly.

Only fixture metadata are used for stratification. No descriptor, matcher, buoy
error, or downstream method score enters the assignment objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = ROOT / "results/iabp_s1_stratified_coverage"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/arctic_tracking_next_experiment/splits/full70_2020"
)
FOLDS = ("development", "confirmation", "final_holdout")

FAMILY_TOTAL_WEIGHTS = {
    "overall": 8.0,
    "observation_month": 2.0,
    "transition_month": 4.0,
    "cadence": 2.0,
    "sic": 1.0,
    "spatial_block": 2.0,
    "path_length": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_buoy_ids(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def _safe_category(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("missing")


def _count_features(
    frame: pd.DataFrame,
    category: str,
    prefix: str,
) -> pd.DataFrame:
    categories = _safe_category(frame[category])
    table = pd.crosstab(frame["buoy_id"], categories)
    table.columns = [f"{prefix}::{value}" for value in table.columns]
    return table.astype(float)


def trajectory_length_band(length: int) -> str:
    if length <= 2:
        return "02"
    if length <= 4:
        return "03_to_04"
    if length <= 7:
        return "05_to_07"
    return "08_plus"


def build_group_features(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return per-buoy metadata counts and feature-family membership."""
    buoy_index = pd.Index(sorted(observations["buoy_id"].unique()), name="buoy_id")
    features = pd.DataFrame(index=buoy_index)
    features["buoy_count"] = 1.0
    features["observation_count"] = observations.groupby("buoy_id").size()
    features["transition_count"] = transitions.groupby("buoy_id").size()
    features["trajectory_count"] = observations.groupby("buoy_id")[
        "experiment_trajectory_id"
    ].nunique()
    features = features.fillna(0.0)

    families: dict[str, list[str]] = {
        "overall": [
            "buoy_count",
            "observation_count",
            "transition_count",
            "trajectory_count",
        ]
    }
    tables = [features]
    specifications = (
        (observations, "month", "observation_month"),
        (transitions, "month", "transition_month"),
        (transitions, "cadence_band", "cadence"),
        (observations, "sic_regime", "sic"),
        (observations, "spatial_block", "spatial_block"),
    )
    for frame, category, family in specifications:
        table = _count_features(frame, category, family)
        families[family] = list(table.columns)
        tables.append(table)

    path_lengths = (
        observations.groupby(["buoy_id", "experiment_trajectory_id"])
        .size()
        .rename("path_observations")
        .reset_index()
    )
    path_lengths["path_length_band"] = path_lengths["path_observations"].map(
        trajectory_length_band
    )
    path_table = _count_features(path_lengths, "path_length_band", "path_length")
    families["path_length"] = list(path_table.columns)
    tables.append(path_table)

    result = pd.concat(tables, axis=1).reindex(buoy_index).fillna(0.0)
    return result.astype(float), families


def feature_weights(
    columns: Iterable[str],
    families: dict[str, list[str]],
) -> np.ndarray:
    by_column: dict[str, float] = {}
    for family, family_columns in families.items():
        if not family_columns:
            continue
        per_column = FAMILY_TOTAL_WEIGHTS[family] / len(family_columns)
        by_column.update({column: per_column for column in family_columns})
    return np.asarray([by_column[column] for column in columns], dtype=float)


def assignment_score(
    matrix: np.ndarray,
    assignment: np.ndarray,
    weights: np.ndarray,
    fold_count: int,
) -> float:
    totals = matrix.sum(axis=0)
    fold_sums = np.vstack(
        [matrix[assignment == fold].sum(axis=0) for fold in range(fold_count)]
    )
    proportions = np.divide(
        fold_sums,
        totals,
        out=np.zeros_like(fold_sums, dtype=float),
        where=totals > 0,
    )
    target = 1.0 / fold_count
    return float(np.sum(weights * np.square(proportions - target)))


def _greedy_assignment(
    matrix: np.ndarray,
    weights: np.ndarray,
    fold_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    totals = matrix.sum(axis=0)
    normalized = np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals > 0,
    )
    magnitude = normalized @ weights
    jitter = rng.uniform(0.95, 1.05, size=len(matrix))
    order = np.argsort(-(magnitude * jitter), kind="stable")
    assignment = np.full(len(matrix), -1, dtype=int)

    for row_index in order:
        candidate_scores = []
        for fold in range(fold_count):
            proposal = assignment.copy()
            proposal[row_index] = fold
            assigned = proposal >= 0
            candidate_scores.append(
                assignment_score(
                    matrix[assigned], proposal[assigned], weights, fold_count
                )
            )
        minimum = min(candidate_scores)
        tied = np.flatnonzero(np.isclose(candidate_scores, minimum, atol=1.0e-14))
        assignment[row_index] = int(rng.choice(tied))
    return assignment


def _improve_by_swaps(
    matrix: np.ndarray,
    assignment: np.ndarray,
    weights: np.ndarray,
    fold_count: int,
    maximum_passes: int = 8,
) -> tuple[np.ndarray, float]:
    result = assignment.copy()
    current = assignment_score(matrix, result, weights, fold_count)
    for _ in range(maximum_passes):
        improved = False
        for left in range(len(result)):
            for right in range(left + 1, len(result)):
                if result[left] == result[right]:
                    continue
                proposal = result.copy()
                proposal[left], proposal[right] = proposal[right], proposal[left]
                score = assignment_score(matrix, proposal, weights, fold_count)
                if score + 1.0e-14 < current:
                    result = proposal
                    current = score
                    improved = True
        if not improved:
            break
    return result, current


def assign_buoys(
    group_features: pd.DataFrame,
    seed: int,
    restarts: int,
    folds: tuple[str, ...] = FOLDS,
) -> tuple[pd.Series, float, np.ndarray]:
    """Assign every buoy to one fold using metadata-only multi-start balancing."""
    if restarts < 1:
        raise ValueError("restarts must be positive")
    columns = list(group_features.columns)
    families: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        family = column.split("::", 1)[0] if "::" in column else "overall"
        families[family].append(column)
    weights = feature_weights(columns, dict(families))
    matrix = group_features.to_numpy(dtype=float)

    best_assignment: np.ndarray | None = None
    best_score = np.inf
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        proposal = _greedy_assignment(matrix, weights, len(folds), rng)
        score = assignment_score(matrix, proposal, weights, len(folds))
        if score < best_score:
            best_assignment = proposal
            best_score = score
    assert best_assignment is not None
    best_assignment, best_score = _improve_by_swaps(
        matrix, best_assignment, weights, len(folds)
    )
    labels = pd.Series(
        [folds[index] for index in best_assignment],
        index=group_features.index,
        name="within_dataset_split",
        dtype="string",
    )
    return labels, best_score, weights


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def linkage_components(observations: pd.DataFrame) -> list[dict[str, object]]:
    linkage = DisjointSet()
    for row in observations.itertuples(index=False):
        buoy = f"buoy:{row.buoy_id}"
        image = f"image:{row.image_id}"
        acquisition_pass = f"pass:{row.acquisition_pass_id}"
        linkage.union(buoy, image)
        linkage.union(image, acquisition_pass)

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(observations.itertuples(index=False)):
        grouped[linkage.find(f"buoy:{row.buoy_id}")].append(index)

    records = []
    for indices in grouped.values():
        group = observations.iloc[indices]
        records.append(
            {
                "observations": int(len(group)),
                "buoys": int(group["buoy_id"].nunique()),
                "images": int(group["image_id"].nunique()),
                "acquisition_passes": int(group["acquisition_pass_id"].nunique()),
                "months": sorted(group["month"].astype(str).unique().tolist()),
            }
        )
    return sorted(records, key=lambda record: record["observations"], reverse=True)


def summarize_splits(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold in FOLDS:
        obs = observations[observations["within_dataset_split"] == fold]
        trans = transitions[transitions["within_dataset_split"] == fold]
        records.append(
            {
                "within_dataset_split": fold,
                "observations": int(len(obs)),
                "buoys": int(obs["buoy_id"].nunique()),
                "images": int(obs["image_id"].nunique()),
                "acquisition_passes": int(obs["acquisition_pass_id"].nunique()),
                "experiment_trajectories": int(
                    obs["experiment_trajectory_id"].nunique()
                ),
                "transitions": int(len(trans)),
                "months": int(obs["month"].nunique()),
                "spatial_blocks": int(obs["spatial_block"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)


def shared_support_summary(observations: pd.DataFrame) -> dict[str, object]:
    image_split_counts = observations.groupby("image_id")[
        "within_dataset_split"
    ].nunique()
    pass_split_counts = observations.groupby("acquisition_pass_id")[
        "within_dataset_split"
    ].nunique()
    return {
        "images_total": int(len(image_split_counts)),
        "images_in_multiple_splits": int((image_split_counts > 1).sum()),
        "image_split_count_distribution": {
            str(key): int(value)
            for key, value in image_split_counts.value_counts().sort_index().items()
        },
        "passes_total": int(len(pass_split_counts)),
        "passes_in_multiple_splits": int((pass_split_counts > 1).sum()),
        "pass_split_count_distribution": {
            str(key): int(value)
            for key, value in pass_split_counts.value_counts().sort_index().items()
        },
    }


def build_splits(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
    seed: int,
    restarts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    obs = observations.copy()
    trans = transitions.copy()
    obs["buoy_id"] = normalize_buoy_ids(obs["buoy_id"])
    trans["buoy_id"] = normalize_buoy_ids(trans["buoy_id"])
    if "eligible_tracking_observation" in obs:
        obs = obs[obs["eligible_tracking_observation"].astype(bool)].copy()

    unknown_transition_buoys = sorted(set(trans["buoy_id"]) - set(obs["buoy_id"]))
    if unknown_transition_buoys:
        raise ValueError(
            "Transitions reference buoy IDs absent from eligible observations: "
            f"{unknown_transition_buoys[:5]}"
        )

    group_features, families = build_group_features(obs, trans)
    labels, score, weights = assign_buoys(group_features, seed, restarts)
    obs["within_dataset_split"] = obs["buoy_id"].map(labels)
    trans["within_dataset_split"] = trans["buoy_id"].map(labels)
    if obs["within_dataset_split"].isna().any() or trans[
        "within_dataset_split"
    ].isna().any():
        raise AssertionError("Every eligible row must receive one split label")

    assignments = group_features.copy()
    assignments.insert(0, "within_dataset_split", labels)
    assignments = assignments.reset_index()
    summary = summarize_splits(obs, trans)
    split_buoy_sets = {
        fold: set(obs.loc[obs["within_dataset_split"] == fold, "buoy_id"])
        for fold in FOLDS
    }
    for index, left in enumerate(FOLDS):
        for right in FOLDS[index + 1 :]:
            if split_buoy_sets[left] & split_buoy_sets[right]:
                raise AssertionError("A buoy ID was assigned to multiple splits")

    manifest = {
        "dataset": "full70_2020",
        "split_type": "whole_buoy_within_dataset",
        "folds": list(FOLDS),
        "seed": int(seed),
        "restarts": int(restarts),
        "objective_score": float(score),
        "stratification_uses_method_scores": False,
        "stratification_families": families,
        "feature_columns": list(group_features.columns),
        "feature_weights": {
            column: float(weight)
            for column, weight in zip(group_features.columns, weights)
        },
        "summary": summary.to_dict(orient="records"),
        "shared_image_pass_support": shared_support_summary(obs),
        "buoy_image_pass_linkage_components": linkage_components(obs),
        "interpretation": (
            "Primary holdout is exclusive by buoy ID but not by SAR image/pass. "
            "Uncertainty must be clustered by buoy and image/pass."
        ),
    }
    return obs, trans, assignments, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_FIXTURE_DIR / "full70_level1_tracking_observations.csv",
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=DEFAULT_FIXTURE_DIR / "full70_level1_tracking_transitions.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--restarts", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = pd.read_csv(args.observations, dtype={"buoy_id": "string"})
    transitions = pd.read_csv(args.transitions, dtype={"buoy_id": "string"})
    obs, trans, assignments, manifest = build_splits(
        observations, transitions, args.seed, args.restarts
    )
    manifest["inputs"] = {
        "observations": str(args.observations.resolve()),
        "observations_sha256": sha256_file(args.observations),
        "transitions": str(args.transitions.resolve()),
        "transitions_sha256": sha256_file(args.transitions),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs.to_csv(args.out_dir / "observations.csv", index=False)
    trans.to_csv(args.out_dir / "transitions.csv", index=False)
    assignments.to_csv(args.out_dir / "buoy_assignments.csv", index=False)
    summarize_splits(obs, trans).to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
