from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate_efficientloftr_batch_acceptance import evaluate


def write_field(path: Path, *, outlier_m: float = 0.0, unavailable: int | None = None) -> None:
    rows = []
    for index, (row, column) in enumerate(np.ndindex(3, 3)):
        rows.append(
            {
                "grid_id": f"cell-{row}-{column}",
                "grid_row": row,
                "grid_column": column,
                "source_x": column * 4_000.0,
                "source_y": row * 4_000.0,
                "available": index != unavailable,
                "selected_vectors": 8,
                "candidate_count": 12,
                "support_radius_m": 6_000.0,
                "proposal_dx_m": 100.0 + (outlier_m if index == 4 else 0.0),
                "proposal_dy_m": -50.0,
                "maximum_vector_residual_m": 200.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_trajectory(path: Path, *, lost: int | None = None, terminal_offset_m: float = 0.0) -> None:
    rows = []
    for index in range(4):
        complete = index != lost
        rows.append(
            {
                "grid_id": f"trajectory-{index}",
                "pair1_valid": 1,
                "pair2_valid": int(complete),
                "complete": int(complete),
                "pair2_target_x_m": 10_000.0 + index + terminal_offset_m,
                "pair2_target_y_m": -20_000.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_truth(path: Path, *, error_offset_m: float = 0.0, lost: int | None = None) -> None:
    rows = [
        {
            "buoy_id": f"buoy-{index}",
            "image_id": 2,
            "available": int(index != lost),
            "error_m": 20.0 + index + error_offset_m,
        }
        for index in range(4)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def fixture(tmp_path: Path):
    baseline_fields = tmp_path / "baseline_fields"
    candidate_fields = tmp_path / "candidate_fields"
    baseline_trajectory = tmp_path / "baseline_trajectory.csv"
    candidate_trajectory = tmp_path / "candidate_trajectory.csv"
    baseline_truth = tmp_path / "baseline_truth.csv"
    candidate_truth = tmp_path / "candidate_truth.csv"
    write_field(baseline_fields / "pair" / "field_4km.csv")
    write_field(candidate_fields / "pair" / "field_4km.csv")
    write_trajectory(baseline_trajectory)
    write_trajectory(candidate_trajectory)
    write_truth(baseline_truth)
    write_truth(candidate_truth)
    return (
        baseline_fields,
        candidate_fields,
        baseline_trajectory,
        candidate_trajectory,
        baseline_truth,
        candidate_truth,
    )


def test_exact_batch_products_pass_all_acceptance_gates(tmp_path):
    inputs = fixture(tmp_path)

    result = evaluate(
        inputs[0], inputs[1], ["pair"], inputs[2], inputs[3],
        baseline_truth=inputs[4], candidate_truth=inputs[5],
    )

    assert result["status"] == "pass"
    assert all(result["checks"].values())
    assert result["fields"]["pair"]["outlier_count"] == 0
    assert result["trajectories"]["reference_only_complete"] == 0


def test_field_and_trajectory_regressions_fail_and_report_outlier(tmp_path):
    inputs = fixture(tmp_path)
    write_field(inputs[1] / "pair" / "field_4km.csv", outlier_m=1_200.0, unavailable=0)
    write_trajectory(inputs[3], lost=0, terminal_offset_m=1_100.0)

    result = evaluate(inputs[0], inputs[1], ["pair"], inputs[2], inputs[3])

    assert result["status"] == "fail"
    field = result["fields"]["pair"]
    assert field["outlier_count"] == 1
    assert field["outliers"][0]["grid_id"] == "cell-1-1"
    assert field["outliers"][0]["vector_error_m"] == 1_200.0
    assert not field["checks"]["vector_maximum"]
    assert not result["trajectories"]["checks"]["no_complete_trajectory_loss"]
    assert not result["trajectories"]["checks"]["terminal_maximum"]


def test_truth_availability_and_error_regressions_fail(tmp_path):
    inputs = fixture(tmp_path)
    write_truth(inputs[5], error_offset_m=25.0, lost=0)

    result = evaluate(
        inputs[0], inputs[1], ["pair"], inputs[2], inputs[3],
        baseline_truth=inputs[4], candidate_truth=inputs[5],
    )

    truth = result["truth"]
    assert result["status"] == "fail"
    assert not truth["checks"]["no_truth_availability_loss"]
    assert not truth["checks"]["median_error"]
    assert not truth["checks"]["p90_error"]
