from datetime import datetime, timedelta, timezone

from limosat.qc import ExtensionQCConfig, score_extensions
from limosat.trajectory import TrajectoryExtension


START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_basic_qc_rejects_only_hard_speed_and_reviews_local_anomaly():
    extensions = []
    for index in range(12):
        dx_m = 1_000.0
        if index == 6:
            dx_m = 25_000.0
        if index == 11:
            dx_m = 70_000.0
        x0_m = index * 4_000.0
        extensions.append(
            TrajectoryExtension(
                trajectory_id=f"trajectory-{index}",
                pair_id="a__b",
                pair_kind="primary",
                source_image_id="a",
                target_image_id="b",
                start_time_utc=START,
                end_time_utc=START + timedelta(days=1),
                x0_m=x0_m,
                y0_m=0.0,
                x1_m=x0_m + dx_m,
                y1_m=0.0,
                target_state="observed",
                target_position_basis="primary_pair_field",
                selected_matches=8.0,
                support_radius_m=500.0,
                maximum_residual_m=20.0,
            )
        )

    scored = score_extensions(
        extensions,
        ExtensionQCConfig(configured_speed_m_per_day=43_200.0),
    )

    assert scored[6].local_neighbour_count >= 8
    assert scored[6].local_neighbour_review
    assert scored[6].qc_status == "review"
    assert not scored[6].hard_speed_reject
    assert scored[11].hard_speed_reject
    assert scored[11].qc_status == "reject"
    assert scored[11].qc_reasons.startswith("hard_speed")
    assert scored[0].local_neighbour_count < 8
    assert scored[0].local_residual_m is None
    assert scored[0].qc_status == "accept"
