from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from limosat import (
    DisplacementField,
    FieldConfig,
    FieldEdge,
    ImagePair,
    ImageRecord,
    MotionMatches,
    PairProductStore,
    PairResult,
    RoutingConfig,
    RunConfig,
    TrajectoryConfig,
)
from limosat.compose import compose_primary_parquet
from limosat.trajectory import iter_global_trajectory_batches


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GRID = np.array(
    [[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]]
)


def _field(displacement=(100.0, 0.0)):
    return DisplacementField(
        pair_id="a__b",
        source_image_id="a",
        target_image_id="b",
        source_time_utc=START,
        target_time_utc=START + timedelta(days=1),
        grid_row=np.array([0, 0, 1, 1]),
        grid_column=np.array([0, 1, 0, 1]),
        source_xy_m=GRID,
        displacement_m=np.tile(displacement, (4, 1)),
        available=np.ones(4, dtype=bool),
        selected_matches=np.full(4, 8),
        candidate_matches=np.full(4, 12),
        support_radius_m=np.full(4, 500.0),
        maximum_residual_m=np.full(4, 20.0),
    )


def _field_config():
    return FieldConfig(
        grid_spacing_m=1_000.0,
        neighbour_count=4,
        minimum_agreeing_matches=3,
        maximum_neighbour_distance_m=2_000.0,
        agreement_distance_m=100.0,
        maximum_triangle_edge_m=1_500.0,
    )


def test_trajectory_batches_expose_direct_extensions_without_a_second_scan():
    images = [
        ImageRecord("a", Path("/tmp/a.tif"), START),
        ImageRecord("b", Path("/tmp/b.tif"), START + timedelta(days=1)),
    ]

    batches = tuple(
        iter_global_trajectory_batches(
            [FieldEdge(_field())], images, _field_config(), TrajectoryConfig()
        )
    )

    assert len(batches) == 2
    assert len(batches[0].extensions) == 0
    assert len(batches[1].extensions) == 4
    assert {value.pair_id for value in batches[1].extensions} == {"a__b"}
    assert {value.target_state for value in batches[1].extensions} == {"observed"}
    for value in batches[1].extensions:
        np.testing.assert_allclose(value.displacement_m, [100.0, 0.0])
        assert value.source_image_id == "a"
        assert value.target_image_id == "b"


def test_compose_uses_pair_products_without_creating_sqlite(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "a.tif"
    target = tmp_path / "b.tif"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    catalogue_path = tmp_path / "catalogue.csv"
    catalogue_path.write_text(
        "image_id,path,time_utc,footprint_wkt\n"
        f'a,{source},2020-01-01T00:00:00+00:00,"POLYGON ((0 0, 40000 0, '
        '40000 40000, 0 40000, 0 0))"\n'
        f'b,{target},2020-01-02T00:00:00+00:00,"POLYGON ((0 0, 40000 0, '
        '40000 40000, 0 40000, 0 0))"\n',
        encoding="utf-8",
    )
    config = RunConfig(
        run_id="parquet-direct",
        catalogue=str(catalogue_path),
        database=str(tmp_path / "must-not-exist.sqlite"),
        output_directory=str(tmp_path / "native-output"),
        pair_product_directory=str(tmp_path / "pair-products"),
        field=_field_config(),
        routing=RoutingConfig(coarse_matching=False, initial="same_center"),
        retain_pair_matches=True,
    )
    images = [
        ImageRecord("a", source, START),
        ImageRecord("b", target, START + timedelta(days=1)),
    ]
    pair = ImagePair(images[0], images[1])
    result = PairResult(
        MotionMatches(
            np.array([[0.0, 0.0]]),
            np.array([[100.0, 0.0]]),
            np.array([0.9]),
            np.array([0]),
            np.array([0]),
        ),
        _field(),
        np.empty(0, dtype=np.int32),
        {"total": 1.0},
        1,
    )
    PairProductStore(config).save(pair, "primary", False, result)

    output = tmp_path / "parquet-output"
    composed = compose_primary_parquet(config, output)

    assert not Path(config.database).exists()
    assert composed["counts"]["trajectory_points"] == 8
    assert composed["counts"]["trajectory_extensions"] == 4
    extensions = pq.read_table(composed["trajectory_extensions"])
    assert extensions.num_rows == 4
    assert extensions.column("qc_status").to_pylist() == ["accept"] * 4
    assert extensions.column("accepted").to_pylist() == [True] * 4
