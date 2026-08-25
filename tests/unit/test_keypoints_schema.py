import pandas as pd
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from tests.helpers_assertions import assert_flags_valid, assert_no_duplicates


@pytest.mark.unit
def test_keypoints_empty_schema():
    from limosat.keypoints import Keypoints
    kp = Keypoints()
    required = {
        'image_id', 'is_last', 'trajectory_id', 'geometry', 'descriptors',
        'angle', 'corr', 'time', 'interpolated', 'orbit_num', 'stopped', 'converged_to'
    }
    assert required.issubset(set(kp.columns))
    assert kp.crs == 'EPSG:3413'
    assert_flags_valid(kp)


@pytest.mark.unit
def test_keypoints_rejects_other_coordinate_systems():
    from limosat.keypoints import Keypoints

    with pytest.raises(ValueError, match="must use EPSG:3413"):
        Keypoints(
            {'trajectory_id': [0]},
            geometry=[Point(0, 0)],
            crs='EPSG:4326',
        )


@pytest.mark.unit
def test_keypoints_append_and_update_semantics():
    from limosat.keypoints import Keypoints

    base = gpd.GeoDataFrame(
        {
            'image_id': [1, 1],
            'is_last': [1, 1],
            'trajectory_id': [0, 1],
            'descriptors': [np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8)],
            'angle': [0.0, 0.0],
            'corr': [0.0, 0.0],
            'time': [pd.Timestamp('2025-01-01')]*2,
            'interpolated': [0, 0],
            'orbit_num': [1, 1],
            'stopped': [False, False],
            'converged_to': [None, None],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs='EPSG:3413',
    )
    kp = Keypoints(base)

    # Append external frame with missing/extra columns
    ext = gpd.GeoDataFrame(
        {
            'image_id': [2],
            'is_last': [1],
            'trajectory_id': [2],
            'time': [pd.Timestamp('2025-01-02')],
        },
        geometry=[Point(2, 2)],
        crs='EPSG:3413',
    )
    before_cols = set(kp.columns)
    kp2 = kp.append(ext)
    assert set(kp2.columns) == before_cols
    assert_no_duplicates(kp2, keys=("trajectory_id", "image_id"))

    # Update existing trajectories should clear previous is_last for those ids
    upd = gpd.GeoDataFrame(
        {
            'image_id': [3],
            'is_last': [1],
            'trajectory_id': [0],
            'time': [pd.Timestamp('2025-01-03')],
        },
        geometry=[Point(0.5, 0.5)], crs='EPSG:3413')
    kp3 = kp2.update(upd)
    # Previous rows for tid=0 should have is_last set to 0 somewhere inside kp3
    prev_rows = kp3[kp3['trajectory_id'] == 0]
    assert (prev_rows['is_last'] == 0).any()
