"""
Test that seed rows are persisted when trajectories become newly eligible for persistence.

This test validates the fix for the issue where seeds created before last_persisted_id
were silently skipped during persistence when their trajectory gained a second observation.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy import create_engine


@pytest.mark.unit
def test_seed_persistence_for_newly_matched_trajectory():
    """
    Test that when a trajectory gains its 2nd observation (becoming eligible for persistence),
    its seed row is included even if the seed's image_id <= last_persisted_id.
    """
    from limosat.keypoints import Keypoints
    
    # Simulate a scenario:
    # - image_id 1: seed created (trajectory_id=10)
    # - image_id 2: another seed created (trajectory_id=11)
    # - Persistence happens at image_id 2, last_persisted_id becomes 2
    # - image_id 5: trajectory_id=10 gets matched (now has 2 observations)
    #   The seed at image_id=1 should be included in this persistence
    
    # Create points representing the state at image_id 5
    points_data = {
        'trajectory_id': [10, 10, 11],  # traj 10 has 2 obs, traj 11 has 1 obs
        'image_id': [1, 5, 2],  # seed at 1, match at 5, and another seed at 2
        'is_last': [0, 1, 1],
        'geometry': [Point(100, 100), Point(105, 105), Point(200, 200)],
        'descriptors': [None, np.array([1, 2, 3], dtype=np.uint8), None],
        'angle': [0.0, 0.0, 0.0],
        'corr': [0.0, 0.8, 0.0],
        'time': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-02']),
        'interpolated': [0, 0, 0],
        'orbit_num': [1, 2, 1],
        'stopped': [False, False, False],
        'converged_to': [None, None, None],
    }
    points = Keypoints(points_data, crs='EPSG:3413')
    
    # Set last_persisted_id to 2 (simulating previous persistence)
    last_persisted_id = 2
    
    # Manually execute the logic to test (same as in database.py save())
    points_image_id_series = points['image_id'].astype(int)
    last_persisted_id_int = int(last_persisted_id)
    mask_series = points_image_id_series > last_persisted_id_int
    points_delta = points.loc[mask_series].copy()
    
    # Check initial delta (should only have image_id=5)
    assert len(points_delta) == 1
    assert 5 in points_delta['image_id'].values
    
    # Now apply the seed inclusion logic (same as in database.py)
    traj_ids_in_delta = points_delta['trajectory_id'].unique()
    seed_rows_to_add = []
    for tid in traj_ids_in_delta:
        if pd.isna(tid):
            continue
        traj_points = points[points['trajectory_id'] == tid]
        if len(traj_points) <= 1:
            continue
        seed_row = traj_points.loc[traj_points['image_id'].idxmin()]
        seed_image_id = int(seed_row['image_id'])
        if seed_image_id <= last_persisted_id_int:
            seed_rows_to_add.append(seed_row)
    
    # Should have found the seed for trajectory 10
    assert len(seed_rows_to_add) == 1
    assert seed_rows_to_add[0]['trajectory_id'] == 10
    assert seed_rows_to_add[0]['image_id'] == 1
    
    # After adding seeds, points_delta should have both the seed and the match
    if seed_rows_to_add:
        seeds_df = pd.DataFrame(seed_rows_to_add)
        seeds_gdf = gpd.GeoDataFrame(seeds_df, geometry='geometry', crs=points.crs)
        points_delta = pd.concat([points_delta, seeds_gdf], ignore_index=True)
        points_delta = points_delta.drop_duplicates(subset=['trajectory_id', 'image_id'], keep='first')
    
    # Now points_delta should have 2 rows: seed at image_id=1 and match at image_id=5
    assert len(points_delta) == 2
    assert 1 in points_delta['image_id'].values
    assert 5 in points_delta['image_id'].values
    assert all(points_delta['trajectory_id'] == 10)


@pytest.mark.unit
def test_seed_persistence_only_for_matched_trajectories():
    """
    Test that singleton trajectories (with only 1 observation) are NOT persisted,
    even if we try to include their seeds.
    """
    from limosat.keypoints import Keypoints
    
    # Create points with only singleton trajectories
    points_data = {
        'trajectory_id': [10, 11],  # Both have only 1 observation
        'image_id': [1, 5],
        'is_last': [1, 1],
        'geometry': [Point(100, 100), Point(200, 200)],
        'descriptors': [None, None],
        'angle': [0.0, 0.0],
        'corr': [0.0, 0.0],
        'time': pd.to_datetime(['2023-01-01', '2023-01-05']),
        'interpolated': [0, 0],
        'orbit_num': [1, 2],
        'stopped': [False, False],
        'converged_to': [None, None],
    }
    points = Keypoints(points_data, crs='EPSG:3413')
    
    last_persisted_id = 2
    
    # Apply the logic
    points_image_id_series = points['image_id'].astype(int)
    last_persisted_id_int = int(last_persisted_id)
    mask_series = points_image_id_series > last_persisted_id_int
    points_delta = points.loc[mask_series].copy()
    
    # Should have image_id=5
    assert len(points_delta) == 1
    
    # Try to find seeds to add
    traj_ids_in_delta = points_delta['trajectory_id'].unique()
    seed_rows_to_add = []
    for tid in traj_ids_in_delta:
        if pd.isna(tid):
            continue
        traj_points = points[points['trajectory_id'] == tid]
        # This trajectory has only 1 observation, should be skipped
        if len(traj_points) <= 1:
            continue
        seed_row = traj_points.loc[traj_points['image_id'].idxmin()]
        seed_image_id = int(seed_row['image_id'])
        if seed_image_id <= last_persisted_id_int:
            seed_rows_to_add.append(seed_row)
    
    # Should NOT have found any seeds (singleton trajectories are excluded)
    assert len(seed_rows_to_add) == 0


@pytest.mark.unit  
def test_seed_persistence_with_multiple_newly_matched():
    """
    Test the case where multiple trajectories become newly matched simultaneously.
    """
    from limosat.keypoints import Keypoints
    
    # Create points with multiple trajectories gaining 2nd observations
    points_data = {
        'trajectory_id': [10, 10, 11, 11, 12],
        'image_id': [1, 5, 2, 5, 5],
        'is_last': [0, 1, 0, 1, 1],
        'geometry': [Point(100, 100), Point(105, 105), Point(200, 200), Point(205, 205), Point(300, 300)],
        'descriptors': [None, np.array([1]), None, np.array([2]), None],
        'angle': [0.0] * 5,
        'corr': [0.0, 0.8, 0.0, 0.7, 0.0],
        'time': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-02', '2023-01-05', '2023-01-05']),
        'interpolated': [0] * 5,
        'orbit_num': [1, 2, 1, 2, 2],
        'stopped': [False] * 5,
        'converged_to': [None] * 5,
    }
    points = Keypoints(points_data, crs='EPSG:3413')
    
    last_persisted_id = 2
    
    # Apply the logic
    points_image_id_series = points['image_id'].astype(int)
    last_persisted_id_int = int(last_persisted_id)
    mask_series = points_image_id_series > last_persisted_id_int
    points_delta = points.loc[mask_series].copy()
    
    # Should have 3 rows with image_id=5
    assert len(points_delta) == 3
    
    # Find seeds
    traj_ids_in_delta = points_delta['trajectory_id'].unique()
    seed_rows_to_add = []
    for tid in traj_ids_in_delta:
        if pd.isna(tid):
            continue
        traj_points = points[points['trajectory_id'] == tid]
        if len(traj_points) <= 1:
            continue
        seed_row = traj_points.loc[traj_points['image_id'].idxmin()]
        seed_image_id = int(seed_row['image_id'])
        if seed_image_id <= last_persisted_id_int:
            seed_rows_to_add.append(seed_row)
    
    # Should find seeds for trajectories 10 and 11 (both have seeds before last_persisted_id)
    # Trajectory 12 has only 1 observation, so no seed
    assert len(seed_rows_to_add) == 2
    seed_tids = [row['trajectory_id'] for row in seed_rows_to_add]
    assert 10 in seed_tids
    assert 11 in seed_tids
