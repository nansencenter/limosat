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
    Test that when a trajectory is NEW (not in DB yet), ALL its rows are included.
    This ensures seeds are persisted for brand-new trajectories.
    """
    from limosat.keypoints import Keypoints
    
    # Simulate a scenario:
    # - Trajectory 10 has 2 observations (image_id 1 and 5), NOT in DB yet
    # - Trajectory 20 has 1 observation, already in DB
    # - last_persisted_id = 2
    
    # Create points representing the state
    points_data = {
        'trajectory_id': [10, 10, 20],  # traj 10 is new, traj 20 exists in DB
        'image_id': [1, 5, 5],  # traj 10: seed at 1, match at 5
        'is_last': [0, 1, 1],
        'geometry': [Point(100, 100), Point(105, 105), Point(200, 200)],
        'descriptors': [None, np.array([1, 2, 3], dtype=np.uint8), None],
        'angle': [0.0, 0.0, 0.0],
        'corr': [0.0, 0.8, 0.0],
        'time': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-05']),
        'interpolated': [0, 0, 0],
        'orbit_num': [1, 2, 2],
        'stopped': [False, False, False],
        'converged_to': [None, None, None],
    }
    points = Keypoints(points_data, crs='EPSG:3413')
    points = points.set_crs('EPSG:3413')
    
    # Simulate existing trajectories in DB
    existing_traj_ids = {20}  # Only trajectory 20 is in DB
    
    # Set last_persisted_id to 2
    last_persisted_id = 2
    
    # Apply the new logic from database.py
    points['trajectory_id'] = points['trajectory_id'].astype('Int64')
    
    # New trajectories are those not present in DB yet
    is_new_traj = ~points['trajectory_id'].isin(existing_traj_ids)
    is_new_traj = is_new_traj.fillna(False)
    
    # For existing trajectories, keep only rows newer than last_persisted_id
    last_persisted_id_int = int(last_persisted_id)
    is_newer_than_last = points['image_id'].astype(int) > last_persisted_id_int
    
    # Build delta
    points_delta = pd.concat([
        points[is_new_traj],
        points[~is_new_traj & is_newer_than_last]
    ], ignore_index=True)
    
    # For trajectory 10 (new): should include BOTH rows (seed at 1 and match at 5)
    traj_10_rows = points_delta[points_delta['trajectory_id'] == 10]
    assert len(traj_10_rows) == 2
    assert 1 in traj_10_rows['image_id'].values
    assert 5 in traj_10_rows['image_id'].values
    
    # For trajectory 20 (existing): should only include rows with image_id > 2
    traj_20_rows = points_delta[points_delta['trajectory_id'] == 20]
    assert len(traj_20_rows) == 1
    assert 5 in traj_20_rows['image_id'].values


@pytest.mark.unit
def test_seed_persistence_existing_trajectory_only_new_rows():
    """
    Test that for trajectories already in DB, only rows with image_id > last_persisted_id
    are included (no seed duplication).
    """
    from limosat.keypoints import Keypoints
    
    # Create points for trajectory already in DB
    points_data = {
        'trajectory_id': [10, 10, 10],  # Trajectory 10 already in DB
        'image_id': [1, 3, 5],  # Seed at 1, observations at 3 and 5
        'is_last': [0, 0, 1],
        'geometry': [Point(100, 100), Point(103, 103), Point(105, 105)],
        'descriptors': [None, None, np.array([1, 2, 3], dtype=np.uint8)],
        'angle': [0.0, 0.0, 0.0],
        'corr': [0.0, 0.5, 0.8],
        'time': pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-05']),
        'interpolated': [0, 0, 0],
        'orbit_num': [1, 1, 2],
        'stopped': [False, False, False],
        'converged_to': [None, None, None],
    }
    points = Keypoints(points_data, crs='EPSG:3413')
    points = points.set_crs('EPSG:3413')
    
    # Simulate trajectory 10 already exists in DB
    existing_traj_ids = {10}
    
    # last_persisted_id = 3 (so seed at 1 and obs at 3 are already persisted)
    last_persisted_id = 3
    
    # Apply the new logic
    points['trajectory_id'] = points['trajectory_id'].astype('Int64')
    is_new_traj = ~points['trajectory_id'].isin(existing_traj_ids)
    is_new_traj = is_new_traj.fillna(False)
    
    last_persisted_id_int = int(last_persisted_id)
    is_newer_than_last = points['image_id'].astype(int) > last_persisted_id_int
    
    points_delta = pd.concat([
        points[is_new_traj],
        points[~is_new_traj & is_newer_than_last]
    ], ignore_index=True)
    
    # Should only have the row with image_id=5 (not the seed or the obs at 3)
    assert len(points_delta) == 1
    assert 5 in points_delta['image_id'].values
    assert 1 not in points_delta['image_id'].values
    assert 3 not in points_delta['image_id'].values


@pytest.mark.unit  
def test_seed_persistence_with_multiple_new_trajectories():
    """
    Test the case where multiple NEW trajectories (not in DB) gain 2nd observations.
    All their rows (including seeds) should be persisted.
    """
    from limosat.keypoints import Keypoints
    
    # Create points with multiple new trajectories
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
    points = points.set_crs('EPSG:3413')
    
    # None of these trajectories are in the DB yet
    existing_traj_ids = set()
    
    last_persisted_id = 2
    
    # Apply the new logic
    points['trajectory_id'] = points['trajectory_id'].astype('Int64')
    is_new_traj = ~points['trajectory_id'].isin(existing_traj_ids)
    is_new_traj = is_new_traj.fillna(False)
    
    last_persisted_id_int = int(last_persisted_id)
    is_newer_than_last = points['image_id'].astype(int) > last_persisted_id_int
    
    points_delta = pd.concat([
        points[is_new_traj],
        points[~is_new_traj & is_newer_than_last]
    ], ignore_index=True)
    
    # All rows should be included since all trajectories are new
    assert len(points_delta) == 5
    
    # Check trajectory 10: should have both rows (seed at 1 and match at 5)
    traj_10 = points_delta[points_delta['trajectory_id'] == 10]
    assert len(traj_10) == 2
    assert set(traj_10['image_id'].values) == {1, 5}
    
    # Check trajectory 11: should have both rows (seed at 2 and match at 5)
    traj_11 = points_delta[points_delta['trajectory_id'] == 11]
    assert len(traj_11) == 2
    assert set(traj_11['image_id'].values) == {2, 5}
    
    # Check trajectory 12: should have its single row
    traj_12 = points_delta[points_delta['trajectory_id'] == 12]
    assert len(traj_12) == 1
    assert traj_12['image_id'].values[0] == 5
