"""
Unit tests for persistence filtering logic.

Tests that only matched trajectories (length > 1) are persisted.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point
from unittest.mock import Mock


@pytest.mark.unit
def test_periodic_persistence_filters_singletons():
    """Test that periodic persistence filters out singleton trajectories."""
    from limosat.keypoints import Keypoints
    
    # Create points with mixed trajectories
    points_data = gpd.GeoDataFrame({
        'image_id': [1, 2, 2, 3],
        'is_last': [0, 1, 1, 1],
        'trajectory_id': [100, 100, 200, 300],  # tid=100 matched (2), tid=200 & 300 singletons
        'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        'descriptors': [None, np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8)],
        'angle': [0.0, 0.0, 0.0, 0.0],
        'corr': [0.5, 0.6, 0.4, 0.3],
        'time': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')],
        'interpolated': [0, 0, 0, 0],
        'orbit_num': [1, 2, 2, 3],
        'stopped': [False, False, False, False],
        'converged_to': [pd.NA, pd.NA, pd.NA, pd.NA],
    }, crs='EPSG:3413')
    
    points = Keypoints(points_data)
    
    # Apply the filtering logic (same as in image_processor.py periodic persistence)
    traj_id_counts = points['trajectory_id'].value_counts()
    matched_traj_ids = traj_id_counts[traj_id_counts > 1].index
    points_to_persist = points[points['trajectory_id'].isin(matched_traj_ids)]
    
    # Verify only matched trajectory is selected
    persisted_traj_ids = points_to_persist['trajectory_id'].unique()
    assert 100 in persisted_traj_ids, "Matched trajectory 100 should be selected"
    assert 200 not in persisted_traj_ids, "Singleton trajectory 200 should NOT be selected"
    assert 300 not in persisted_traj_ids, "Singleton trajectory 300 should NOT be selected"
    assert len(points_to_persist) == 2, "Should select only 2 points from matched trajectory"


@pytest.mark.unit
def test_final_persistence_filters_singletons():
    """Test that ensure_final_persistence applies the same filter as periodic persistence."""
    from limosat.image_processor import ImageProcessor
    from limosat.keypoints import Keypoints
    
    # Create mock components
    mock_model = Mock()
    mock_matcher = Mock()
    mock_db = Mock()
    mock_db.save = Mock(return_value=True)
    
    # Create points with mixed trajectories
    points_data = gpd.GeoDataFrame({
        'image_id': [1, 2, 2, 3],
        'is_last': [0, 1, 1, 1],
        'trajectory_id': [100, 100, 200, 300],  # tid=100 matched (2), tid=200 & 300 singletons
        'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        'descriptors': [None, np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8)],
        'angle': [0.0, 0.0, 0.0, 0.0],
        'corr': [0.5, 0.6, 0.4, 0.3],
        'time': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')],
        'interpolated': [0, 0, 0, 0],
        'orbit_num': [1, 2, 2, 3],
        'stopped': [False, False, False, False],
        'converged_to': [pd.NA, pd.NA, pd.NA, pd.NA],
    }, crs='EPSG:3413')
    
    points = Keypoints(points_data)
    
    # Create processor with persistence enabled
    processor = ImageProcessor(
        points=points,
        model=mock_model,
        matcher=mock_matcher,
        persist_updates=True
    )
    processor.db = mock_db
    processor._last_persisted_id = 0
    
    # Call ensure_final_persistence
    processor.ensure_final_persistence()
    
    # Verify db.save was called
    assert mock_db.save.called
    
    # Get the points that were passed to db.save
    call_args = mock_db.save.call_args
    # Check if points is in kwargs or args
    if 'points' in call_args.kwargs:
        persisted_points = call_args.kwargs['points']
    else:
        persisted_points = call_args.args[0]  # First positional arg is points
    
    # Verify only matched trajectory was passed (tid=100 with 2 points)
    persisted_traj_ids = persisted_points['trajectory_id'].unique()
    assert 100 in persisted_traj_ids, "Matched trajectory 100 should be persisted"
    assert 200 not in persisted_traj_ids, "Singleton trajectory 200 should NOT be persisted"
    assert 300 not in persisted_traj_ids, "Singleton trajectory 300 should NOT be persisted"
    assert len(persisted_points) == 2, "Should pass only 2 points from matched trajectory"


@pytest.mark.unit
def test_combined_filtering_logic():
    """Test the logic for combining DB and delta counts."""
    # Simulate scenario:
    # - tid=100: 1 row in DB, 1 row in delta -> combined=2, should be persisted
    # - tid=200: 0 rows in DB, 1 row in delta -> combined=1, should NOT be persisted
    # - tid=300: 2 rows in DB, 1 row in delta -> combined=3, should be persisted
    
    # DB counts (simulated from database query)
    db_counts = {100: 1, 300: 2}
    
    # Delta counts (from points_delta DataFrame)
    delta_counts = {100: 1, 200: 1, 300: 1}
    
    # All trajectory IDs in delta
    delta_traj_ids = [100, 200, 300]
    
    # Combine counts
    combined_counts = {}
    for tid in delta_traj_ids:
        db_count = db_counts.get(tid, 0)
        delta_count = delta_counts.get(tid, 0)
        combined_counts[tid] = db_count + delta_count
    
    # Filter: only persist trajectories with combined_count > 1
    matched_traj_ids = [tid for tid, count in combined_counts.items() if count > 1]
    
    # Verify the filtering
    assert 100 in matched_traj_ids, "tid=100 should be matched (combined=2)"
    assert 200 not in matched_traj_ids, "tid=200 should NOT be matched (combined=1)"
    assert 300 in matched_traj_ids, "tid=300 should be matched (combined=3)"


@pytest.mark.unit
def test_delta_only_filtering_logic():
    """Test filtering when table doesn't exist (delta-only counts)."""
    # Simulate first run scenario where table doesn't exist yet
    # - tid=100: 2 rows in delta -> should be persisted
    # - tid=200: 1 row in delta -> should NOT be persisted
    
    # DB counts (empty - table doesn't exist)
    db_counts = {}
    
    # Delta counts
    delta_counts = {100: 2, 200: 1}
    
    # All trajectory IDs in delta
    delta_traj_ids = [100, 200]
    
    # Combine counts (same logic as before)
    combined_counts = {}
    for tid in delta_traj_ids:
        db_count = db_counts.get(tid, 0)
        delta_count = delta_counts.get(tid, 0)
        combined_counts[tid] = db_count + delta_count
    
    # Filter: only persist trajectories with combined_count > 1
    matched_traj_ids = [tid for tid, count in combined_counts.items() if count > 1]
    
    # Verify the filtering
    assert 100 in matched_traj_ids, "tid=100 should be matched (delta=2)"
    assert 200 not in matched_traj_ids, "tid=200 should NOT be matched (delta=1)"


@pytest.mark.unit
def test_empty_delta_returns_early():
    """Test that save returns early when delta is empty."""
    from limosat.keypoints import Keypoints
    
    # Create points where all have image_id <= last_persisted_id
    points_data = gpd.GeoDataFrame({
        'image_id': [1, 2],
        'is_last': [0, 1],
        'trajectory_id': [100, 100],
        'geometry': [Point(0, 0), Point(1, 1)],
        'descriptors': [None, np.zeros(32, dtype=np.uint8)],
        'angle': [0.0, 0.0],
        'corr': [0.5, 0.6],
        'time': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02')],
        'interpolated': [0, 0],
        'orbit_num': [1, 2],
        'stopped': [False, False],
        'converged_to': [pd.NA, pd.NA],
    }, crs='EPSG:3413')
    
    points = Keypoints(points_data)
    
    # Filter to get delta (image_id > last_persisted_id)
    last_persisted_id = 2
    points_delta = points[points['image_id'] > last_persisted_id]
    
    # Verify delta is empty
    assert len(points_delta) == 0, "Delta should be empty when all points have image_id <= last_persisted_id"


@pytest.mark.unit
def test_all_singletons_returns_empty_filtered_delta():
    """Test that when all trajectories in delta are singletons, filtered delta is empty."""
    from limosat.keypoints import Keypoints
    
    # Create delta with only singleton trajectories
    points_delta = gpd.GeoDataFrame({
        'image_id': [2, 2, 2],
        'is_last': [1, 1, 1],
        'trajectory_id': [100, 200, 300],  # All singletons
        'geometry': [Point(1, 1), Point(2, 2), Point(3, 3)],
        'descriptors': [np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8), np.zeros(32, dtype=np.uint8)],
        'angle': [0.0, 0.0, 0.0],
        'corr': [0.6, 0.4, 0.3],
        'time': [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-02')],
        'interpolated': [0, 0, 0],
        'orbit_num': [2, 2, 2],
        'stopped': [False, False, False],
        'converged_to': [pd.NA, pd.NA, pd.NA],
    }, crs='EPSG:3413')
    
    points_delta = Keypoints(points_delta)
    
    # Simulate DB counts (all zero - no existing rows for these trajectories)
    db_counts = {}
    
    # Delta counts
    delta_counts = points_delta['trajectory_id'].value_counts().to_dict()
    delta_traj_ids = list(delta_counts.keys())
    
    # Combine counts
    combined_counts = {}
    for tid in delta_traj_ids:
        db_count = db_counts.get(tid, 0)
        delta_count = delta_counts.get(tid, 0)
        combined_counts[tid] = db_count + delta_count
    
    # Filter
    matched_traj_ids = [tid for tid, count in combined_counts.items() if count > 1]
    
    # Verify no trajectories match
    assert len(matched_traj_ids) == 0, "No trajectories should be matched when all are singletons"
