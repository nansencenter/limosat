import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import cv2
from shapely.geometry import Point

from limosat.matcher import Matcher
from tests.factories import make_keypoints


@pytest.mark.unit
def test_dynamic_velocity_filter_removes_fast_points():
    """Test that the dynamic velocity filter removes points exceeding velocity threshold."""
    # Create matcher with dynamic velocity filter enabled
    matcher = Matcher(
        use_dynamic_velocity_filter=True,
        max_valid_speed_m_per_day=10000.0,  # 10 km/day threshold
        use_model_estimation=False  # Disable model estimation for simpler test
    )
    
    # Create test points at t0
    n_points = 5
    t0 = pd.Timestamp('2025-01-01 00:00:00')
    points_poly = make_keypoints(n_points, image_id=1, t0=t0)
    
    # Create test points at t1 (1 day later)
    t1 = t0 + pd.Timedelta(days=1)
    points_grid = make_keypoints(n_points, image_id=2, t0=t1)
    
    # Modify positions to create different velocities:
    # Point 0: no movement (0 m/day) - should pass
    points_grid.loc[0, 'geometry'] = Point(10, 10)
    
    # Point 1: slow movement (5 km/day) - should pass
    points_grid.loc[1, 'geometry'] = Point(11 + 5, 10)  # moved 5 km
    
    # Point 2: moderate movement (10 km/day) - should pass (at threshold)
    points_grid.loc[2, 'geometry'] = Point(12 + 10, 10)  # moved 10 km
    
    # Point 3: fast movement (20 km/day) - should be filtered out
    points_grid.loc[3, 'geometry'] = Point(13 + 20, 10)  # moved 20 km
    
    # Point 4: very fast movement (100 km/day) - should be filtered out
    points_grid.loc[4, 'geometry'] = Point(14 + 100, 10)  # moved 100 km
    
    # Extract positions
    pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
    pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
    
    # Create dummy matches (all points matched)
    matches = [cv2.DMatch(i, i, 0) for i in range(n_points)]
    
    # Apply filter
    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, points_poly, points_grid)
    
    # Should keep points 0, 1, 2 (3 points total)
    assert len(idx0) == 3, f"Expected 3 points to pass filter, got {len(idx0)}"
    assert len(idx1) == 3
    
    # Check that the right points passed
    assert 0 in idx0  # no movement
    assert 1 in idx0  # slow movement
    assert 2 in idx0  # at threshold
    assert 3 not in idx0  # too fast
    assert 4 not in idx0  # way too fast


@pytest.mark.unit
def test_dynamic_velocity_filter_handles_zero_time_diff():
    """Test that the filter handles zero time difference gracefully."""
    matcher = Matcher(
        use_dynamic_velocity_filter=True,
        max_valid_speed_m_per_day=10000.0,
        use_model_estimation=False
    )
    
    # Create test points at the same time
    t0 = pd.Timestamp('2025-01-01 00:00:00')
    points_poly = make_keypoints(3, image_id=1, t0=t0)
    points_grid = make_keypoints(3, image_id=2, t0=t0)  # Same time!
    
    # Modify positions (movement with zero time should result in infinite speed)
    points_grid.loc[0, 'geometry'] = Point(10, 10)  # no movement
    points_grid.loc[1, 'geometry'] = Point(15, 10)  # moved 5 m
    points_grid.loc[2, 'geometry'] = Point(100, 10)  # moved 90 m
    
    pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
    pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
    
    matches = [cv2.DMatch(i, i, 0) for i in range(3)]
    
    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, points_poly, points_grid)
    
    # With zero time diff, any movement results in infinite speed and should be filtered
    # Only the point with no movement should pass
    assert len(idx0) == 1
    assert 0 in idx0


@pytest.mark.unit
def test_dynamic_velocity_filter_disabled():
    """Test that filtering works correctly when dynamic velocity filter is disabled."""
    matcher = Matcher(
        use_dynamic_velocity_filter=False,  # Disabled
        max_valid_speed_m_per_day=10000.0,
        spatial_distance_max=50.0,  # Use spatial filter instead
        use_model_estimation=False
    )
    
    # Create test points
    t0 = pd.Timestamp('2025-01-01 00:00:00')
    t1 = t0 + pd.Timedelta(days=1)
    points_poly = make_keypoints(3, image_id=1, t0=t0)
    points_grid = make_keypoints(3, image_id=2, t0=t1)
    
    # Create movements: one within spatial limit, two beyond
    points_grid.loc[0, 'geometry'] = Point(10 + 30, 10)  # 30m - within spatial limit
    points_grid.loc[1, 'geometry'] = Point(11 + 60, 10)  # 60m - exceeds spatial limit
    points_grid.loc[2, 'geometry'] = Point(12 + 100, 10)  # 100m - exceeds spatial limit
    
    pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
    pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
    
    matches = [cv2.DMatch(i, i, 0) for i in range(3)]
    
    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, points_poly, points_grid)
    
    # Only point 0 should pass the spatial filter
    assert len(idx0) == 1
    assert 0 in idx0


@pytest.mark.unit
def test_dynamic_velocity_filter_with_multiple_time_gaps():
    """Test dynamic velocity filter with different time gaps."""
    matcher = Matcher(
        use_dynamic_velocity_filter=True,
        max_valid_speed_m_per_day=10000.0,  # 10 km/day
        use_model_estimation=False
    )
    
    # Create points with different time gaps
    base_time = pd.Timestamp('2025-01-01 00:00:00')
    
    # Point 0: 1 day gap, 5 km movement = 5 km/day - should pass
    t0_poly = base_time
    t0_grid = base_time + pd.Timedelta(days=1)
    
    # Point 1: 2 day gap, 15 km movement = 7.5 km/day - should pass
    t1_poly = base_time
    t1_grid = base_time + pd.Timedelta(days=2)
    
    # Point 2: 0.5 day gap, 8 km movement = 16 km/day - should be filtered
    t2_poly = base_time
    t2_grid = base_time + pd.Timedelta(hours=12)
    
    # Create GeoDataFrames with different times
    points_poly = gpd.GeoDataFrame(
        {
            'image_id': [1, 1, 1],
            'trajectory_id': [0, 1, 2],
            'time': [t0_poly, t1_poly, t2_poly],
            'descriptors': [np.zeros((32,), dtype=np.uint8) for _ in range(3)],
        },
        geometry=[Point(10, 10), Point(11, 10), Point(12, 10)],
        crs='EPSG:3413',
    )
    
    points_grid = gpd.GeoDataFrame(
        {
            'image_id': [2, 2, 2],
            'trajectory_id': [0, 1, 2],
            'time': [t0_grid, t1_grid, t2_grid],
            'descriptors': [np.zeros((32,), dtype=np.uint8) for _ in range(3)],
        },
        geometry=[Point(15, 10), Point(26, 10), Point(20, 10)],  # moved 5, 15, 8 km respectively
        crs='EPSG:3413',
    )
    
    pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
    pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
    
    matches = [cv2.DMatch(i, i, 0) for i in range(3)]
    
    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, points_poly, points_grid)
    
    # Points 0 and 1 should pass, point 2 should be filtered
    assert len(idx0) == 2
    assert 0 in idx0
    assert 1 in idx0
    assert 2 not in idx0


@pytest.mark.unit
def test_dynamic_velocity_filter_missing_time_column():
    """Test that filter falls back gracefully when time column is missing."""
    matcher = Matcher(
        use_dynamic_velocity_filter=True,
        max_valid_speed_m_per_day=10000.0,
        spatial_distance_max=50.0,
        use_model_estimation=False
    )
    
    # Create points without time column
    points_poly = gpd.GeoDataFrame(
        {
            'image_id': [1, 1],
            'trajectory_id': [0, 1],
            'descriptors': [np.zeros((32,), dtype=np.uint8) for _ in range(2)],
        },
        geometry=[Point(10, 10), Point(11, 10)],
        crs='EPSG:3413',
    )
    
    points_grid = gpd.GeoDataFrame(
        {
            'image_id': [2, 2],
            'trajectory_id': [0, 1],
            'descriptors': [np.zeros((32,), dtype=np.uint8) for _ in range(2)],
        },
        geometry=[Point(15, 10), Point(100, 10)],  # 5m and 89m movements
        crs='EPSG:3413',
    )
    
    pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
    pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
    
    matches = [cv2.DMatch(i, i, 0) for i in range(2)]
    
    # Should fall back to spatial filter when time column is missing
    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, points_poly, points_grid)
    
    # Only the first point should pass spatial filter (5m < 50m)
    assert len(idx0) == 1
    assert 0 in idx0
