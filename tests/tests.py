import pytest
import sys
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import MagicMock, patch
import warnings
import xarray as xr
from limosat import Keypoints, ImageProcessor, extract_templates, compute_descriptors
from test_helpers import mockimage, create_test_image_pair
import cv2
from nansat import NSR

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

nsr3413 = NSR(3413)

# Existing Keypoints tests
def test_keypoints_initialization_empty():
    # Test initializing without arguments
    kp = Keypoints()
    assert len(kp) == 0
    assert isinstance(kp, Keypoints)
    assert kp.srs.wkt == nsr3413.wkt
    assert list(kp.columns) == [
        'image_id', 'is_last', 'trajectory_id',
        'geometry', 'descriptors', 'angle', 'corr', 'time', 'interpolated'
    ]

def test_keypoints_initialization_with_data():
    # Test initializing with data
    data = {
        'image_id': [1, 2],
        'is_last': [1, 0],
        'trajectory_id': [0, 1],
        'geometry': [Point(0, 0), Point(1, 1)],
        'descriptors': [np.array([1, 2]), np.array([3, 4])],
        'angle': [45, 90],
        'corr': [0.5, 0.7],
        'time': [datetime.now(timezone.utc), datetime.now(timezone.utc)],
        'interpolated': [0, 1]
    }
    gdf = gpd.GeoDataFrame(data, crs='EPSG:3413')
    kp = Keypoints(gdf)
    assert len(kp) == 2
    assert isinstance(kp, Keypoints)
    assert kp.srs.wkt == nsr3413.wkt
    pd.testing.assert_frame_equal(kp, gdf)

def test_keypoints_from_gdf():
    # Test _from_gdf method
    data = {
        'image_id': [1],
        'is_last': [1],
        'trajectory_id': [0],
        'geometry': [Point(0, 0)],
        'descriptors': [np.array([1, 2])],
        'angle': [45],
        'corr': [0.5],
        'time': [datetime.now(timezone.utc)],
        'interpolated': [0]
    }
    gdf = gpd.GeoDataFrame(data, crs='EPSG:3413')
    kp = Keypoints._from_gdf(gdf)
    assert isinstance(kp, Keypoints)
    pd.testing.assert_frame_equal(kp, gdf)

def test_keypoints_in_poly():
    # Test in_poly method
    data = {
        'trajectory_id': [0, 1, 2],
        'geometry': [Point(0, 0), Point(10, 10), Point(20, 20)]
    }
    kp = Keypoints(gpd.GeoDataFrame(data, crs='EPSG:3413'))
    poly_geom = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    class MockImage:
        def __init__(self, poly):
            self.poly = poly
    img = MockImage(poly_geom)
    points_in_poly = kp.in_poly(img)
    assert len(points_in_poly) == 1
    assert points_in_poly.iloc[0]['trajectory_id'] == 1

def test_keypoints_last():
    # Test last method
    data = {
        'trajectory_id': [0, 1, 2],
        'is_last': [1, 0, 1],
        'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
    }
    kp = Keypoints(gpd.GeoDataFrame(data, crs='EPSG:3413'))
    last_points = kp.last()
    assert len(last_points) == 2
    assert set(last_points['trajectory_id']) == {0, 2}

def test_keypoints_append():
    # Test append method
    data_existing = {
        'trajectory_id': [0],
        'geometry': [Point(0, 0)]
    }
    kp_existing = Keypoints(gpd.GeoDataFrame(data_existing, crs='EPSG:3413'))
    data_new = {
        'geometry': [Point(1, 1), Point(2, 2)]
    }
    kp_new = Keypoints(gpd.GeoDataFrame(data_new, crs='EPSG:3413'))
    kp_appended = kp_existing.append(kp_new)
    assert len(kp_appended) == 3
    assert set(kp_appended['trajectory_id']) == {0, 1, 2}
    assert kp_appended.iloc[1]['trajectory_id'] == 1
    assert kp_appended.iloc[2]['trajectory_id'] == 2

@pytest.fixture
def templates():
    """Create empty templates xarray."""
    return xr.DataArray(
        dims=("trajectory_id", "height", "width"),
        coords={
            "trajectory_id": range(0),
            "height": np.arange(33),
            "width": np.arange(33)
        }
    )

@pytest.fixture
def image_processor(templates):
    """Create ImageProcessor with test configuration."""
    # Create a mock model that returns proper keypoints and descriptors
    mock_model = MagicMock()
    def compute_side_effect(img, keypoints):
        # Return proper format for OpenCV keypoints and descriptors
        descriptors = np.random.randint(0, 255, (len(keypoints), 32), dtype=np.uint8)
        return keypoints, descriptors
    mock_model.compute = MagicMock(side_effect=compute_side_effect)

    # Create a mock matcher
    mock_matcher = MagicMock()
    mock_matcher.match_with_grid.return_value = (None, None)

    return ImageProcessor(
        points=Keypoints(),
        templates=templates,
        model=mock_model,
        matcher=mock_matcher,
        window_size=32,
        border_size=32
    )

def test_process_new_points(image_processor):
    """Test processing of new points."""
    test_image = mockimage(size=(200, 200))
    points_final = Keypoints()
    
    with patch.object(image_processor.keypoint_detector, 'detect_new_keypoints') as mock_detect:
        # Create proper OpenCV keypoint
        keypoint = cv2.KeyPoint()
        keypoint.pt = (100, 100)
        keypoint.size = 32
        keypoint.angle = 45.0
        keypoint.response = 0.8
        keypoint.octave = 1
        mock_detect.return_value = [keypoint]
        
        with patch('limosat.limosat.extract_templates') as mock_extract:
            # Mock template extraction
            mock_templates = np.zeros((1, 33, 33), dtype=np.uint8)
            mock_extract.return_value = (mock_templates, np.array([0]))
            
            # We also need to patch compute_descriptors
            with patch('limosat.limosat.compute_descriptors') as mock_compute:
                # Mock the descriptors computation
                mock_compute.return_value = (np.array([[100.0, 100.0]]), np.array([[1, 2, 3, 4]]))
                
                # Process new points
                image_processor._process_new_points(points_final, test_image, 1, "test.tiff")
    
    # Verify points were added
    assert len(image_processor.points) > 0
    # Verify point properties
    point = image_processor.points.iloc[0]
    assert point.image_id == 1
    assert point.is_last == 1
    assert isinstance(point.descriptors, np.ndarray)

def test_match_points_between_images(image_processor):
    """Test matching points between two related images."""
    img1, img2 = create_test_image_pair(drift_pixels=5)
    points_final = Keypoints()
    
    # First create some points in the first image
    with patch.object(image_processor.keypoint_detector, 'detect_new_keypoints') as mock_detect:
        keypoint = cv2.KeyPoint()
        keypoint.pt = (100, 100)
        keypoint.size = 32
        keypoint.angle = 45.0
        keypoint.response = 0.8
        keypoint.octave = 1
        mock_detect.return_value = [keypoint]
        
        with patch('limosat.limosat.extract_templates') as mock_extract:
            mock_templates = np.zeros((1, 33, 33), dtype=np.uint8)
            mock_extract.return_value = (mock_templates, np.array([0]))
            
            # We also need to patch compute_descriptors
            with patch('limosat.limosat.compute_descriptors') as mock_compute:
                # Mock the descriptors computation
                mock_compute.return_value = (np.array([[100.0, 100.0]]), np.array([[1, 2, 3, 4]]))
                
                image_processor._process_new_points(points_final, img1, 1, "test1.tiff")
    
    # Set up matcher to return some matches
    points_img1 = image_processor.points.copy()
    image_processor.matcher.match_with_grid.return_value = (points_img1, points_img1)
    
    # For matching points, we need to patch additional functions
    with patch('limosat.limosat.pattern_matching') as mock_pattern:
        # Mock pattern matching with reasonable return values
        mock_pattern.return_value = (np.array([[105.0, 105.0]]), np.array([0.9]))
        
        # Try to match points in second image
        result = image_processor._match_existing_points(points_img1, img2, image_id=2)
    
    assert result is not None
    assert len(result) > 0

if __name__ == '__main__':
    pytest.main(['-v'])