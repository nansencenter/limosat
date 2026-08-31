"""Test land mask filtering logic in detect_gridded_points method."""
import numpy as np


def test_landmask_filtering_logic():
    """
    Test the core logic of landmask filtering for grid points.
    This test validates the algorithm without needing the full KeypointDetector class.
    """
    # Create a simple landmask
    landmask = np.zeros((50, 50), dtype=np.uint8)
    # Mark region 10:20, 10:20 as excluded (noncanonical value 253).
    landmask[10:20, 10:20] = 253
    
    stride = 10
    
    # Simulate grid point generation WITH landmask filtering
    keypoints_with_filter = []
    for r in range(0, landmask.shape[0], stride):
        for c in range(0, landmask.shape[1], stride):
            if landmask is not None and landmask[r, c] >= 2:
                continue  # skip land cells
            keypoints_with_filter.append((c, r))
    
    # Simulate grid point generation WITHOUT landmask filtering
    keypoints_without_filter = []
    for r in range(0, landmask.shape[0], stride):
        for c in range(0, landmask.shape[1], stride):
            keypoints_without_filter.append((c, r))
    
    # Verify that we have fewer keypoints with filtering
    assert len(keypoints_with_filter) < len(keypoints_without_filter), \
        f"Expected fewer keypoints with filter, got {len(keypoints_with_filter)} vs {len(keypoints_without_filter)}"
    
    # Verify specific keypoint at land location (10, 10) is filtered out
    assert (10, 10) in keypoints_without_filter, "Keypoint (10,10) should exist without filter"
    assert (10, 10) not in keypoints_with_filter, "Keypoint (10,10) should be filtered out with land mask"
    
    # Verify that we filtered out exactly the right number of points
    # Grid points at: 0, 10, 20, 30, 40 (5 per dimension = 25 total)
    # Land cells at (10, 10) - only 1 cell is marked as land at a grid point
    expected_filtered = 1
    actual_filtered = len(keypoints_without_filter) - len(keypoints_with_filter)
    assert actual_filtered == expected_filtered, \
        f"Expected {expected_filtered} filtered keypoints, got {actual_filtered}"
    
    print(f"✓ Test passed: {actual_filtered} land keypoints filtered out of {len(keypoints_without_filter)} total")


def test_landmask_none_handling():
    """Test that the code handles landmask=None correctly."""
    landmask = None
    stride = 10
    img_shape = (50, 50)
    
    # Simulate grid point generation with landmask=None
    keypoints = []
    for r in range(0, img_shape[0], stride):
        for c in range(0, img_shape[1], stride):
            if landmask is not None and landmask[r, c] >= 2:
                continue  # skip land cells
            keypoints.append((c, r))
    
    # Should generate all grid points when landmask is None
    expected_count = len(range(0, img_shape[0], stride)) * len(range(0, img_shape[1], stride))
    assert len(keypoints) == expected_count, \
        f"Expected {expected_count} keypoints with no landmask, got {len(keypoints)}"
    
    print(f"✓ Test passed: {len(keypoints)} keypoints generated with landmask=None")


def test_large_land_region_filtering():
    """Test filtering with a larger land region."""
    # Create landmask with larger land region
    landmask = np.zeros((100, 100), dtype=np.uint8)
    # Mark region 20:60, 20:60 as land
    landmask[20:60, 20:60] = 2
    
    stride = 10
    
    # Simulate grid point generation WITH filtering
    keypoints_with_filter = []
    for r in range(0, landmask.shape[0], stride):
        for c in range(0, landmask.shape[1], stride):
            if landmask is not None and landmask[r, c] >= 2:
                continue  # skip land cells
            keypoints_with_filter.append((c, r))
    
    # Simulate grid point generation WITHOUT filtering
    keypoints_without_filter = []
    for r in range(0, landmask.shape[0], stride):
        for c in range(0, landmask.shape[1], stride):
            keypoints_without_filter.append((c, r))
    
    # Count land grid points
    # Grid points: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 (10 per dimension = 100 total)
    # Land region: 20-60 (inclusive on start, exclusive on end)
    # Land grid points: 20, 30, 40, 50 (4 per dimension = 16 total)
    expected_land_points = 16
    actual_filtered = len(keypoints_without_filter) - len(keypoints_with_filter)
    
    assert actual_filtered == expected_land_points, \
        f"Expected {expected_land_points} filtered keypoints, got {actual_filtered}"
    
    # Verify some specific land points are filtered
    land_grid_points = [(20, 20), (30, 30), (40, 40), (50, 50)]
    for point in land_grid_points:
        assert point not in keypoints_with_filter, f"Land point {point} should be filtered out"
        assert point in keypoints_without_filter, f"Land point {point} should exist without filter"
    
    print(f"✓ Test passed: {actual_filtered} land keypoints filtered out of {len(keypoints_without_filter)} total")


if __name__ == '__main__':
    test_landmask_filtering_logic()
    test_landmask_none_handling()
    test_large_land_region_filtering()
    print("\n✓ All tests passed!")
