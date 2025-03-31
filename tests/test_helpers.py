import numpy as np
from datetime import datetime, timezone, timedelta
from shapely.geometry import Polygon, Point
import geopandas as gpd

class mockimage:
    """A lightweight Image class for testing that mimics SAR data characteristics"""
    def __init__(self, size=(100, 100), angle=45.0, poly=None, add_features=True):
        """
        Initialize test image with SAR-like values.
        
        Parameters:
        -----------
        size : tuple
            Size of the image (height, width)
        angle : float
            Rotation angle of the image
        poly : Polygon
            Custom polygon for the image bounds
        add_features : bool
            If True, adds some bright features for tracking
        """
        self._angle = angle
        self.date = datetime.now(timezone.utc)
        from nansat import NSR
        self.srs = NSR(3413)
        
        # Create base arrays with exact means
        hh_base = np.random.normal(110, 20, size=size)
        hv_base = np.random.normal(90, 20, size=size)
        
        # Ensure non-negative values and convert to uint8
        hh_base = np.clip(hh_base, 0, 255).astype(np.uint8)
        hv_base = np.clip(hv_base, 0, 255).astype(np.uint8)
        
        if add_features:
            # Add some brighter features for tracking
            for _ in range(3):
                # Random position for feature
                y = np.random.randint(10, size[0]-10)
                x = np.random.randint(10, size[1]-10)
                
                # Create bright feature (5x5 pixels)
                # Use higher values for features
                feature_value_hh = np.random.randint(180, 256)
                feature_value_hv = np.random.randint(160, 256)
                
                # Add feature with gaussian falloff
                y_idx, x_idx = np.ogrid[-2:3, -2:3]
                feature_mask = np.exp(-(x_idx**2 + y_idx**2) / 2)
                
                hh_base[y-2:y+3, x-2:x+3] = np.clip(
                    feature_value_hh * feature_mask, 
                    hh_base[y-2:y+3, x-2:x+3], 
                    255
                ).astype(np.uint8)
                
                hv_base[y-2:y+3, x-2:x+3] = np.clip(
                    feature_value_hv * feature_mask, 
                    hv_base[y-2:y+3, x-2:x+3], 
                    255
                ).astype(np.uint8)
        
        self.data = {
            's0_HH': hh_base,
            's0_HV': hv_base
        }
        
        # Create default polygon if none provided
        if poly is None:
            self._poly = Polygon([
                (-200000, -200000),
                (-200000, 200000),
                (200000, 200000),
                (200000, -200000)
            ])
        else:
            self._poly = poly
            
    def __getitem__(self, key):
        return self.data[key].copy()
        
    @property
    def angle(self):
        return self._angle
        
    @property
    def poly(self):
        return self._poly
        
    def transform_points(self, x, y, DstToSrc=0, dst_srs=None):
        """
        Simplified transform that maintains realistic coordinate scales.
        """
        if DstToSrc == 1:
            # Transform from meters to pixels
            x_scaled = (x + 200000) * self.data['s0_HH'].shape[1] / 400000
            y_scaled = (y + 200000) * self.data['s0_HH'].shape[0] / 400000
            return x_scaled, y_scaled
        else:
            # Transform from pixels to meters
            x_meters = x * 400000 / self.data['s0_HH'].shape[1] - 200000
            y_meters = y * 400000 / self.data['s0_HH'].shape[0] - 200000
            return x_meters, y_meters

def create_test_image_pair(time_difference=timedelta(hours=1), drift_pixels=5):
    """
    Creates a pair of test images with some matching features.
    """
    # Create first image
    img1 = mockimage(size=(100, 100), angle=45.0)
    
    # Create second image by shifting features from first
    img2_hh = np.roll(img1.data['s0_HH'], drift_pixels, axis=(0, 1))
    img2_hv = np.roll(img1.data['s0_HV'], drift_pixels, axis=(0, 1))
    
    # Add some noise while maintaining means
    noise_hh = np.random.normal(0, 5, img2_hh.shape)
    noise_hv = np.random.normal(0, 5, img2_hv.shape)
    
    img2_hh = np.clip(img2_hh + noise_hh, 0, 255).astype(np.uint8)
    img2_hv = np.clip(img2_hv + noise_hv, 0, 255).astype(np.uint8)
    
    img2 = mockimage(size=(100, 100), angle=45.0)
    img2.data = {'s0_HH': img2_hh, 's0_HV': img2_hv}
    img2.date = img1.date + time_difference  # Use date instead of time_coverage_start
    
    return img1, img2

def create_test_points(num_points=5):
    """
    Creates test keypoints with realistic coordinates and attributes.
    """
    data = {
        'geometry': [Point(np.random.uniform(-150000, 150000), 
                         np.random.uniform(-150000, 150000)) 
                    for _ in range(num_points)],
        'descriptors': [np.random.randint(0, 255, 32, dtype=np.uint8) 
                       for _ in range(num_points)],
        'image_id': [1] * num_points,
        'is_last': [1] * num_points,
        'trajectory_id': range(num_points),
        'angle': [45.0] * num_points,
        'corr': np.random.uniform(0.7, 0.9, num_points),
        'time': [datetime.now(timezone.utc)] * num_points
    }
    
    return gpd.GeoDataFrame(data, crs='EPSG:3413')

# Quick verification of means
if __name__ == "__main__":
    img = mockimage()
    print(f"HH mean: {np.mean(img.data['s0_HH']):.1f}")  # Should be close to 110
    print(f"HV mean: {np.mean(img.data['s0_HV']):.1f}")  # Should be close to 90
    
    # Create and verify an image pair
    img1, img2 = create_test_image_pair()
    print("\nImage pair test:")
    print(f"Time difference: {img2.time_coverage_start - img1.time_coverage_start}")
    print(f"HH means: {np.mean(img1.data['s0_HH']):.1f} -> {np.mean(img2.data['s0_HH']):.1f}")
    print(f"HV means: {np.mean(img1.data['s0_HV']):.1f} -> {np.mean(img2.data['s0_HV']):.1f}")