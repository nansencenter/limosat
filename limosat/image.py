# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import os
import re
from functools import cached_property
import numpy as np
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from nansat import Nansat, NSR
from .utils import extract_date, logger

class Image(Nansat):
    """
    Extension of Nansat class with extra functionality and transform caching.

    Attributes:
    -----------
    file_path : str
        Path to the image file.
    srs : object
        Spatial reference system for the image.
    date : datetime
        Date extracted from the file path.
    _transform_cache : dict
        Internal cache for transform_points results.
    _max_cache_entries : int
        Maximum number of entries allowed in the cache.

    Methods:
    --------
    __init__(file_path, srs=NSR(3413)):
        Initializes the Image object with the given file path and SRS.
    angle:
        Returns the angle of rotation of the image in degrees.
    poly:
        Performs coordinate transformations and sets up a polygon for the image.
    transform_points(x, y, DstToSrc=0, dst_srs=None):
        Transforms points with caching for large coordinate arrays.
    """

    def __init__(self, file_path, srs=NSR(3413)):
        """
        Initialize the Image object with the given file path and spatial reference system (SRS).

        Parameters:
        file_path (str): Path to the image file.
        srs (object): Spatial reference system for the image. Default is NSR(3413).
        """
        super().__init__(file_path)
        self.vrt.tps = True # Assuming this is desired behaviour
        self.srs = srs
        self.date = extract_date(file_path)
        # Initialize cache attributes
        self._transform_cache = {}
        self._max_cache_entries = 50  # Limit cache size to prevent excessive memory use

    @cached_property
    def orbit_num(self):
        """Extracts the orbit number from the filename."""
        match = re.search(r'_\d{8}T\d{6}_(\d{6})_', os.path.basename(self.filename))
        if match:
            return int(match.group(1))
        logger.warning(f"Could not extract orbit number from {os.path.basename(self.filename)}")
        return None

    @cached_property
    def angle(self):
        ''' Returns angle <alpha> of rotation of n'''
        nps_proj = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
        pc_proj = ccrs.PlateCarree()
        # get corners
        corners_n_lons, corners_n_lats = self.get_corners()
        # transform corners into lat, lon of other image
        cx, cy = nps_proj.transform_points(pc_proj, corners_n_lons, corners_n_lats).T[:2]
        # Calculate angle using arctan2 for robustness
        alpha = np.degrees(np.arctan2(cx[1] - cx[0], cy[1] - cy[0]))
        return alpha

    @cached_property
    def poly(self):
        """
        Perform coordinate transformations and polygon setup for the image extent.

        Returns:
        Polygon: A shapely Polygon object representing the image footprint in the target SRS.
        """
        # Get corners in longitude and latitude
        corners_n_lons, corners_n_lats = self.get_corners()

        # Transform points from lon/lat to image pixel coordinates (source)
        # DstToSrc=1 means LonLat (source) to Pixel (destination)
        cols, rows = super().transform_points(corners_n_lons, corners_n_lats, DstToSrc=1)

        # Transform points from image pixel coordinates (source) to meters in the specified SRS (destination)
        # DstToSrc=0 means Pixel (source) to Target SRS (destination)
        corners_nx_meters, corners_ny_meters = super().transform_points(cols, rows, DstToSrc=0, dst_srs=self.srs)

        # Construct coordinates in the required order to form a polygon (e.g., clockwise or counter-clockwise)
        coords = np.vstack([
            corners_nx_meters[[0, 2, 3, 1]],
            corners_ny_meters[[0, 2, 3, 1]]
        ]).T

        # Create and return the polygon
        poly = Polygon(coords)

        return poly
