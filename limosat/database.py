# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
database.py

Class and methods for persisting keypoints and templates
"""

import os
import json
import numpy as np
from sqlalchemy import Float, Text, DateTime
from .utils import logger

class DriftDatabase:
    """
    Initialize the database repository.
    
    Args:
        engine: SQLAlchemy engine for database connection
        zarr_path: Path to zarr storage location for templates
        run_name: Identifier for this run (used for table naming)
    """

    def __init__(self, engine=None, zarr_path=None, run_name=None):
        self.engine = engine
        self.zarr_path = zarr_path
        self.run_name = run_name

        # Define database column types
        self.dtype = {
            'image_id': Float(),
            'is_last': Float(),
            'trajectory_id': Float(),
            'geometry': 'geometry',
            'descriptors': Text(),
            'angle': Float(),
            'corr': Float(),
            'time': DateTime(timezone=True),
            'interpolated': Float()
        }

        logger.info(f"Initialized database for: {self.run_name}")

    def save(self, points, templates, insitu_points=None):
        """
        Save both points and templates as a single atomic operation.

        Args:
            points: The keypoints to persist.
            templates: The templates to persist.
            insitu_points: Optional validation points.

        Returns:
            bool: True if the save operation was successful
        """
        # Persist points to PostgreSQL
        points_out = points.copy().set_crs('EPSG:3413')
        points_out['descriptors'] = points_out['descriptors'].apply(self._serialize_descriptors)
        points_out.to_postgis(self.run_name, self.engine, if_exists='replace', dtype=self.dtype)

        # Persist templates to Zarr
        templates.to_dataset(name="template_data").to_zarr(self.zarr_path, mode="w")

        # Save validation metadata if provided
        if insitu_points is not None:
            self._save_validation_metadata(insitu_points)

        logger.info(f"Successfully persisted {len(points)} points and associated templates")
        return True

    def _serialize_descriptors(self, descriptor_list):
        """
        Serialize a list of numpy arrays (descriptors) into a JSON string.
        
        Args:
            descriptor_list: List of numpy array descriptors
            
        Returns:
            str: JSON string representation of the descriptors
        """
        return json.dumps(np.array(descriptor_list).tolist())

    def _save_validation_metadata(self, insitu_points):
        """
        Save validation metadata as GeoJSON.
        
        Args:
            insitu_points: GeoDataFrame containing validation points
        """
        # Create validation directory if it doesn't exist
        validation_dir = 'validation'
        os.makedirs(validation_dir, exist_ok=True)

        # Save to GeoJSON
        output_file = os.path.join(validation_dir, f"{self.run_name}_validation.geojson")

        # Ensure CRS is set correctly
        validation_data = insitu_points.copy()
        if validation_data.crs is None:
            validation_data.set_crs('EPSG:3413', inplace=True)

        validation_data.to_file(output_file, driver='GeoJSON')
        logger.info(f"Saved validation metadata to {output_file}")