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
import pandas as pd 
from sqlalchemy import text, Float, Text, DateTime, inspect
from .utils import logger, log_execution_time 

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
            'time': DateTime(timezone=False),
            'interpolated': Float()
        }
        
    @log_execution_time
    def save(self, points, templates, last_persisted_id, insitu_points=None):
        if points.empty:
             logger.info("No points to save.")
             if templates is not None and templates.trajectory_ids.size > 0 and self.zarr_path:
                 try:
                      logger.info("Saving empty points state, but saving templates...")
                      templates.data.to_dataset(name="template_data").to_zarr(self.zarr_path, mode="w")
                      logger.info(f"Successfully persisted {templates.trajectory_ids.size} templates (with 0 new points).")
                      return True
                 except Exception as e:
                      logger.error(f"Failed to save templates even with no points: {e}", exc_info=True)
                      return False
             return True


        try:
            # --- 1. Prepare Points Delta ---
            points_image_id_series = points['image_id'].astype(float)
            last_persisted_id_float = float(last_persisted_id)
            mask_series = points_image_id_series > last_persisted_id_float
            points_delta = points.loc[mask_series].copy()

            if points_delta.empty:
                logger.info("No new points detected since last save.")
                if templates is not None and templates.trajectory_ids.size > 0 and self.zarr_path:
                    logger.info("Saving templates only (no new points).")
                    templates.data.to_dataset(name="template_data").to_zarr(self.zarr_path, mode="w") 
                    return True
                return True

            logger.info(f"Processing {len(points_delta)} new/updated points for persistence.")
            points_delta = points_delta.set_crs('EPSG:3413')
            if 'descriptors' in points_delta.columns:
                 points_delta['descriptors'] = points_delta['descriptors'].apply(self._serialize_descriptors)

            with self.engine.connect() as connection:
                with connection.begin():
                    updated_traj_ids = points_delta['trajectory_id'].unique().tolist()
                    updated_traj_ids = [int(tid) for tid in updated_traj_ids if pd.notna(tid)]

                    inspector = inspect(connection.engine) # Use the engine from the connection
                    if updated_traj_ids and inspector.has_table(self.run_name, schema=connection.dialect.default_schema_name):
                        update_sql = text(f"""
                            UPDATE {self.run_name}
                            SET is_last = 0
                            WHERE is_last = 1 AND trajectory_id = ANY(:traj_ids)
                        """)
                        result = connection.execute(update_sql, {"traj_ids": updated_traj_ids})
                        logger.debug(f"Updated is_last=0 for {result.rowcount} previous points in DB.")
                    elif updated_traj_ids:
                        # This condition is met on the very first save operation for this run_name
                        logger.debug(f"Table '{self.run_name}' does not exist yet or is empty. Skipping initial 'is_last' update.")

                    points_delta.to_postgis(
                        self.run_name,
                        connection,
                        if_exists='append',
                        index=False,
                        dtype=self.dtype
                    )
                    logger.debug(f"Appended {len(points_delta)} points to database.")

            if templates is not None and templates.trajectory_ids.size > 0:
                templates_to_save = templates.data.copy()
                if hasattr(templates_to_save, 'encoding') and 'chunks' in templates_to_save.encoding:
                    del templates_to_save.encoding['chunks']
                templates_to_save.to_dataset(name="template_data").to_zarr(self.zarr_path, mode="w")
                logger.debug("Saved templates to Zarr (overwrite mode).")

            if insitu_points is not None:
                self._save_validation_metadata(insitu_points)

            logger.info(f"Successfully appended {len(points_delta)} points and saved templates.")
            return True

        except Exception as e:
            logger.error(f"Database save/append operation failed: {e}", exc_info=True)
            return False
        
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

    @staticmethod
    def deserialize_descriptors(json_string):
        """
        Deserialize a JSON string back into a numpy array of descriptors.
        """
        return np.array(json.loads(json_string), dtype=np.uint8)
