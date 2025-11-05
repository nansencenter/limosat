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
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from sqlalchemy import text, Float, Text, DateTime, inspect, Integer
from sqlalchemy import Boolean
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
            'image_id': Integer(),
            'is_last': Integer(),
            'trajectory_id': Integer(), 
            'geometry': 'geometry',
            'descriptors': Text(),
            'angle': Float(),
            'corr': Float(),
            'time': DateTime(timezone=False),
            'interpolated': Integer(),
            'orbit_num': Integer(),
            'stopped': Boolean(),
            'converged_to': Integer(),
        }

    def prepare_run_state(self, clear_existing_data=False, temporal_window_days=None):
        """
        Prepares the database and Zarr store for a run.

        If clear_existing_data is True, it wipes any previous data.
        Otherwise, it attempts to load the latest state to resume the run.

        Args:
            clear_existing_data (bool): Flag to clear data before starting.
            temporal_window_days (int): The number of days to look back for active points when resuming.

        Returns:
            tuple: A tuple containing (points, templates) for the ImageProcessor.
                   - points (Keypoints): Initial keypoints (empty for a fresh run).
                   - templates (Templates): Initial templates object (empty for a fresh run).
        """
        # Local imports to avoid circular dependencies at the module level
        from .keypoints import Keypoints
        from .templates import Templates

        if clear_existing_data:
            logger.info(f"Starting fresh run for '{self.run_name}'. Wiping existing data.")
            self._clear_storage()
            self._ensure_schema()
            return Keypoints(), Templates()
        else:
            logger.info(f"Attempting to resume run '{self.run_name}'...")
            if not temporal_window_days:
                raise ValueError("A temporal window (days) must be provided for resuming a run.")
            return self._load_latest_state(temporal_window_days)
        
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
            points_image_id_series = points['image_id'].astype(int)
            last_persisted_id_int = int(last_persisted_id)
            mask_series = points_image_id_series > last_persisted_id_int
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
                points_delta['descriptors'] = points_delta.apply(
                    lambda row: self._serialize_descriptors(row['descriptors']) if row['is_last'] == 1 else None,
                    axis=1
                )

            # --- 2. Compute Combined Trajectory Counts (DB + Delta) ---
            with self.engine.connect() as connection:
                with connection.begin():
                    delta_traj_ids = [
                        int(tid) for tid in points_delta['trajectory_id'].unique().tolist()
                        if pd.notna(tid)
                    ]
                    inspector = inspect(connection.engine)
                    table_exists = inspector.has_table(self.run_name, schema=connection.dialect.default_schema_name)
                    
                    # Count trajectories in delta
                    delta_counts = points_delta['trajectory_id'].value_counts().to_dict()
                    
                    # Fetch existing counts from DB if table exists
                    if table_exists and delta_traj_ids:
                        count_sql = text(f"""
                            SELECT trajectory_id, COUNT(*) as count
                            FROM {self.run_name}
                            WHERE trajectory_id = ANY(:traj_ids)
                            GROUP BY trajectory_id
                        """)
                        result = connection.execute(count_sql, {"traj_ids": delta_traj_ids})
                        db_counts = {int(row[0]): int(row[1]) for row in result}
                    else:
                        db_counts = {}
                    
                    # Combine counts: DB count + delta count
                    combined_counts = {}
                    for tid in delta_traj_ids:
                        db_count = db_counts.get(tid, 0)
                        delta_count = delta_counts.get(tid, 0)
                        combined_counts[tid] = db_count + delta_count
                    
                    # Filter: only persist trajectories with combined_count > 1
                    matched_traj_ids = [tid for tid, count in combined_counts.items() if count > 1]
                    
                    if not matched_traj_ids:
                        logger.info("No matched trajectories (length > 1) to persist after combined filtering.")
                        # Still save templates if available
                        if templates is not None and templates.trajectory_ids.size > 0 and self.zarr_path:
                            templates_to_save = templates.data.copy()
                            if hasattr(templates_to_save, 'encoding') and 'chunks' in templates_to_save.encoding:
                                del templates_to_save.encoding['chunks']
                            templates_to_save.to_dataset(name="template_data").to_zarr(self.zarr_path, mode="w")
                            logger.debug("Saved templates to Zarr (no matched trajectories to persist).")
                        return True
                    
                    # Filter points_delta to only matched trajectories
                    points_delta_filtered = points_delta[points_delta['trajectory_id'].isin(matched_traj_ids)].copy()
                    logger.info(
                        f"Filtered {len(points_delta)} delta points to {len(points_delta_filtered)} "
                        f"points from {len(matched_traj_ids)} matched trajectories (combined DB+delta count > 1)."
                    )
                    
                    # Update is_last flag only for matched trajectory_ids
                    if matched_traj_ids and table_exists:
                        update_sql = text(f"""
                            UPDATE {self.run_name}
                            SET is_last = 0
                            WHERE is_last = 1 AND trajectory_id = ANY(:traj_ids)
                        """)
                        result = connection.execute(update_sql, {"traj_ids": matched_traj_ids})
                        logger.debug(f"Updated is_last=0 for {result.rowcount} previous points in DB.")
                    elif matched_traj_ids:
                        logger.debug(f"Table '{self.run_name}' does not exist yet. It will be created.")

                    points_delta_filtered.to_postgis(
                        self.run_name,
                        connection,
                        if_exists='append',
                        index=False,
                        dtype=self.dtype
                    )
                    logger.debug(f"Appended {len(points_delta_filtered)} points to database.")

            if templates is not None and templates.trajectory_ids.size > 0:
                templates_to_save = templates.data.copy()
                if hasattr(templates_to_save, 'encoding') and 'chunks' in templates_to_save.encoding:
                    del templates_to_save.encoding['chunks']
                templates_to_save.to_dataset(name="template_data").to_zarr(self.zarr_path, mode="w")
                logger.debug("Saved templates to Zarr (overwrite mode).")

            if insitu_points is not None:
                self._save_validation_metadata(insitu_points)

            logger.info(f"Successfully appended {len(points_delta_filtered)} points and saved templates.")
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
        if json_string is None:
            return None
        return np.array(json.loads(json_string), dtype=np.uint8)

    def _clear_storage(self):
        """Drops the database table and removes the Zarr directory for the run."""
        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    connection.execute(text(f'DROP TABLE IF EXISTS "{self.run_name}";'))
                logger.info(f"Table {self.run_name} dropped.")
        except Exception as e:
            logger.warning(f"Could not drop table {self.run_name} (it may not exist): {e}")

        if self.zarr_path and os.path.exists(self.zarr_path):
            shutil.rmtree(self.zarr_path)
            logger.info(f"Zarr store at {self.zarr_path} removed.")

    def _ensure_schema(self):
        """Ensures the database table and index exist for the run."""
        inspector = inspect(self.engine)
        if inspector.has_table(self.run_name):
            logger.info(f"Table '{self.run_name}' already exists.")
            return

        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    # Create table using an empty GeoDataFrame to define schema
                    empty_gdf = gpd.GeoDataFrame({
                        'image_id': pd.Series(dtype='int64'),
                        'is_last': pd.Series(dtype='int64'),
                        'trajectory_id': pd.Series(dtype='int64'),
                        'geometry': gpd.GeoSeries(dtype='geometry'),
                        'descriptors': pd.Series(dtype='object'),
                        'angle': pd.Series(dtype='float64'),
                        'corr': pd.Series(dtype='float64'),
                        'time': pd.Series(dtype='datetime64[ns]'),
                        'interpolated': pd.Series(dtype='int64'),
                        'orbit_num': pd.Series(dtype='int64'),
                        'stopped': pd.Series(dtype='bool'),
                        'converged_to': pd.Series(dtype='Int64'),
                    }, crs='EPSG:3413')
                    
                    empty_gdf.to_postgis(
                        self.run_name,
                        connection,
                        if_exists='fail',
                        index=False,
                        dtype=self.dtype
                    )
                    logger.info(f"Table '{self.run_name}' created successfully.")

                    # Create performance index
                    index_name = f"idx_{self.run_name}_traj_last"
                    schema = connection.dialect.default_schema_name
                    index_sql = text(f'CREATE INDEX IF NOT EXISTS {index_name} ON "{schema}"."{self.run_name}" (trajectory_id, is_last)')
                    connection.execute(index_sql)
                    logger.info(f"Performance index '{index_name}' created.")
        except Exception as e:
            logger.error(f"Failed to create schema for run '{self.run_name}': {e}", exc_info=True)
            raise

    def _load_latest_state(self, temporal_window_days):
        """Loads the last known state from storage to resume a run."""
        from .keypoints import Keypoints
        from .templates import Templates
        
        inspector = inspect(self.engine)
        if not inspector.has_table(self.run_name):
            raise FileNotFoundError(f"Cannot resume: Database table '{self.run_name}' not found.")
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(f"Cannot resume: Zarr path not found at '{self.zarr_path}'")

        sql_query = f"SELECT * FROM {self.run_name} WHERE is_last = 1"
        points_last_from_db = gpd.read_postgis(sql_query, self.engine, geom_col='geometry', coerce_float=False)
        
        if points_last_from_db.empty:
            raise ValueError("Cannot resume: No points with is_last=1 found in database.")

        points_last_from_db['time'] = pd.to_datetime(points_last_from_db['time'])
        if points_last_from_db['time'].dt.tz is not None:
            points_last_from_db['time'] = points_last_from_db['time'].dt.tz_convert('UTC').dt.tz_localize(None)

        latest_time_in_db = points_last_from_db['time'].max()
        time_threshold = latest_time_in_db - pd.Timedelta(days=temporal_window_days)
        active_points_gdf = points_last_from_db[points_last_from_db['time'] >= time_threshold].copy()

        if active_points_gdf.empty:
            raise ValueError(f"Cannot resume: No 'is_last=1' points survived time filter (>= {time_threshold})")

        active_points_gdf['descriptors'] = active_points_gdf['descriptors'].apply(self.deserialize_descriptors)
        points = Keypoints._from_gdf(active_points_gdf)

        templates = Templates()
        with xr.open_zarr(self.zarr_path) as ds:
            templates.data = ds["template_data"].load()
            if templates.data.trajectory_id.size > 0:
                templates._initialized = True

        logger.info(f"Resume successful: Loaded {len(points)} points and {len(templates)} templates.")
        return points, templates