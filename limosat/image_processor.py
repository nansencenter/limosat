# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
from scipy.spatial import cKDTree
import time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.transform import AffineTransform
import logging
from .utils import log_execution_time, logger
from .database import DriftDatabase
from .deformation import filter_and_interpolate_flipped_triangles
from .image import Image
from .keypoints import Keypoints
from .keypoint_detector import KeypointDetector
from .templates import Templates
from .processing import interpolate_drift, pattern_matching

class ImageProcessor:
    """
    Image processor for ice drift analysis.
    """
    def __init__(self,
                 points,
                 model,
                 matcher,
                 config=None,
                 engine=None,
                 zarr_path=None,
                 run_name=None,
                 insitu_points=None,
                 return_insitu_points_on_completion=False,
                 **kwargs
                ):
        self.points = points
        self.templates = Templates()
        self.model = model
        self.matcher = matcher
        self.run_name = run_name
        self.insitu_points = insitu_points
        self.return_insitu_points_on_completion = return_insitu_points_on_completion

        # Define default parameters
        default_params = {
            'persist_updates': True,
            'persist_interval': 10,
            'pruning_interval': 10,
            'temporal_window': 3,
            'convergence_radius_pixels': 5,
            'candidate_search_max_daily_drift_m': 10000,
            'window_size': 64,
            'border_size': 128,
            'border_matched': 16,
            'border_interpolated': 64,
            'stride': 15,
            'octave': 8,
            'min_correlation': 0.4,
            'response_threshold': 0.0001,
            'template_size': 16,
            'use_interpolation': True,
            'max_interpolation_time_gap_hours': 25,
            'max_valid_speed_m_per_day': 50000.0,
            'window_border': 0,
        }

        # Start with defaults, update from config, then from kwargs
        proc_params = default_params.copy()
        if config and 'image_processor_params' in config:
            proc_params.update(config['image_processor_params'])
        proc_params.update(kwargs)

        # Set attributes from the final parameters for clarity and static analysis
        self.persist_updates = proc_params['persist_updates']
        self.persist_interval = proc_params['persist_interval']
        self.pruning_interval = proc_params['pruning_interval']
        self.temporal_window = proc_params['temporal_window']
        self.convergence_radius_pixels = proc_params['convergence_radius_pixels']
        self.candidate_search_max_daily_drift_m = proc_params['candidate_search_max_daily_drift_m']
        self.window_size = proc_params['window_size']
        self.border_size = proc_params['border_size']
        self.border_matched = proc_params['border_matched']
        self.border_interpolated = proc_params['border_interpolated']
        self.stride = proc_params['stride']
        self.octave = proc_params['octave']
        self.min_correlation = proc_params['min_correlation']
        self.response_threshold = proc_params['response_threshold']
        self.template_size = proc_params['template_size']
        self.use_interpolation = proc_params['use_interpolation']
        self.max_interpolation_time_gap_hours = proc_params['max_interpolation_time_gap_hours']
        self.max_valid_speed_m_per_day = proc_params['max_valid_speed_m_per_day']
        self.window_border = proc_params['window_border']  # 0 disables weighting
        self._last_persisted_id = 0
        
        # Initialize the KeypointDetector
        self.keypoint_detector = KeypointDetector(model=model)

        # Initialize trajectory_id column in insitu_points if in validation mode
        if self.insitu_points is not None:
            if 'trajectory_id' not in self.insitu_points.columns:
                self.insitu_points['trajectory_id'] = pd.NA
                self.insitu_points['trajectory_id'] = self.insitu_points['trajectory_id'].astype(pd.Int64Dtype())
            logger.info("Validation mode: 'trajectory_id' column ensured in self.insitu_points.")

        # Create DriftDatabase instance if persistence is enabled
        if self.persist_updates:
            if engine is None or zarr_path is None:
                raise ValueError("engine and zarr_path must be provided when persist_updates=True")
            self.db = DriftDatabase(engine=engine, zarr_path=zarr_path, run_name=run_name)

        logger.info(f"Initialized ImageProcessor" + (f" for: {self.run_name}" if run_name else ""))
        logger.info(f"Interpolation: {'enabled' if self.use_interpolation else 'disabled'}")
        if self.insitu_points is not None:
            logger.info("Validation mode enabled with in-situ points")

    def process_image(self, image_id, filename):
        """
        Process a single image and update internal points/templates.
        
        This method modifies the internal state of the processor by updating
        self.points and self.templates.
        
        Parameters:
        -----------
        image_id : int
            Unique identifier for the image being processed
        filename : str
            Path to the image file
        """
        basename = os.path.basename(filename)

        # Skip if already processed
        if image_id <= self.points.last_image_id:
            logger.info(f"Skipping image {image_id}: {basename}")
            return

        logger.info(f"Processing image {image_id}: {basename}")

        # Create Nansat Image object from file
        img = Image(filename)
        
        # Calculate buffer to capture points drifting INTO image
        max_possible_drift = self.candidate_search_max_daily_drift_m * self.temporal_window 
        buffer_distance = min(max_possible_drift, self.matcher.spatial_distance_max)
        logger.debug(f"Using buffer distance: {buffer_distance/1000.0:.2f} km "
                    f"(Min of {max_possible_drift/1000.0:.1f}km theoretical max drift over {self.temporal_window} days "
                    f"and {self.matcher.spatial_distance_max/1000.0:.1f}km spatial match limit)")
    
        # Filter for active points within the buffered polygon and time threshold
        buffered_image_poly = img.poly.buffer(buffer_distance)
        time_threshold = img.date - pd.Timedelta(days=self.temporal_window)
        points_last = self.points.last()
        points_poly = points_last[
            points_last.within(buffered_image_poly) &
            (points_last['time'] >= time_threshold)
        ]

        # this is to prevent memory errors for rare instances with many overlapping active points
        CANDIDATE_POINT_LIMIT = 40000
        if len(points_poly) > CANDIDATE_POINT_LIMIT:
            logger.info(
                f"Candidate points ({len(points_poly)}) exceed limit. "
                f"Taking a random sample of {CANDIDATE_POINT_LIMIT} points"
            )
            points_poly = points_poly.sample(n=CANDIDATE_POINT_LIMIT, random_state=42)

        if len(points_poly) == 0:
            logger.info("No overlapping points found")
            points_final = Keypoints()
            occupancy_points = Keypoints()  # nothing to exclude
        else:
            logger.info(f"{len(points_poly)} overlapping points found")
            points_final, failed_predictions = self._match_existing_points(points_poly, img, image_id, img.orbit_num)

            # --- Build occupancy set to avoid seeding near ANY active last point (matched or unmatched) ---
            if not points_poly.empty:
                orbit_num_current = getattr(img, 'orbit_num', None)

                same_orbit_mask = pd.Series(False, index=points_poly.index)
                if orbit_num_current is not None and 'orbit_num' in points_poly.columns:
                    same_orbit_mask = (points_poly['orbit_num'] == orbit_num_current)

                stopped_mask = pd.Series(False, index=points_poly.index)
                if 'stopped' in points_poly.columns:
                    stopped_mask = points_poly['stopped'].astype(bool)

                combined_exclude_mask = same_orbit_mask | stopped_mask
                candidates = points_poly[~combined_exclude_mask]

                same_orbit_excluded = int(same_orbit_mask.sum())
                stopped_excluded = int(stopped_mask.sum())

                if points_final.empty:
                    unmatched = candidates
                else:
                    matched_tids = set(points_final.trajectory_id.values)
                    unmatched = candidates[~candidates.trajectory_id.isin(matched_tids)]

                # Substitute unmatched geometry with failed interpolated predictions when available
                if isinstance(failed_predictions, pd.DataFrame) and not failed_predictions.empty:
                    pred_map = dict(zip(failed_predictions['trajectory_id'], failed_predictions['geometry_pred']))
                    if pred_map and not unmatched.empty:
                        unmatched = unmatched.copy()
                        unmatched['geometry'] = unmatched.apply(
                            lambda r: pred_map.get(r.trajectory_id, r.geometry), axis=1
                        )

                if not points_final.empty:
                    occupancy = pd.concat([points_final, unmatched], ignore_index=True)
                else:
                    occupancy = unmatched

                occupancy_points = Keypoints._from_gdf(occupancy)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Occupancy build: candidates=%d matched=%d unmatched=%d excl_same_orbit=%d excl_stopped=%d",
                        len(candidates),
                        len(points_final),
                        len(unmatched),
                        same_orbit_excluded,
                        stopped_excluded
                    )
            else:
                occupancy_points = Keypoints()

            if not points_final.empty and 'interpolated' in points_final.columns:
                interp_count = points_final['interpolated'].sum()
                total_count = len(points_final)
                if interp_count > 0:
                    logger.debug(f"Interpolation stats: {interp_count}/{total_count} points ({interp_count/total_count:.1%}) were interpolated")

        # Process new points (now excluding both matched and unmatched existing active trajectories)
        self._process_new_points(points_final, img, image_id, basename, occupancy_points=occupancy_points)

        # Template pruning
        if image_id > 0 and image_id % self.pruning_interval == 0 and len(self.templates) > 0:
            try:
                time_threshold_prune = img.date.to_numpy() - np.timedelta64(self.temporal_window, 'D')
                active_traj_ids = self.points.loc[self.points['is_last'] == 1, 'trajectory_id'].unique()
                self.templates.prune(active_traj_ids, time_threshold_prune)
            except Exception as e:
                logger.error(f"Error during template pruning: {e}", exc_info=True)
                logger.warning("Skipping template pruning for this interval due to error.")

        # Persist database every X images
        if self.persist_updates:
            current_points_image_id = self.points.last_image_id
            images_since_persist = current_points_image_id - self._last_persisted_id

            if images_since_persist >= self.persist_interval:
                current_last_image_id = current_points_image_id
                
                # --- 1. Filter points to save only matched trajectories ---
                
                # Identify trajectory_ids that appear more than once (i.e., have been matched)
                traj_id_counts = self.points['trajectory_id'].value_counts()
                matched_traj_ids = traj_id_counts[traj_id_counts > 1].index
                
                # Create a filtered dataframe for persistence
                points_to_persist = self.points[self.points['trajectory_id'].isin(matched_traj_ids)]
                
                logger.info(f"Found {len(matched_traj_ids)} matched trajectories. "
                            f"Persisting {len(points_to_persist)} points.")

                # --- 2. Attempt to save data to the database ---
                save_successful = self.db.save(
                    points_to_persist,
                    self.templates,
                    self._last_persisted_id,
                    self.insitu_points
                )

                # --- 3. Prune in-memory data only if save was successful ---
                if save_successful:
                    try:
                        num_before_prune = len(self.points)
                        keep_mask = (self.points['is_last'] == 1) & \
                                    (self.points['time'] >= time_threshold)
                        
                        self.points = Keypoints._from_gdf(self.points[keep_mask])
                        
                        num_after_prune = len(self.points)
                        logger.debug(f"In-memory points pruned: {num_before_prune} -> {num_after_prune}")
                        
                        self._last_persisted_id = current_last_image_id
                    except Exception as e:
                        logger.error(f"Error during in-memory point pruning after successful save: {e}", exc_info=True)
                        logger.critical("CRITICAL: In-memory pruning failed. State may be inconsistent. Continuing without pruning.")
                else:
                    logger.warning("Database save reported failure. Skipping in-memory point pruning to prevent data loss.")


    def _handle_trajectory_convergence(self, points_matched):
        """
        Identifies and handles converging trajectories using connected components.

        When multiple trajectories converge, this method finds clusters of nearby points.
        For each cluster, it identifies a single 'winner' trajectory and marks all
        others in the cluster as 'losers'. The winner is chosen based on the longest
        full trajectory history, with tie-breaking by correlation.

        Args:
            points_matched (pd.DataFrame): A GeoDataFrame of points successfully matched.

        Returns:
            pd.DataFrame: The input DataFrame with points from loser trajectories removed.
        """
        if len(points_matched) < 2:
            return points_matched

        # 1. Find nearby points using cKDTree
        coords = np.vstack(points_matched.geometry.apply(lambda p: (p.x, p.y)))
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=self.convergence_radius_pixels, output_type='ndarray')

        if pairs.size == 0:
            return points_matched
                
        logger.info(f"Found {pairs.shape[0]} converging trajectory pairs, forming clusters.")

        # 2. Find connected components using SciPy's sparse graph tools
        n_nodes = len(points_matched)
        adj_matrix = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_nodes, n_nodes))
        
        n_components, labels = connected_components(
            csgraph=adj_matrix, directed=False, return_labels=True
        )

        # 3. Group indices by component label, keeping only clusters of size > 1
        unique_labels, counts = np.unique(labels, return_counts=True)
        multi_node_labels = unique_labels[counts > 1]
        
        if multi_node_labels.size == 0:
            return points_matched

        label_to_indices = {label: np.where(labels == label)[0] for label in multi_node_labels}
        clusters = list(label_to_indices.values())

        # 4. Determine winner/loser for each cluster
        loser_tids = set()
        loser_to_winner_map = {}

        all_involved_indices = np.concatenate(clusters)
        all_involved_tids = points_matched.iloc[all_involved_indices].trajectory_id.unique()
        traj_lengths = self.points[self.points['trajectory_id'].isin(all_involved_tids)]['trajectory_id'].value_counts()

        for comp_indices in clusters:
            cluster_df = points_matched.iloc[comp_indices].copy()
            
            # Add the 'length' column to this temporary dataframe for sorting
            cluster_df['length'] = cluster_df['trajectory_id'].map(traj_lengths).fillna(1).astype(int)
            
            # Robustly find the winner using multi-level sorting:
            # 1. Sort by trajectory length (descending).
            # 2. Then, sort by correlation (descending) to break ties.
            sorted_cluster = cluster_df.sort_values(
                by=['length', 'corr'], 
                ascending=[False, False]
            )
            
            # The winner is unambiguously the first row of the sorted DataFrame.
            winner_row = sorted_cluster.iloc[0]
            winner_tid = int(winner_row['trajectory_id'])
            
            # Mark all others in the cluster as losers
            tids_in_cluster = cluster_df['trajectory_id'].unique()
            cluster_losers = set(tids_in_cluster) - {winner_tid}

            if cluster_losers:
                loser_tids.update(cluster_losers)
                for loser in cluster_losers:
                    loser_to_winner_map[loser] = winner_tid

        # 5. Apply all updates in a single, vectorized operation
        if loser_tids:
            logger.info(f"Stopping {len(loser_tids)} trajectories due to convergence.")
            
            mask = self.points['trajectory_id'].isin(loser_tids)
            self.points.loc[mask, 'stopped'] = True
            
            converged_to_values = self.points.loc[mask, 'trajectory_id'].map(loser_to_winner_map)
            self.points.loc[mask, 'converged_to'] = converged_to_values

            return points_matched[~points_matched['trajectory_id'].isin(loser_tids)]

        return points_matched

    @log_execution_time
    def _process_new_points(self, points_final, img, image_id, basename, occupancy_points=None):
        """
        Process new (seeded) points and update templates.

        occupancy_points (Keypoints): Combined set of positions to exclude when seeding:
            - matched trajectories with updated positions (points_final)
            - unmatched active trajectories with their previous positions
            If None, falls back to using points_final (legacy behaviour).
        """
        # Use combined occupancy set if provided
        exclusion_points = occupancy_points if occupancy_points is not None else points_final

        # Detect new keypoints avoiding both matched and unmatched active trajectories
        raw_kps_new = self.keypoint_detector.detect_new_keypoints(
            points=exclusion_points,
            img=img,
            window_size=self.window_size,
            border_size=self.border_size,
            response_threshold=self.response_threshold,
            octave=self.octave,
            compute_descriptors=False,
            window_border=self.window_border,
        )
        logger.debug(f"Detected {len(raw_kps_new)} new raw keypoints from image {image_id} after occupancy exclusion")

        if self.insitu_points is not None:
            matching_insitu_points = self.insitu_points.loc[
                self.insitu_points['image_filepath'].isin([basename])
            ]

            if len(matching_insitu_points) > 0:
                logger.info(f"Found {len(matching_insitu_points)} matching buoy observations in image {basename}")
                buoy_kps = self.keypoint_detector.keypoint_from_point(matching_insitu_points, octave=self.octave,img=img, response_threshold=self.response_threshold)
                if buoy_kps is None:
                    logger.error("keypoint_from_point returned None unexpectedly!")
                    buoy_kps = []
                raw_kps_new = raw_kps_new + buoy_kps
                logger.info(f"Added {len(buoy_kps)} buoy keypoints to detection set")

        # Now, compute descriptors for the combined list of new and buoy keypoints
        keypoints_coords_arr, descriptors_arr, surviving_tags = self.keypoint_detector.compute_descriptors(
            raw_kps_new,
            img,
            polarisation=1,
            normalize=False
        )

        if keypoints_coords_arr is not None and descriptors_arr is not None:
            points_new = Keypoints.create(keypoints_coords_arr, descriptors_arr, img, image_id, img.orbit_num)
            
            current_self_points_len = len(self.points)
            self.points = self.points.append(points_new) # This assigns final trajectory_ids to the points_new portion
            
            appended_points_gdf = self.points.iloc[current_self_points_len:]
            logger.info(f"Added {len(appended_points_gdf)} new points (total: {len(self.points)})")

            # Link insitu_points to final trajectory_ids using surviving_tags
            if self.insitu_points is not None and surviving_tags is not None and not appended_points_gdf.empty:
                appended_points_gdf_reset = appended_points_gdf.reset_index(drop=True)
                
                if len(surviving_tags) == len(appended_points_gdf_reset):
                    for i, original_df_idx_tag in enumerate(surviving_tags):
                        if original_df_idx_tag is not None: # Check if the tag is an actual index
                            final_tid = appended_points_gdf_reset.iloc[i]['trajectory_id']
                            self.insitu_points.loc[original_df_idx_tag, 'trajectory_id'] = final_tid
                            logger.debug(f"Linked insitu_point (original index {original_df_idx_tag}) to trajectory_id {final_tid}")
                else:
                    logger.warning(f"Mismatch between surviving_tags ({len(surviving_tags)}) and "
                                   f"appended_points_gdf_reset ({len(appended_points_gdf_reset)}). Skipping insitu linking for this batch.")

            # OPTIONAL: defensive removal of any newly seeded points that slipped within convergence radius
            if len(appended_points_gdf) > 0 and exclusion_points is not None and not exclusion_points.empty:
                try:
                    # Fast spatial culling using cKDTree
                    new_xy = np.vstack(appended_points_gdf.geometry.apply(lambda p: (p.x, p.y)))
                    old_xy = np.vstack(exclusion_points.geometry.apply(lambda p: (p.x, p.y)))
                    tree = cKDTree(old_xy)
                    nn_dist, _ = tree.query(new_xy, k=1, workers=-1)
                    keep_mask = nn_dist > self.convergence_radius_pixels
                    if not np.all(keep_mask):
                        removed = (~keep_mask).sum()
                        if removed > 0:
                            logger.debug(f"Removed {removed} newly seeded points inside convergence radius of existing trajectories")
                            # Drop from self.points
                            to_remove_index = appended_points_gdf.index[~keep_mask]
                            self.points = Keypoints._from_gdf(self.points.drop(index=to_remove_index))
                            # Reset appended_points_gdf to survivors for template creation / linking
                            appended_points_gdf = appended_points_gdf.loc[keep_mask]
                except Exception as e:
                    logger.warning(f"Proximity purge of new seeds failed: {e}")

            # Add templates only for surviving new points
            if len(appended_points_gdf) > 0:
                self.templates.add(appended_points_gdf, img, self.template_size, band=1)
        # ...existing code...
        if len(points_final) > 0:
            self.points = self.points.update(points_final)
            # Update templates for the points that were successfully matched and updated
            self.templates.update(points_final, img, self.template_size, band=1)
            
    @log_execution_time
    def _match_existing_points(self, points_poly, img, image_id, current_orbit_num):
        """
        Match points and return (final_points, failed_predictions).
        failed_predictions contains columns ['trajectory_id', 'geometry_pred'] with predicted positions
        for candidates that were interpolated but failed the recheck.
        """
        # Goal: Filter out points from trajectories that have already been stopped.
        stopped_tids = self.points[self.points['stopped'] == True]['trajectory_id'].unique()
        if len(stopped_tids) > 0:
            num_before_stop_filter = len(points_poly)
            points_poly = points_poly[~points_poly['trajectory_id'].isin(stopped_tids)]
            num_after_stop_filter = len(points_poly)
            if num_after_stop_filter < num_before_stop_filter:
                logger.info(f"Removed {num_before_stop_filter - num_after_stop_filter} points from stopped trajectories.")

        # Goal: Filter points_poly to exclude points from the same orbit.

        # 1. Check for the prerequisite column.
        if 'orbit_num' not in points_poly.columns:
            logger.error("FATAL: 'orbit_num' column not found in points_poly. Cannot apply orbit filter.")
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        # 2. Create and apply the filter mask.
        previous_orbit_nums = points_poly['orbit_num']
        orbit_filter_mask = previous_orbit_nums != current_orbit_num
        points_poly_filtered = points_poly[orbit_filter_mask]

        # 3. Log the operation and handle the empty case.
        if points_poly_filtered.empty:
            total_candidates_in_buffer = len(points_poly)
            logger.info(
                f"Found {total_candidates_in_buffer} points in buffer, but all were from the same orbit ({current_orbit_num}). "
                "Skipping matching."
            )
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 
        else:
            logger.info(
                f"{len(points_poly_filtered)} valid candidate points found for matching from previous orbits."
            )

        # 4. Ensure the rest of the method uses the filtered data.
        # Detect gridded keypoints and compute their descriptors in one step
        keypoints_coords_arr_grid, descriptors_grid, _ = self.keypoint_detector.detect_gridded_points(
            img,
            stride=self.stride,
            border_size=self.border_size,
            octave=self.octave,
        )

        if keypoints_coords_arr_grid is None or descriptors_grid is None:
            logger.warning("Failed to compute descriptors for grid points. Skipping matching.")
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        points_grid = Keypoints.create(keypoints_coords_arr_grid, descriptors_grid, img, image_id=image_id, orbit_num=img.orbit_num)

        # Match and filter points
        points_fg1, points_fg2, _ = self.matcher.match_with_grid(
            points_poly_filtered,
            points_grid
        )

        if points_fg1 is None or points_fg2 is None or points_fg1.empty or points_fg2.empty:
            logger.info("Insufficient match quality or no matches found, skipping point matching step.")
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        # Assign trajectory IDs to matched points
        points_fg2['trajectory_id'] = points_fg1.trajectory_id.values
        points_fg2['interpolated'] = 0
        points_matched = points_fg2

        # Interpolate drift if needed
        if self.use_interpolation and len(points_fg1) < len(points_poly_filtered):
            points_matched = interpolate_drift(
                points_poly=points_poly_filtered,
                points_fg1=points_fg1,
                points_fg2=points_matched,
                img=img,
                max_interpolation_time_gap_hours=self.max_interpolation_time_gap_hours,
                border_size=self.border_size,
                model_type=AffineTransform
            )
            if points_matched is None or points_matched.empty:
                logger.warning("Interpolation resulted in no valid points. Proceeding with only matched points.")
                points_matched = points_fg2  # Fall back to only matched points
        else:
            if not self.use_interpolation:
                logger.info("Interpolation disabled, using only matched points")
            elif len(points_fg1) >= len(points_poly_filtered):
                logger.info("All points matched, no interpolation needed")

        # GLOBAL VELOCITY FILTER for both matched and interpolated points
        if not points_matched.empty:
            num_before_speed_filter = len(points_matched)
            
            # Merge with `points_poly_filtered` which contains the previous state for all candidates
            candidates_with_history = pd.merge(
                points_matched,
                points_poly_filtered[['trajectory_id', 'geometry', 'time']],
                on='trajectory_id',
                suffixes=('', '_prev')
            )

            # Calculate speed
            # The current time for all points in `points_matched` is the time of the new image
            time_diff_days = (img.date - candidates_with_history['time_prev']).dt.total_seconds() / 86400.0
            distance_m = candidates_with_history.geometry.distance(candidates_with_history.geometry_prev)
            
            # Avoid division by zero
            speed_m_per_day = np.divide(
                distance_m, 
                time_diff_days, 
                out=np.full_like(time_diff_days, np.inf), 
                where=time_diff_days > 1e-9
            )

            # Apply filter
            speed_filter_mask = speed_m_per_day <= self.max_valid_speed_m_per_day
            valid_trajectory_ids = candidates_with_history.loc[speed_filter_mask, 'trajectory_id']
            
            # Filter the original `points_matched` dataframe
            final_mask = points_matched['trajectory_id'].isin(valid_trajectory_ids)
            points_matched = points_matched[final_mask]
            
            num_after_speed_filter = len(points_matched)
            if num_after_speed_filter < num_before_speed_filter:
                num_removed = num_before_speed_filter - num_after_speed_filter
                logger.info(
                    f"Global velocity filter: Removed {num_removed} outlier points "
                    f"exceeding {self.max_valid_speed_m_per_day} m/day."
                )

        # CONVERGENCE HANDLING
        if not points_matched.empty:
            points_matched = self._handle_trajectory_convergence(points_matched)
            if points_matched.empty:
                logger.info("All points removed after convergence handling.")
                return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        # Filter out points that don't have templates
        all_traj_ids = points_matched.trajectory_id.values
        available_traj_ids = self.templates.trajectory_ids
        has_template_mask = np.isin(all_traj_ids, available_traj_ids)

        if not np.all(has_template_mask):
            points_matched = points_matched[has_template_mask]
            all_traj_ids = points_matched.trajectory_id.values
            
        if points_matched.empty:
            logger.debug("No points remaining after template filtering.")
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        # Get original points and templates for pattern matching
        points_orig = points_poly_filtered[points_poly_filtered.trajectory_id.isin(all_traj_ids)]
        templates_all = self.templates.get_by_id(all_traj_ids)
        
        if not (len(points_matched) == len(points_orig) == len(templates_all.trajectory_id)):
            logger.error(f"Length mismatch before pattern_matching! "
                        f"points_matched: {len(points_matched)}, "
                        f"points_orig: {len(points_orig)}, "
                        f"templates_all: {len(templates_all.trajectory_id)}")
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        # Perform pattern matching
        keypoints_corrected_xy, keypoints_corrected_rc, corr_values = pattern_matching(
            points_matched,
            img,
            templates_all,
            points_orig,
            hs=self.template_size,
            border_matched=self.border_matched,
            border_interpolated=self.border_interpolated,
            band=1
        )

        # Initial correlation filter
        points_matched['corr'] = corr_values
        correlation_mask = corr_values >= self.min_correlation
        
        if not np.any(correlation_mask):
            logger.debug("No points passed correlation filter.")
            return Keypoints(), pd.DataFrame(columns=["trajectory_id", "geometry_pred"]) 

        # Apply correlation filter
        points_matched = points_matched[correlation_mask]
        keypoints_corrected_xy = keypoints_corrected_xy[correlation_mask]
        
        # Prepare data for triangulation filter
        points_matched = pd.merge(
            points_matched,
            points_orig[['trajectory_id', 'geometry']].rename(columns={'geometry': 'geometry_orig'}),
            on='trajectory_id', how='left'
        )
        
        # Call the filter to mask bad vectors and propose interpolated positions
        x1_interp, y1_interp, was_interpolated_mask = filter_and_interpolate_flipped_triangles(
            points_matched.geometry_orig.x.to_numpy(),
            points_matched.geometry_orig.y.to_numpy(),
            keypoints_corrected_xy[:, 0],
            keypoints_corrected_xy[:, 1],
        )

        # Separate points into "good" (not interpolated) and "re-check" (interpolated)
        good_mask = ~was_interpolated_mask & ~np.isnan(x1_interp)
        
        # Handle good points
        points_good = points_matched[good_mask]
        xy_good = np.column_stack((x1_interp[good_mask], y1_interp[good_mask]))
        rc_good = keypoints_corrected_rc[correlation_mask][good_mask]

        # Handle points that need re-checking
        failed_predictions = pd.DataFrame(columns=["trajectory_id", "geometry_pred"])  # collect failed recheck predictions

        if np.any(was_interpolated_mask):
            logger.info(f"Re-validating {np.sum(was_interpolated_mask)} interpolated vectors...")
            
            points_to_recheck = points_matched[was_interpolated_mask]
            points_to_recheck['geometry'] = gpd.points_from_xy(x1_interp[was_interpolated_mask], y1_interp[was_interpolated_mask])
            templates_recheck = self.templates.get_by_id(points_to_recheck.trajectory_id.values)
            
            points_fg1_recheck = points_matched[was_interpolated_mask][['trajectory_id', 'geometry_orig', 'angle']]
            points_fg1_recheck.rename(columns={'geometry_orig': 'geometry'}, inplace=True)

            xy_rechecked, rc_rechecked, corr_rechecked = pattern_matching(
                points_to_recheck,
                img,
                templates_recheck,
                points_fg1_recheck,
                hs=self.template_size,
                border_matched=self.border_matched,
                border_interpolated=self.border_interpolated,
                band=1
            )
            
            recheck_passed_mask = corr_rechecked >= self.min_correlation

            # collect failed (interpolated but did not pass recheck)
            if (~recheck_passed_mask).any():
                failed_tids = points_to_recheck.loc[~recheck_passed_mask, 'trajectory_id'].values
                xs = x1_interp[was_interpolated_mask][~recheck_passed_mask]
                ys = y1_interp[was_interpolated_mask][~recheck_passed_mask]
                failed_geom = gpd.points_from_xy(xs, ys).tolist()
                failed_predictions = pd.DataFrame({
                    'trajectory_id': failed_tids,
                    'geometry_pred': failed_geom
                })

            num_passed = np.sum(recheck_passed_mask)
            logger.debug(f"Re-validation: {num_passed}/{len(points_to_recheck)} interpolated vectors passed.")

            # Combine survivors from both groups
            points_rechecked = points_to_recheck[recheck_passed_mask]
            points_rechecked['corr'] = corr_rechecked[recheck_passed_mask]

            # Combine final results
            points_matched = pd.concat([points_good, points_rechecked], ignore_index=True)
            corrected_xy = np.vstack([xy_good, xy_rechecked[recheck_passed_mask]])
            corrected_rc = np.vstack([rc_good, rc_rechecked[recheck_passed_mask]])
        else:
            # No re-checking needed - use good points directly
            points_matched = points_good
            corrected_xy = xy_good
            corrected_rc = rc_good

        # Final checks
        if points_matched.empty:
            logger.debug("No points survived filtering.")
            return Keypoints(), failed_predictions
                        
        # Update geometry with corrected positions
        points_matched = points_matched.drop(columns=['geometry_orig'])
        points_matched['geometry'] = gpd.points_from_xy(corrected_xy[:, 0], corrected_xy[:, 1])
        
        logger.info(f"Pattern matching kept {len(points_matched)} points (correlation >= {self.min_correlation})")

        # Recompute descriptors for final points
        if not points_matched.empty:
            keypoints_list_with_tags_recompute = [
                # The size parameter is required, but will be overwritten by the detector's patch size
                (cv2.KeyPoint(px_c, px_r, size=31, angle=img.angle, octave=self.octave), None)
                for px_c, px_r in corrected_rc
            ]
            
            _raw_kps_recomputed, new_descriptors, _surviving_tags_recomputed = self.keypoint_detector.compute_descriptors(
                keypoints_list_with_tags_recompute,
                img,
                polarisation=1,
                normalize=False
            )
        else:
            new_descriptors = None

        original_count = len(points_matched)

        if new_descriptors is not None and len(new_descriptors) == len(points_matched):
            points_matched['descriptors'] = list(new_descriptors)
            
            # Filter out points with invalid descriptors
            valid_mask_desc = points_matched['descriptors'].apply(lambda d: isinstance(d, np.ndarray))
            points_matched = points_matched[valid_mask_desc]
            
            if len(points_matched) < original_count:
                logger.debug(f"Removed {original_count - len(points_matched)} points with invalid descriptors")
        elif original_count > 0:
            logger.warning("Descriptor computation failed; removing all remaining points.")
            points_matched = points_matched.iloc[0:0]  # Empty DataFrame

        # Update templates for surviving points
        if not points_matched.empty:
            self.templates.update(points_matched, img, self.template_size, band=1)

        logger.debug(f"Returning {len(points_matched)} final points")
        return points_matched, failed_predictions

    def ensure_final_persistence(self):
        """Ensure final persistence of any remaining unprocessed data."""
        if self.persist_updates:
            images_since_persist = self.points.last_image_id - self._last_persisted_id
            if images_since_persist > 0:
                logger.info(f"Performing final persistence for remaining {images_since_persist} images")
                save_successful = self.db.save(
                    points=self.points,
                    templates=self.templates,
                    last_persisted_id=self._last_persisted_id,
                    insitu_points=self.insitu_points
                )
                if save_successful:
                    self._last_persisted_id = self.points.last_image_id
                    logger.info(f"Final persistence completed. Last persisted ID set to {self._last_persisted_id}")
                else:
                    logger.error("Final persistence FAILED. _last_persisted_id not updated.")
                    logger.error("Final persistence FAILED. _last_persisted_id not updated.")
