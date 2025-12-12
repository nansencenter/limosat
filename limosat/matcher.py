# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import cv2
from skimage.transform import AffineTransform
from matplotlib import pyplot as plt
from collections import defaultdict
from .utils import log_execution_time, logger

class Matcher:
    def __init__(self,
                 # General matching parameters
                 norm=cv2.NORM_HAMMING2,
                 descriptor_distance_max=120,
                 spatial_distance_max=100000,

                 # Homography estimation parameters
                 model=AffineTransform,
                 model_threshold=10000,
                 use_model_estimation=True,
                 estimation_method="USAC_MAGSAC",
                 min_homography_inliers=10,

                 # Lowe's ratio test parameter
                 lowe_ratio=0.9,
                 knn_k=4,

                 # Visualization
                 plot=False,
                 
                 # Debug recording
                 debug_recorder=None):

        # General matching parameters
        self.norm = norm
        self.descriptor_distance_max = descriptor_distance_max
        self.spatial_distance_max = spatial_distance_max

        # Homography estimation parameters
        self.model = model
        self.model_threshold = model_threshold
        self.use_model_estimation = use_model_estimation
        self.min_homography_inliers = min_homography_inliers

        # Store the method name and get its value
        if estimation_method.upper() == "DEGENSAC":
            self.estimation_method_name = "DEGENSAC"
            self.estimation_method = None
        else:
            self.estimation_method_name = estimation_method
            self.estimation_method = getattr(cv2, estimation_method)

        # Additional parameters
        self.lowe_ratio = lowe_ratio
        self.knn_k = knn_k
        self.plot = plot
        
        # Debug recording
        self.debug_recorder = debug_recorder

    def plot_quiver(self, pos0, pos1, dist):
        u = pos1[:, 0] - pos0[:, 0]
        v = pos1[:, 1] - pos0[:, 1]
        spd = np.hypot(u,v)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(dist, spd, '.', alpha=0.5)
        qui = axs[1].quiver(pos0[:, 0], pos0[:, 1], pos1[:, 0] - pos0[:, 0], pos1[:, 1] - pos0[:, 1], dist, cmap='jet', clim=np.percentile(dist, [1, 99]), angles='xy', scale_units='xy', scale=1)
        plt.colorbar(qui, ax=axs[0], shrink=0.5)
        plt.show()

    @log_execution_time
    def match_with_grid(self, points_poly, points_grid):
        """
        Match points between polygon and grid representations. Match globally, filter locally.

        Parameters:
        points_poly: Points from polygon representation (GeoDataFrame)
        points_grid: Points from grid representation (GeoDataFrame)

        Returns:
        tuple: (points_fg1, points_fg2, residuals) matched points, or (None, None, None) if matching fails.
        """
        # 1. Setup: Extract positions and descriptors
        pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
        pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
        x0 = np.vstack(points_poly['descriptors'].values)
        x1 = np.vstack(points_grid['descriptors'].values)

        # 2. Generate an enhanced set of candidate matches
        # Start with the high-confidence cross-check matches...
        base_matches = self.match_with_crosscheck(x0, x1)
        num_crosscheck_matches = len(base_matches)
        candidate_matches = self.match_with_lowe_ratio(base_matches, x0, x1, pos0, pos1)
        num_lowe_additional = len(candidate_matches) - num_crosscheck_matches
        logger.info(f"Total candidate matches after cross-check and Lowe's ratio: {len(candidate_matches)}")
        
        # Record matching statistics
        if self.debug_recorder:
            image_id = points_grid['image_id'].iloc[0] if 'image_id' in points_grid.columns and len(points_grid) > 0 else None
            self.debug_recorder.record(
                stage="bf_matcher",
                event_type="info",
                message="candidate matches generated",
                step=image_id,
                num_crosscheck_matches=num_crosscheck_matches,
                num_lowe_additional=num_lowe_additional,
                num_total_candidates=len(candidate_matches),
            )

        # 3. Group all candidate matches by their source image_id
        matches_by_group = defaultdict(list)
        traj_ids0 = points_poly['trajectory_id'].to_numpy() if 'trajectory_id' in points_poly.columns else None
        for match in candidate_matches:
            # Check if queryIdx is valid before accessing points_poly
            if match.queryIdx < len(points_poly):
                image_id = points_poly.iloc[match.queryIdx]['image_id']
                matches_by_group[image_id].append(match)
            else:
                logger.warning(f"Invalid queryIdx {match.queryIdx} found in a match. Skipping.")


        # 4. Loop through each group and apply the 'filter' function
        all_inliers_idx0, all_inliers_idx1, all_residuals = [], [], []
        for image_id, group_matches in matches_by_group.items():
            # image_id here represents the source image of points_poly for this group, used for debug recording
            rc_idx0_group, rc_idx1_group, residuals_group = self.filter(group_matches, pos0, pos1, image_id=image_id, traj_ids0=traj_ids0)
            
            if rc_idx0_group is not None and rc_idx0_group.size > 0:
                all_inliers_idx0.append(rc_idx0_group)
                all_inliers_idx1.append(rc_idx1_group)
                all_residuals.append(residuals_group)

        # 5. Check if any valid groups were found at all
        if not all_inliers_idx0:
            logger.warning("No valid inlier groups found after filtering.")
            if self.debug_recorder:
                grid_image_id = points_grid['image_id'].iloc[0] if 'image_id' in points_grid.columns and len(points_grid) > 0 else None
                self.debug_recorder.record(
                    stage="bf_matcher",
                    event_type="failure",
                    message="no valid inlier groups found after filtering",
                    step=grid_image_id,
                    num_groups=len(matches_by_group),
                )
            return None, None, None

        # 6. Aggregate results and return
        final_rc_idx0 = np.concatenate(all_inliers_idx0)
        final_rc_idx1 = np.concatenate(all_inliers_idx1)
        residuals = np.concatenate(all_residuals)
        
        points_fg1 = points_poly.iloc[final_rc_idx0]
        points_fg2 = points_grid.iloc[final_rc_idx1]
        
        logger.info(f"Total aggregated inliers from all groups: {len(points_fg1)}")
        
        return points_fg1, points_fg2, residuals

    @log_execution_time
    def match_with_crosscheck(self, x0, x1):
        """
        Matches descriptors using cross-checking.

        Parameters:
            x0 (ndarray): Descriptors from the first image.
            x1 (ndarray): Descriptors from the second image.

        Returns:
            list: Matches that pass the cross-check.
        """
        bf = cv2.BFMatcher(self.norm, crossCheck=True)
        matches = bf.match(x0, x1)
        return list(matches)

    @log_execution_time
    def match_with_lowe_ratio(self, matches_bf_initial, x0, x1, pos0, pos1):
        """
        Applies a Lowe's ratio-like test using k=4 neighbors to find additional matches.
        Relies on the subsequent self.filter call for descriptor and spatial distance filtering.
        """
        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        search_params = {}
        knn_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # Get k nearest neighbors for each descriptor in x0 from x1
        all_knn_matches = knn_matcher.knnMatch(x0, x1, k=self.knn_k)

        new_matches_from_lowe_variant = []

        for mmm in all_knn_matches: 
            if len(mmm) < 2: # Need at least two matches to perform a ratio test
                continue

            # Iterate through the first k-1 candidates (m_candidate)
            # and compare each to its immediate successor (m_next)
            for midx, m_candidate in enumerate(mmm[:-1]):
                m_next = mmm[midx+1] # The DMatch object to compare against

                # Apply the Lowe's ratio test
                if m_candidate.distance < self.lowe_ratio * m_next.distance:
                    new_matches_from_lowe_variant.append(m_candidate)
                    break
        
        # Combine initial crosscheck matches with the new ones from the Lowe's variant
        combined_matches = list(matches_bf_initial) # Start with a copy
        existing_query_indices_from_bf = {m.queryIdx for m in matches_bf_initial}

        for m_lowe in new_matches_from_lowe_variant:
            if m_lowe.queryIdx not in existing_query_indices_from_bf:
                combined_matches.append(m_lowe)
        
        return combined_matches

    @log_execution_time
    def filter(self, matches, pos0, pos1, image_id=None, traj_ids0=None):
        bf_idx0 = np.array([m.queryIdx for m in matches]) # Indices into pos0 for ALL 'matches'
        bf_idx1 = np.array([m.trainIdx for m in matches]) # Indices into pos1 for ALL 'matches'
        
        num_initial = len(matches)
        num_spatial_passed = 0
        num_homography_inliers = 0
        inlier_ratio = 0.0  # initialize for logging

        def traj_ids_sample(indices):
            if traj_ids0 is None or indices is None:
                return None
            try:
                return list(traj_ids0[indices][:20])
            except Exception:
                return None

        def record_per_traj(tids, event_type, message, **extra):
            """Emit per-trajectory matcher_filter events with shared metrics."""
            if self.debug_recorder is None or tids is None:
                return
            unique_tids = [int(t) for t in np.unique(tids) if t is not None and not np.isnan(t)]
            for tid in unique_tids:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=tid,
                    step=image_id,
                    event_type=event_type,
                    message=message,
                    num_initial_matches=num_initial,
                    num_descriptor_passed=extra.get("num_descriptor_passed", num_descriptor_passed),
                    num_spatial_passed=extra.get("num_spatial_passed", num_spatial_passed),
                    num_homography_inliers=extra.get("num_homography_inliers", 0),
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                    inlier_ratio=extra.get("inlier_ratio", inlier_ratio),
                    residual_median=extra.get("residual_median"),
                    residual_mean=extra.get("residual_mean"),
                    num_traj_descriptor=extra.get("num_traj_descriptor"),
                    num_traj_spatial=extra.get("num_traj_spatial"),
                    num_traj_inliers=extra.get("num_traj_inliers"),
                    trajectory_ids_sample=extra.get("trajectory_ids_sample"),
                )

        # Filter by descriptor distance
        # descriptor_distance is already calculated for ALL 'matches' (initial crosscheck)
        descriptor_distance = np.array([m.distance for m in matches])
        gpi0_desc_filter_mask = descriptor_distance < self.descriptor_distance_max 
        dd_idx0 = bf_idx0[gpi0_desc_filter_mask] # Indices of points passing descriptor distance filter
        dd_idx1 = bf_idx1[gpi0_desc_filter_mask]
        num_descriptor_passed = dd_idx0.size
        num_traj_descriptor = len(np.unique(traj_ids0[dd_idx0])) if traj_ids0 is not None and dd_idx0.size > 0 else None
        traj_ids_initial = traj_ids0[bf_idx0] if traj_ids0 is not None else None
        
        if dd_idx0.size == 0: # No points passed descriptor distance filter
            if self.debug_recorder:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=None,
                    step=image_id,
                    event_type="failure",
                    message="no matches passed descriptor distance filter",
                    num_initial_matches=num_initial,
                    num_descriptor_passed=0,
                    num_spatial_passed=0,
                    num_homography_inliers=0,
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                    inlier_ratio=inlier_ratio,
                    num_traj_descriptor=num_traj_descriptor,
                    trajectory_ids_sample=traj_ids_sample(dd_idx0),
                )
                record_per_traj(
                    traj_ids_initial,
                    "failure",
                    "no matches passed descriptor distance filter",
                    num_descriptor_passed=0,
                    num_spatial_passed=0,
                    num_homography_inliers=0,
                    num_traj_descriptor=num_traj_descriptor,
                    trajectory_ids_sample=traj_ids_sample(dd_idx0),
                )
            if not self.use_model_estimation: return dd_idx0, dd_idx1, None
            return None, None, None
        
        # Filter by spatial distance
        # Calculate spatial distances ONLY for those that passed the descriptor distance filter
        current_spatial_distances = np.hypot(pos1[dd_idx1, 0] - pos0[dd_idx0, 0], 
                                            pos1[dd_idx1, 1] - pos0[dd_idx0, 1])
        gpi1_spatial_filter_mask = current_spatial_distances < self.spatial_distance_max # Mask relative to dd_idx arrays
        md_idx0 = dd_idx0[gpi1_spatial_filter_mask] # Indices of points passing spatial (and descriptor) filter
        md_idx1 = dd_idx1[gpi1_spatial_filter_mask]
        num_spatial_passed = md_idx0.size
        inlier_ratio = 0.0  # default, updated later
        num_traj_spatial = len(np.unique(traj_ids0[md_idx0])) if traj_ids0 is not None and md_idx0.size > 0 else None

        if md_idx0.size == 0:
            if self.debug_recorder:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=None,
                    step=image_id,
                    event_type="failure",
                    message="no matches passed spatial distance filter",
                    num_initial_matches=num_initial,
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=0,
                    num_homography_inliers=0,
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                    inlier_ratio=inlier_ratio,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                    trajectory_ids_sample=traj_ids_sample(md_idx0),
                )
                record_per_traj(
                    traj_ids0[dd_idx0] if traj_ids0 is not None else None,
                    "failure",
                    "no matches passed spatial distance filter",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=0,
                    num_homography_inliers=0,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                    trajectory_ids_sample=traj_ids_sample(md_idx0),
                )
            if not self.use_model_estimation:
                return md_idx0, md_idx1, None
            return None, None, None

        if not self.use_model_estimation:
            return md_idx0, md_idx1, None # md_idx0, md_idx1 are the final indices if no model estimation

        if md_idx0.size < 4:
            logger.warning("Warning: Insufficient matches for model estimation (minimum 4 required)")
            if self.debug_recorder:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=None,
                    step=image_id,
                    event_type="failure",
                    message="insufficient matches for model estimation",
                    num_initial_matches=num_initial,
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                    inlier_ratio=inlier_ratio,
                )
                record_per_traj(
                    traj_ids0[md_idx0] if traj_ids0 is not None else None,
                    "failure",
                    "insufficient matches for model estimation",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                )
            return None, None, None

        try:
            # H, inliers is your gpi2, applied to md_idx0/md_idx1
            # The `inliers` mask returned by findHomography is relative to the points fed into it (pos0[md_idx0], pos1[md_idx1])
            if self.estimation_method_name.upper() == "DEGENSAC":
                try:
                    import pydegensac
                    H, inliers_mask_homography_relative = pydegensac.findHomography(pos0[md_idx0], pos1[md_idx1], self.model_threshold)
                except ImportError:
                    logger.warning("pydegensac not found, falling back to cv2.USAC_MAGSAC")
                    self.estimation_method_name = "USAC_MAGSAC"
                    self.estimation_method = cv2.USAC_MAGSAC
                    H, inliers_mask_homography_relative = cv2.findHomography(pos0[md_idx0], pos1[md_idx1], 
                                                                            self.estimation_method, self.model_threshold)
            else:
                H, inliers_mask_homography_relative = cv2.findHomography(pos0[md_idx0], pos1[md_idx1], 
                                                                        self.estimation_method, self.model_threshold)

            if H is None:
                logger.warning("Warning: Model estimation failed")
                if self.debug_recorder:
                    self.debug_recorder.record_matcher_filter(
                        stage="matcher_filter",
                        trajectory_id=None,
                        step=image_id,
                        event_type="failure",
                        message="model estimation failed / H is None",
                        num_initial_matches=num_initial,
                        num_descriptor_passed=num_descriptor_passed,
                        num_spatial_passed=num_spatial_passed,
                        num_homography_inliers=0,
                        descriptor_distance_max=self.descriptor_distance_max,
                        spatial_distance_max=self.spatial_distance_max,
                        model_threshold=self.model_threshold,
                        min_homography_inliers=self.min_homography_inliers,
                        estimation_method=self.estimation_method_name,
                    )
                record_per_traj(
                    traj_ids0[md_idx0] if traj_ids0 is not None else None,
                    "failure",
                    "model estimation failed / H is None",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                )
                return None, None, None
            
            inliers_mask_homography_relative = inliers_mask_homography_relative.ravel().astype(bool)

            # Points KEPT by homography
            # these are indices into the original pos0/pos1 arrays
            rc_idx0 = md_idx0[inliers_mask_homography_relative]
            rc_idx1 = md_idx1[inliers_mask_homography_relative]
            num_homography_inliers = rc_idx0.size
            if num_spatial_passed > 0:
                inlier_ratio = num_homography_inliers / num_spatial_passed
            else:
                inlier_ratio = 0.0
            num_traj_inliers = len(np.unique(traj_ids0[rc_idx0])) if traj_ids0 is not None and rc_idx0.size > 0 else None

            if rc_idx0.size < self.min_homography_inliers:
                logger.warning(f"Warning: Not enough inliers after homography estimation (minimum {self.min_homography_inliers} required)")
                if self.debug_recorder:
                    self.debug_recorder.record_matcher_filter(
                        stage="matcher_filter",
                        trajectory_id=None,
                        step=image_id,
                        event_type="failure",
                        message="not enough inliers after homography estimation",
                        num_initial_matches=num_initial,
                        num_descriptor_passed=num_descriptor_passed,
                        num_spatial_passed=num_spatial_passed,
                        num_homography_inliers=num_homography_inliers,
                        descriptor_distance_max=self.descriptor_distance_max,
                        spatial_distance_max=self.spatial_distance_max,
                        model_threshold=self.model_threshold,
                        min_homography_inliers=self.min_homography_inliers,
                        estimation_method=self.estimation_method_name,
                        inlier_ratio=inlier_ratio,
                        num_traj_descriptor=num_traj_descriptor,
                        num_traj_spatial=num_traj_spatial,
                        num_traj_inliers=num_traj_inliers,
                        trajectory_ids_sample=traj_ids_sample(rc_idx0),
                    )
                record_per_traj(
                    traj_ids0[rc_idx0] if traj_ids0 is not None else None,
                    "failure",
                    "not enough inliers after homography estimation",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=num_homography_inliers,
                    inlier_ratio=inlier_ratio,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                    num_traj_inliers=num_traj_inliers,
                    trajectory_ids_sample=traj_ids_sample(rc_idx0),
                )
                return None, None, None
                
            model = self.model(H) # Assuming self.model is AffineTransform class
            residuals = model.residuals(pos0[rc_idx0], pos1[rc_idx1])
            residual_median = float(np.median(residuals)) if residuals is not None and len(residuals) > 0 else None
            residual_mean = float(np.mean(residuals)) if residuals is not None and len(residuals) > 0 else None
            logger.info(
                f'{self.estimation_method_name}: Found {rc_idx0.size} inliers from {len(matches)} initial candidates.'
            )
            
            # Record successful filtering
            if self.debug_recorder:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=None,
                    step=image_id,
                    event_type="info",
                    message="filter successful",
                    num_initial_matches=num_initial,
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=num_homography_inliers,
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                    inlier_ratio=inlier_ratio,
                    residual_median=residual_median,
                    residual_mean=residual_mean,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                    num_traj_inliers=num_traj_inliers,
                    trajectory_ids_sample=traj_ids_sample(rc_idx0),
                )
                record_per_traj(
                    traj_ids0[rc_idx0] if traj_ids0 is not None else None,
                    "info",
                    "filter successful",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=num_homography_inliers,
                    inlier_ratio=inlier_ratio,
                    residual_median=residual_median,
                    residual_mean=residual_mean,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                    num_traj_inliers=num_traj_inliers,
                    trajectory_ids_sample=traj_ids_sample(rc_idx0),
                )
            
            if self.plot:
                self.plot_quiver(pos0[rc_idx0], pos1[rc_idx1], residuals)
                
            return rc_idx0, rc_idx1, residuals

        except cv2.error as e:
            logger.error(f"Warning: OpenCV error during model estimation: {str(e)}")
            if self.debug_recorder:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=None,
                    step=image_id,
                    event_type="failure",
                    message=f"OpenCV error during model estimation: {str(e)}",
                    num_initial_matches=num_initial,
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                )
                record_per_traj(
                    traj_ids0[md_idx0] if traj_ids0 is not None else None,
                    "failure",
                    f"OpenCV error during model estimation: {str(e)}",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                )
            return None, None, None
        except Exception as e:
            logger.error(f"Warning: Unexpected error during model estimation: {str(e)}")
            if self.debug_recorder:
                self.debug_recorder.record_matcher_filter(
                    stage="matcher_filter",
                    trajectory_id=None,
                    step=image_id,
                    event_type="failure",
                    message=f"Unexpected error during model estimation: {str(e)}",
                    num_initial_matches=num_initial,
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    descriptor_distance_max=self.descriptor_distance_max,
                    spatial_distance_max=self.spatial_distance_max,
                    model_threshold=self.model_threshold,
                    min_homography_inliers=self.min_homography_inliers,
                    estimation_method=self.estimation_method_name,
                )
                record_per_traj(
                    traj_ids0[md_idx0] if traj_ids0 is not None else None,
                    "failure",
                    f"Unexpected error during model estimation: {str(e)}",
                    num_descriptor_passed=num_descriptor_passed,
                    num_spatial_passed=num_spatial_passed,
                    num_homography_inliers=0,
                    num_traj_descriptor=num_traj_descriptor,
                    num_traj_spatial=num_traj_spatial,
                )
            return None, None, None
