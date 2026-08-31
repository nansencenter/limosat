# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import cv2
import pandas as pd
from skimage.transform import AffineTransform
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.spatial import cKDTree
from .utils import log_execution_time, logger

class Matcher:
    CANDIDATE_SELECTION_MODES = {
        "global_descriptor_first",
        "global_then_local_physics_fallback",
    }
    MODEL_ESTIMATORS = {
        "legacy_homography",
        "configured_affine",
        "homography_affine_union",
        "homography_kilometre_coordinates",
    }

    def __init__(self,
                 # General matching parameters
                 norm=cv2.NORM_HAMMING2,
                 descriptor_distance_max=120,
                 spatial_distance_max=100000,
                 max_speed_m_per_day=None,

                 # Homography estimation parameters
                 model=AffineTransform,
                 model_threshold=10000,
                 use_model_estimation=True,
                 estimation_method="USAC_MAGSAC",
                 min_homography_inliers=10,
                 model_coordinate_scale_m=1.0,
                 model_estimator="legacy_homography",

                 # Lowe's ratio test parameter
                 lowe_ratio=0.9,
                 knn_k=4,
                 candidate_selection="global_descriptor_first",

                 # Visualization
                 plot=False,

                 # Optional experiment-only append-only audit sink
                 audit_sink=None):

        # General matching parameters
        self.norm = norm
        self.descriptor_distance_max = descriptor_distance_max
        self.spatial_distance_max = spatial_distance_max
        self.max_speed_m_per_day = max_speed_m_per_day

        # Homography estimation parameters
        self.model = model
        self.model_threshold = model_threshold
        self.use_model_estimation = use_model_estimation
        self.min_homography_inliers = min_homography_inliers
        self.model_coordinate_scale_m = float(model_coordinate_scale_m)
        if (
            not np.isfinite(self.model_coordinate_scale_m)
            or self.model_coordinate_scale_m <= 0
        ):
            raise ValueError("model_coordinate_scale_m must be finite and positive")
        if model_estimator not in self.MODEL_ESTIMATORS:
            raise ValueError(
                f"Unsupported model_estimator {model_estimator!r}; expected one of "
                f"{sorted(self.MODEL_ESTIMATORS)}"
            )
        if (
            model_estimator in {"configured_affine", "homography_affine_union"}
            and model is not AffineTransform
        ):
            raise ValueError(f"{model_estimator} requires skimage AffineTransform")
        self.model_estimator = model_estimator
        if model_estimator == "homography_kilometre_coordinates":
            if self.model_coordinate_scale_m not in {1.0, 1_000.0}:
                raise ValueError(
                    "homography_kilometre_coordinates fixes "
                    "model_coordinate_scale_m at 1000"
                )
            self.model_coordinate_scale_m = 1_000.0
        elif (
            model_estimator != "legacy_homography"
            and self.model_coordinate_scale_m != 1.0
        ):
            raise ValueError(
                "model_coordinate_scale_m is currently supported only for "
                "homography estimation"
            )

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
        if candidate_selection not in self.CANDIDATE_SELECTION_MODES:
            raise ValueError(
                f"Unsupported candidate_selection {candidate_selection!r}; expected one of "
                f"{sorted(self.CANDIDATE_SELECTION_MODES)}"
            )
        self.candidate_selection = candidate_selection
        self.plot = plot
        self.audit_sink = audit_sink

    def plot_quiver(self, pos0, pos1, dist):
        u = pos1[:, 0] - pos0[:, 0]
        v = pos1[:, 1] - pos0[:, 1]
        spd = np.hypot(u,v)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(dist, spd, '.', alpha=0.5)
        qui = axs[1].quiver(pos0[:, 0], pos0[:, 1], pos1[:, 0] - pos0[:, 0], pos1[:, 1] - pos0[:, 1], dist, cmap='jet', clim=np.percentile(dist, [1, 99]), angles='xy', scale_units='xy', scale=1)
        plt.colorbar(qui, ax=axs[0], shrink=0.5)
        plt.show()

    @staticmethod
    def _single_time_value(times, context):
        valid_times = pd.Series(times).dropna().unique()
        if len(valid_times) == 0:
            return None
        if len(valid_times) != 1:
            raise ValueError(f"{context} must contain exactly one timestamp.")
        return pd.Timestamp(valid_times[0])

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
        candidate_matches = self.match_with_lowe_ratio(base_matches, x0, x1, pos0, pos1)
        crosscheck_query_indices = {match.queryIdx for match in base_matches}
        logger.info(f"Total candidate matches after cross-check and Lowe's ratio: {len(candidate_matches)}")

        # 3. Group all candidate matches by their source image_id
        matches_by_group = defaultdict(list)
        for match in candidate_matches:
            # Check if queryIdx is valid before accessing points_poly
            if match.queryIdx < len(points_poly):
                image_id = points_poly.iloc[match.queryIdx]['image_id']
                matches_by_group[image_id].append(match)
            else:
                logger.warning(f"Invalid queryIdx {match.queryIdx} found in a match. Skipping.")


        # 4. Loop through each group and apply the 'filter' function
        all_inliers_idx0, all_inliers_idx1, all_residuals = [], [], []
        current_time = None
        if 'time' in points_grid.columns and len(points_grid) > 0:
            current_time = self._single_time_value(points_grid['time'], "Current frame points")
        source_image_ids = points_poly["image_id"].to_numpy()
        target_tree = (
            cKDTree(pos1)
            if self.candidate_selection == "global_then_local_physics_fallback"
            else None
        )
        for image_id in pd.unique(source_image_ids):
            group_matches = matches_by_group.get(image_id, [])
            group_query_idx = np.flatnonzero(source_image_ids == image_id)
            previous_time = None
            if current_time is not None and len(group_query_idx) > 0 and 'time' in points_poly.columns:
                previous_time = self._single_time_value(
                    points_poly.iloc[group_query_idx]['time'],
                    f"Matches for previous image_id {image_id}",
                )
            max_distance_m = self.motion_distance_limit(previous_time, current_time)
            fallback_origins = {}
            if target_tree is not None:
                group_matches, fallback_origins = self._add_local_physics_fallback(
                    group_matches=group_matches,
                    group_query_indices=group_query_idx,
                    x0=x0,
                    x1=x1,
                    pos0=pos0,
                    pos1=pos1,
                    target_tree=target_tree,
                    max_distance_m=max_distance_m,
                )
            if not group_matches:
                continue
            rc_idx0_group, rc_idx1_group, residuals_group = self.filter(
                group_matches,
                pos0,
                pos1,
                max_distance_m=max_distance_m,
                audit_context={
                    "source_image_id": image_id,
                    "target_image_id": (
                        int(points_grid.iloc[0]["image_id"])
                        if "image_id" in points_grid.columns and len(points_grid)
                        else None
                    ),
                    "source_trajectory_ids": points_poly["trajectory_id"].to_numpy(),
                    "candidate_origins": {
                        (match.queryIdx, match.trainIdx): (
                            "crosscheck"
                            if match.queryIdx in crosscheck_query_indices
                            else "lowe_ratio"
                        )
                        for match in group_matches
                    }
                    | fallback_origins,
                },
            )
            
            if rc_idx0_group is not None and rc_idx0_group.size > 0:
                all_inliers_idx0.append(rc_idx0_group)
                all_inliers_idx1.append(rc_idx1_group)
                all_residuals.append(residuals_group)

        # 5. Check if any valid groups were found at all
        if not all_inliers_idx0:
            logger.warning("No valid inlier groups found after filtering.")
            return None, None, None

        # 6. Aggregate results and return
        final_rc_idx0 = np.concatenate(all_inliers_idx0)
        final_rc_idx1 = np.concatenate(all_inliers_idx1)
        residuals = np.concatenate(all_residuals)
        
        points_fg1 = points_poly.iloc[final_rc_idx0]
        points_fg2 = points_grid.iloc[final_rc_idx1]
        
        logger.info(f"Total aggregated inliers from all groups: {len(points_fg1)}")
        
        return points_fg1, points_fg2, residuals

    def _add_local_physics_fallback(
        self,
        group_matches,
        group_query_indices,
        x0,
        x1,
        pos0,
        pos1,
        target_tree,
        max_distance_m,
    ):
        """Add one local descriptor candidate when the global choice violates physics.

        Existing global candidates are retained for audit. A fallback is added only
        when none for that source passes both descriptor and motion limits.
        """
        if (
            max_distance_m is None
            or not np.isfinite(max_distance_m)
            or max_distance_m < 0
        ):
            raise ValueError(
                "global_then_local_physics_fallback requires a finite non-negative motion limit"
            )

        result = list(group_matches)
        origins = {}
        matches_by_query = defaultdict(list)
        for match in group_matches:
            matches_by_query[int(match.queryIdx)].append(match)

        matcher = cv2.BFMatcher(self.norm, crossCheck=False)
        for query_index in np.asarray(group_query_indices, dtype=int):
            existing = matches_by_query.get(int(query_index), [])
            has_valid_global = any(
                match.distance < self.descriptor_distance_max
                and np.hypot(
                    pos1[match.trainIdx, 0] - pos0[query_index, 0],
                    pos1[match.trainIdx, 1] - pos0[query_index, 1],
                )
                <= max_distance_m
                for match in existing
            )
            if has_valid_global:
                continue

            local_indices = np.sort(
                np.asarray(
                    target_tree.query_ball_point(pos0[query_index], max_distance_m),
                    dtype=int,
                )
            )
            if local_indices.size == 0:
                continue
            local_match = matcher.match(
                x0[query_index : query_index + 1], x1[local_indices]
            )[0]
            target_index = int(local_indices[local_match.trainIdx])
            if any(match.trainIdx == target_index for match in existing):
                continue
            fallback = cv2.DMatch(
                int(query_index), target_index, 0, float(local_match.distance)
            )
            result.append(fallback)
            origins[(int(query_index), target_index)] = "local_physics_fallback"
        return result, origins

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
        if x1 is None or x1.shape[0] < self.knn_k:
            n_train = 0 if x1 is None else int(x1.shape[0])
            logger.debug(
                f"Skipping Lowe's ratio test: train descriptors ({n_train}) < knn_k ({self.knn_k})."
            )
            return list(matches_bf_initial)

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

    def motion_distance_limit(self, previous_time=None, current_time=None):
        if self.max_speed_m_per_day is not None and previous_time is not None and current_time is not None:
            delta_t_days = max((current_time - previous_time).total_seconds() / 86400.0, 0.0)
            return float(self.max_speed_m_per_day) * delta_t_days
        if self.spatial_distance_max is None:
            return None
        return float(self.spatial_distance_max)

    def _emit_filter_audit(
        self,
        matches,
        pos0,
        pos1,
        descriptor_distance,
        descriptor_pass,
        spatial_distance,
        spatial_pass,
        model_inlier,
        accepted,
        residual,
        group_status,
        max_distance_m,
        audit_context,
    ):
        if self.audit_sink is None or audit_context is None:
            return
        source_image_id = audit_context.get("source_image_id")
        target_image_id = audit_context.get("target_image_id")
        trajectory_ids = audit_context.get("source_trajectory_ids")
        origins = audit_context.get("candidate_origins", {})
        model_inlier_count = int(np.sum(model_inlier))
        records = []
        for index, match in enumerate(matches):
            query_index = int(match.queryIdx)
            train_index = int(match.trainIdx)
            trajectory_id = (
                int(trajectory_ids[query_index]) if trajectory_ids is not None else None
            )
            if not descriptor_pass[index]:
                reason = "descriptor_distance"
            elif not spatial_pass[index]:
                reason = "motion_distance"
            elif accepted[index]:
                reason = "accepted"
            elif group_status == "model_success":
                reason = "model_outlier"
            else:
                reason = group_status
            records.append(
                {
                    "candidate_id": (
                        f"{target_image_id}:{source_image_id}:{trajectory_id}:"
                        f"{query_index}:{train_index}"
                    ),
                    "source_image_id": source_image_id,
                    "target_image_id": target_image_id,
                    "trajectory_id": trajectory_id,
                    "query_index": query_index,
                    "train_index": train_index,
                    "candidate_origin": origins.get(
                        (query_index, train_index),
                        origins.get(query_index, "unknown"),
                    ),
                    "source_x": pos0[query_index, 0],
                    "source_y": pos0[query_index, 1],
                    "target_x": pos1[train_index, 0],
                    "target_y": pos1[train_index, 1],
                    "descriptor_distance": descriptor_distance[index],
                    "descriptor_pass": descriptor_pass[index],
                    "spatial_distance_m": spatial_distance[index],
                    "max_distance_m": max_distance_m,
                    "motion_pass": spatial_pass[index],
                    "model_inlier": model_inlier[index],
                    "model_residual_m": residual[index],
                    "accepted": accepted[index],
                    "rejection_reason": reason,
                    "group_status": group_status,
                    "group_candidate_count": len(matches),
                    "group_model_inlier_count": model_inlier_count,
                }
            )
        self.audit_sink.emit("matcher_candidates", records)

    @log_execution_time
    def filter(self, matches, pos0, pos1, max_distance_m=None, audit_context=None):
        bf_idx0 = np.array([m.queryIdx for m in matches]) # Indices into pos0 for ALL 'matches'
        bf_idx1 = np.array([m.trainIdx for m in matches]) # Indices into pos1 for ALL 'matches'

        # Filter by descriptor distance
        # descriptor_distance is already calculated for ALL 'matches' (initial crosscheck)
        descriptor_distance = np.array([m.distance for m in matches])
        gpi0_desc_filter_mask = descriptor_distance < self.descriptor_distance_max 
        dd_idx0 = bf_idx0[gpi0_desc_filter_mask] # Indices of points passing descriptor distance filter
        dd_idx1 = bf_idx1[gpi0_desc_filter_mask]
        spatial_distances_all = np.hypot(
            pos1[bf_idx1, 0] - pos0[bf_idx0, 0],
            pos1[bf_idx1, 1] - pos0[bf_idx0, 1],
        )
        spatial_pass_all = gpi0_desc_filter_mask.copy()
        model_inlier_all = np.zeros(len(matches), dtype=bool)
        accepted_all = np.zeros(len(matches), dtype=bool)
        residual_all = np.full(len(matches), np.nan, dtype=float)

        def emit(group_status):
            self._emit_filter_audit(
                matches,
                pos0,
                pos1,
                descriptor_distance,
                gpi0_desc_filter_mask,
                spatial_distances_all,
                spatial_pass_all,
                model_inlier_all,
                accepted_all,
                residual_all,
                group_status,
                max_distance_m,
                audit_context,
            )

        if dd_idx0.size == 0: # No points passed descriptor distance filter
            spatial_pass_all[:] = False
            emit("no_descriptor_candidates")
            if not self.use_model_estimation: return dd_idx0, dd_idx1, None
            return None, None, None
        
        if max_distance_m is not None and np.isfinite(max_distance_m) and max_distance_m >= 0:
            current_spatial_distances = np.hypot(
                pos1[dd_idx1, 0] - pos0[dd_idx0, 0],
                pos1[dd_idx1, 1] - pos0[dd_idx0, 1],
            )
            gpi1_spatial_filter_mask = current_spatial_distances <= max_distance_m
            md_idx0 = dd_idx0[gpi1_spatial_filter_mask]
            md_idx1 = dd_idx1[gpi1_spatial_filter_mask]
            spatial_pass_all = gpi0_desc_filter_mask & (
                spatial_distances_all <= max_distance_m
            )
        else:
            md_idx0 = dd_idx0
            md_idx1 = dd_idx1

        if not self.use_model_estimation:
            accepted_all = spatial_pass_all.copy()
            emit("model_disabled")
            return md_idx0, md_idx1, None # md_idx0, md_idx1 are the final indices if no model estimation

        if md_idx0.size < 4:
            logger.warning("Warning: Insufficient matches for model estimation (minimum 4 required)")
            emit("insufficient_model_samples")
            return None, None, None

        try:
            # H, inliers is your gpi2, applied to md_idx0/md_idx1
            # The `inliers` mask returned by findHomography is relative to the points fed into it (pos0[md_idx0], pos1[md_idx1])
            model_matrices = []
            if self.model_estimator == "configured_affine":
                if self.estimation_method_name.upper() == "DEGENSAC":
                    raise ValueError("DEGENSAC does not provide affine estimation")
                affine, inliers_mask_homography_relative = cv2.estimateAffine2D(
                    pos0[md_idx0],
                    pos1[md_idx1],
                    method=self.estimation_method,
                    ransacReprojThreshold=self.model_threshold,
                )
                H = (
                    np.vstack((affine, np.array([0.0, 0.0, 1.0])))
                    if affine is not None
                    else None
                )
                if H is not None:
                    model_matrices.append(H)
            elif self.model_estimator == "homography_affine_union":
                if self.estimation_method_name.upper() == "DEGENSAC":
                    raise ValueError("DEGENSAC does not provide affine estimation")
                homography, homography_mask = cv2.findHomography(
                    pos0[md_idx0],
                    pos1[md_idx1],
                    self.estimation_method,
                    self.model_threshold,
                )
                affine, affine_mask = cv2.estimateAffine2D(
                    pos0[md_idx0],
                    pos1[md_idx1],
                    method=self.estimation_method,
                    ransacReprojThreshold=self.model_threshold,
                )
                masks = []
                if homography is not None and homography_mask is not None:
                    model_matrices.append(homography)
                    masks.append(homography_mask.ravel().astype(bool))
                if affine is not None and affine_mask is not None:
                    model_matrices.append(
                        np.vstack((affine, np.array([0.0, 0.0, 1.0])))
                    )
                    masks.append(affine_mask.ravel().astype(bool))
                H = model_matrices[0] if model_matrices else None
                inliers_mask_homography_relative = (
                    np.logical_or.reduce(masks) if masks else None
                )
            elif self.model_estimator in {
                "legacy_homography",
                "homography_kilometre_coordinates",
            }:
                coordinate_scale_m = self.model_coordinate_scale_m
                source_model = pos0[md_idx0] / coordinate_scale_m
                target_model = pos1[md_idx1] / coordinate_scale_m
                threshold_model = self.model_threshold / coordinate_scale_m
                if self.estimation_method_name.upper() == "DEGENSAC":
                    try:
                        import pydegensac

                        scaled_homography, inliers_mask_homography_relative = (
                            pydegensac.findHomography(
                                source_model,
                                target_model,
                                threshold_model,
                            )
                        )
                    except ImportError:
                        logger.warning(
                            "pydegensac not found, falling back to cv2.USAC_MAGSAC"
                        )
                        self.estimation_method_name = "USAC_MAGSAC"
                        self.estimation_method = cv2.USAC_MAGSAC
                        scaled_homography, inliers_mask_homography_relative = (
                            cv2.findHomography(
                                source_model,
                                target_model,
                                self.estimation_method,
                                threshold_model,
                            )
                        )
                else:
                    scaled_homography, inliers_mask_homography_relative = (
                        cv2.findHomography(
                            source_model,
                            target_model,
                            self.estimation_method,
                            threshold_model,
                        )
                    )
                if scaled_homography is None:
                    H = None
                else:
                    input_to_model = np.diag(
                        [1.0 / coordinate_scale_m, 1.0 / coordinate_scale_m, 1.0]
                    )
                    H = (
                        np.linalg.inv(input_to_model)
                        @ scaled_homography
                        @ input_to_model
                    )
                    model_matrices.append(H)

            if H is None or inliers_mask_homography_relative is None:
                logger.warning("Warning: Model estimation failed")
                emit("model_estimation_failed")
                return None, None, None

            if not model_matrices:
                model_matrices.append(H)
            
            inliers_mask_homography_relative = inliers_mask_homography_relative.ravel().astype(bool)

            # Points KEPT by homography
            # these are indices into the original pos0/pos1 arrays
            rc_idx0 = md_idx0[inliers_mask_homography_relative]
            rc_idx1 = md_idx1[inliers_mask_homography_relative]

            spatial_candidate_indices = np.flatnonzero(spatial_pass_all)
            model_inlier_all[
                spatial_candidate_indices[inliers_mask_homography_relative]
            ] = True
            model_residuals = np.vstack(
                [
                    self.model(matrix).residuals(
                        pos0[md_idx0], pos1[md_idx1]
                    )
                    for matrix in model_matrices
                ]
            )
            combined_residuals = np.min(model_residuals, axis=0)
            if self.audit_sink is not None:
                residual_all[spatial_candidate_indices] = combined_residuals

            if rc_idx0.size < self.min_homography_inliers:
                logger.warning(f"Warning: Not enough inliers after homography estimation (minimum {self.min_homography_inliers} required)")
                emit("below_minimum_model_inliers")
                return None, None, None
                
            residuals = combined_residuals[inliers_mask_homography_relative]
            accepted_all = model_inlier_all.copy()
            emit("model_success")
            logger.info(
                f'{self.estimation_method_name}/{self.model_estimator}: Found '
                f'{rc_idx0.size} inliers from {len(matches)} initial candidates.'
            )
            
            if self.plot:
                self.plot_quiver(pos0[rc_idx0], pos1[rc_idx1], residuals)
                
            return rc_idx0, rc_idx1, residuals

        except cv2.error as e:
            logger.error(f"Warning: OpenCV error during model estimation: {str(e)}")
            emit("model_opencv_error")
            return None, None, None
        except Exception as e:
            logger.error(f"Warning: Unexpected error during model estimation: {str(e)}")
            emit("model_unexpected_error")
            return None, None, None
