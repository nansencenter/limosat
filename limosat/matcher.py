# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import cv2
from skimage.transform import AffineTransform
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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
                 plot=False):

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

    def plot_filter_steps(self,
                          all_pos0, all_pos1,
                          desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                          spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1,
                          post_pos0=None, post_pos1=None, post_residuals=None):
        """
        Multi-panel figure showing pipeline steps with shared axes and a shared legend:
        Panels:
          (a) Initial matching        -> shows all initial matches in the valid color
          (b) Descriptor distance filter -> valid (same highlight color) vs removed (grey)
          (c) Spatial distance filter    -> valid (same highlight color) vs removed (grey)
          (d) MAGSAC inliers             -> post_pos colored by residuals (viridis)
        Legend: two entries for the whole figure: 'valid matches' (proceed to next step) and 'removed' (grey).
        """
        def has(a): return (a is not None) and (getattr(a, "size", 0) > 0)

        fig, axs = plt.subplots(1, 4, figsize=(22, 6), sharex=True, sharey=True)
        titles = [
            "(a) Initial matching",
            "(b) Descriptor distance filter",
            "(c) Spatial distance filter",
            "(d) MAGSAC inliers"
        ]

        for ax, title in zip(axs, titles):
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("x")
        axs[0].set_ylabel("y")

        # Color choices: use the same highlight color for valid matches across the first three panels
        valid_color = 'tab:blue'
        removed_color = 'lightgray'

        # Panel (a): initial matching — draw all initial matches in the valid color (no split)
        if has(all_pos0) and has(all_pos1):
            u_all = all_pos1[:, 0] - all_pos0[:, 0]
            v_all = all_pos1[:, 1] - all_pos0[:, 1]
            axs[0].quiver(all_pos0[:, 0], all_pos0[:, 1], u_all, v_all,
                          color=valid_color, alpha=0.85, angles='xy', scale_units='xy', scale=1, linewidth=0.7)

        # Panel (b): descriptor filter
        # - survivors: desc_survive_pos (valid_color)
        # - removed: desc_removed_pos (grey)
        if has(desc_removed_pos0):
            u_dr = desc_removed_pos1[:, 0] - desc_removed_pos0[:, 0]
            v_dr = desc_removed_pos1[:, 1] - desc_removed_pos0[:, 1]
            axs[1].quiver(desc_removed_pos0[:, 0], desc_removed_pos0[:, 1], u_dr, v_dr,
                          color=removed_color, alpha=0.6, angles='xy', scale_units='xy', scale=1, linewidth=0.4)
        if has(desc_survive_pos0):
            u_ds = desc_survive_pos1[:, 0] - desc_survive_pos0[:, 0]
            v_ds = desc_survive_pos1[:, 1] - desc_survive_pos0[:, 1]
            axs[1].quiver(desc_survive_pos0[:, 0], desc_survive_pos0[:, 1], u_ds, v_ds,
                          color=valid_color, alpha=0.85, angles='xy', scale_units='xy', scale=1, linewidth=0.7)

        # Panel (c): spatial filter
        # - survivors: spatial_survive_pos (valid_color)
        # - removed: spatial_removed_pos (grey)
        if has(spatial_removed_pos0):
            u_sr = spatial_removed_pos1[:, 0] - spatial_removed_pos0[:, 0]
            v_sr = spatial_removed_pos1[:, 1] - spatial_removed_pos0[:, 1]
            axs[2].quiver(spatial_removed_pos0[:, 0], spatial_removed_pos0[:, 1], u_sr, v_sr,
                          color=removed_color, alpha=0.6, angles='xy', scale_units='xy', scale=1, linewidth=0.4)
        if has(spatial_survive_pos0):
            u_ss = spatial_survive_pos1[:, 0] - spatial_survive_pos0[:, 0]
            v_ss = spatial_survive_pos1[:, 1] - spatial_survive_pos0[:, 1]
            axs[2].quiver(spatial_survive_pos0[:, 0], spatial_survive_pos0[:, 1], u_ss, v_ss,
                          color=valid_color, alpha=0.85, angles='xy', scale_units='xy', scale=1, linewidth=0.7)

        # Panel (d): final model inliers colored by residuals
        if has(post_pos0) and has(post_pos1) and has(post_residuals):
            u_post = post_pos1[:, 0] - post_pos0[:, 0]
            v_post = post_pos1[:, 1] - post_pos0[:, 1]
            clim = np.percentile(post_residuals, [1, 99]) if post_residuals.size > 2 else (post_residuals.min(), post_residuals.max())
            qui = axs[3].quiver(post_pos0[:, 0], post_pos0[:, 1], u_post, v_post, post_residuals,
                                cmap='viridis', clim=clim, angles='xy', scale_units='xy', scale=1)
            cbar = fig.colorbar(qui, ax=axs[3], shrink=0.9)
            cbar.set_label('Residuals (m)', fontsize=10)

        # Create a shared legend with exactly two entries: valid matches (proceed) and removed (grey)
        proxy_valid = Line2D([0], [0], color=valid_color, linewidth=6, label='valid matches')
        proxy_removed = Line2D([0], [0], color=removed_color, linewidth=6, label='removed')
        axs[-1].legend(handles=[proxy_valid, proxy_removed], loc='lower right', fontsize=10, frameon=False)

        plt.tight_layout()
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
        candidate_matches = self.match_with_lowe_ratio(base_matches, x0, x1, pos0, pos1)
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
        for image_id, group_matches in matches_by_group.items():
            rc_idx0_group, rc_idx1_group, residuals_group = self.filter(group_matches, pos0, pos1)
            
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
    def filter(self, matches, pos0, pos1):
        bf_idx0 = np.array([m.queryIdx for m in matches]) # Indices into pos0 for ALL 'matches'
        bf_idx1 = np.array([m.trainIdx for m in matches]) # Indices into pos1 for ALL 'matches'

        # Filter by descriptor distance
        # descriptor_distance is already calculated for ALL 'matches' (initial crosscheck)
        descriptor_distance = np.array([m.distance for m in matches])
        gpi0_desc_filter_mask = descriptor_distance < self.descriptor_distance_max 
        dd_idx0 = bf_idx0[gpi0_desc_filter_mask] # Indices of points passing descriptor distance filter
        dd_idx1 = bf_idx1[gpi0_desc_filter_mask]
        if dd_idx0.size == 0: # No points passed descriptor distance filter
            if not self.use_model_estimation: return dd_idx0, dd_idx1, None
            # Prepare arrays for plotting (initial + descriptor results) if requested
            if self.plot:
                all_pos0 = pos0[bf_idx0] if bf_idx0.size else np.empty((0,2))
                all_pos1 = pos1[bf_idx1] if bf_idx1.size else np.empty((0,2))
                desc_survive_pos0 = np.empty((0,2))
                desc_survive_pos1 = np.empty((0,2))
                desc_removed_pos0 = all_pos0.copy()
                desc_removed_pos1 = all_pos1.copy()
                spatial_survive_pos0 = np.empty((0,2))
                spatial_survive_pos1 = np.empty((0,2))
                spatial_removed_pos0 = np.empty((0,2))
                spatial_removed_pos1 = np.empty((0,2))
                self.plot_filter_steps(all_pos0, all_pos1,
                                       desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                       spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
            return None, None, None
        
        # Filter by spatial distance
        # Calculate spatial distances ONLY for those that passed the descriptor distance filter
        current_spatial_distances = np.hypot(pos1[dd_idx1, 0] - pos0[dd_idx0, 0], 
                                            pos1[dd_idx1, 1] - pos0[dd_idx0, 1])
        gpi1_spatial_filter_mask = current_spatial_distances < self.spatial_distance_max # Mask relative to dd_idx arrays
        md_idx0 = dd_idx0[gpi1_spatial_filter_mask] # Indices of points passing spatial (and descriptor) filter
        md_idx1 = dd_idx1[gpi1_spatial_filter_mask]

        # Prepare arrays and index sets for plotting
        all_idx0, all_idx1 = bf_idx0, bf_idx1
        desc_idx0, desc_idx1 = dd_idx0, dd_idx1
        spatial_idx0, spatial_idx1 = md_idx0, md_idx1

        # Removed at descriptor stage: present in all but not in desc
        desc_removed_idx0 = np.setdiff1d(all_idx0, desc_idx0, assume_unique=False)
        # Build map from query index to train index for all matches
        query_to_train = dict(zip(bf_idx0.tolist(), bf_idx1.tolist()))
        if desc_removed_idx0.size:
            desc_removed_idx1 = np.array([query_to_train[i] for i in desc_removed_idx0])
        else:
            desc_removed_idx1 = np.empty((0,), dtype=int)

        # Removed at spatial stage: present in desc but not in spatial
        spatial_removed_idx0 = np.setdiff1d(desc_idx0, spatial_idx0, assume_unique=False)
        if spatial_removed_idx0.size:
            spatial_removed_idx1 = np.array([query_to_train[i] for i in spatial_removed_idx0])
        else:
            spatial_removed_idx1 = np.empty((0,), dtype=int)

        # Prepare position arrays
        # all positions (for panel a fallback)
        all_pos0 = pos0[all_idx0] if all_idx0.size else np.empty((0,2))
        all_pos1 = pos1[all_idx1] if all_idx1.size else np.empty((0,2))

        # descriptor survivors and removed
        desc_survive_pos0 = pos0[desc_idx0] if desc_idx0.size else np.empty((0,2))
        desc_survive_pos1 = pos1[desc_idx1] if desc_idx1.size else np.empty((0,2))
        desc_removed_pos0 = pos0[desc_removed_idx0] if desc_removed_idx0.size else np.empty((0,2))
        desc_removed_pos1 = pos1[desc_removed_idx1] if desc_removed_idx1.size else np.empty((0,2))

        # spatial survivors and removed
        spatial_survive_pos0 = pos0[spatial_idx0] if spatial_idx0.size else np.empty((0,2))
        spatial_survive_pos1 = pos1[spatial_idx1] if spatial_idx1.size else np.empty((0,2))
        spatial_removed_pos0 = pos0[spatial_removed_idx0] if spatial_removed_idx0.size else np.empty((0,2))
        spatial_removed_pos1 = pos1[spatial_removed_idx1] if spatial_removed_idx1.size else np.empty((0,2))

        if not self.use_model_estimation:
            if self.plot:
                self.plot_filter_steps(all_pos0, all_pos1,
                                       desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                       spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
            return md_idx0, md_idx1, None # md_idx0, md_idx1 are the final indices if no model estimation

        if md_idx0.size < 4:
            logger.warning("Warning: Insufficient matches for model estimation (minimum 4 required)")
            if self.plot:
                self.plot_filter_steps(all_pos0, all_pos1,
                                       desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                       spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
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
                if self.plot:
                    self.plot_filter_steps(all_pos0, all_pos1,
                                           desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                           spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
                return None, None, None
            
            inliers_mask_homography_relative = inliers_mask_homography_relative.ravel().astype(bool)

            # Points KEPT by homography
            # these are indices into the original pos0/pos1 arrays
            rc_idx0 = md_idx0[inliers_mask_homography_relative]
            rc_idx1 = md_idx1[inliers_mask_homography_relative]

            if rc_idx0.size < self.min_homography_inliers:
                logger.warning(f"Warning: Not enough inliers after homography estimation (minimum {self.min_homography_inliers} required)")
                if self.plot:
                    self.plot_filter_steps(all_pos0, all_pos1,
                                           desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                           spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
                return None, None, None
                
            model = self.model(H) # Assuming self.model is AffineTransform class
            residuals = model.residuals(pos0[rc_idx0], pos1[rc_idx1])
            logger.info(
                f'{self.estimation_method_name}: Found {rc_idx0.size} inliers from {len(matches)} initial candidates.'
            )
            
            if self.plot:
                # final post arrays
                post_pos0 = pos0[rc_idx0]
                post_pos1 = pos1[rc_idx1]
                post_residuals = residuals
                self.plot_filter_steps(all_pos0, all_pos1,
                                       desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                       spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1,
                                       post_pos0, post_pos1, post_residuals)
                
            return rc_idx0, rc_idx1, residuals

        except cv2.error as e:
            logger.error(f"Warning: OpenCV error during model estimation: {str(e)}")
            if self.plot:
                self.plot_filter_steps(all_pos0, all_pos1,
                                       desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                       spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
            return None, None, None
        except Exception as e:
            logger.error(f"Warning: Unexpected error during model estimation: {str(e)}")
            if self.plot:
                self.plot_filter_steps(all_pos0, all_pos1,
                                       desc_survive_pos0, desc_survive_pos1, desc_removed_pos0, desc_removed_pos1,
                                       spatial_survive_pos0, spatial_survive_pos1, spatial_removed_pos0, spatial_removed_pos1)
            return None, None, None