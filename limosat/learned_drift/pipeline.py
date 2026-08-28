"""Small pair and sequence facade for learned sea-ice drift."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import shapely
import torch
from kornia.feature import ALIKED
from shapely.geometry.base import BaseGeometry

from .config import ALIKEDConfig
from .features import extract_image_features, restrict_features
from .field import estimate_field as estimate_regular_field
from .field import reject_folds
from .imagery import projected_footprint
from .matching import DirectALIKEDLightGlue, match_features
from .types import DriftField, ImageFeatures, MotionMatches, PairResult


class ALIKEDDrift:
    """Extract, match, estimate, and topology-filter ALIKED sea-ice drift."""

    def __init__(
        self,
        config: ALIKEDConfig | None = None,
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
        model_cache: str | Path | None = None,
        model=None,
        matcher=None,
    ):
        self.config = config or ALIKEDConfig()
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        torch.manual_seed(self.config.random_seed)
        if model_cache is not None:
            torch.hub.set_dir(str(Path(model_cache) / "hub"))
        if model is None:
            model = ALIKED.from_pretrained(
                model_name=self.config.model_name,
                max_num_keypoints=self.config.features_per_tile,
                detection_threshold=self.config.detection_threshold,
                device=self.device,
            )
        self.model = model.to(self.device).eval()
        self.matcher = (
            matcher
            if matcher is not None
            else DirectALIKEDLightGlue(self.config)
        ).to(self.device).eval()

    def extract(
        self,
        image_path: str,
        domain: BaseGeometry | None = None,
    ) -> ImageFeatures:
        if domain is None:
            domain = projected_footprint(image_path, self.config.analysis_epsg)
        return extract_image_features(
            image_path,
            domain,
            self.model,
            self.device,
            self.config,
            self.cache_dir,
        )

    def match(
        self,
        source: ImageFeatures,
        target: ImageFeatures,
        elapsed_hours: float,
        prior_displacement_m: tuple[float, float] | None = None,
        prior_uncertainty_m: float | None = None,
    ) -> MotionMatches:
        return match_features(
            source,
            target,
            elapsed_hours,
            self.matcher,
            self.device,
            self.config,
            prior_displacement_m,
            prior_uncertainty_m,
        )

    def estimate_field(
        self,
        matches: MotionMatches,
        domain: BaseGeometry,
    ) -> tuple[DriftField, np.ndarray]:
        field = estimate_regular_field(matches, domain, self.config)
        return reject_folds(field, self.config.grid_spacing_m)

    def track_pair(
        self,
        source: ImageFeatures,
        target: ImageFeatures,
        elapsed_hours: float,
        domain: BaseGeometry | None = None,
        prior_displacement_m: tuple[float, float] | None = None,
        prior_uncertainty_m: float | None = None,
    ) -> PairResult:
        if domain is None:
            domain = source.domain
        started = time.perf_counter()
        matches = self.match(
            source,
            target,
            elapsed_hours,
            prior_displacement_m,
            prior_uncertainty_m,
        )
        matching_seconds = time.perf_counter() - started
        started = time.perf_counter()
        field, rejected = self.estimate_field(matches, domain)
        field_seconds = time.perf_counter() - started
        return PairResult(
            matches=matches,
            field=field,
            fold_rejected_indices=rejected,
            matching_seconds=matching_seconds,
            field_seconds=field_seconds,
            prior_displacement_m=prior_displacement_m,
        )

    def track_images(
        self,
        source_path: str,
        target_path: str,
        elapsed_hours: float,
    ) -> PairResult:
        source_domain, target_domain = self._pair_domains(
            source_path, target_path, elapsed_hours
        )
        source = self.extract(source_path, source_domain)
        target = self.extract(target_path, target_domain)
        return self.track_pair(source, target, elapsed_hours, source_domain)

    def track_sequence(
        self,
        image_paths: Sequence[str],
        elapsed_hours: Sequence[float],
        sequential_prior_uncertainty_m: float | None = None,
    ) -> tuple[PairResult, ...]:
        """Track adjacent images, extracting every unique image only once."""
        if len(image_paths) < 2 or len(elapsed_hours) != len(image_paths) - 1:
            raise ValueError("a sequence needs one elapsed time per adjacent pair")
        pair_domains = [
            self._pair_domains(source, target, hours)
            for source, target, hours in zip(
                image_paths[:-1], image_paths[1:], elapsed_hours, strict=True
            )
        ]
        domains_by_image: dict[str, list[BaseGeometry]] = {
            path: [] for path in image_paths
        }
        for index, (source_domain, target_domain) in enumerate(pair_domains):
            domains_by_image[image_paths[index]].append(source_domain)
            domains_by_image[image_paths[index + 1]].append(target_domain)
        extracted = {
            path: self.extract(path, shapely.union_all(domains))
            for path, domains in domains_by_image.items()
        }

        results = []
        previous_field = None
        previous_elapsed_days = None
        for index, hours in enumerate(elapsed_hours):
            source_domain, target_domain = pair_domains[index]
            source = restrict_features(extracted[image_paths[index]], source_domain)
            target = restrict_features(
                extracted[image_paths[index + 1]], target_domain
            )
            prior = None
            if sequential_prior_uncertainty_m is not None:
                prior = _field_velocity_prior(
                    previous_field,
                    previous_elapsed_days,
                    hours / 24.0,
                    self.config.minimum_agreeing_matches,
                )
            result = self.track_pair(
                source,
                target,
                hours,
                source_domain,
                prior,
                sequential_prior_uncertainty_m if prior is not None else None,
            )
            results.append(result)
            previous_field = result.field
            previous_elapsed_days = hours / 24.0
        return tuple(results)

    def _pair_domains(
        self,
        source_path: str,
        target_path: str,
        elapsed_hours: float,
    ) -> tuple[BaseGeometry, BaseGeometry]:
        if elapsed_hours <= 0:
            raise ValueError("elapsed hours must be positive")
        maximum_displacement_m = self.config.maximum_displacement_m(elapsed_hours)
        source_footprint = projected_footprint(
            source_path, self.config.analysis_epsg
        )
        target_footprint = projected_footprint(
            target_path, self.config.analysis_epsg
        )
        source_domain = source_footprint.intersection(
            target_footprint.buffer(maximum_displacement_m)
        )
        target_domain = target_footprint.intersection(
            source_domain.buffer(maximum_displacement_m)
        )
        if source_domain.is_empty or target_domain.is_empty:
            raise ValueError("source and target have no physics-reachable overlap")
        return source_domain, target_domain


def _field_velocity_prior(
    field: DriftField | None,
    previous_elapsed_days: float | None,
    current_elapsed_days: float,
    minimum_nodes: int,
) -> tuple[float, float] | None:
    if field is None or previous_elapsed_days is None or previous_elapsed_days <= 0:
        return None
    displacement = field.displacement_m[field.available]
    finite = np.isfinite(displacement).all(axis=1)
    if finite.sum() < minimum_nodes:
        return None
    velocity_m_per_day = np.median(displacement[finite], axis=0) / previous_elapsed_days
    prior = velocity_m_per_day * current_elapsed_days
    return float(prior[0]), float(prior[1])
