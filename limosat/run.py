"""Sequence and component orchestration."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Sequence

from .catalog import ImageCatalogue, ImagePair, load_catalogue
from .config import RunConfig
from .deformation import deformation_from_field
from .efficientloftr import EfficientLoFTR
from .manifest import write_manifest
from .models import FieldEdge
from .pairs import PairProcessor
from .store import RunStore
from .trajectory import build_trajectories, targeted_recovery_positions


class LiMOSATRun:
    """Execute catalogue components with pair-level deterministic resume."""

    def __init__(
        self,
        config: RunConfig,
        catalogue: ImageCatalogue | None = None,
        processor: PairProcessor | None = None,
    ) -> None:
        self.config = config
        self.catalogue = catalogue or load_catalogue(
            config.catalogue, config.analysis_epsg
        )
        self.store = RunStore(config)
        self.processor = processor or PairProcessor(
            config, EfficientLoFTR(config.matcher)
        )

    def execute(self, command: Sequence[str] | None = None) -> dict:
        started_clock = time.perf_counter()
        started_utc = datetime.now(timezone.utc)
        self.store.register_catalogue(self.catalogue)
        self.store.start_run()
        resumed_pairs = 0
        computed_pairs = 0
        try:
            for component_id, images in self.catalogue.components().items():
                if len(images) < 2:
                    continue
                edges: list[FieldEdge] = []
                adjacent_fields = []
                previous_field = None
                previous_elapsed = None
                for pair in self.catalogue.adjacent_pairs(component_id):
                    field, resumed = self._obtain_pair(
                        pair,
                        component_id,
                        "adjacent",
                        False,
                        previous_field,
                        previous_elapsed,
                    )
                    resumed_pairs += int(resumed)
                    computed_pairs += int(not resumed)
                    adjacent_fields.append(field)
                    edges.append(FieldEdge(field))
                    previous_field, previous_elapsed = field, pair.elapsed_seconds

                trajectories = build_trajectories(
                    edges,
                    images,
                    self.config.field,
                    self.config.trajectories,
                )
                if self.config.routing.targeted_recovery:
                    adjacent_pairs = self.catalogue.adjacent_pairs(component_id)
                    for skipped_images in range(
                        1, self.config.routing.maximum_skip_images + 1
                    ):
                        for target_index in range(skipped_images + 1, len(images)):
                            source_index = target_index - skipped_images - 1
                            positions = targeted_recovery_positions(
                                trajectories,
                                images[source_index].image_id,
                                images[target_index].image_id,
                            )
                            if not len(positions):
                                continue
                            pair = ImagePair(
                                images[source_index], images[target_index]
                            )
                            prior = (
                                adjacent_fields[source_index - 1]
                                if source_index
                                else None
                            )
                            prior_elapsed = (
                                adjacent_pairs[source_index - 1].elapsed_seconds
                                if source_index
                                else None
                            )
                            field, resumed = self._obtain_pair(
                                pair,
                                component_id,
                                "recovery",
                                True,
                                prior,
                                prior_elapsed,
                                positions,
                            )
                            resumed_pairs += int(resumed)
                            computed_pairs += int(not resumed)
                            edges.append(
                                FieldEdge(
                                    field, skipped_images=skipped_images
                                )
                            )
                        trajectories = build_trajectories(
                            edges,
                            images,
                            self.config.field,
                            self.config.trajectories,
                        )
                self.store.replace_trajectories(component_id, trajectories)
                for field in adjacent_fields:
                    self.store.replace_deformation(
                        field.pair_id,
                        deformation_from_field(
                            field, self.config.field.maximum_triangle_edge_m
                        ),
                    )
            completed_utc = datetime.now(timezone.utc)
            runtime_seconds = time.perf_counter() - started_clock
            path, checksum = write_manifest(
                self.config,
                self.catalogue,
                self.store,
                started_utc,
                completed_utc,
                runtime_seconds,
                tuple(command or sys.argv),
            )
            self.store.finish_run(runtime_seconds, path, checksum)
            return {
                "manifest": str(path),
                "manifest_sha256": checksum,
                "computed_pairs": computed_pairs,
                "resumed_pairs": resumed_pairs,
                "runtime_seconds": runtime_seconds,
            }
        except Exception as error:
            self.store.fail_run(error)
            raise

    def _obtain_pair(
        self,
        pair,
        component_id,
        kind,
        targeted,
        previous_field,
        previous_elapsed,
        targeted_positions=None,
    ):
        completed = self.store.load_field(pair.pair_id)
        if completed is not None:
            return completed, True
        self.store.claim_pair(pair, component_id, kind, targeted)
        try:
            result = self.processor.process(
                pair,
                previous_field,
                previous_elapsed,
                targeted_positions,
            )
            self.store.save_pair(pair, result)
            return result.field, False
        except Exception as error:
            self.store.fail_pair(pair.pair_id, error)
            raise
