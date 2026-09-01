"""Deterministic global catalogue coordination."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Sequence

from .catalog import ImageCatalogue, load_catalogue
from .config import RunConfig
from .deformation import deformation_from_field
from .efficientloftr import EfficientLoFTR
from .manifest import write_manifest
from .models import FieldEdge
from .pairs import PairProcessor
from .planning import plan_candidate_pairs, recovery_candidates
from .store import RunStore
from .trajectory import build_trajectories, targeted_recovery_positions


class LiMOSATRun:
    """Plan independent image pairs and recompose one global trajectory product."""

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
        planned = plan_candidate_pairs(self.catalogue, self.config.routing)
        self.store.register_catalogue(self.catalogue)
        self.store.register_candidate_pairs(planned)
        self.store.start_run()
        resumed_pairs = 0
        computed_pairs = 0
        try:
            edges: list[FieldEdge] = []
            primary_fields = []
            primary_items = [
                item for item in planned if item.selection == "primary"
            ]
            with ThreadPoolExecutor(
                max_workers=self.config.pair_workers
            ) as pool:
                primary_results = tuple(
                    pool.map(self._obtain_primary, primary_items)
                )
            for field, resumed in primary_results:
                resumed_pairs += int(resumed)
                computed_pairs += int(not resumed)
                primary_fields.append(field)
                edges.append(FieldEdge(field))

            trajectories = build_trajectories(
                edges,
                self.catalogue.chronological(),
                self.config.field,
                self.config.trajectories,
            )
            if self.config.routing.targeted_recovery:
                recovery_work = []
                for item in recovery_candidates(
                    planned, self.config.routing.maximum_skip_images
                ):
                    positions = targeted_recovery_positions(
                        trajectories,
                        item.pair.source.image_id,
                        item.pair.target.image_id,
                    )
                    if not len(positions):
                        continue
                    recovery_work.append((item, positions))
                with ThreadPoolExecutor(
                    max_workers=self.config.pair_workers
                ) as pool:
                    recovery_results = tuple(
                        pool.map(self._obtain_recovery, recovery_work)
                    )
                for field, resumed in recovery_results:
                    resumed_pairs += int(resumed)
                    computed_pairs += int(not resumed)
                    edges.append(
                        FieldEdge(
                            field,
                            pair_kind="recovery",
                            skipped_images=1,
                        )
                    )
                trajectories = build_trajectories(
                    edges,
                    self.catalogue.chronological(),
                    self.config.field,
                    self.config.trajectories,
                )
            self.store.replace_global_trajectories(trajectories)
            for field in primary_fields:
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

    def _obtain_primary(self, item):
        return self._obtain_pair(
            item.pair,
            item.planning_component_id,
            "primary",
            False,
            None,
            None,
        )

    def _obtain_recovery(self, work):
        item, positions = work
        return self._obtain_pair(
            item.pair,
            item.planning_component_id,
            "recovery",
            True,
            None,
            None,
            positions,
        )

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
