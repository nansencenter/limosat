"""Platform-neutral execution stages for pair workers and CPU composition."""

from __future__ import annotations

import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from .catalog import ImageCatalogue, load_catalogue
from .config import RunConfig
from .deformation import deformation_from_field
from .efficientloftr import EfficientLoFTR
from .manifest import write_manifest
from .models import FieldEdge
from .pair_products import PAIR_PRODUCT_SCHEMA_VERSION, PairProductStore
from .pairs import PairProcessor
from .planning import PlannedPair, build_candidate_plan, recovery_candidates
from .store import RunStore
from .trajectory import (
    TrajectoryPoint,
    audit_trajectory_convergence,
    iter_global_trajectory_points,
)


@dataclass(frozen=True)
class PairWork:
    planned: PlannedPair
    targeted_positions_xy_m: np.ndarray | None = None


class RunStages:
    """Execute independently resumable stages without scheduler assumptions."""

    def __init__(
        self,
        config: RunConfig,
        catalogue: ImageCatalogue | None = None,
        processor: PairProcessor | None = None,
    ) -> None:
        self.config = config
        self._catalogue = catalogue
        self._processor = processor
        self._pair_products: PairProductStore | None = None

    @property
    def catalogue(self) -> ImageCatalogue:
        if self._catalogue is None:
            self._catalogue = load_catalogue(
                self.config.catalogue, self.config.analysis_epsg
            )
        return self._catalogue

    @property
    def processor(self) -> PairProcessor:
        if self._processor is None:
            self._processor = PairProcessor(
                self.config, EfficientLoFTR(self.config.matcher)
            )
        return self._processor

    @property
    def pair_products(self) -> PairProductStore:
        if self._pair_products is None:
            self._pair_products = PairProductStore(self.config)
        return self._pair_products

    def prepare(self) -> dict:
        """Freeze the catalogue and candidate image-pair plan in SQLite."""
        plan = build_candidate_plan(
            self.catalogue,
            self.config.routing,
            grid_spacing_m=self.config.field.grid_spacing_m,
            maximum_speed_m_per_day=self.config.matcher.maximum_speed_m_per_day,
        )
        store = RunStore(self.config)
        store.register_catalogue(self.catalogue)
        store.register_candidate_pairs(plan.pairs)
        store.register_planning_counts(plan.exclusion_counts)
        store.start_run()
        return {
            "catalogue_images": len(self.catalogue.records),
            "candidate_pairs": len(plan.pairs),
            "primary_pairs": sum(
                item.selection == "primary" for item in plan.pairs
            ),
            "planning_counts": dict(plan.exclusion_counts),
        }

    def process_pairs(
        self,
        kind: str,
        *,
        batch_index: int = 0,
        batch_count: int = 1,
    ) -> dict:
        """Measure one deterministic batch and publish immutable pair products."""
        _validate_batch(batch_index, batch_count)
        store = RunStore(self.config, read_only=True)
        candidates = self._pair_candidates(store, kind)
        selected = self._work(
            store,
            kind,
            candidates=candidates,
            batch_index=batch_index,
            batch_count=batch_count,
        )

        def obtain(item: PairWork) -> str:
            pair = item.planned.pair
            if store.load_field(pair.pair_id) is not None:
                return "sqlite"
            completed = self.pair_products.load(
                pair,
                kind,
                kind == "recovery",
                item.targeted_positions_xy_m,
            )
            if completed is not None:
                return "pair_product"
            result = self.processor.process(
                pair,
                None,
                None,
                item.targeted_positions_xy_m,
            )
            self.pair_products.save(
                pair,
                kind,
                kind == "recovery",
                result,
                item.targeted_positions_xy_m,
            )
            return "computed"

        outcomes = _bounded_map(obtain, selected, self.config.pair_workers)
        return {
            "kind": kind,
            "batch_index": batch_index,
            "batch_count": batch_count,
            "planned_pairs": len(candidates),
            "assigned_pairs": len(outcomes),
            "computed_pairs": outcomes.count("computed"),
            "resumed_pair_products": outcomes.count("pair_product"),
            "resumed_sqlite_pairs": outcomes.count("sqlite"),
        }

    def _import_pair_products(
        self,
        store: RunStore,
        kind: str,
    ) -> tuple[dict, tuple[PlannedPair, ...]]:
        """Import verified worker outputs through the single SQLite writer."""
        imported = 0
        resumed = 0
        planned = []
        missing = []
        for item in self._work(store, kind):
            planned.append(item.planned)
            pair = item.planned.pair
            field = store.load_field(pair.pair_id)
            if field is None:
                product = self.pair_products.load(
                    pair,
                    kind,
                    kind == "recovery",
                    item.targeted_positions_xy_m,
                )
                if product is None:
                    missing.append(pair.pair_id)
                    continue
                if not store.claim_pair(
                    pair,
                    item.planned.planning_component_id,
                    kind,
                    kind == "recovery",
                ):
                    raise RuntimeError(
                        f"pair completed while the coordinator was importing: {pair.pair_id}"
                    )
                result = replace(
                    product.result,
                    diagnostics={
                        **product.result.diagnostics,
                        "pair_product_schema_version": PAIR_PRODUCT_SCHEMA_VERSION,
                        "pair_product_sha256": product.sha256,
                        "pair_product_content_sha256": product.content_sha256,
                    },
                )
                store.save_pair(
                    pair,
                    result,
                    match_count=product.match_count,
                )
                field = result.field
                imported += 1
            else:
                resumed += 1
            if kind == "primary":
                store.replace_deformation(
                    pair.pair_id,
                    deformation_from_field(
                        field, self.config.field.maximum_triangle_edge_m
                    ),
                )
        if missing:
            sample = ", ".join(missing[:5])
            raise RuntimeError(
                f"{len(missing)} {kind} pair products are incomplete; first: {sample}"
            )
        return (
            {
                "kind": kind,
                "expected_pair_fields": len(planned),
                "imported_pair_products": imported,
                "resumed_pair_fields": resumed,
            },
            tuple(planned),
        )

    def compose(
        self,
        phase: str,
        command: Sequence[str] | None = None,
    ) -> dict:
        """Stream primary or final global trajectory rows into SQLite."""
        if phase not in {"primary", "final"}:
            raise ValueError("composition phase must be primary or final")
        store = RunStore(self.config)
        planned = store.planned_pairs(self.catalogue)
        primary = tuple(item for item in planned if item.selection == "primary")
        pair_import = {
            "kind": "recovery" if phase == "final" else "primary",
            "expected_pair_fields": 0,
            "imported_pair_products": 0,
            "resumed_pair_fields": 0,
        }
        if phase == "primary":
            pair_import, _ = self._import_pair_products(store, "primary")
        self._require_complete(store, "primary", primary)
        recovery = ()
        if phase == "final" and self.config.routing.targeted_recovery:
            pair_import, recovery = self._import_pair_products(
                store, "recovery"
            )
            self._require_complete(
                store,
                "recovery",
                recovery,
            )

        edges = [
            FieldEdge(self._required_field(store, item.pair.pair_id))
            for item in primary
        ]
        if recovery:
            edges.extend(
                FieldEdge(
                    self._required_field(store, item.pair.pair_id),
                    pair_kind="recovery",
                    skipped_images=item.skipped_images,
                )
                for item in recovery
            )

        state_counts: dict[str, int] = {}
        point_count = 0
        retained_points: list[TrajectoryPoint] | None = (
            []
            if self.config.trajectories.convergence_audit_radius_m is not None
            else None
        )

        def batches():
            nonlocal point_count
            for batch in iter_global_trajectory_points(
                edges,
                self.catalogue.chronological(),
                self.config.field,
                self.config.trajectories,
            ):
                point_count += len(batch)
                for point in batch:
                    state_counts[point.state] = state_counts.get(point.state, 0) + 1
                if retained_points is not None:
                    retained_points.extend(batch)
                yield batch

        clock = time.perf_counter()
        store.replace_global_trajectory_batches(batches())
        events = (
            audit_trajectory_convergence(
                retained_points,
                self.config.trajectories.convergence_audit_radius_m,
            )
            if retained_points is not None
            else ()
        )
        store.replace_convergence_events(events)
        result = {
            "phase": phase,
            "primary_pairs": len(primary),
            "recovery_pairs": len(recovery),
            "trajectory_points": point_count,
            "trajectory_states": state_counts,
            "convergence_events": len(events),
            "composition_seconds": time.perf_counter() - clock,
            "pair_product_import": pair_import,
        }
        if phase == "final":
            completed_utc = datetime.now(timezone.utc)
            run = store.run_record()
            started_utc = datetime.fromisoformat(run["started_utc"])
            runtime_seconds = (completed_utc - started_utc).total_seconds()
            path, checksum = write_manifest(
                self.config,
                self.catalogue,
                store,
                started_utc,
                completed_utc,
                runtime_seconds,
                tuple(command or sys.argv),
            )
            store.finish_run(runtime_seconds, path, checksum)
            result.update(
                manifest=str(path),
                manifest_sha256=checksum,
                runtime_seconds=runtime_seconds,
            )
        return result

    def run_all(self, command: Sequence[str] | None = None) -> dict:
        """Run the same stages sequentially on a single machine."""
        try:
            preparation = self.prepare()
            primary = self.process_pairs("primary")
            primary_composition = self.compose("primary")
            recovery = {
                "computed_pairs": 0,
                "resumed_pair_products": 0,
                "resumed_sqlite_pairs": 0,
            }
            if self.config.routing.targeted_recovery:
                recovery = self.process_pairs("recovery")
            final = self.compose("final", command)
            return {
                **final,
                "candidate_pairs": preparation["candidate_pairs"],
                "computed_pairs": (
                    primary["computed_pairs"] + recovery["computed_pairs"]
                ),
                "resumed_pairs": (
                    primary["resumed_pair_products"]
                    + primary["resumed_sqlite_pairs"]
                    + recovery["resumed_pair_products"]
                    + recovery["resumed_sqlite_pairs"]
                ),
                "primary_composition_seconds": primary_composition[
                    "composition_seconds"
                ],
            }
        except Exception as error:
            RunStore(self.config).fail_run(error)
            raise

    def _pair_candidates(
        self, store: RunStore, kind: str
    ) -> tuple[PlannedPair, ...]:
        planned = store.planned_pairs(self.catalogue)
        if kind == "primary":
            return tuple(
                item for item in planned if item.selection == "primary"
            )
        if kind != "recovery":
            raise ValueError("pair kind must be primary or recovery")
        if not self.config.routing.targeted_recovery:
            return ()
        return recovery_candidates(
            planned,
            self.config.routing.maximum_recovery_elapsed_hours,
        )

    def _work(
        self,
        store: RunStore,
        kind: str,
        *,
        candidates: Sequence[PlannedPair] | None = None,
        batch_index: int = 0,
        batch_count: int = 1,
    ):
        candidates = (
            self._pair_candidates(store, kind)
            if candidates is None
            else candidates
        )
        assigned = tuple(
            item
            for index, item in enumerate(candidates)
            if index % batch_count == batch_index
        )
        if kind == "primary":
            yield from (PairWork(item) for item in assigned)
            return
        by_id = {item.pair.pair_id: item for item in assigned}
        for pair, positions in store.iter_targeted_recovery_positions(
            item.pair for item in assigned
        ):
            if len(positions):
                yield PairWork(by_id[pair.pair_id], positions)

    @staticmethod
    def _required_field(store: RunStore, pair_id: str):
        field = store.load_field(pair_id)
        if field is None:
            raise RuntimeError(f"completed pair field is missing: {pair_id}")
        return field

    @staticmethod
    def _require_complete(
        store: RunStore,
        kind: str,
        planned: Sequence[PlannedPair],
    ) -> None:
        complete = store.completed_pair_ids(kind)
        missing = [
            item.pair.pair_id
            for item in planned
            if item.pair.pair_id not in complete
        ]
        if missing:
            sample = ", ".join(missing[:5])
            raise RuntimeError(
                f"{len(missing)} {kind} pair fields are incomplete; first: {sample}"
            )


def _validate_batch(batch_index: int, batch_count: int) -> None:
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if batch_index < 0 or batch_index >= batch_count:
        raise ValueError("batch_index must satisfy 0 <= index < count")


def _bounded_map(function, values, workers: int) -> tuple[str, ...]:
    """Map with at most ``workers`` recovery-position arrays resident."""
    iterator = iter(values)
    if workers == 1:
        return tuple(function(value) for value in iterator)
    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _index in range(workers):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            outcomes.append(pending.popleft().result())
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass
    return tuple(outcomes)
