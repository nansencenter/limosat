"""Platform-neutral execution stages: prepare, primary pairs and composition."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .catalog import ImageCatalogue, load_catalogue
from .compose import compose_primary_parquet
from .config import RunConfig
from .efficientloftr import EfficientLoFTR
from .pair_identity import write_scene_checksums
from .pair_products import PairProductStore
from .pairs import PairProcessor
from .planning import PlannedPair, run_plan
from .store import RunStore


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
        plan = run_plan(self.config, self.catalogue)
        store = RunStore(self.config)
        store.register_catalogue(self.catalogue)
        store.register_candidate_pairs(plan.pairs)
        store.register_planning_counts(plan.exclusion_counts)
        store.start_run()
        checksums = (
            write_scene_checksums(self.catalogue.records, self.config.scene_checksums)
            if self.config.scene_checksums
            else None
        )
        return {
            "scene_checksums": checksums,
            "catalogue_images": len(self.catalogue.records),
            "candidate_pairs": len(plan.pairs),
            "primary_pairs": sum(
                item.selection == "primary" for item in plan.pairs
            ),
            "planning_counts": dict(plan.exclusion_counts),
        }

    def process_pairs(self, *, batch_index: int = 0, batch_count: int = 1) -> dict:
        """Measure one deterministic batch of primary pairs and publish immutable products."""
        _validate_batch(batch_index, batch_count)
        store = RunStore(self.config, read_only=True)
        primary = self._primary_pairs(store)
        assigned = [
            item for index, item in enumerate(primary) if index % batch_count == batch_index
        ]

        def obtain(item: PlannedPair) -> str:
            pair = item.pair
            if self.pair_products.load(pair) is not None:
                return "pair_product"
            self.pair_products.save(pair, self.processor.process(pair))
            return "computed"

        outcomes = _bounded_map(obtain, assigned, self.config.pair_workers)
        return {
            "batch_index": batch_index,
            "batch_count": batch_count,
            "planned_pairs": len(primary),
            "assigned_pairs": len(outcomes),
            "computed_pairs": outcomes.count("computed"),
            "resumed_pair_products": outcomes.count("pair_product"),
        }

    def compose(self, destination: str | Path | None = None) -> dict:
        """Compose primary trajectories from the pair products into Parquet."""
        return compose_primary_parquet(
            self.config,
            destination or Path(self.config.output_directory) / "primary-composition",
        )

    def run_all(self) -> dict:
        """Run the same stages sequentially on a single machine."""
        try:
            preparation = self.prepare()
            pairs = self.process_pairs()
            composition = self.compose()
        except Exception as error:
            RunStore(self.config).fail_run(error)
            raise
        return {
            "candidate_pairs": preparation["candidate_pairs"],
            "primary_pairs": preparation["primary_pairs"],
            "computed_pairs": pairs["computed_pairs"],
            "resumed_pairs": pairs["resumed_pair_products"],
            **composition,
        }

    def _primary_pairs(self, store: RunStore) -> tuple[PlannedPair, ...]:
        return tuple(
            item for item in store.planned_pairs(self.catalogue)
            if item.selection == "primary"
        )


def _validate_batch(batch_index: int, batch_count: int) -> None:
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if batch_index < 0 or batch_index >= batch_count:
        raise ValueError("batch_index must satisfy 0 <= index < count")


def _bounded_map(function, values, workers: int) -> tuple[str, ...]:
    """Map with at most ``workers`` pairs in flight."""
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
