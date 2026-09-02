"""Simple single-machine facade over staged LiMOSAT execution."""

from __future__ import annotations

import sys
from typing import Sequence

from .catalog import ImageCatalogue
from .config import RunConfig
from .pairs import PairProcessor
from .stages import RunStages
from .store import RunStore


class LiMOSATRun:
    """Run planning, pair measurement, and global composition sequentially."""

    def __init__(
        self,
        config: RunConfig,
        catalogue: ImageCatalogue | None = None,
        processor: PairProcessor | None = None,
    ) -> None:
        self.config = config
        self.stages = RunStages(config, catalogue, processor)
        self._store: RunStore | None = None

    @property
    def catalogue(self) -> ImageCatalogue:
        return self.stages.catalogue

    @property
    def processor(self) -> PairProcessor:
        return self.stages.processor

    @property
    def store(self) -> RunStore:
        if self._store is None:
            self._store = RunStore(self.config)
        return self._store

    def execute(self, command: Sequence[str] | None = None) -> dict:
        return self.stages.run_all(tuple(command or sys.argv))
