"""Optional append-only audit output for tracking experiments.

The production pipeline does not instantiate this class. Experiment runners can
attach it to Matcher/ImageProcessor to save candidate fates without changing
selection logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


class TrackingAuditSink:
    """Write named event streams as newline-delimited JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}
        self.counts: dict[str, int] = {}

    def emit(self, stream: str, records: Iterable[Mapping[str, Any]]) -> None:
        rows = list(records)
        if not rows:
            return
        handle = self._handles.get(stream)
        if handle is None:
            handle = (self.output_dir / f"{stream}.jsonl").open("a", buffering=1)
            self._handles[stream] = handle
        for row in rows:
            payload = {key: _json_value(value) for key, value in row.items()}
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.counts[stream] = self.counts.get(stream, 0) + len(rows)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
