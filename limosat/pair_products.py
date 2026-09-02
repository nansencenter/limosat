"""Immutable, portable products emitted by independent pair workers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .catalog import ImagePair
from .config import RunConfig
from .models import DisplacementField, MotionMatches, PairResult
from .store import file_sha256, implementation_sha256


PAIR_PRODUCT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PairProduct:
    """Verified pair result plus metadata needed for SQLite import."""

    result: PairResult
    match_count: int
    path: Path
    sha256: str
    content_sha256: str


class PairProductStore:
    """Write pair results once and validate them before every reuse."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.root = config.pair_products
        self.implementation_sha256 = implementation_sha256()
        self.model_sha256 = (
            file_sha256(config.matcher.checkpoint)
            if config.matcher.checkpoint
            and Path(config.matcher.checkpoint).is_file()
            else None
        )

    def save(
        self,
        pair: ImagePair,
        kind: str,
        targeted: bool,
        result: PairResult,
        targeted_positions_xy_m: np.ndarray | None = None,
    ) -> PairProduct:
        """Atomically publish a pair product without replacing an existing one."""
        self._validate_identity(pair, kind, targeted, result)
        positions_sha256 = _positions_sha256(targeted_positions_xy_m)
        if kind == "recovery" and (
            targeted_positions_xy_m is None or not len(targeted_positions_xy_m)
        ):
            raise ValueError("recovery pair products require measured source positions")
        arrays = _arrays(result, include_matches=self.config.retain_pair_matches)
        content_sha256 = _content_sha256(arrays)
        existing = self.load(
            pair,
            kind,
            targeted,
            targeted_positions_xy_m=targeted_positions_xy_m,
        )
        if existing is not None:
            if (
                existing.content_sha256 != content_sha256
                or existing.match_count != len(result.matches)
            ):
                raise ValueError(
                    f"immutable pair product already differs: {pair.pair_id}"
                )
            return existing

        data_path, marker_path = self._paths(pair.pair_id, kind)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = data_path.with_name(
            f".{data_path.name}.writing.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            data_sha256 = _publish_data(
                temporary, data_path, content_sha256
            )
        finally:
            temporary.unlink(missing_ok=True)

        metadata = {
            "pair_product_schema_version": PAIR_PRODUCT_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "config_sha256": self.config.sha256,
            "implementation_sha256": self.implementation_sha256,
            "model_sha256": self.model_sha256,
            "pair_id": pair.pair_id,
            "kind": kind,
            "targeted": bool(targeted),
            "source_image_id": pair.source.image_id,
            "target_image_id": pair.target.image_id,
            "source_time_utc": pair.source.time_utc.isoformat(),
            "target_time_utc": pair.target.time_utc.isoformat(),
            "elapsed_seconds": float(pair.elapsed_seconds),
            "targeted_positions_sha256": positions_sha256,
            "field_sha256": result.field.checksum,
            "content_sha256": content_sha256,
            "match_count": len(result.matches),
            "matches_included": bool(self.config.retain_pair_matches),
            "fold_rejected_count": len(result.fold_rejected_indices),
            "runtime_seconds": result.runtime_seconds,
            "matcher_calls": int(result.matcher_calls),
            "diagnostics": result.diagnostics,
            "ancillary_inputs": result.ancillary_inputs,
            "data_file": data_path.name,
            "data_sha256": data_sha256,
            "data_size_bytes": data_path.stat().st_size,
        }
        metadata["marker_content_sha256"] = _metadata_sha256(metadata)
        marker_bytes = (
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        marker_temporary = marker_path.with_name(
            f".{marker_path.name}.writing.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            with marker_temporary.open("wb") as stream:
                stream.write(marker_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            _publish_marker(marker_temporary, marker_path)
        finally:
            marker_temporary.unlink(missing_ok=True)
        product = self.load(
            pair,
            kind,
            targeted,
            targeted_positions_xy_m=targeted_positions_xy_m,
        )
        if product is None:  # pragma: no cover - guarded by publication above
            raise RuntimeError(f"pair product was not published: {pair.pair_id}")
        if (
            product.content_sha256 != content_sha256
            or product.match_count != len(result.matches)
        ):
            raise ValueError(
                f"immutable pair product already differs: {pair.pair_id}"
            )
        return product

    def load(
        self,
        pair: ImagePair,
        kind: str,
        targeted: bool,
        targeted_positions_xy_m: np.ndarray | None = None,
    ) -> PairProduct | None:
        """Load a completed pair product, or ``None`` without a marker."""
        data_path, marker_path = self._paths(pair.pair_id, kind)
        if not marker_path.exists():
            return None
        if not data_path.is_file():
            raise ValueError(f"pair product marker has no data file: {pair.pair_id}")
        if kind == "recovery" and (
            targeted_positions_xy_m is None or not len(targeted_positions_xy_m)
        ):
            raise ValueError("recovery pair products require measured source positions")
        metadata = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"pair product marker is not an object: {pair.pair_id}")
        marker_content_sha256 = metadata.get("marker_content_sha256")
        marker_content = {
            name: value
            for name, value in metadata.items()
            if name != "marker_content_sha256"
        }
        if marker_content_sha256 != _metadata_sha256(marker_content):
            raise ValueError(f"pair product marker failed checksum: {pair.pair_id}")
        expected = {
            "pair_product_schema_version": PAIR_PRODUCT_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "config_sha256": self.config.sha256,
            "implementation_sha256": self.implementation_sha256,
            "model_sha256": self.model_sha256,
            "pair_id": pair.pair_id,
            "kind": kind,
            "targeted": bool(targeted),
            "source_image_id": pair.source.image_id,
            "target_image_id": pair.target.image_id,
            "source_time_utc": pair.source.time_utc.isoformat(),
            "target_time_utc": pair.target.time_utc.isoformat(),
            "elapsed_seconds": float(pair.elapsed_seconds),
            "targeted_positions_sha256": _positions_sha256(
                targeted_positions_xy_m
            ),
        }
        changed = [
            name for name, value in expected.items() if metadata.get(name) != value
        ]
        if changed:
            raise ValueError(
                f"pair product identity changed for {pair.pair_id}: {changed}"
            )
        if metadata.get("data_file") != data_path.name:
            raise ValueError(f"pair product data name changed: {pair.pair_id}")
        data_sha256 = file_sha256(data_path)
        if data_sha256 != metadata.get("data_sha256"):
            raise ValueError(f"pair product failed checksum: {pair.pair_id}")
        if data_path.stat().st_size != metadata.get("data_size_bytes"):
            raise ValueError(f"pair product size changed: {pair.pair_id}")
        with np.load(data_path, allow_pickle=False) as values:
            arrays = {name: _array(values, name) for name in _ARRAY_NAMES}
        content_sha256 = _content_sha256(arrays)
        if content_sha256 != metadata.get("content_sha256"):
            raise ValueError(f"pair product content failed checksum: {pair.pair_id}")
        result = _result_from_arrays(metadata, arrays)
        self._validate_identity(pair, kind, targeted, result)
        if result.field.checksum != metadata.get("field_sha256"):
            raise ValueError(f"pair field failed checksum: {pair.pair_id}")
        match_count = int(metadata["match_count"])
        if match_count < 0:
            raise ValueError(f"pair match count is negative: {pair.pair_id}")
        matches_included = metadata.get("matches_included")
        if matches_included is not self.config.retain_pair_matches:
            raise ValueError(f"pair match retention changed: {pair.pair_id}")
        if matches_included and len(result.matches) != match_count:
            raise ValueError(f"pair match count changed: {pair.pair_id}")
        if not matches_included and len(result.matches):
            raise ValueError(f"unexpected retained matches: {pair.pair_id}")
        if len(result.fold_rejected_indices) != metadata.get("fold_rejected_count"):
            raise ValueError(f"fold-rejected count changed: {pair.pair_id}")
        return PairProduct(
            result,
            match_count,
            data_path,
            data_sha256,
            content_sha256,
        )

    def count(self, kind: str) -> int:
        if kind not in {"primary", "recovery"}:
            raise ValueError("pair kind must be primary or recovery")
        directory = self.root / kind
        return (
            sum(1 for _path in directory.glob("*.json"))
            if directory.is_dir()
            else 0
        )

    def _paths(self, pair_id: str, kind: str) -> tuple[Path, Path]:
        if kind not in {"primary", "recovery"}:
            raise ValueError("pair kind must be primary or recovery")
        identity = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
        base = self.root / kind / identity
        return base.with_suffix(".npz"), base.with_suffix(".json")

    @staticmethod
    def _validate_identity(
        pair: ImagePair,
        kind: str,
        targeted: bool,
        result: PairResult,
    ) -> None:
        if kind not in {"primary", "recovery"}:
            raise ValueError("pair kind must be primary or recovery")
        if targeted != (kind == "recovery"):
            raise ValueError("only recovery pair products may be targeted")
        field = result.field
        if (
            field.pair_id != pair.pair_id
            or field.source_image_id != pair.source.image_id
            or field.target_image_id != pair.target.image_id
            or field.source_time_utc != pair.source.time_utc
            or field.target_time_utc != pair.target.time_utc
        ):
            raise ValueError("pair product field identity differs from its image pair")


def _arrays(result: PairResult, include_matches: bool) -> dict[str, np.ndarray]:
    field = result.field
    matches = result.matches if include_matches else MotionMatches.empty()
    return {
        "grid_row": np.asarray(field.grid_row, dtype="<i4"),
        "grid_column": np.asarray(field.grid_column, dtype="<i4"),
        "source_xy_m": np.asarray(field.source_xy_m, dtype="<f8"),
        "displacement_m": np.asarray(field.displacement_m, dtype="<f8"),
        "available": np.asarray(field.available, dtype=bool),
        "selected_matches": np.asarray(field.selected_matches, dtype="<i4"),
        "candidate_matches": np.asarray(field.candidate_matches, dtype="<i4"),
        "support_radius_m": np.asarray(field.support_radius_m, dtype="<f8"),
        "maximum_residual_m": np.asarray(field.maximum_residual_m, dtype="<f8"),
        "match_source_xy_m": np.asarray(matches.source_xy_m, dtype="<f8"),
        "match_target_xy_m": np.asarray(matches.target_xy_m, dtype="<f8"),
        "match_score": np.asarray(matches.score, dtype="<f8"),
        "match_source_tile": np.asarray(matches.source_tile, dtype="<i4"),
        "match_target_tile": np.asarray(matches.target_tile, dtype="<i4"),
        "fold_rejected_indices": np.asarray(
            result.fold_rejected_indices, dtype="<i4"
        ),
    }


def _result_from_arrays(metadata: dict, arrays: dict[str, np.ndarray]) -> PairResult:
    field = DisplacementField(
        pair_id=metadata["pair_id"],
        source_image_id=metadata["source_image_id"],
        target_image_id=metadata["target_image_id"],
        source_time_utc=datetime.fromisoformat(metadata["source_time_utc"]),
        target_time_utc=datetime.fromisoformat(metadata["target_time_utc"]),
        grid_row=arrays["grid_row"],
        grid_column=arrays["grid_column"],
        source_xy_m=arrays["source_xy_m"],
        displacement_m=arrays["displacement_m"],
        available=arrays["available"],
        selected_matches=arrays["selected_matches"],
        candidate_matches=arrays["candidate_matches"],
        support_radius_m=arrays["support_radius_m"],
        maximum_residual_m=arrays["maximum_residual_m"],
    )
    matches = MotionMatches(
        arrays["match_source_xy_m"],
        arrays["match_target_xy_m"],
        arrays["match_score"],
        arrays["match_source_tile"],
        arrays["match_target_tile"],
    )
    return PairResult(
        matches=matches,
        field=field,
        fold_rejected_indices=arrays["fold_rejected_indices"],
        runtime_seconds=dict(metadata.get("runtime_seconds") or {}),
        matcher_calls=int(metadata["matcher_calls"]),
        diagnostics=dict(metadata.get("diagnostics") or {}),
        ancillary_inputs=dict(metadata.get("ancillary_inputs") or {}),
    )


def _positions_sha256(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("targeted positions must have shape (n, 2)")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _metadata_sha256(metadata: dict) -> str:
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_ARRAY_NAMES = (
    "grid_row",
    "grid_column",
    "source_xy_m",
    "displacement_m",
    "available",
    "selected_matches",
    "candidate_matches",
    "support_radius_m",
    "maximum_residual_m",
    "match_source_xy_m",
    "match_target_xy_m",
    "match_score",
    "match_source_tile",
    "match_target_tile",
    "fold_rejected_indices",
)


def _array(values, name: str) -> np.ndarray:
    if name not in values.files:
        raise ValueError(f"pair product array is missing: {name}")
    return np.asarray(values[name]).copy()


def _content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _ARRAY_NAMES:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _publish_data(
    temporary: Path,
    destination: Path,
    content_sha256: str,
) -> str:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        with np.load(destination, allow_pickle=False) as values:
            arrays = {name: _array(values, name) for name in _ARRAY_NAMES}
        if _content_sha256(arrays) != content_sha256:
            raise ValueError(f"immutable pair product already differs: {destination}")
    return file_sha256(destination)


def _publish_marker(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        pass
