"""Immutable, portable products emitted by independent pair workers."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .catalog import ImagePair
from .config import RunConfig
from .models import DisplacementField, MotionMatches, PairResult
from .pair_identity import (
    MATCHING_VERSION,
    SceneChecksums,
    identity_sha256,
    matching_identity,
)
from .store import file_sha256, implementation_sha256


# Schema 3 identifies a product by its matching fingerprint (pair_identity.py) and the
# checksums of its scene and SIC files, so any run with the same fingerprint can reuse
# it. (Schema 2 products, tied to the run that made them, are no longer read.)
PAIR_PRODUCT_SCHEMA_VERSION = 3
# How long a writer that lost a publication race waits for the winner's marker.
PUBLICATION_WAIT_SECONDS = 600.0
# Recorded in every marker; recovery products of the abandoned continuation passes were
# the only other kind.
_KIND = "primary"


@dataclass(frozen=True)
class PairProduct:
    """Verified pair result plus metadata needed for SQLite import."""

    result: PairResult
    match_count: int
    path: Path
    sha256: str
    content_sha256: str
    producer_implementation_sha256: str


class PairProductStore:
    """Write pair results once and validate them before every reuse.

    Products live in ``config.pair_store_directory/<matching_sha256>/primary`` when a
    shared store is configured, otherwise in ``<pair_product_directory>/primary``. Every
    product is a primary product of the whole pair overlap; its marker still records
    ``kind``, ``targeted`` and ``targeted_positions_sha256`` (always "primary", false and
    null) so that stored products keep their identity.
    """

    def __init__(
        self, config: RunConfig, scene_checksums: SceneChecksums | None = None
    ) -> None:
        self.config = config
        self.root = config.pair_products
        self.implementation_sha256 = implementation_sha256()
        self.model_sha256 = (
            file_sha256(config.matcher.checkpoint)
            if config.matcher.checkpoint
            and Path(config.matcher.checkpoint).is_file()
            else None
        )
        self.scene_checksums = scene_checksums or SceneChecksums(
            config.scene_checksums or None
        )
        self._matching_identity = None
        self._matching_sha256 = None
        self._sic_index = None

    @property
    def matching_identity(self) -> dict:
        """Settings and input checksums this store's products depend on (computed once,
        since it reads the model and land raster)."""
        if self._matching_identity is None:
            self._matching_identity = matching_identity(self.config)
            self._matching_sha256 = identity_sha256(self._matching_identity)
        return self._matching_identity

    @property
    def matching_sha256(self) -> str:
        self.matching_identity
        return self._matching_sha256

    @property
    def directory(self) -> Path:
        if self.config.pair_store_directory:
            return Path(self.config.pair_store_directory) / self.matching_sha256 / _KIND
        return self.root / _KIND

    def input_checksums(self, pair: ImagePair) -> dict:
        """Content checksums of the per-pair inputs: both scenes and their SIC files."""
        sic = {"source": None, "target": None}
        if self.config.open_water.enabled:
            from .tile_gates import SicFileIndex, sic_file_sha256

            if self._sic_index is None:
                self._sic_index = SicFileIndex(self.config.open_water.sic_root)
            for role, image in (("source", pair.source), ("target", pair.target)):
                path = self._sic_index.resolve(
                    image.time_utc, self.config.open_water.maximum_age_days
                )
                sic[role] = None if path is None else sic_file_sha256(path)
        return {
            "scene_sha256": {
                "source": self.scene_checksums.sha256(pair.source),
                "target": self.scene_checksums.sha256(pair.target),
            },
            "sic_sha256": sic,
        }

    def save(self, pair: ImagePair, result: PairResult) -> PairProduct:
        """Atomically publish a pair product without replacing an existing one."""
        self._validate_identity(pair, result)
        arrays = _arrays(result, include_matches=self.config.retain_pair_matches)
        content_sha256 = _content_sha256(arrays)
        existing = self.load(pair)
        if existing is not None:
            # Products are immutable and first-writer-wins: another worker or run with
            # the same fingerprint already measured this pair, so its result is kept.
            return existing
        inputs = self.input_checksums(pair)

        data_path, marker_path = self._paths(pair.pair_id)
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
                temporary, data_path, marker_path, content_sha256
            )
        finally:
            temporary.unlink(missing_ok=True)
        if data_sha256 is None:
            # Another writer published different data first; adopt its product.
            adopted = self.load(pair)
            if adopted is None:  # pragma: no cover - _publish_data waited for it
                raise RuntimeError(f"pair product was not published: {pair.pair_id}")
            return adopted

        metadata = {
            "pair_product_schema_version": PAIR_PRODUCT_SCHEMA_VERSION,
            "matching_version": MATCHING_VERSION,
            "matching_sha256": self.matching_sha256,
            "matching_identity": self.matching_identity,
            **inputs,
            "producer_run_id": self.config.run_id,
            "producer_config_sha256": self.config.sha256,
            "implementation_sha256": self.implementation_sha256,
            "model_sha256": self.model_sha256,
            "pair_id": pair.pair_id,
            "kind": _KIND,
            "targeted": False,
            "source_image_id": pair.source.image_id,
            "target_image_id": pair.target.image_id,
            "source_time_utc": pair.source.time_utc.isoformat(),
            "target_time_utc": pair.target.time_utc.isoformat(),
            "elapsed_seconds": float(pair.elapsed_seconds),
            "targeted_positions_sha256": None,
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
        product = self.load(pair)
        if product is None:  # pragma: no cover - guarded by publication above
            raise RuntimeError(f"pair product was not published: {pair.pair_id}")
        return product

    def load(self, pair: ImagePair) -> PairProduct | None:
        """Load a completed pair product, or ``None`` without a marker.

        Products are accepted by any run with the same matching fingerprint and
        identical scene and SIC files; ``MATCHING_VERSION`` decides when code changes
        invalidate them, so the producing code itself may differ (its hash is kept).
        """
        data_path, marker_path = self._paths(pair.pair_id)
        if not marker_path.exists():
            return None
        if not data_path.is_file():
            raise ValueError(f"pair product marker has no data file: {pair.pair_id}")
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
        producer_implementation_sha256 = metadata.get("implementation_sha256")
        if not _is_sha256(producer_implementation_sha256):
            raise ValueError(
                f"pair product producer hash is invalid: {pair.pair_id}"
            )
        schema = metadata.get("pair_product_schema_version")
        if schema != PAIR_PRODUCT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported pair product schema {schema!r}: {pair.pair_id}"
            )
        expected = {
            "matching_version": MATCHING_VERSION,
            "matching_sha256": self.matching_sha256,
            **self.input_checksums(pair),
        }
        expected.update({
            "pair_id": pair.pair_id,
            "kind": _KIND,
            "targeted": False,
            "source_image_id": pair.source.image_id,
            "target_image_id": pair.target.image_id,
            "source_time_utc": pair.source.time_utc.isoformat(),
            "target_time_utc": pair.target.time_utc.isoformat(),
            "elapsed_seconds": float(pair.elapsed_seconds),
            "targeted_positions_sha256": None,
        })
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
        self._validate_identity(pair, result)
        if result.field.checksum != metadata.get("field_sha256"):
            raise ValueError(f"pair field failed checksum: {pair.pair_id}")
        match_count = int(metadata["match_count"])
        if match_count < 0:
            raise ValueError(f"pair match count is negative: {pair.pair_id}")
        matches_included = metadata.get("matches_included")
        if not isinstance(matches_included, bool) or (
            self.config.retain_pair_matches and not matches_included
        ):
            # A product with retained matches also serves runs that do not keep them.
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
            producer_implementation_sha256,
        )

    def field_nodes(self, pair: ImagePair) -> np.ndarray:
        """The pair field's nodes with a displacement (source coordinates), read without
        the matches; the full product is verified when it is sampled."""
        data_path, _marker_path = self._paths(pair.pair_id)
        with np.load(data_path, allow_pickle=False) as values:
            xy = np.asarray(values["source_xy_m"], dtype=np.float64)
            available = np.asarray(values["available"], dtype=bool)
        return xy[available]

    def count(self) -> int:
        """Completed products in the store (a shared store also holds other runs' pairs)."""
        directory = self.directory
        return (
            sum(1 for _path in directory.glob("*.json"))
            if directory.is_dir()
            else 0
        )

    def _paths(self, pair_id: str) -> tuple[Path, Path]:
        identity = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
        base = self.directory / identity
        return base.with_suffix(".npz"), base.with_suffix(".json")

    @staticmethod
    def _validate_identity(pair: ImagePair, result: PairResult) -> None:
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


def _metadata_sha256(metadata: dict) -> str:
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    marker: Path,
    content_sha256: str,
) -> str | None:
    """Publish data atomically; ``None`` when another writer's product wins.

    Matching is not bit-reproducible on GPUs, so two writers of one pair (two batch
    tasks, or two runs sharing a store) may produce different data. The first data
    file published wins; a later writer waits for the winner's marker and adopts it.
    """
    try:
        os.link(temporary, destination)
    except FileExistsError:
        with np.load(destination, allow_pickle=False) as values:
            arrays = {name: _array(values, name) for name in _ARRAY_NAMES}
        if _content_sha256(arrays) == content_sha256:
            return file_sha256(destination)
        deadline = time.monotonic() + PUBLICATION_WAIT_SECONDS
        while not marker.exists():
            if time.monotonic() > deadline:
                raise ValueError(
                    f"pair product data differs and its marker never appeared: {destination}"
                )
            time.sleep(5.0)
        return None
    return file_sha256(destination)


def _publish_marker(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        pass
