"""Catalogue-driven image identity and UTC chronology."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pyproj import Transformer
from shapely import from_wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


@dataclass(frozen=True)
class ImageRecord:
    """One immutable catalogue image."""

    image_id: str
    path: Path
    time_utc: datetime
    component_id: str = "default"
    footprint: BaseGeometry | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id cannot be empty")
        if not self.component_id.strip():
            raise ValueError("component_id cannot be empty")
        if self.time_utc.tzinfo is None or self.time_utc.utcoffset() is None:
            raise ValueError("image chronology must be timezone-aware")
        object.__setattr__(self, "time_utc", self.time_utc.astimezone(timezone.utc))
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True)
class ImagePair:
    source: ImageRecord
    target: ImageRecord

    def __post_init__(self) -> None:
        if self.source.component_id != self.target.component_id:
            raise ValueError("pair images must belong to one component")
        if self.target.time_utc <= self.source.time_utc:
            raise ValueError("pair chronology must be strictly increasing")

    @property
    def elapsed_seconds(self) -> float:
        return (self.target.time_utc - self.source.time_utc).total_seconds()

    @property
    def pair_id(self) -> str:
        return f"{self.source.image_id}__{self.target.image_id}"


class ImageCatalogue:
    """Validated images grouped into deterministic chronological components."""

    def __init__(self, records: Iterable[ImageRecord]) -> None:
        ordered = sorted(
            records,
            key=lambda item: (item.component_id, item.time_utc, item.image_id),
        )
        if not ordered:
            raise ValueError("catalogue contains no images")
        ids = [item.image_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("catalogue image_id values must be globally unique")
        self.records = tuple(ordered)
        for component in self.components().values():
            times = [item.time_utc for item in component]
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("component image times must be strictly increasing")

    def components(self) -> dict[str, tuple[ImageRecord, ...]]:
        grouped: dict[str, list[ImageRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.component_id, []).append(record)
        return {name: tuple(values) for name, values in grouped.items()}

    def adjacent_pairs(self, component_id: str) -> tuple[ImagePair, ...]:
        images = self.components()[component_id]
        return tuple(ImagePair(a, b) for a, b in zip(images, images[1:]))


def load_catalogue(path: str | Path, analysis_epsg: int = 3413) -> ImageCatalogue:
    """Read CSV or GeoJSON with image_id, path, time_utc, and component_id."""
    catalogue_path = Path(path).resolve()
    if catalogue_path.suffix.lower() in {".json", ".geojson"}:
        document = json.loads(catalogue_path.read_text(encoding="utf-8"))
        source_epsg = _geojson_epsg(document)
        rows = []
        for feature in document.get("features", []):
            row = dict(feature.get("properties") or {})
            geometry = feature.get("geometry")
            row["footprint"] = None if geometry is None else shape(geometry)
            rows.append(row)
    else:
        source_epsg = analysis_epsg
        with catalogue_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["footprint"] = (
                from_wkt(row["footprint_wkt"]) if row.get("footprint_wkt") else None
            )
    projector = (
        None
        if source_epsg == analysis_epsg
        else Transformer.from_crs(
            source_epsg, analysis_epsg, always_xy=True
        ).transform
    )
    records = []
    for row in rows:
        row = dict(row)
        row["path"] = row.get("path") or row.get("filepath")
        row["time_utc"] = row.get("time_utc") or row.get("timestamp")
        missing = [
            name for name in ("image_id", "path", "time_utc") if not row.get(name)
        ]
        if missing:
            raise ValueError(f"catalogue row is missing values: {missing}")
        image_path = Path(str(row["path"])).expanduser()
        if not image_path.is_absolute():
            image_path = catalogue_path.parent / image_path
        footprint = row.get("footprint")
        if footprint is not None and projector is not None:
            footprint = transform(projector, footprint)
        records.append(
            ImageRecord(
                image_id=str(row["image_id"]),
                path=image_path,
                time_utc=_parse_utc(str(row["time_utc"])),
                component_id=str(row.get("component_id") or "default"),
                footprint=footprint,
            )
        )
    return ImageCatalogue(records)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a UTC offset: {value}")
    return parsed.astimezone(timezone.utc)


def _geojson_epsg(document: dict) -> int:
    name = str(
        (document.get("crs") or {}).get("properties", {}).get("name", "EPSG:3413")
    )
    try:
        return int(name.rsplit(":", 1)[-1])
    except ValueError as error:
        raise ValueError(f"unsupported GeoJSON CRS: {name}") from error
