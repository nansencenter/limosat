#!/usr/bin/env python3
"""Audit OSI-455 as a search-window prior for long-gap SAR matching.

The external drift is used only to propose a target-window displacement.  It
never enters the accepted EfficientLoFTR displacement or field estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from limosat.learned_drift.config import EfficientLoFTRConfig
from limosat.learned_drift.imagery import projected_footprint
from limosat.learned_drift.routing import coarse_phase_translation


DEFAULT_CASES = (
    ROOT / "experiments/configs/osisaf455_routing_prior_audit_20260831.json"
)
DEFAULT_OUTPUT = ROOT / "results/osisaf_routing_prior_audit_20260831"
SENTINEL1_TIME_RE = re.compile(r"_(\d{8}T\d{6})_")
OSI_FILENAME = "ice_drift_nh_ease2-750_cdr-v1p0_24h-{date}1200.nc"
USABLE_OSI_FLAGS = frozenset({20, 21, 22, 23, 24, 25, 30})


@dataclass(frozen=True)
class PairCase:
    case_id: str
    cohort: str
    source_image_id: int
    target_image_id: int
    source_time: datetime
    target_time: datetime
    source_path: str
    target_path: str
    elapsed_hours: float
    truth_path: Path


@dataclass(frozen=True)
class DailySegment:
    product_end: datetime
    overlap_start: datetime
    overlap_end: datetime

    @property
    def fraction_of_day(self) -> float:
        return (self.overlap_end - self.overlap_start).total_seconds() / 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download required OSI-455 files from the configured mirror.",
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument(
        "--skip-phase",
        action="store_true",
        help="Require an existing phase_pairs.csv instead of recomputing it.",
    )
    parser.add_argument(
        "--refresh-phase",
        action="store_true",
        help="Ignore existing phase results and recompute every pair.",
    )
    return parser.parse_args()


def utc_datetime(value: str | datetime | pd.Timestamp) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def acquisition_time(path: str) -> datetime:
    match = SENTINEL1_TIME_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Sentinel-1 time is absent from {path}")
    parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    return parsed.replace(tzinfo=timezone.utc)


def load_reference_manifest(path: Path, cohort: str) -> PairCase:
    manifest = json.loads(path.read_text())
    truth_path = path.parent / "buoy_results.csv"
    if not truth_path.exists():
        raise FileNotFoundError(truth_path)
    source_time = utc_datetime(
        manifest.get("source_image_time")
        or acquisition_time(manifest["source_image_filepath"])
    )
    target_time = utc_datetime(
        manifest.get("target_image_time")
        or acquisition_time(manifest["target_image_filepath"])
    )
    source_id = int(manifest["source_image_id"])
    target_id = int(manifest["target_image_id"])
    return PairCase(
        case_id=f"{cohort}_{source_id}_{target_id}",
        cohort=cohort,
        source_image_id=source_id,
        target_image_id=target_id,
        source_time=source_time,
        target_time=target_time,
        source_path=str(manifest["source_image_filepath"]),
        target_path=str(manifest["target_image_filepath"]),
        elapsed_hours=float(manifest["elapsed_hours"]),
        truth_path=truth_path,
    )


def load_pair_output(path: Path, cohort: str) -> PairCase:
    truth_path = path / "buoy_results.csv"
    rows = pd.read_csv(truth_path)
    if rows.empty:
        raise ValueError(f"empty buoy truth file: {truth_path}")
    first = rows.iloc[0]
    source_path = str(first["source_image_filepath"])
    target_path = str(first["target_image_filepath"])
    source_id = int(first["source_image_id"])
    target_id = int(first["target_image_id"])
    return PairCase(
        case_id=f"{cohort}_{source_id}_{target_id}",
        cohort=cohort,
        source_image_id=source_id,
        target_image_id=target_id,
        source_time=acquisition_time(source_path),
        target_time=acquisition_time(target_path),
        source_path=source_path,
        target_path=target_path,
        elapsed_hours=float(first["elapsed_hours"]),
        truth_path=truth_path,
    )


def load_cases(config: dict) -> list[PairCase]:
    cases: list[PairCase] = []
    for source in config["reference_roots"]:
        root = Path(source["path"])
        manifests = sorted(root.glob(source.get("glob", "pair_*/run_manifest.json")))
        if not manifests:
            raise FileNotFoundError(f"no pair manifests below {root}")
        cases.extend(
            load_reference_manifest(path, source["cohort"]) for path in manifests
        )
    for source in config.get("reference_pairs", []):
        cases.append(
            load_reference_manifest(Path(source["manifest"]), source["cohort"])
        )
    for source in config.get("pair_outputs", []):
        cases.append(load_pair_output(Path(source["path"]), source["cohort"]))
    identities = [case.case_id for case in cases]
    if len(identities) != len(set(identities)):
        duplicates = sorted({value for value in identities if identities.count(value) > 1})
        raise ValueError(f"duplicate case ids: {duplicates}")
    for case in cases:
        measured_hours = (case.target_time - case.source_time).total_seconds() / 3600.0
        if not np.isclose(case.elapsed_hours, measured_hours, atol=1.0 / 3600.0):
            raise ValueError(f"elapsed time mismatch for {case.case_id}")
        if not Path(case.source_path).exists() or not Path(case.target_path).exists():
            raise FileNotFoundError(f"SAR input missing for {case.case_id}")
    return cases


def daily_segments(start: datetime, end: datetime) -> list[DailySegment]:
    """Split an arbitrary interval over OSI-455 noon-to-noon products."""
    start = utc_datetime(start)
    end = utc_datetime(end)
    if end <= start:
        raise ValueError("target time must follow source time")
    noon = start.replace(hour=12, minute=0, second=0, microsecond=0)
    if noon <= start:
        noon += timedelta(days=1)
    segments = []
    while noon - timedelta(days=1) < end:
        overlap_start = max(start, noon - timedelta(days=1))
        overlap_end = min(end, noon)
        if overlap_end > overlap_start:
            segments.append(DailySegment(noon, overlap_start, overlap_end))
        noon += timedelta(days=1)
    covered = sum(segment.fraction_of_day for segment in segments) * 24.0
    expected = (end - start).total_seconds() / 3600.0
    if not np.isclose(covered, expected, atol=1.0e-9):
        raise AssertionError("daily OSI segments do not cover the pair interval")
    return segments


def osi_path(cache_dir: Path, product_end: datetime) -> Path:
    filename = OSI_FILENAME.format(date=product_end.strftime("%Y%m%d"))
    return cache_dir / filename


def osi_url(url_template: str, product_end: datetime) -> str:
    return url_template.format(
        year=product_end.strftime("%Y"),
        month=product_end.strftime("%m"),
        filename=OSI_FILENAME.format(date=product_end.strftime("%Y%m%d")),
    )


def required_product_ends(cases: list[PairCase]) -> list[datetime]:
    return sorted(
        {
            segment.product_end
            for case in cases
            for segment in daily_segments(case.source_time, case.target_time)
        }
    )


def _download_one(url: str, path: Path) -> tuple[Path, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "limosat-osi455-audit/1"})
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path, path.stat().st_size


def acquire_osi_files(
    product_ends: list[datetime],
    cache_dir: Path,
    url_template: str,
    download_missing: bool,
    workers: int,
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [date for date in product_ends if not osi_path(cache_dir, date).exists()]
    if missing and not download_missing:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} OSI-455 files are missing ({preview}); use --download-missing"
        )
    if workers < 1:
        raise ValueError("download workers must be positive")
    if missing:
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = {
                pool.submit(
                    _download_one,
                    osi_url(url_template, date),
                    osi_path(cache_dir, date),
                ): date
                for date in missing
            }
            for future in as_completed(futures):
                date = futures[future]
                path, size = future.result()
                print(f"downloaded {date.date()} {size} bytes -> {path}", flush=True)
    paths = [osi_path(cache_dir, date) for date in product_ends]
    for path in paths:
        with xr.open_dataset(path) as dataset:
            if dataset.attrs.get("product_id") != "OSI-455":
                raise ValueError(f"unexpected product in {path}")
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_osi_field(
    dataset: xr.Dataset,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly sample vectors, with a valid nearest-cell fallback."""
    coordinates = {
        "xc": xr.DataArray(np.asarray(x_m) / 1000.0, dims="point"),
        "yc": xr.DataArray(np.asarray(y_m) / 1000.0, dims="point"),
    }
    dx = dataset["dX"].isel(time=0).interp(coordinates, method="linear").values
    dy = dataset["dY"].isel(time=0).interp(coordinates, method="linear").values
    uncertainty = (
        dataset["uncert_dX_and_dY"]
        .isel(time=0)
        .interp(coordinates, method="linear")
        .values
    )
    flag = (
        dataset["status_flag"]
        .isel(time=0)
        .sel(coordinates, method="nearest")
        .values.astype(np.int16)
    )
    nearest_dx = dataset["dX"].isel(time=0).sel(coordinates, method="nearest").values
    nearest_dy = dataset["dY"].isel(time=0).sel(coordinates, method="nearest").values
    nearest_uncertainty = (
        dataset["uncert_dX_and_dY"]
        .isel(time=0)
        .sel(coordinates, method="nearest")
        .values
    )
    linear_valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(uncertainty)
    nearest_valid = np.isin(flag, list(USABLE_OSI_FLAGS))
    fallback = ~linear_valid & nearest_valid
    dx[fallback] = nearest_dx[fallback]
    dy[fallback] = nearest_dy[fallback]
    uncertainty[fallback] = nearest_uncertainty[fallback]
    valid = (
        np.isfinite(dx)
        & np.isfinite(dy)
        & np.isfinite(uncertainty)
        & nearest_valid
    )
    return dx, dy, uncertainty, np.where(valid, flag, -1)


def advect_with_osi455(
    source_xy_m: np.ndarray,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    analysis_epsg: int = 3413,
) -> dict[str, np.ndarray]:
    """Integrate daily mean OSI-455 velocity over an arbitrary SAR interval."""
    source_xy_m = np.asarray(source_xy_m, dtype=np.float64)
    if source_xy_m.ndim != 2 or source_xy_m.shape[1] != 2:
        raise ValueError("source positions must have shape (n, 2)")
    to_osi = Transformer.from_crs(analysis_epsg, 6931, always_xy=True)
    from_osi = Transformer.from_crs(6931, analysis_epsg, always_xy=True)
    current_x, current_y = to_osi.transform(source_xy_m[:, 0], source_xy_m[:, 1])
    current_x = np.asarray(current_x, dtype=np.float64)
    current_y = np.asarray(current_y, dtype=np.float64)
    available = np.ones(len(source_xy_m), dtype=bool)
    uncertainty_m = np.zeros(len(source_xy_m), dtype=np.float64)
    wind_fraction = np.zeros(len(source_xy_m), dtype=np.float64)
    flags: list[list[int]] = [[] for _ in range(len(source_xy_m))]
    for segment in daily_segments(start, end):
        path = osi_path(cache_dir, segment.product_end)
        with xr.open_dataset(path) as dataset:
            dx_km, dy_km, sigma_km, step_flag = sample_osi_field(
                dataset, current_x, current_y
            )
        step_valid = step_flag >= 0
        available &= step_valid
        fraction = segment.fraction_of_day
        valid_indices = np.flatnonzero(step_valid)
        current_x[valid_indices] += dx_km[valid_indices] * 1000.0 * fraction
        current_y[valid_indices] += dy_km[valid_indices] * 1000.0 * fraction
        uncertainty_m[valid_indices] += (
            sigma_km[valid_indices] * 1000.0 * fraction
        )
        wind_fraction[valid_indices] += (
            np.isin(step_flag[valid_indices], [23, 24, 25]).astype(float) * fraction
        )
        for index in valid_indices:
            flags[index].append(int(step_flag[index]))
    target_x, target_y = from_osi.transform(current_x, current_y)
    target_xy = np.column_stack((target_x, target_y)).astype(np.float64)
    displacement = target_xy - source_xy_m
    displacement[~available] = np.nan
    uncertainty_m[~available] = np.nan
    wind_fraction[~available] = np.nan
    return {
        "displacement_m": displacement,
        "available": available,
        "uncertainty_m": uncertainty_m,
        "wind_fraction": wind_fraction,
        "flags": np.asarray(
            [";".join(map(str, values)) if values else "" for values in flags],
            dtype=object,
        ),
    }


def phase_for_case(case: PairCase, maximum_speed_m_per_day: float) -> dict:
    config = EfficientLoFTRConfig(maximum_speed_m_per_day=maximum_speed_m_per_day)
    maximum_displacement_m = config.maximum_displacement_m(case.elapsed_hours)
    source_footprint = projected_footprint(case.source_path, config.analysis_epsg)
    target_footprint = projected_footprint(case.target_path, config.analysis_epsg)
    source_domain = source_footprint.intersection(
        target_footprint.buffer(maximum_displacement_m)
    )
    if source_domain.is_empty:
        raise ValueError("source and target have no physics-reachable overlap")
    started = time.perf_counter()
    phase = coarse_phase_translation(
        case.source_path,
        case.target_path,
        source_domain,
        maximum_displacement_m,
        config.analysis_epsg,
        config.transform_grid_spacing_px,
    )
    magnitude = float(np.linalg.norm(phase.displacement_m))
    return {
        "case_id": case.case_id,
        "phase_dx_m": phase.displacement_m[0],
        "phase_dy_m": phase.displacement_m[1],
        "phase_magnitude_m": magnitude,
        "phase_response": phase.response,
        "phase_overlap_fraction": phase.overlap_fraction,
        "phase_pixel_size_m": phase.pixel_size_m,
        "phase_pixels": phase.pixels,
        "phase_clipped": bool(
            np.isclose(magnitude, maximum_displacement_m, rtol=0, atol=1.0)
        ),
        "phase_seconds": time.perf_counter() - started,
        "phase_error": "",
    }


def compute_phase_table(
    cases: list[PairCase],
    path: Path,
    maximum_speed_m_per_day: float,
    skip_phase: bool,
    refresh_phase: bool,
) -> pd.DataFrame:
    if path.exists() and not refresh_phase:
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame()
    if skip_phase:
        missing = set(case.case_id for case in cases) - set(existing.get("case_id", []))
        if missing:
            raise ValueError(f"phase cache is missing {len(missing)} cases")
        return existing
    records = [] if refresh_phase else existing.to_dict("records")
    completed = {str(row["case_id"]) for row in records}
    for index, case in enumerate(cases, start=1):
        if case.case_id in completed:
            continue
        try:
            row = phase_for_case(case, maximum_speed_m_per_day)
        except Exception as exc:  # Preserve failures as audit evidence.
            row = {
                "case_id": case.case_id,
                "phase_dx_m": np.nan,
                "phase_dy_m": np.nan,
                "phase_magnitude_m": np.nan,
                "phase_response": np.nan,
                "phase_overlap_fraction": np.nan,
                "phase_pixel_size_m": np.nan,
                "phase_pixels": np.nan,
                "phase_clipped": False,
                "phase_seconds": np.nan,
                "phase_error": f"{type(exc).__name__}: {exc}",
            }
        records.append(row)
        pd.DataFrame(records).to_csv(path, index=False)
        print(f"phase {index}/{len(cases)} {case.case_id}", flush=True)
    return pd.DataFrame(records)


def duration_band(hours: float) -> str:
    if hours < 12:
        return "00_to_12h"
    if hours < 30:
        return "12_to_30h"
    if hours < 60:
        return "30_to_60h"
    if hours < 96:
        return "60_to_96h"
    return "96h_plus"


def sic_stratum(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    if "pack" in text or "ge80" in text:
        return "pack_ice"
    if "marginal" in text or "miz" in text or "15_80" in text:
        return "miz"
    if "fast" in text:
        return "fast_ice"
    if "open" in text or "lt15" in text:
        return "open_water"
    return "unknown"


def agreement_band(distance_m: float) -> str:
    if not np.isfinite(distance_m):
        return "unavailable"
    if distance_m <= 5_000:
        return "agree_le05km"
    if distance_m <= 10_000:
        return "differ_05_10km"
    if distance_m <= 20_000:
        return "differ_10_20km"
    return "differ_gt20km"


def response_band(response: float) -> str:
    if not np.isfinite(response):
        return "unavailable"
    if response < 0.02:
        return "very_low_lt0p02"
    if response < 0.05:
        return "low_0p02_0p05"
    if response < 0.15:
        return "moderate_0p05_0p15"
    return "high_ge0p15"


def provenance(flags: str) -> str:
    values = {int(value) for value in str(flags).split(";") if value}
    if not values:
        return "unavailable"
    if 24 in values:
        return "wind_filled"
    if 25 in values:
        return "wind_blended"
    if 23 in values:
        return "wind_parameter_gapfilled"
    if values & {20, 21, 22}:
        return "processed_satellite"
    if values == {30}:
        return "nominal_satellite"
    return "mixed_other"


def truth_rows(case: PairCase) -> pd.DataFrame:
    rows = pd.read_csv(case.truth_path, dtype={"buoy_id": str})
    required = {"buoy_id", "source_x", "source_y", "truth_dx_m", "truth_dy_m"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"{case.truth_path} is missing {sorted(missing)}")
    rows = rows.dropna(subset=["source_x", "source_y", "truth_dx_m", "truth_dy_m"])
    if rows.empty:
        raise ValueError(f"no usable truth rows for {case.case_id}")
    return rows.reset_index(drop=True)


def routing_rows(
    case: PairCase,
    phase: pd.Series,
    cache_dir: Path,
    analysis_epsg: int,
) -> pd.DataFrame:
    rows = truth_rows(case)
    source_xy = rows[["source_x", "source_y"]].to_numpy(np.float64)
    truth = rows[["truth_dx_m", "truth_dy_m"]].to_numpy(np.float64)
    osi = advect_with_osi455(
        source_xy, case.source_time, case.target_time, cache_dir, analysis_epsg
    )
    phase_vector = np.array(
        [phase["phase_dx_m"], phase["phase_dy_m"]], dtype=np.float64
    )
    phase_vectors = np.repeat(phase_vector[None, :], len(rows), axis=0)
    phase_error = np.linalg.norm(phase_vectors - truth, axis=1)
    osi_error = np.linalg.norm(osi["displacement_m"] - truth, axis=1)
    same_error = np.linalg.norm(truth, axis=1)
    phase_osi_distance = np.linalg.norm(
        phase_vectors - osi["displacement_m"], axis=1
    )
    output = pd.DataFrame(
        {
            "case_id": case.case_id,
            "cohort": case.cohort,
            "source_image_id": case.source_image_id,
            "target_image_id": case.target_image_id,
            "source_time": case.source_time.isoformat(),
            "target_time": case.target_time.isoformat(),
            "elapsed_hours": case.elapsed_hours,
            "duration_band": duration_band(case.elapsed_hours),
            "buoy_id": rows["buoy_id"].astype(str),
            "source_x_m": source_xy[:, 0],
            "source_y_m": source_xy[:, 1],
            "truth_dx_m": truth[:, 0],
            "truth_dy_m": truth[:, 1],
            "same_center_error_m": same_error,
            "phase_dx_m": phase_vector[0],
            "phase_dy_m": phase_vector[1],
            "phase_error_m": phase_error,
            "phase_response": phase["phase_response"],
            "phase_response_band": response_band(float(phase["phase_response"])),
            "phase_clipped": bool(phase["phase_clipped"]),
            "osi455_dx_m": osi["displacement_m"][:, 0],
            "osi455_dy_m": osi["displacement_m"][:, 1],
            "osi455_available": osi["available"],
            "osi455_error_m": osi_error,
            "osi455_uncertainty_m": osi["uncertainty_m"],
            "osi455_wind_fraction": osi["wind_fraction"],
            "osi455_flags": osi["flags"],
            "phase_osi455_distance_m": phase_osi_distance,
        }
    )
    output["phase_osi455_agreement"] = output["phase_osi455_distance_m"].map(
        agreement_band
    )
    output["osi455_provenance"] = output["osi455_flags"].map(provenance)
    output.loc[~output["osi455_available"], "osi455_provenance"] = "unavailable"
    output["oracle_either_error_m"] = np.fmin(phase_error, osi_error)
    output["osi455_normalized_error"] = (
        output["osi455_error_m"] / output["osi455_uncertainty_m"]
    )
    output["experiment_split"] = (
        rows["experiment_split"]
        if "experiment_split" in rows
        else "external_holdout"
    )
    source_regime = rows.get(
        "source_sic_regime", pd.Series("unknown", index=rows.index)
    )
    output["sic_stratum"] = source_regime.map(sic_stratum)
    for threshold_km in (2.56, 5, 10, 15, 20):
        label = str(threshold_km).replace(".", "p")
        for method in ("same_center", "phase", "osi455", "oracle_either"):
            output[f"{method}_le_{label}km"] = (
                output[f"{method}_error_m"] <= threshold_km * 1000.0
            )
    return output


def method_summary(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records = []
    group_key = groups[0] if len(groups) == 1 else groups
    keys: list[tuple] = (
        [()] if not groups else list(rows.groupby(group_key, dropna=False).groups)
    )
    for key in keys:
        if not isinstance(key, tuple):
            key = (key,)
        subset = rows
        for column, value in zip(groups, key):
            subset = subset.loc[subset[column] == value]
        for method in ("same_center", "phase", "osi455", "oracle_either"):
            errors = subset[f"{method}_error_m"]
            valid = errors[np.isfinite(errors)]
            record = {column: value for column, value in zip(groups, key)}
            record.update(
                {
                    "method": method,
                    "expected": len(subset),
                    "available": len(valid),
                    "available_fraction": len(valid) / len(subset),
                    "median_error_km": (
                        float(valid.median() / 1000.0) if len(valid) else np.nan
                    ),
                    "p90_error_km": (
                        float(valid.quantile(0.9) / 1000.0) if len(valid) else np.nan
                    ),
                    "within_5km_fraction_expected": float(
                        (errors <= 5_000).fillna(False).mean()
                    ),
                    "within_10km_fraction_expected": float(
                        (errors <= 10_000).fillna(False).mean()
                    ),
                    "within_15km_fraction_expected": float(
                        (errors <= 15_000).fillna(False).mean()
                    ),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def paired_summary(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    paired = rows.dropna(subset=["phase_error_m", "osi455_error_m"])
    records = []
    group_key = groups[0] if len(groups) == 1 else groups
    grouped = [((), paired)] if not groups else paired.groupby(group_key, dropna=False)
    for key, subset in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        record = {column: value for column, value in zip(groups, key)}
        delta = subset["phase_error_m"] - subset["osi455_error_m"]
        record.update(
            {
                "paired_samples": len(subset),
                "phase_median_error_km": subset["phase_error_m"].median() / 1000.0,
                "osi455_median_error_km": subset["osi455_error_m"].median() / 1000.0,
                "osi455_win_fraction": float((delta > 0).mean()),
                "median_phase_minus_osi455_error_km": delta.median() / 1000.0,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def pair_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for case_id, subset in rows.groupby("case_id", sort=False):
        first = subset.iloc[0]
        paired = subset.dropna(subset=["phase_error_m", "osi455_error_m"])
        records.append(
            {
                "case_id": case_id,
                "cohort": first["cohort"],
                "source_image_id": first["source_image_id"],
                "target_image_id": first["target_image_id"],
                "elapsed_hours": first["elapsed_hours"],
                "duration_band": first["duration_band"],
                "buoys": len(subset),
                "phase_response": first["phase_response"],
                "phase_dx_km": first["phase_dx_m"] / 1000.0,
                "phase_dy_km": first["phase_dy_m"] / 1000.0,
                "osi455_available_fraction": subset["osi455_available"].mean(),
                "same_center_median_error_km": subset["same_center_error_m"].median()
                / 1000.0,
                "phase_median_error_km": subset["phase_error_m"].median() / 1000.0,
                "osi455_median_error_km": subset["osi455_error_m"].median() / 1000.0,
                "osi455_win_fraction": (
                    float((paired["phase_error_m"] > paired["osi455_error_m"]).mean())
                    if len(paired)
                    else np.nan
                ),
                "phase_osi455_median_distance_km": subset[
                    "phase_osi455_distance_m"
                ].median()
                / 1000.0,
                "osi455_provenance": ";".join(
                    sorted(set(subset["osi455_provenance"]))
                ),
            }
        )
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "(no rows)"
    printable = frame.copy()
    numeric = printable.select_dtypes(include=[np.number]).columns
    printable[numeric] = printable[numeric].round(digits)
    headings = [str(column) for column in printable.columns]
    lines = ["| " + " | ".join(headings) + " |"]
    lines.append("| " + " | ".join("---" for _ in headings) + " |")
    for values in printable.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    cases: list[PairCase],
    samples: pd.DataFrame,
    phase: pd.DataFrame,
    osi_files: list[Path],
    config: dict,
) -> dict:
    overall = method_summary(samples, [])
    by_duration = method_summary(samples, ["duration_band"])
    paired_duration = paired_summary(samples, ["duration_band"])
    paired_agreement = paired_summary(samples, ["phase_osi455_agreement"])
    paired_sic = paired_summary(samples, ["sic_stratum"])
    paired_split = paired_summary(samples, ["experiment_split"])
    paired_provenance = paired_summary(samples, ["osi455_provenance"])
    pairs = pair_summary(samples)
    overall.to_csv(output_dir / "method_summary_overall.csv", index=False)
    by_duration.to_csv(output_dir / "method_summary_by_duration.csv", index=False)
    paired_duration.to_csv(output_dir / "paired_summary_by_duration.csv", index=False)
    paired_agreement.to_csv(output_dir / "paired_summary_by_agreement.csv", index=False)
    paired_sic.to_csv(output_dir / "paired_summary_by_sic.csv", index=False)
    paired_split.to_csv(output_dir / "paired_summary_by_split.csv", index=False)
    paired_provenance.to_csv(
        output_dir / "paired_summary_by_provenance.csv", index=False
    )
    pairs.to_csv(output_dir / "pair_summary.csv", index=False)
    long_pairs = pairs.loc[pairs["elapsed_hours"] >= 30].sort_values("elapsed_hours")
    report = "\n".join(
        [
            "# OSI-455 routing-prior audit",
            "",
            "OSI-455 and phase correlation are evaluated only as target-window priors; "
            "the buoy truth is not used to form either prior.",
            "",
            "Dataset DOI: https://doi.org/10.15770/EUM_SAF_OSI_0012. "
            "Files were retrieved from the configured academic THREDDS mirror and hashed.",
            "",
            f"- Pair cases: {len(cases)}",
            f"- Buoy-pair samples: {len(samples)}",
            f"- OSI-455 daily files: {len(osi_files)}",
            f"- Phase failures: {int(phase['phase_dx_m'].isna().sum())}",
            "- Analysis CRS: EPSG:3413; displacement/error units: metres in samples, kilometres in summaries.",
            "- OSI native CRS/resolution: EPSG:6931, 75 km; daily noon-to-noon vectors are fractionally integrated.",
            "- OSI uncertainty accumulation is conservative linear addition across temporal segments.",
            "- 2.56 km is the guaranteed per-axis routing slack between the 35.84 km tile core and 40.96 km patch; "
            "5/10/15 km are diagnostic tolerances, not guarantees.",
            "",
            "## Overall",
            "",
            markdown_table(overall),
            "",
            "## Paired phase versus OSI-455 by gap duration",
            "",
            markdown_table(paired_duration),
            "",
            "## Paired phase versus OSI-455 by their agreement",
            "",
            markdown_table(paired_agreement),
            "",
            "## Paired phase versus OSI-455 by sea-ice stratum",
            "",
            markdown_table(paired_sic),
            "",
            "## Paired phase versus OSI-455 by frozen experiment split",
            "",
            markdown_table(paired_split),
            "",
            "## Paired phase versus OSI-455 by OSI provenance",
            "",
            markdown_table(paired_provenance),
            "",
            "## Cases at least 30 hours",
            "",
            markdown_table(long_pairs),
            "",
            "## Interpretation guardrails",
            "",
            "- The full-70 archive is overwhelmingly pack ice; MIZ estimates are sparse and fast ice is absent.",
            "- IABP/N-ICE buoys validate routing accuracy but do not measure spatial field continuity.",
            "- The broad audit may select mechanistic MPS confirmation cases, but those confirmations are not an independent test set.",
            "- Pure/blended wind flags are reported separately because OSI-455 is not wholly satellite-derived there.",
        ]
    )
    (output_dir / "REPORT.md").write_text(report + "\n")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases_config": str(config.get("config_path", "")),
        "case_count": len(cases),
        "sample_count": len(samples),
        "analysis_epsg": int(config["analysis_epsg"]),
        "maximum_speed_m_per_day": float(config["maximum_speed_m_per_day"]),
        "osi455_url_template": config["osi455_url_template"],
        "osi455_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in osi_files
        ],
        "outputs": {
            "samples": "routing_samples.csv",
            "phase": "phase_pairs.csv",
            "pairs": "pair_summary.csv",
            "report": "REPORT.md",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.cases.read_text())
    config["config_path"] = str(args.cases)
    cases = load_cases(config)
    product_ends = required_product_ends(cases)
    cache_dir = args.output_dir / "osisaf_cache"
    osi_files = acquire_osi_files(
        product_ends,
        cache_dir,
        config["osi455_url_template"],
        args.download_missing,
        args.download_workers,
    )
    phase = compute_phase_table(
        cases,
        args.output_dir / "phase_pairs.csv",
        float(config["maximum_speed_m_per_day"]),
        args.skip_phase,
        args.refresh_phase,
    ).set_index("case_id")
    sample_tables = []
    for index, case in enumerate(cases, start=1):
        sample_tables.append(
            routing_rows(
                case,
                phase.loc[case.case_id],
                cache_dir,
                int(config["analysis_epsg"]),
            )
        )
        print(f"OSI audit {index}/{len(cases)} {case.case_id}", flush=True)
    samples = pd.concat(sample_tables, ignore_index=True)
    samples.to_csv(args.output_dir / "routing_samples.csv", index=False)
    manifest = write_report(
        args.output_dir,
        cases,
        samples,
        phase.reset_index(),
        osi_files,
        config,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
