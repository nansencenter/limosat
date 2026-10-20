"""Per-link quality attributes for a composed run with chained continuation passes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import RunConfig
from .store import file_sha256

ANNOTATED_LINKS_SCHEMA_VERSION = 1
_DAY_US = 86_400_000_000


def annotate_links(
    config: RunConfig,
    final_base: str | Path,
    destination: str | Path,
    *,
    recovery_compositions: dict[int, str | Path] | None = None,
    maximum_neighbour_gap_days: float = 4.0,
    minimum_neighbour_gap_days: float = 1.0 / 24.0,
) -> dict:
    """Write every link with its pass and an along-track consistency check.

    Links are the base extensions (pass 0) plus each continuation pass's
    augmentation extensions, found through the final base's lineage unless
    ``recovery_compositions`` maps pass numbers to recovery composition
    directories explicitly. For each link, the displacement predicted by the
    same track's previous and next links (each at most
    ``maximum_neighbour_gap_days`` away, velocities averaged and scaled to this
    link's lag) is recorded with the deviation from it. Neighbours closer than
    ``minimum_neighbour_gap_days`` (default 1 h, e.g. frames of one pass) are
    skipped in favour of the next earlier or later position. Nothing is rejected.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    base = Path(final_base)
    output = Path(destination)
    links_path = output / "annotated-links-v1.parquet"
    manifest_path = output / "annotated-links-manifest-v1.json"
    for path in (links_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    base_manifest = json.loads((base / "composition-manifest-v1.json").read_text(encoding="utf-8"))
    if base_manifest.get("config_sha256") != config.sha256:
        raise ValueError("final base configuration differs")
    passes = {
        int(entry["pass_index"]): Path(entry["recovery_manifest"]).parent
        for entry in base_manifest.get("continuation_lineage", [])
    }
    for index, directory in (recovery_compositions or {}).items():
        passes[int(index)] = Path(directory)
    sources = [(0, base / "trajectory-extensions-qc-v1.parquet")] + [
        (index, directory / "trajectory-augmentation-extensions-qc-v1.parquet")
        for index, directory in sorted(passes.items())
    ]
    tables = []
    for index, path in sources:
        table = pq.read_table(path)
        table = table.cast(pa.schema(
            [f.with_type(pa.large_string()) if f.type == pa.string() else f for f in table.schema]
        ))
        tables.append(table.append_column("link_pass", pa.array(np.full(table.num_rows, index, np.int16))))
    links = pa.concat_tables(tables, promote_options="default")
    del tables

    points = pq.read_table(
        base / "trajectory-points-v1.parquet",
        columns=["trajectory_id", "time_utc", "state", "x_m", "y_m"],
        filters=[("state", "in", ["created", "observed", "reappeared"])],
    )
    codes = pc.dictionary_encode(
        pa.chunked_array([points["trajectory_id"].cast(pa.large_string()),
                          links["trajectory_id"]])
    ).combine_chunks().indices.to_numpy().astype(np.int64)
    point_code, link_code = codes[: points.num_rows], codes[points.num_rows:]
    point_t = points["time_utc"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy() / _DAY_US
    point_xy = np.c_[points["x_m"].to_numpy(), points["y_m"].to_numpy()]
    order = np.lexsort((point_t, point_code))
    point_code, point_t, point_xy = point_code[order], point_t[order], point_xy[order]
    span = float(np.ceil(max(point_t.max(), 1.0) + 10.0))
    point_key = point_code * span + point_t
    t0 = links["start_time_utc"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy() / _DAY_US
    t1 = links["end_time_utc"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy() / _DAY_US
    xy0 = np.c_[links["x0_m"].to_numpy(), links["y0_m"].to_numpy()]
    xy1 = np.c_[links["x1_m"].to_numpy(), links["y1_m"].to_numpy()]
    eps = 1e-6
    gap = max(minimum_neighbour_gap_days, eps)
    before = np.searchsorted(point_key, link_code * span + t0 - gap, side="right") - 1
    after = np.searchsorted(point_key, link_code * span + t1 + gap, side="left")
    velocity_sum = np.zeros((links.num_rows, 2))
    used = np.zeros(links.num_rows, np.int8)
    previous_gap = np.full(links.num_rows, np.nan)
    next_gap = np.full(links.num_rows, np.nan)
    ok = (before >= 0)
    b = np.clip(before, 0, len(point_t) - 1)
    ok &= (point_code[b] == link_code) & (t0 - point_t[b] >= gap) & (t0 - point_t[b] <= maximum_neighbour_gap_days)
    velocity_sum[ok] += (xy0[ok] - point_xy[b[ok]]) / (t0[ok] - point_t[b[ok]])[:, None]
    previous_gap[ok] = t0[ok] - point_t[b[ok]]
    used += ok
    a = np.clip(after, 0, len(point_t) - 1)
    ok = (after < len(point_t)) & (point_code[a] == link_code) & (point_t[a] - t1 >= gap) & (point_t[a] - t1 <= maximum_neighbour_gap_days)
    velocity_sum[ok] += (point_xy[a[ok]] - xy1[ok]) / (point_t[a[ok]] - t1[ok])[:, None]
    next_gap[ok] = point_t[a[ok]] - t1[ok]
    used += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        predicted = velocity_sum / used[:, None] * (t1 - t0)[:, None]
    predicted[used == 0] = np.nan
    deviation = np.hypot(*(xy1 - xy0 - predicted).T)
    links = links.append_column("track_neighbour_links", pa.array(used))
    links = links.append_column("track_predicted_dx_m", pa.array(predicted[:, 0]))
    links = links.append_column("track_predicted_dy_m", pa.array(predicted[:, 1]))
    links = links.append_column("track_deviation_m", pa.array(deviation))
    links = links.append_column("track_previous_gap_days", pa.array(previous_gap))
    links = links.append_column("track_next_gap_days", pa.array(next_gap))
    output.mkdir(parents=True, exist_ok=True)
    temporary = links_path.with_name(links_path.name + f".tmp-{os.getpid()}")
    pq.write_table(links, temporary, compression="zstd")
    os.replace(temporary, links_path)
    counts = {
        "links": links.num_rows,
        "links_by_pass": {str(k): int(v) for k, v in zip(*np.unique(links["link_pass"].to_numpy(), return_counts=True))},
        "links_with_track_check": int((used > 0).sum()),
    }
    manifest = {
        "annotated_links_schema_version": ANNOTATED_LINKS_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_sha256": config.sha256,
        "final_base_manifest_sha256": file_sha256(base / "composition-manifest-v1.json"),
        "sources": [{"pass": index, "path": str(path), "sha256": file_sha256(path)} for index, path in sources],
        "maximum_neighbour_gap_days": maximum_neighbour_gap_days,
        "minimum_neighbour_gap_days": minimum_neighbour_gap_days,
        "counts": counts,
        "links": str(links_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), **counts}
