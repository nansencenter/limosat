#!/usr/bin/env python3
"""Resolve the QC-filtered Sentinel-1 acquisition queue to ASF URLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACQUISITIONS = (
    ROOT
    / "results/iabp_s1_stratified_coverage/"
    "sentinel1_acquisition_manifest_qc_filtered.csv"
)
DEFAULT_SOURCE_MANIFESTS = Path(
    "/Users/seachu/projects/arktalas-deployment/s1_download_manifests"
)
DEFAULT_DATA_ROOT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020"
)


def product_name_from_url(url: str) -> str:
    return Path(urlparse(url).path).stem


def logical_acquisition_key(product_name: str) -> str:
    parts = product_name.split("_")
    if len(parts) < 9:
        raise ValueError(f"Unexpected Sentinel-1 product name: {product_name}")
    return "_".join(parts[:-1])


def load_url_index(manifest_dir: Path) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    index: dict[str, str] = {}
    logical_lists: dict[str, list[str]] = {}
    for path in sorted(manifest_dir.glob("S1_EW_ARCTIC_*.txt")):
        for line in path.read_text().splitlines():
            url = line.strip()
            if not url:
                continue
            name = product_name_from_url(url)
            existing = index.get(name)
            if existing is not None and existing != url:
                raise ValueError(f"Conflicting URLs for {name}")
            index[name] = url
            key = logical_acquisition_key(name)
            logical_lists.setdefault(key, [])
            if url not in logical_lists[key]:
                logical_lists[key].append(url)
    return index, {key: tuple(values) for key, values in logical_lists.items()}


def unique_logical_url(
    product_name: str, logical_url_index: dict[str, tuple[str, ...]]
) -> str | None:
    candidates = logical_url_index.get(logical_acquisition_key(product_name), ())
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous logical-publication substitution for {product_name}: "
            f"{len(candidates)} ASF URLs"
        )
    return candidates[0] if candidates else None


def resolve_downloads(
    acquisitions: pd.DataFrame,
    url_index: dict[str, str],
    logical_url_index: dict[str, tuple[str, ...]],
    maximum_priority_tier: int,
    data_root: Path,
) -> pd.DataFrame:
    selected = acquisitions[
        acquisitions["priority_tier"].le(maximum_priority_tier)
        & acquisitions["download_decision"].eq("ready_for_restore_or_download")
        & ~acquisitions["standard_vae_pixels_local"].astype(bool)
    ].copy()
    selected["asf_url"] = selected["sentinel1_product_name"].map(url_index)
    missing_exact = selected["asf_url"].isna()
    selected.loc[missing_exact, "asf_url"] = selected.loc[
        missing_exact, "sentinel1_product_name"
    ].map(lambda value: unique_logical_url(value, logical_url_index))
    missing = selected[selected["asf_url"].isna()]["sentinel1_product_name"].tolist()
    if missing:
        raise ValueError(f"{len(missing)} selected products missing ASF URLs: {missing[:3]}")
    selected["resolved_product_name"] = selected["asf_url"].map(product_name_from_url)
    selected["logical_publication_substitution"] = selected[
        "resolved_product_name"
    ].ne(selected["sentinel1_product_name"])
    selected["raw_zip_path"] = selected.apply(
        lambda row: str(
            data_root
            / "sentinel1"
            / "raw"
            / pd.Timestamp(row["image_time"]).strftime("%Y")
            / pd.Timestamp(row["image_time"]).strftime("%m")
            / f"{row['resolved_product_name']}.zip"
        ),
        axis=1,
    )
    selected["standard_vae_output_path"] = selected.apply(
        lambda row: str(
            data_root
            / "sentinel1"
            / "standard_vae"
            / pd.Timestamp(row["image_time"]).strftime("%Y")
            / pd.Timestamp(row["image_time"]).strftime("%m")
            / f"{row['resolved_product_name']}.tiff"
        ),
        axis=1,
    )
    return selected.sort_values(["priority_tier", "image_time", "image_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=Path, default=DEFAULT_ACQUISITIONS)
    parser.add_argument("--source-manifests", type=Path, default=DEFAULT_SOURCE_MANIFESTS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--maximum-priority-tier", type=int, default=1)
    parser.add_argument(
        "--output-prefix",
        default="tier1",
        help="Prefix for inventory, URL-list, and plan filenames.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/iabp_s1_stratified_coverage",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    acquisitions = pd.read_csv(args.acquisitions)
    url_index, logical_url_index = load_url_index(args.source_manifests)
    resolved = resolve_downloads(
        acquisitions,
        url_index,
        logical_url_index,
        args.maximum_priority_tier,
        args.data_root,
    )
    prefix = args.output_prefix
    resolved.to_csv(
        args.out_dir / f"{prefix}_sentinel1_download_inventory.csv", index=False
    )
    (args.out_dir / f"{prefix}_asf_urls.txt").write_text(
        "\n".join(resolved["asf_url"].astype(str)) + "\n"
    )
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "acquisition_manifest": str(args.acquisitions),
        "source_manifest_dir": str(args.source_manifests),
        "data_root": str(args.data_root),
        "maximum_priority_tier": args.maximum_priority_tier,
        "products": len(resolved),
        "logical_publication_substitutions": int(
            resolved["logical_publication_substitution"].sum()
        ),
        "destination_is_kingston": str(args.data_root).startswith("/Volumes/KINGSTON/"),
    }
    (args.out_dir / f"{prefix}_download_plan.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(resolved[["image_time", "sentinel1_product_name", "raw_zip_path"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
