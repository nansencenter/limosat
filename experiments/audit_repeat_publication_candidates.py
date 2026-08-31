#!/usr/bin/env python3
"""Validate catalogue repeat-publication candidates against official ASF search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"
ASF_SEARCH_ENDPOINT = "https://api.daac.asf.alaska.edu/services/search/param"


def logical_product_key(product_name: str) -> str:
    return "_".join(str(product_name).split("_")[:-1])


def official_grd_names(logical_key: str, timeout_seconds: float) -> list[str]:
    query = urlencode(
        {
            "granule_list": f"{logical_key}*",
            "maxResults": 20,
            "output": "geojson",
        }
    )
    with urlopen(f"{ASF_SEARCH_ENDPOINT}?{query}", timeout=timeout_seconds) as stream:
        payload = json.load(stream)
    return sorted(
        {
            feature["properties"]["sceneName"]
            for feature in payload.get("features", [])
            if feature.get("properties", {}).get("processingLevel") == "GRD_MD"
        }
    )


def audit_candidates(
    candidates: pd.DataFrame,
    timeout_seconds: float,
) -> pd.DataFrame:
    official_by_key: dict[str, list[str]] = {}
    records: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        logical_key = logical_product_key(row.primary_product_name)
        if logical_key not in official_by_key:
            official_by_key[logical_key] = official_grd_names(
                logical_key, timeout_seconds
            )
        official = official_by_key[logical_key]
        primary_is_official = row.primary_product_name in official
        candidate_is_official = row.repeat_product_name in official
        records.append(
            {
                "repeat_control_id": row.repeat_control_id,
                "primary_product_name": row.primary_product_name,
                "candidate_repeat_product_name": row.repeat_product_name,
                "official_grd_md_product_names": ";".join(official),
                "primary_is_official_asf_grd_md": primary_is_official,
                "candidate_repeat_is_official_asf_grd_md": candidate_is_official,
                "candidate_direct_url": row.repeat_asf_url,
                "candidate_audit_status": (
                    "available_repeat_publication"
                    if primary_is_official and candidate_is_official
                    else "stale_catalog_duplicate_not_official_repeat"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()
    candidates = pd.read_csv(
        args.results_dir / "full70_repeat_publication_controls.csv"
    )
    audited = audit_candidates(candidates, args.timeout_seconds)
    audited.to_csv(
        args.results_dir / "full70_repeat_publication_candidate_audit.csv",
        index=False,
    )
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "official_api": ASF_SEARCH_ENDPOINT,
        "catalog_candidates": len(audited),
        "official_repeat_publications": int(
            audited["candidate_repeat_is_official_asf_grd_md"].sum()
        ),
        "status_counts": audited["candidate_audit_status"].value_counts().to_dict(),
        "conclusion": (
            "A catalogue duplicate is not a repeat-publication control unless both "
            "product names are returned as GRD_MD scenes by official ASF search."
        ),
    }
    (
        args.results_dir / "full70_repeat_publication_candidate_audit_summary.json"
    ).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
