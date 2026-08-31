#!/usr/bin/env python3
"""Replay fixed ALIKED proposals through bounded pattern-matching variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_aliked_orb_northup import error_m, pattern_refine


VARIANTS = (
    ("direct_no_pattern", None, "none", "integer"),
    ("legacy_border48", 48, "none", "integer"),
    ("aligned_border4_integer_template", 4, "aligned_integer", "integer"),
    ("aligned_border4_bilinear_template", 4, "aligned_integer", "bilinear"),
    ("quadratic_border2_integer_template", 2, "quadratic", "integer"),
    ("quadratic_border4_integer_template", 4, "quadratic", "integer"),
    ("quadratic_border8_integer_template", 8, "quadratic", "integer"),
    ("quadratic_border4_bilinear_template", 4, "quadratic", "bilinear"),
    ("quadratic_border8_bilinear_template", 8, "quadratic", "bilinear"),
    ("continuous_border4_bilinear_template", 4, "continuous", "bilinear"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: pd.DataFrame) -> list[dict]:
    summary = []
    for (panel, variant), group in rows.groupby(["panel", "variant"], sort=True):
        accepted = group["accepted"].fillna(False)
        errors = group.loc[accepted, "error_m"].dropna()
        record = {
            "panel": panel,
            "variant": variant,
            "cases": int(len(group)),
            "accepted": int(accepted.sum()),
            "median_error_m": float(errors.median()) if len(errors) else None,
            "p90_error_m": float(errors.quantile(0.90)) if len(errors) else None,
            "mean_seconds": float(group["seconds"].mean()),
        }
        for threshold in (100, 250, 500, 1000, 2000):
            record[f"within_{threshold}m"] = int((accepted & group["error_m"].le(threshold)).sum())
        summary.append(record)
    return summary


def add_q4_direct_fallback(rows: pd.DataFrame) -> pd.DataFrame:
    """Retain a direct proposal when tight quadratic refinement is rejected."""
    direct_name = "direct_no_pattern"
    q4_name = "quadratic_border4_bilinear_template"
    direct = rows.loc[rows["variant"].eq(direct_name)].copy()
    q4 = rows.loc[rows["variant"].eq(q4_name)].copy()
    if direct.empty or q4.empty:
        return rows
    keys = ["transition_id", "panel"]
    direct = direct.set_index(keys)
    q4 = q4.set_index(keys)
    if not direct.index.equals(q4.index):
        raise ValueError("direct and q4 rows do not describe identical cases")
    fallback = q4.copy()
    rejected = ~fallback["accepted"].fillna(False)
    q4_seconds = fallback["seconds"].copy()
    fallback.loc[rejected] = direct.loc[rejected]
    fallback["seconds"] = q4_seconds
    fallback["variant"] = "quadratic_border4_bilinear_with_direct_fallback"
    return pd.concat([rows, fallback.reset_index()], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--proposal-results", type=Path)
    parser.add_argument("--proposal-policy")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant[0] for variant in VARIANTS],
        default=[variant[0] for variant in VARIANTS],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input, low_memory=False)
    if args.proposal_results is not None:
        if not args.proposal_policy:
            parser.error("--proposal-policy is required with --proposal-results")
        proposals = pd.read_csv(args.proposal_results, low_memory=False)
        proposals = proposals.loc[proposals["policy"].eq(args.proposal_policy)].copy()
        proposal_columns = [
            "transition_id",
            "available",
            "proposal_dx_m",
            "proposal_dy_m",
        ]
        proposals = proposals[proposal_columns].drop_duplicates()
        if proposals["transition_id"].duplicated().any():
            raise ValueError("proposal results disagree across panel labels")
        source = source.drop(
            columns=[
                "aliked_available",
                "aliked_proposal_dx_m",
                "aliked_proposal_dy_m",
            ]
        ).merge(proposals, on="transition_id", how="left", validate="one_to_one")
        source = source.rename(
            columns={
                "available": "aliked_available",
                "proposal_dx_m": "aliked_proposal_dx_m",
                "proposal_dy_m": "aliked_proposal_dy_m",
            }
        )
    source = source.loc[source["aliked_available"].fillna(False)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in source.itertuples(index=False):
        source_xy = np.array(
            [
                getattr(row, "tracking_source_x", row.source_x),
                getattr(row, "tracking_source_y", row.source_y),
            ],
            dtype=float,
        )
        proposal_xy = source_xy + np.array(
            [row.aliked_proposal_dx_m, row.aliked_proposal_dy_m], dtype=float
        )
        truth_xy = np.array([row.source_x, row.source_y], dtype=float) + np.array(
            [row.truth_dx_m, row.truth_dy_m], dtype=float
        )
        panels = []
        if row.representative_panel:
            panels.append("representative")
        if row.challenge_panel:
            panels.append("challenge")
        for name, border, subpixel, sampling in VARIANTS:
            if name not in args.variants:
                continue
            started = time.perf_counter()
            if border is None:
                result = {
                    "available": True,
                    "accepted": True,
                    "corrected_x": proposal_xy[0],
                    "corrected_y": proposal_xy[1],
                    "correlation": np.nan,
                    "subpixel_status": "not_run",
                }
            else:
                result = pattern_refine(
                    row.source_image_filepath,
                    row.target_image_filepath,
                    source_xy,
                    proposal_xy,
                    template_half_size=16,
                    search_border=border,
                    subpixel_method=subpixel,
                    template_sampling=sampling,
                )
            elapsed = time.perf_counter() - started
            accepted = bool(result.get("accepted", False))
            endpoint_error = (
                error_m(result["corrected_x"], result["corrected_y"], truth_xy)
                if result.get("available", False)
                else np.nan
            )
            for panel in panels:
                records.append(
                    {
                        "transition_id": row.transition_id,
                        "panel": panel,
                        "variant": name,
                        "border_pixels": border,
                        "subpixel_method": subpixel,
                        "template_sampling": sampling,
                        "available": bool(result.get("available", False)),
                        "accepted": accepted,
                        "correlation": result.get("correlation", np.nan),
                        "error_m": endpoint_error,
                        "correction_pixels": result.get("correction_pixels", np.nan),
                        "subpixel_col": result.get("subpixel_col", np.nan),
                        "subpixel_row": result.get("subpixel_row", np.nan),
                        "subpixel_status": result.get("subpixel_status"),
                        "seconds": elapsed,
                    }
                )

    results = pd.DataFrame.from_records(records)
    results = add_q4_direct_fallback(results)
    results.to_csv(args.output_dir / "variant_results.csv", index=False)
    summary = summarize(results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "proposal_results": (
            str(args.proposal_results.resolve()) if args.proposal_results else None
        ),
        "proposal_results_sha256": (
            sha256(args.proposal_results) if args.proposal_results else None
        ),
        "proposal_policy": args.proposal_policy,
        "variants": args.variants,
        "rows": int(len(results)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
