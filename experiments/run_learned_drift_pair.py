#!/usr/bin/env python3
"""Run the selected ALIKED workflow for one image pair."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from limosat.learned_drift import ALIKEDConfig, ALIKEDDrift, topology_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--elapsed-hours", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--target-tile-limit", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = replace(
        ALIKEDConfig(), target_tile_limit=args.target_tile_limit
    )
    started = time.perf_counter()
    tracker = ALIKEDDrift(
        config=config,
        device=args.device,
        cache_dir=args.feature_cache,
        model_cache=args.model_cache,
    )
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = tracker.track_images(args.source, args.target, args.elapsed_hours)
    total_seconds = time.perf_counter() - started

    result.matches.to_frame().to_csv(args.output_dir / "matches.csv", index=False)
    result.field.to_frame().to_csv(args.output_dir / "field.csv", index=False)
    summary = {
        "source": args.source,
        "target": args.target,
        "elapsed_hours": args.elapsed_hours,
        "config": asdict(config),
        "matches": len(result.matches),
        "grid_nodes": len(result.field),
        "available_nodes": int(result.field.available.sum()),
        "coverage_fraction": float(result.field.available.mean()),
        "fold_rejected_nodes": int(len(result.fold_rejected_indices)),
        "topology": topology_summary(result.field, config.grid_spacing_m),
        "timing_seconds": {
            "model_setup": setup_seconds,
            "matching": result.matching_seconds,
            "field_and_topology": result.field_seconds,
            "track_images_total": total_seconds,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
