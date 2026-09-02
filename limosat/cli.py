"""LiMOSAT command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .catalog import load_catalogue
from .config import load_config
from .finalize import finalize_products
from .pair_products import PairProductStore
from .planning import build_candidate_plan, recovery_candidates, select_overlap_probe
from .run import LiMOSATRun
from .stages import RunStages
from .store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="limosat")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run the complete global catalogue workflow"
    )
    run.add_argument("config", type=Path)
    prepare = commands.add_parser(
        "prepare", help="register the catalogue and candidate image-pair plan"
    )
    prepare.add_argument("config", type=Path)
    pairs = commands.add_parser(
        "pairs", help="measure one deterministic batch of image pairs"
    )
    pairs.add_argument("config", type=Path)
    pairs.add_argument("--kind", choices=("primary", "recovery"), required=True)
    pairs.add_argument("--batch-index", type=int, default=0)
    pairs.add_argument("--batch-count", type=int, default=1)
    compose = commands.add_parser(
        "compose", help="import completed pair products and compose trajectories"
    )
    compose.add_argument("config", type=Path)
    compose.add_argument("--phase", choices=("primary", "final"), required=True)
    status = commands.add_parser("status", help="show durable local run state")
    status.add_argument("config", type=Path)
    plan = commands.add_parser(
        "plan", help="inspect candidate image pairs without loading the matcher"
    )
    plan.add_argument("config", type=Path)
    plan.add_argument(
        "--probe-bin",
        action="append",
        default=[],
        metavar="LOWER:UPPER",
        help="select a bounded overlap stratum; may be repeated",
    )
    plan.add_argument("--maximum-per-bin", type=int, default=5)
    finalize = commands.add_parser(
        "finalize", help="validate and package a completed native run"
    )
    finalize.add_argument("config", type=Path)
    finalize.add_argument(
        "--skip-parquet",
        action="store_true",
        help="write the assessment summary without the optional Parquet export",
    )
    finalize.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command == "run":
        result = LiMOSATRun(config).execute(["limosat", "run", str(arguments.config)])
    elif arguments.command == "prepare":
        result = RunStages(config).prepare()
    elif arguments.command == "pairs":
        result = RunStages(config).process_pairs(
            arguments.kind,
            batch_index=arguments.batch_index,
            batch_count=arguments.batch_count,
        )
    elif arguments.command == "compose":
        result = RunStages(config).compose(
            arguments.phase,
            [
                "limosat",
                "compose",
                str(arguments.config),
                "--phase",
                arguments.phase,
            ],
        )
    elif arguments.command == "status":
        result = RunStore(config, read_only=True).status()
        products = PairProductStore(config)
        result["pair_products"] = {
            kind: products.count(kind) for kind in ("primary", "recovery")
        }
    elif arguments.command == "finalize":
        result = finalize_products(
            config,
            export_parquet=not arguments.skip_parquet,
            batch_size=arguments.batch_size,
        )
    else:
        catalogue = load_catalogue(config.catalogue, config.analysis_epsg)
        candidate_plan = build_candidate_plan(
            catalogue,
            config.routing,
            grid_spacing_m=config.field.grid_spacing_m,
            maximum_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
        )
        pairs = candidate_plan.pairs
        recovery = recovery_candidates(
            pairs,
            config.routing.maximum_recovery_elapsed_hours,
        )
        primary_per_target: dict[str, int] = {}
        for item in pairs:
            if item.selection == "primary":
                target_id = item.pair.target.image_id
                primary_per_target[target_id] = primary_per_target.get(target_id, 0) + 1
        primary_count_distribution: dict[str, int] = {}
        for count in primary_per_target.values():
            key = str(count)
            primary_count_distribution[key] = (
                primary_count_distribution.get(key, 0) + 1
            )
        result = {
            "catalogue_images": len(catalogue.records),
            "candidate_pairs": len(pairs),
            "primary_pairs": sum(item.selection == "primary" for item in pairs),
            "targets_with_primary_pairs": len(primary_per_target),
            "maximum_primary_pairs_per_target": max(
                primary_per_target.values(), default=0
            ),
            "primary_pairs_per_target_distribution": primary_count_distribution,
            "eligible_recovery_pairs": len(recovery),
            "planning_counts": candidate_plan.exclusion_counts,
        }
        if arguments.probe_bin:
            bins = tuple(_parse_overlap_bin(value) for value in arguments.probe_bin)
            probe = select_overlap_probe(pairs, bins, arguments.maximum_per_bin)
            result["overlap_probe"] = {
                label: [
                    {
                        "pair_id": item.pair.pair_id,
                        "overlap_fraction": item.overlap_fraction,
                        "overlap_area_m2": item.overlap_area_m2,
                        "elapsed_seconds": item.pair.elapsed_seconds,
                        "skipped_images": item.skipped_images,
                    }
                    for item in values
                ]
                for label, values in probe.items()
            }
            result["candidate_pair_ids"] = sorted(
                {
                    item.pair.pair_id
                    for values in probe.values()
                    for item in values
                }
            )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _parse_overlap_bin(value: str) -> tuple[float, float]:
    try:
        lower, upper = (float(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid overlap bin {value!r}; expected LOWER:UPPER"
        ) from error
    return lower, upper
