#!/usr/bin/env python3
"""Run matched direct-phase and OSI-455 long-gap pairs on Apple MPS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROOT = ROOT / "results/osisaf_routing_prior_audit_20260831"
DEFAULT_PLAN = DEFAULT_AUDIT_ROOT / "expanded_gap_references/expansion_plan.json"
DEFAULT_REPO = Path("/private/tmp/limosat_efficientloftr_official")
DEFAULT_CHECKPOINT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/"
    "model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--efficientloftr-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--efficientloftr-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run only selected case IDs; repeat to select more than one.",
    )
    return parser.parse_args()


def commands_for_case(args: argparse.Namespace, case: dict) -> tuple[list[str], list[str], Path, Path]:
    source_id = int(case["source_image_id"])
    target_id = int(case["target_image_id"])
    reference = Path(case["reference_pair_dir"])
    phase_root = (
        args.audit_root
        / "phase_gap_baselines"
        / f"full70_simulated_gap_{source_id}_{target_id}_phase"
    )
    baseline_summary = phase_root / f"pair_{source_id}_{target_id}" / "summary.json"
    assisted_root = (
        args.audit_root
        / "mps_confirmations"
        / f"full70_simulated_gap_{source_id}_{target_id}_osi_samecenter_fallback"
    )
    common = [
        "--efficientloftr-repo",
        str(args.efficientloftr_repo),
        "--efficientloftr-checkpoint",
        str(args.efficientloftr_checkpoint),
        "--device",
        "mps",
    ]
    phase = [
        sys.executable,
        str(ROOT / "experiments/run_efficientloftr_sequence.py"),
        "--reference-pair-run-dir",
        str(reference),
        "--output-dir",
        str(phase_root),
        "--routing-mode",
        "sequential_local",
        "--initial-routing",
        "phase_correlation",
        *common,
    ]
    assisted = [
        sys.executable,
        str(ROOT / "experiments/run_efficientloftr_osisaf_pair.py"),
        "--pair-dir",
        str(reference),
        "--cohort",
        "full70_simulated_gap",
        "--role",
        "predeclared_expanded_long_gap",
        "--cluster",
        str(case["sequence"]),
        "--baseline-summary",
        str(baseline_summary),
        "--output-dir",
        str(assisted_root),
        "--fallback",
        "same_center",
        *common,
    ]
    return phase, assisted, baseline_summary, assisted_root / "run_manifest.json"


def run_command(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND\n{' '.join(command)}\n\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr}"
    )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")
    return elapsed


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    selected = plan["selected"]
    if args.case_id:
        requested = set(args.case_id)
        selected = [case for case in selected if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in selected}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    if not selected:
        raise ValueError("no cases selected")

    batch_path = args.audit_root / "expanded_gap_mps_batch.json"
    log_root = args.audit_root / "expanded_gap_mps_logs"
    batch = {
        "status": "running",
        "device": "mps",
        "selection_rule": plan["selection_rule"],
        "requested_cases": len(selected),
        "completed": [],
    }
    batch_path.write_text(json.dumps(batch, indent=2) + "\n")
    total_started = time.perf_counter()
    for index, case in enumerate(selected, start=1):
        case_id = case["case_id"]
        phase, assisted, baseline_summary, assisted_manifest = commands_for_case(
            args, case
        )
        print(f"[{index}/{len(selected)}] {case_id}: direct phase", flush=True)
        phase_seconds = run_command(phase, log_root / f"{case_id}_phase.log")
        if not baseline_summary.exists():
            raise FileNotFoundError(baseline_summary)
        print(f"[{index}/{len(selected)}] {case_id}: OSI-455", flush=True)
        assisted_seconds = run_command(assisted, log_root / f"{case_id}_osi455.log")
        if not assisted_manifest.exists():
            raise FileNotFoundError(assisted_manifest)
        batch["completed"].append(
            {
                "case_id": case_id,
                "cluster": case["sequence"],
                "elapsed_hours": case["elapsed_hours"],
                "phase_seconds": phase_seconds,
                "osi455_seconds": assisted_seconds,
                "baseline_summary": str(baseline_summary),
                "assisted_manifest": str(assisted_manifest),
            }
        )
        batch_path.write_text(json.dumps(batch, indent=2) + "\n")
    batch["status"] = "complete"
    batch["elapsed_seconds"] = time.perf_counter() - total_started
    batch_path.write_text(json.dumps(batch, indent=2) + "\n")
    print(f"completed {len(selected)} matched cases in {batch['elapsed_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
