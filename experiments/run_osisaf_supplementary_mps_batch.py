#!/usr/bin/env python3
"""Complete previously audited >=30 h cases with matched direct-phase controls."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.osisaf_routing_prior_audit import load_pair_output
from experiments.run_osisaf_long_gap_mps_batch import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_REPO,
    run_command,
)


DEFAULT_CASES = (
    ROOT
    / "experiments/configs/osisaf455_supplementary_long_gap_mps_20260831.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--efficientloftr-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--efficientloftr-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    return parser.parse_args()


def materialize_reference(pair_dir: Path, cohort: str, reference_root: Path) -> Path:
    if (pair_dir / "run_manifest.json").exists():
        return pair_dir
    case = load_pair_output(pair_dir, cohort)
    output = reference_root / f"pair_{case.source_image_id}_{case.target_image_id}"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(case.truth_path, output / "buoy_results.csv")
    manifest = {
        "status": "complete",
        "source_image_id": case.source_image_id,
        "target_image_id": case.target_image_id,
        "source_image_time": case.source_time.isoformat(),
        "target_image_time": case.target_time.isoformat(),
        "source_image_filepath": case.source_path,
        "target_image_filepath": case.target_path,
        "elapsed_hours": case.elapsed_hours,
        "truth_source": "copied_from_existing_pair_output",
        "analysis_crs": "EPSG:3413",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def direct_phase_command(args: argparse.Namespace, reference: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments/run_efficientloftr_sequence.py"),
        "--reference-pair-run-dir",
        str(reference),
        "--efficientloftr-repo",
        str(args.efficientloftr_repo),
        "--efficientloftr-checkpoint",
        str(args.efficientloftr_checkpoint),
        "--output-dir",
        str(output),
        "--routing-mode",
        "sequential_local",
        "--initial-routing",
        "phase_correlation",
        "--device",
        "mps",
    ]


def assisted_command(
    args: argparse.Namespace,
    case: dict,
    reference: Path,
    baseline: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments/run_efficientloftr_osisaf_pair.py"),
        "--pair-dir",
        str(reference),
        "--cohort",
        case["cohort"],
        "--role",
        case["role"],
        "--cluster",
        case["cluster"],
        "--baseline-summary",
        str(baseline),
        "--efficientloftr-repo",
        str(args.efficientloftr_repo),
        "--efficientloftr-checkpoint",
        str(args.efficientloftr_checkpoint),
        "--output-dir",
        str(output),
        "--device",
        "mps",
        "--fallback",
        "same_center",
    ]


def main() -> int:
    args = parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    reference_root = args.audit_root / "supplementary_references"
    phase_root = args.audit_root / "phase_gap_baselines"
    confirmation_root = args.audit_root / "mps_confirmations"
    log_root = args.audit_root / "supplementary_mps_logs"
    report_path = args.audit_root / "supplementary_gap_mps_batch.json"
    report = {"status": "running", "device": "mps", "completed": []}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        pair_dir = Path(case["pair_dir"])
        reference = materialize_reference(pair_dir, case["cohort"], reference_root)
        manifest = json.loads((reference / "run_manifest.json").read_text())
        source_id = int(manifest["source_image_id"])
        target_id = int(manifest["target_image_id"])
        case_id = f"{case['cohort']}_{source_id}_{target_id}"
        baseline_value = case.get("baseline_summary")
        if baseline_value:
            baseline = Path(baseline_value)
            phase_seconds = 0.0
        else:
            output = phase_root / f"{case_id}_phase"
            baseline = output / f"pair_{source_id}_{target_id}" / "summary.json"
            print(f"[{index}/{len(cases)}] {case_id}: direct phase", flush=True)
            phase_seconds = run_command(
                direct_phase_command(args, reference, output),
                log_root / f"{case_id}_phase.log",
            )
        summary = json.loads(baseline.read_text())
        if set(summary["routing"]["source_counts"]) != {"coarse_phase_translation"}:
            raise ValueError(f"baseline is not direct phase: {baseline}")
        assisted_output = (
            confirmation_root / f"{case_id}_osi_samecenter_fallback"
        )
        print(f"[{index}/{len(cases)}] {case_id}: OSI-455", flush=True)
        assisted_seconds = run_command(
            assisted_command(args, case, reference, baseline, assisted_output),
            log_root / f"{case_id}_osi455.log",
        )
        report["completed"].append(
            {
                "case_id": case_id,
                "cluster": case["cluster"],
                "phase_seconds": phase_seconds,
                "osi455_seconds": assisted_seconds,
                "baseline_summary": str(baseline),
                "assisted_manifest": str(assisted_output / "run_manifest.json"),
            }
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"completed {len(cases)} supplementary cases in {report['elapsed_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
