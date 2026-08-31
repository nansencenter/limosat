#!/usr/bin/env python3
"""Summarize frozen March ICESat-2 expansion per track and SAR pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def event_row(result_dir: Path, pair_id: str, event: str) -> dict:
    summary = read_json(result_dir / "summary.json")
    product = summary["product"]
    common = summary.get("common_support_comparison")
    row = {
        "pair_id": pair_id,
        "event": event,
        "product": product,
        "status": summary["status"],
        "result_dir": str(result_dir),
        "common_bins": int(common["bins"]) if common is not None else 0,
    }
    if product == "ATL07" and common is not None:
        orb = summary["methods"]["orb"]["common_support"]
        aliked = summary["methods"]["aliked"]["common_support"]
        row.update(
            {
                "ridge_events": int(orb["ridge_events"]),
                "orb_shear_roughness_rho": orb[
                    "spearman_shear_vs_relative_roughness"
                ],
                "aliked_shear_roughness_rho": aliked[
                    "spearman_shear_vs_relative_roughness"
                ],
                "orb_spatial_null_p": (
                    summary.get("common_support_spatial_nulls", {})
                    .get("orb", {})
                    .get("shear_vs_relative_roughness", {})
                    .get("one_sided_p")
                ),
                "aliked_spatial_null_p": (
                    summary.get("common_support_spatial_nulls", {})
                    .get("aliked", {})
                    .get("shear_vs_relative_roughness", {})
                    .get("one_sided_p")
                ),
            }
        )
    elif product == "ATL10" and common is not None:
        orb = summary["methods"]["orb"]["common_support"]
        aliked = summary["methods"]["aliked"]["common_support"]
        row.update(
            {
                "orb_lead_bins": int(orb["bins_with_leads"]),
                "aliked_lead_bins": int(aliked["bins_with_leads"]),
                "orb_divergence_lead_rho": orb[
                    "spearman_divergence_vs_lead_fraction"
                ],
                "aliked_divergence_lead_rho": aliked[
                    "spearman_divergence_vs_lead_fraction"
                ],
            }
        )
    return row


def configured_events(root: Path) -> list[tuple[Path, str, str]]:
    ice = root / "icesat2_validation/results"
    selected = ice / "selected_expansion_v1"
    events = []
    for pair_id in ("10245_10341", "10245_10352"):
        pair_dir = selected / f"pair_{pair_id}"
        for result_dir in sorted(
            path
            for path in pair_dir.glob("*_4000m")
            if path.is_dir() and not path.name.startswith("._")
        ):
            event = result_dir.name.replace("_4000m", "")
            events.append((result_dir, pair_id, event))
    events.extend(
        [
            (
                ice / "drift_aware_v2/pair_10245_10352/atl07_0040_4km",
                "10245_10352",
                "atl07_0040_existing",
            ),
            (
                ice / "drift_aware_v2/pair_10245_10352/atl07_0044_4km",
                "10245_10352",
                "atl07_0044_existing",
            ),
            (
                ice / "pair_10245_10352/atl07_0039_v1",
                "10245_10352",
                "atl07_0039_existing",
            ),
            (
                ice / "pair_10245_10341/atl07_0039_v1",
                "10245_10341",
                "atl07_0039_existing",
            ),
        ]
    )
    return events


def plot_atl07(output: Path, table: pd.DataFrame) -> None:
    atl07 = table.loc[table["product"].eq("ATL07")].copy()
    atl07["label"] = atl07["pair_id"] + " / " + atl07["event"].str.extract(
        r"(\d{4})", expand=False
    ).fillna(atl07["event"])
    atl07 = atl07.sort_values(["pair_id", "event"]).reset_index(drop=True)
    positions = np.arange(len(atl07))
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    for index, row in atl07.iterrows():
        orb = row.get("orb_shear_roughness_rho")
        aliked = row.get("aliked_shear_roughness_rho")
        if pd.notna(orb) and pd.notna(aliked):
            ax.plot([orb, aliked], [index, index], color="0.7", linewidth=1.2)
            ax.scatter(orb, index, color="#386cb0", marker="o", s=42)
            ax.scatter(aliked, index, color="#f0027f", marker="s", s=42)
        else:
            ax.scatter(0.0, index, color="0.7", marker="x", s=45)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_yticks(positions, atl07["label"])
    ax.set_xlabel("Spearman shear vs ATL07 relative roughness on exact common 4 km bins")
    ax.set_title("Per-track March multisensor result (no pooled footprint inference)")
    ax.scatter([], [], color="#386cb0", marker="o", label="ORB")
    ax.scatter([], [], color="#f0027f", marker="s", label="ALIKED")
    ax.scatter([], [], color="0.7", marker="x", label="insufficient support")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output / "atl07_per_track_4km.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [event_row(*event) for event in configured_events(args.root)]
    table = pd.DataFrame(rows).sort_values(["product", "pair_id", "event"])
    table.to_csv(args.output_dir / "event_summary.csv", index=False)
    plot_atl07(args.output_dir, table)

    long_atl07 = table[
        table["product"].eq("ATL07") & table["pair_id"].eq("10245_10352")
    ]
    component = table[
        table["product"].eq("ATL07") & table["pair_id"].eq("10245_10341")
    ]
    atl10 = table[table["product"].eq("ATL10")]
    lines = [
        "# March multisensor expansion",
        "",
        f"The frozen expansion contains {len(table)} four-kilometre event/pair results: {len(long_atl07)} long-pair ATL07, {len(component)} component-pair ATL07, and {len(atl10)} ATL10.",
        "",
        "On 10245-to-10352, new ATL07 0030 is positive (ORB 0.182; ALIKED 0.414), while new ATL07 0041 is negative (ORB -0.202; ALIKED -0.258). The component-pair applications of both new tracks are insufficient, so neither establishes cross-pair replication.",
        "",
        "All added ATL10 event/pair applications are lead-insufficient (zero to two exact-common lead bins). They are retained as feasibility/null results and do not replicate or contradict the CryoSat-2 shear/lead result.",
        "",
        "Results remain per track and SAR pair; no pooled footprint statistic is used.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
