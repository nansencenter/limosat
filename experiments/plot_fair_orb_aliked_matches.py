#!/usr/bin/env python3
"""Plot spatial match coverage for the frozen fair ORB/ALIKED benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from experiments.validate_icesat2_deformation import load_orb_vectors
else:
    from validate_icesat2_deformation import load_orb_vectors


PAIR_SPECS = (
    ("10245_10341", 1, 2, 2.0),
    ("10341_10352", 2, 3, 6.0),
)
METHOD_COLOURS = {"ORB": "#d95f02", "ALIKED": "#1b9e77"}


def one_path(paths, description: str) -> Path:
    selected = [path for path in paths if not path.name.startswith("._")]
    if len(selected) != 1:
        raise ValueError(f"Expected one {description}, found {selected}")
    return selected[0]


def load_pair_matches(
    benchmark_root: Path,
    label: str,
    pair_id: str,
    source_run_id: int,
    target_run_id: int,
) -> dict[str, pd.DataFrame]:
    manifest_path = one_path(
        benchmark_root.glob(f"orb/{label}/runs/*/run_manifest.json"),
        "ORB run manifest",
    )
    manifest = json.loads(manifest_path.read_text())
    database = Path(manifest["engine_url"].removeprefix("sqlite:///"))
    orb = load_orb_vectors(
        database,
        manifest["effective_run_name"],
        source_run_id,
        target_run_id,
    )
    aliked = pd.read_csv(
        benchmark_root / "aliked" / label / f"pair_{pair_id}" / "matches.csv"
    )
    required = {"source_x", "source_y", "target_x", "target_y", "dx_m", "dy_m"}
    for method, rows in (("ORB", orb), ("ALIKED", aliked)):
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"{method} {pair_id} is missing {sorted(missing)}")
    return {"ORB": orb, "ALIKED": aliked}


def spatially_balanced_sample(
    rows: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    maximum: int = 350,
) -> pd.DataFrame:
    """Select at most one match nearest each regular spatial-cell centre."""
    if len(rows) <= maximum:
        return rows.copy()
    xmin, xmax, ymin, ymax = bounds
    aspect = max((xmax - xmin) / max(ymax - ymin, 1.0), 0.25)
    ny = max(8, int(np.sqrt(maximum / aspect)))
    nx = max(8, int(np.ceil(maximum / ny)))
    xwidth = max((xmax - xmin) / nx, 1.0)
    ywidth = max((ymax - ymin) / ny, 1.0)
    work = rows.copy()
    work["_ix"] = np.clip(
        np.floor((work["source_x"] - xmin) / xwidth).astype(int), 0, nx - 1
    )
    work["_iy"] = np.clip(
        np.floor((work["source_y"] - ymin) / ywidth).astype(int), 0, ny - 1
    )
    centres_x = xmin + (work["_ix"] + 0.5) * xwidth
    centres_y = ymin + (work["_iy"] + 0.5) * ywidth
    work["_centre_distance"] = np.hypot(
        work["source_x"] - centres_x,
        work["source_y"] - centres_y,
    )
    selected = (
        work.sort_values(["_iy", "_ix", "_centre_distance"])
        .groupby(["_iy", "_ix"], sort=True, as_index=False)
        .first()
        .sort_values(["_iy", "_ix"])
    )
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected) - 1, maximum).round().astype(int)
        selected = selected.iloc[indices]
    return selected.drop(columns=["_ix", "_iy", "_centre_distance"])


def pair_bounds(matches: dict[str, pd.DataFrame], display_scale: float):
    x_values = []
    y_values = []
    for rows in matches.values():
        x_values.extend(
            [
                rows["source_x"].to_numpy(dtype=float),
                (
                    rows["source_x"].to_numpy(dtype=float)
                    + display_scale * rows["dx_m"].to_numpy(dtype=float)
                ),
            ]
        )
        y_values.extend(
            [
                rows["source_y"].to_numpy(dtype=float),
                (
                    rows["source_y"].to_numpy(dtype=float)
                    + display_scale * rows["dy_m"].to_numpy(dtype=float)
                ),
            ]
        )
    xmin, xmax = np.nanmin(np.concatenate(x_values)), np.nanmax(np.concatenate(x_values))
    ymin, ymax = np.nanmin(np.concatenate(y_values)), np.nanmax(np.concatenate(y_values))
    padding = 0.035 * max(xmax - xmin, ymax - ymin)
    return xmin - padding, xmax + padding, ymin - padding, ymax + padding


def density_histograms(
    matches: dict[str, pd.DataFrame],
    bounds: tuple[float, float, float, float],
    bins: int = 32,
):
    xmin, xmax, ymin, ymax = bounds
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)
    histograms = {}
    for method, rows in matches.items():
        histogram, _, _ = np.histogram2d(
            rows["source_x"], rows["source_y"], bins=(xedges, yedges)
        )
        histogram[histogram == 0] = np.nan
        histograms[method] = histogram.T
    return xedges, yedges, histograms


def plot_pair_row(
    axes,
    figure,
    pair_id: str,
    matches: dict[str, pd.DataFrame],
    display_scale: float,
    show_x_label: bool,
):
    bounds = pair_bounds(matches, display_scale)
    xedges, yedges, histograms = density_histograms(matches, bounds)
    maximum_count = max(
        float(np.nanmax(histogram)) for histogram in histograms.values()
    )
    normalizer = colors.LogNorm(vmin=1.0, vmax=max(maximum_count, 2.0))
    density_artist = None
    for axis, method in zip(axes, ("ORB", "ALIKED"), strict=True):
        rows = matches[method]
        density_artist = axis.pcolormesh(
            xedges / 1000.0,
            yedges / 1000.0,
            histograms[method],
            cmap="Greys",
            norm=normalizer,
            shading="flat",
            rasterized=True,
        )
        sample = spatially_balanced_sample(rows, bounds)
        axis.quiver(
            sample["source_x"] / 1000.0,
            sample["source_y"] / 1000.0,
            display_scale * sample["dx_m"] / 1000.0,
            display_scale * sample["dy_m"] / 1000.0,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=METHOD_COLOURS[method],
            alpha=0.88,
            width=0.0022,
            headwidth=3.2,
            headlength=4.0,
            headaxislength=3.7,
        )
        magnitude = np.hypot(rows["dx_m"], rows["dy_m"])
        axis.set_title(
            f"{method}: {len(rows):,} matches; "
            f"median {np.median(magnitude) / 1000.0:.1f} km",
            fontsize=11,
        )
        axis.text(
            0.015,
            0.018,
            f"{len(sample):,} spatially balanced arrows shown at {display_scale:g}×",
            transform=axis.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.82},
        )
        axis.set_xlim(bounds[0] / 1000.0, bounds[1] / 1000.0)
        axis.set_ylim(bounds[2] / 1000.0, bounds[3] / 1000.0)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="0.85", linewidth=0.5)
        if show_x_label:
            axis.set_xlabel("EPSG:3413 x (km)")
    axes[0].set_ylabel(f"{pair_id.replace('_', ' → ')}\nEPSG:3413 y (km)")
    colourbar = figure.colorbar(
        density_artist,
        ax=list(axes),
        location="right",
        fraction=0.025,
        pad=0.015,
    )
    colourbar.set_label("source matches per spatial bin")


def plot_combined(
    output: Path,
    benchmark_root: Path,
    label: str,
    pair_data: list[tuple[str, float, dict[str, pd.DataFrame]]],
):
    figure, axes = plt.subplots(
        len(pair_data),
        2,
        figsize=(13.5, 11.0),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (pair_id, display_scale, matches) in enumerate(pair_data):
        plot_pair_row(
            axes[row],
            figure,
            pair_id,
            matches,
            display_scale,
            show_x_label=row == len(pair_data) - 1,
        )
    figure.suptitle(
        "Spatial distribution of frozen ORB and ALIKED correspondences",
        fontsize=16,
    )
    figure.supxlabel(
        "Grey bins show all source matches; arrows are a deterministic spatial sample. "
        "Arrow labels state the display magnification.\n"
        "ORB: persisted production trajectories after pattern matching; "
        "ALIKED: physics-valid LightGlue correspondences before field estimation.",
        fontsize=9,
    )
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--label", default="warm_rep1")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.benchmark_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_data = []
    for pair_id, source_id, target_id, display_scale in PAIR_SPECS:
        pair_data.append(
            (
                pair_id,
                display_scale,
                load_pair_matches(
                    args.benchmark_root,
                    args.label,
                    pair_id,
                    source_id,
                    target_id,
                ),
            )
        )
    output = output_dir / f"match_spatial_comparison_{args.label}.png"
    plot_combined(output, args.benchmark_root, args.label, pair_data)
    summary = {
        "benchmark_root": str(args.benchmark_root),
        "label": args.label,
        "output": str(output),
        "semantics": {
            "ORB": "paired persisted trajectories after production descriptor, geometry, interpolation, and pattern-matching stages",
            "ALIKED": "physics-valid LightGlue feature correspondences before nearest-12 field estimation and fold rejection",
        },
        "pairs": {
            pair_id: {
                "display_vector_scale": display_scale,
                **{
                    method.lower(): {
                        "matches": int(len(rows)),
                        "median_displacement_m": float(
                            np.median(np.hypot(rows["dx_m"], rows["dy_m"]))
                        ),
                    }
                    for method, rows in matches.items()
                },
            }
            for pair_id, display_scale, matches in pair_data
        },
    }
    (output_dir / f"match_spatial_comparison_{args.label}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
