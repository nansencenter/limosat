# Primary composition

`limosat compose-parquet CONFIG OUTPUT_DIRECTORY` reads the immutable pair products
of every primary pair in the plan and writes trajectory points and directly
measured trajectory extensions to Parquet. It changes neither the run database nor
the pair products, so composition can be repeated with other composition settings
without rematching.

At each image (in UTC order) a trajectory continues from its most recent position
through the incoming primary pair field that supports it best: the most recent
source, then more agreeing matches, lower residual and support radius. New
trajectories are seeded at the field nodes of the image's outgoing primary pairs
(`add_as_coverage_enters`) unless an existing trajectory is within
`new_point_exclusion_radius_m` (production 3 km). Gaps are not stored: a trajectory
has no row on an image that a pair from its last position reaches without
supporting it, and it may continue later through another pair.

The local field defaults are 4 km grid spacing and a 6.4 km maximum triangle
edge; pair planning remains on a fixed 4 km grid. Recomposition of an existing
archive must use that archive's frozen configuration, not current defaults.

Every pair of the deterministic primary plan must have a product; a shared pair
store may hold other runs' pairs as well, which are ignored. Every marker, NPZ
file, array-content checksum, field checksum, matching fingerprint, scene and SIC
checksum, and pair identity is verified. The producer implementation hashes are
retained as provenance but do not have to equal the downstream consumer code
hash, and products from several producer versions may be composed together when
they share one matching fingerprint. This allows composition and publication code
to evolve without invalidating completed GPU measurements.

The command writes these files atomically:

- `trajectory-points-v1.parquet`, with created and observed rows;
- `trajectory-extensions-qc-v1.parquet`, containing only directly measured
  source-to-target extensions; and
- `trajectory-merges-v1.parquet`, only with merging on (below); and
- `composition-manifest-v1.json`, written last as the completion marker.

Before choosing an incoming pair, composition checks each inferred endpoint
against that target image's raster bounds and invalid-pixel band. An endpoint
outside the raster or on a pixel with mask value at least 2 cannot become an
observed point; another valid incoming pair may still continue the trajectory.
If none does, the trajectory has a gap there. An endpoint that is invalid on the
target frame but valid on another frame of the same acquisition pass (same
platform and absolute orbit, seconds apart) is accepted: in the March 2020
archive about a quarter of gate rejections were such frame-seam positions.

The optional `surface` section adds land and point open-water exclusion. With
`land_raster` (an EPSG:3413 inland-distance `.npz`, such as the 1 km
OpenStreetMap raster), pixels at least `land_minimum_inland_km` inland are
removed from both matching tiles, and such endpoints are invalid; the coastal
band stays matchable so drift is kept up to the coast. With `point_sic_root`
(daily AMSR2 ASI GeoTIFFs, nearest cell, optional coarser fallback), endpoints
and new seeds below `point_open_water_threshold_percent` on the image date are
rejected; a missing concentration never rejects. Seeds are filtered by both.
The tile-level OSI SAF open-water gate is unchanged. On a Fram Strait week
(1-7 March 2020) this removed all 1,264 land or open-water positions (0.3 %)
while ice cell-days fell by 0.04 %.

## Gap filling and shear point estimates

Both act during composition only and are not in the matching fingerprint; both
need retained matches (`retain_pair_matches: true`).

Gap filling (`field.gap_fill_*`) applies where a point has no usable triangle:
first a rigid triangle of accepted nodes up to `gap_fill_rigid_edge_m` apart whose
displacements differ by at most `gap_fill_rigid_spread_m`, then the point's own raw
matches under the node rule with the `gap_fill_point_*` settings (point-local
consensus). Production: 12.8 km / 300 m and 12 matches within 3 km, 8 agreeing
within 500 m. On the January 2020 test week lost tracks fell 5.9 %; over MOSAiC
January-April 2020 13 %; at MOSAiC buoys the added samples had median errors of
56-61 m against 47-51 m for ordinary samples.

Shear point estimates (`field.shear_point_estimates`, production on): a point is
normally moved by linear interpolation between three 4 km nodes, which blends the
two sides of a lead or shear line running between them. Where those nodes disagree
by more than the point's own point-local estimate disagrees with its matches, that
estimate is used instead; points without a sample get their own estimate before any
rigid-triangle bridge. On the January-April 2020 stored pairs the p90 displacement
error at MOSAiC and IABP buoys fell from 152 to 124 m.

The missing-node fallback (`field.missing_node_fallback`) is off: at MOSAiC and
Beaufort buoys the points it adds were 2-3x less accurate than measured nodes.

## Merging duplicate trajectories (optional, off)

With `trajectories.merge_radius_m` set, when positions of two or more trajectories
in one image lie within that distance (chained), only the one with the longest
history continues (then more selected matches, lower residual, identity); the others
end there and are recorded in `trajectory-merges-v1.parquet`. With
`merge_maximum_younger_age_hours` only young duplicates are merged: a trajectory is
ended only if it is at most that old and was never positioned in the same image as
its survivor. `scripts/prepare_chain_olivia.py --merge-radius-m 250
--merge-maximum-younger-age-hours 48` turns this on.

## Initial QC policy

This branch does not import the ORB production QC protocol wholesale. Its first
ELoFTR-specific policy is intentionally small:

- reject extensions above 60 km/day;
- mark extensions above the configured matcher speed for review;
- calculate leave-one-out diagnostics using only extensions from the exact same
  source/target image pair; and
- mark strong local anomalies for review without rejecting them.

The neighbour defaults are a 20 km search radius, 8 minimum and 24 maximum
neighbours. The ELoFTR fields are already built from locally consistent matches
on a 4 km grid, so trajectory neighbours are correlated evidence. Automatic
local rejection requires a separate validation experiment. Sparse locations
with fewer than eight neighbours receive no local decision and remain subject
only to the speed gate.

`trajectory-extensions-qc-v1.parquet` records accepted, review and rejected rows.
The `accepted` column excludes only hard-speed failures. PySIDA and public
exporters can select `accepted`, while validation can examine the review rows.
