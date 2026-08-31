# Operating LiMOSAT

## Catalogue

LiMOSAT accepts CSV or GeoJSON. Every row or feature requires:

| Field | Meaning |
| --- | --- |
| `image_id` | Globally unique stable image identity |
| `path` | Raster path, absolute or relative to the catalogue |
| `time_utc` | ISO-8601 timestamp with an explicit UTC offset |
| `component_id` | Optional sequence/component identity; defaults to `default` |

CSV may include `footprint_wkt` in EPSG:3413. GeoJSON geometry is transformed
from its declared CRS to EPSG:3413. If a footprint is absent, LiMOSAT derives
it from the raster affine transform or ground-control points. Existing
catalogues using `filepath` and `timestamp` are accepted as direct aliases.

Within each component, images are ordered by UTC time and must be strictly
chronological. Adjacent pairs are always processed. Separate components never
share fields or trajectories.

## Matching and recovery

The first pair uses coarse phase correlation unless `initial: same_center` is
configured. Later pairs use only earlier accepted fields: a global median prior
with a local supported-field refinement by default. EfficientLoFTR runs on
north-up tiles whose non-overlapping source cores prevent duplicate ownership.
Endpoint validity, elapsed-time speed, local vector consensus, and fold gates
are applied in EPSG:3413 metres.

Residual-edge recovery may rerun a tile when a large routing residual and
target-edge pressure agree. Sequence recovery is separate: after adjacent
trajectories are composed, LiMOSAT identifies points measured at an earlier
image but dormant at a later image and matches only source-tile cores within
the configured 6.4 km buffer. The default permits one skipped image. Recovery
fields can reconnect trajectories, but only adjacent full fields generate
deformation cells.

## Resume and outputs

SQLite is the source of run truth. A pair is marked `running` before inference;
its field nodes and completion record are committed in one transaction. A
stopped `running` or `failed` pair is safe to retry. A `complete` pair is loaded
and never overwritten by normal resume.

The output directory contains `run-manifest-v1.json`. Scientific arrays and
tables live in SQLite; no result products belong in Git. `limosat status`
reports the run and pair status counts.

For a controlled restart under a changed configuration, choose a new `run_id`
and database/output path. Reusing a `run_id` with a different resolved config is
rejected.
