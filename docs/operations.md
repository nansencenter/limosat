# Operating LiMOSAT

## Catalogue

LiMOSAT accepts CSV or GeoJSON. Every row or feature requires:

| Field | Meaning |
| --- | --- |
| `image_id` | Globally unique stable image identity |
| `path` | Raster path, absolute or relative to the catalogue |
| `time_utc` | ISO-8601 timestamp with an explicit UTC offset |
| `component_id` | Optional compute-planning label; defaults to `default` |

CSV may include `footprint_wkt` in EPSG:3413. GeoJSON geometry is transformed
from its declared CRS to EPSG:3413. If a footprint is absent, LiMOSAT derives
it from the raster affine transform or ground-control points. Existing
catalogues using `filepath` and `timestamp` are accepted as direct aliases.

Images have one deterministic global UTC chronology. A component label can
group pair computation, but it is not a scientific trajectory boundary.
Parcels can continue through measured image pairs whose images carry different
component labels.

## Candidate and primary image pairs

Candidate image pairs are frozen deterministically before inference. Defaults
require 1--96 hours elapsed time and at least 0.25 overlap relative to the
smaller footprint. For each target image, candidates from the most recent
source acquisition are primary pairs. Equal-time sources are retained so the
global composer can use measured field quality to choose deterministically.
The remaining candidates are available only for measured-loss recovery.

Primary pair fields have no preceding-pair dependency and can be claimed and
processed independently. `pair_workers` controls local concurrency. Complete
fields are immutable and are loaded on resume.

## Matching and recovery

Each independently scheduled primary pair uses coarse phase correlation unless
`initial: same_center` is configured. EfficientLoFTR runs on north-up tiles
whose non-overlapping source cores prevent duplicate ownership. Endpoint
validity, elapsed-time speed, local vector consensus, and fold gates are
applied in EPSG:3413 metres.

Residual field recovery may rerun a tile when a large routing residual and
target-boundary pressure agree. Catalogue recovery is separate: after all
primary pair fields are composed globally, LiMOSAT identifies parcels measured
at an earlier source image but dormant at a candidate target image and matches
only source-tile cores within the configured 6.4 km buffer. The default checks
the most recent one unselected candidate per target. Recovery fields can
reconnect trajectories, but only full primary pair fields generate deformation
cells. After recovery completes, trajectories are recomposed from all
completed fields.

At a target image, the composer evaluates parcels whose last measured row is
at the source of an incoming completed pair field. A parcel unsupported by all
such fields gets a dormant row with SQL `NULL` coordinates. Images with no
incoming measurement do not cause catalogue-wide Cartesian dormant rows. New
seeds are compared with all measured parcel positions active at that image,
irrespective of compute label.

## Resume and outputs

SQLite is the source of run truth. A pair is marked `running` before inference;
its field nodes and completion record are committed in one transaction. A
stopped `running` or `failed` pair is safe to retry. A `complete` pair is loaded
and never overwritten by normal resume.

The output directory contains `run-manifest-v2.json`. Scientific arrays and
tables live in SQLite; no result products belong in Git. `limosat status`
reports the run and pair status counts.

For a controlled restart under a changed configuration, choose a new `run_id`
and database/output path. Reusing a `run_id` with a different resolved config is
rejected.

## Read-only production field replay

The replay script consumes completed production field CSVs and their recorded
checksums without opening source imagery or invoking EfficientLoFTR:

```bash
python scripts/replay_global_catalogue_fields.py \
  --source /path/to/completed-production-run \
  --output /path/outside/git/global-field-replay
python scripts/render_global_catalogue.py \
  --database /path/outside/git/global-field-replay/global-trajectories.sqlite \
  --source /path/to/completed-production-run \
  --output /path/outside/git/global-field-replay/visuals
```

The replay is explicitly labelled `FIELD REPLAY`; its provenance records the
source state, plan, environment, individual completed-field verification, and
ordered field-set checksum. It does not reconstruct deformation, recovery
pairs, or a native run manifest. The renderer chooses at most 30,000
multi-observation trajectories by deterministic spatially balanced sampling
and draws only measured source-to-target segments, so dormant intervals have
no visual bridge.
