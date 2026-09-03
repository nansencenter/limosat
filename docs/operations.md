# Operating LiMOSAT

## Catalogue

LiMOSAT accepts CSV or GeoJSON. Every row or feature requires:

| Field | Meaning |
| --- | --- |
| `image_id` | Globally unique stable image identity |
| `path` | Raster path, absolute or relative to the catalogue |
| `time_utc` | ISO-8601 timestamp with an explicit UTC offset |
| `component_id` | Optional compute-planning label; defaults to `default` |
| `platform` | Sentinel platform, for example `S1A`; inferred from standard product IDs |
| `absolute_orbit` | Positive absolute orbit; `orbit_number` and `orbit_num` are aliases |

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
require 1--96 hours elapsed time, at least 0.05 overlap relative to the smaller
footprint, and at least 1,024 km2 direct overlap. The absolute threshold is the
area of 64 nominal 4 km field cells; it is a minimum compute-support gate, not
a claim that every such cell will have a measurement. For every fixed 4 km
planning cell, the most recent eligible source is selected. The union of those
cell choices defines the target's primary image pairs; equal-time alternatives
remain available for deterministic field-quality selection. Older candidates
are available only for measured-loss recovery.

When `exclude_same_acquisition_pass` is true, an image pair is removed only if
both platform and absolute orbit agree. Set `require_orbit_metadata: true` for
a native production run so missing metadata fails during planning, before GPU
inference. Exclusion counts, direct overlap area, overlap fraction, and the
number of intervening global catalogue images are stored durably. The
intervening-image count is diagnostic only.

An explicit `candidate_pair_ids` list is a diagnostic allowlist. Every accepted
listed image pair is treated as primary so a small diagnostic run actually
processes every selected pair. Use a separate run ID/database and clear the
allowlist for a full run.

`primary_maximum_pairs_per_target` is a separate diagnostic compute bound. It
first applies the normal newest-source-per-cell rule, then greedily retains the
pairs contributing the most complementary selected cells. Source recency,
direct overlap area, and stable image-pair identity break ties. It cannot be
combined with `candidate_pair_ids`, and its production default is `null`.

Primary pair fields have no preceding-pair dependency and can be processed
independently. `pair_workers` controls CPU concurrency within one process.
CUDA configurations require `pair_workers: 1`; `limosat pairs --batch-index I
--batch-count N` provides deterministic, non-overlapping work for one process
per GPU. Each worker writes an immutable pair product and does not write the
global SQLite database.

## Matching and recovery

Each independently scheduled production pair estimates coarse phase
correlation. EfficientLoFTR runs on north-up tiles whose non-overlapping source
cores prevent duplicate ownership. Endpoint validity, elapsed-time speed,
local vector consensus, and fold gates are applied in EPSG:3413 metres.

The phase-correlation domain is the direct overlap. Its coarse resolution is at
least 1 km/pixel and the translation is clipped by the configured maximum ice
speed. A phase response below `phase_correlation_minimum_response` evaluates
both the phase-shifted and zero-shift same-centre hypotheses. The result with
the most available fold-gated field cells wins, followed by spatial tile
support, agreeing matches, residual, and stable phase preference. All matcher
calls and both candidate-support counts are recorded. With
`phase_correlation_failure: same_center`, insufficient coarse support falls
back to zero initial shift; `error` instead fails the image pair.

Inference tiles cover the source footprint intersected with the target
footprint buffered by the elapsed-time physical displacement limit. A validity
gate avoids inference only where source-core and target support cannot contain
a physically reachable endpoint pair. Optional OSI SAF filtering prefers
`ice_conc_unfiltered` and skips only when a complete sample grid is below the
configured SIC threshold on both dates. Missing, stale, partial, or one-date
evidence keeps the tile. SIC paths and SHA256 checksums and every gate count
are recorded with the completed pair.

Residual field recovery may rerun a tile when a large routing residual and
target-boundary pressure agree. Catalogue recovery is separate: after all
primary pair fields are composed globally, LiMOSAT identifies parcels measured
at an earlier source image but dormant at a candidate target image and matches
only source-tile cores within the configured 6.4 km buffer.
`maximum_recovery_elapsed_hours` bounds recovery from the parcel's measured
source time to the candidate target time. Images elsewhere in the pan-Arctic
catalogue do not affect eligibility. Every candidate is discarded unless the
primary composition contains a parcel that was measured at its source and is
dormant at its target. Eligible work is ranked recent-first and each recovery
field is spatially targeted to those measured source positions. Recovery
fields can reconnect trajectories, but only full primary pair fields generate
deformation cells. After recovery completes, trajectories are recomposed from
all completed fields.

At a target image, the composer evaluates parcels whose last measured row is
at the source of an incoming completed pair field. A parcel unsupported by all
such fields gets a dormant row with SQL `NULL` coordinates. Images with no
incoming measurement do not cause catalogue-wide Cartesian dormant rows. New
seeds are compared with all measured parcel positions active at that image,
irrespective of compute label.

The optional convergence audit inherits production LiMOSAT's deterministic
principle: within a measured-position cluster, prefer the longest observed
history and then measurement quality. It records candidate-to-winner evidence
in metres but does not merge, stop, or move either scientific parcel. The audit
is disabled until `convergence_audit_radius_m` is set; the old production
numeric radius is not reused because its coordinate semantics were ambiguous.

## Resume and outputs

The run begins with `limosat prepare`, which freezes the catalogue and
candidate image-pair plan in SQLite. Pair workers then write atomic NPZ data
and JSON completion markers under `pair_product_directory`. A completion
marker binds the product to the run, configuration, source implementation,
checkpoint, image-pair identity, and—for recovery—the exact measured source
positions. Existing completed products are verified and never overwritten.

`limosat compose` is the only pair-product SQLite writer. It checksum-verifies
and imports every expected worker product before streaming per-image
trajectory batches into one transaction. An interruption rolls back the
incomplete trajectory composition; already imported immutable pair fields
remain valid. Primary composition derives deformation only from primary pair
fields. Recovery workers run only after primary composition supplies genuine
measured-loss positions.
When `targeted_recovery` is false, no recovery pair work is selected; the
Olivia launcher also omits the recovery GPU stage entirely.

`limosat run CONFIG` executes all stages sequentially and is the recommended
interface on a local or other single-GPU machine. Scheduler launchers may call
the stages separately without changing their scientific behavior. SQLite is
the authoritative scientific and finalized resume product after composition;
pair-product files are intermediate compute-resume state.

The output directory contains `run-manifest-v4.json`. Scientific arrays and
tables live in SQLite; no result products belong in Git. `limosat status`
reports the run and pair status counts. Assessment runs should set
`retain_pair_matches: true`; production runs may disable it after the field
policy is frozen.

Once `limosat status` reports `complete`, run `limosat finalize CONFIG` to
write the compact global trajectory Parquet catalogue and a checksummed
assessment summary. This command does not alter scientific rows. Use
`--skip-parquet` only where PyArrow is unavailable and a validation/summary
report is sufficient.

For a controlled restart under a changed configuration, choose a new `run_id`
and database/output path. Reusing a `run_id` with a different resolved config is
rejected.

After successful finalization and transfer of any required assessment
products, the pair-product directory can be removed to reclaim duplicated work
space. Keep it until the run and its raw-match retention policy have been
verified; removing it prevents worker-level resume but does not alter the
completed SQLite catalogue.

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
