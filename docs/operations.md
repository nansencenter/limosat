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

## Planning: next observation

Candidate image pairs are frozen deterministically before inference. Defaults
require 1--96 hours elapsed time, at least 0.05 overlap relative to the smaller
footprint, and at least 1,024 km2 direct overlap (the area of 64 nominal 4 km
field cells). When `exclude_same_acquisition_pass` is true, a pair is removed
only if both platform and absolute orbit agree; set `require_orbit_metadata:
true` for production so missing metadata fails during planning, before GPU work.

Primary pairs are chosen by next observation (`planning.py`
`_next_observation_selection`). Each source image's 4 km planning cells that a
later candidate image also images (no drift allowance) go to the earliest such
image, so every piece of ice is carried to its next observation and composition
chains the steps. A pair is kept when it is the next observation for at least
`next_observation_minimum_cells` (25) cells. Cells whose next observation is more
than `next_observation_backup_after_hours` away (production 30 h; longer pairs fail
more often, mostly in fast drift) also go to the following observation, as a
backup. With `plan_surface_cells: true` (production) cells that the surface gate
excludes at the source time (land at least `land_minimum_inland_km` inland, ASI
open water) are dropped first. On January-April 2020 this gave the same survival
and coverage as the earlier per-target plan with completion, fewer duplicate
tracks and lower buoy errors (paper repository, `notes/reports/2026-11-20-next-observation.md`).

`limosat plan CONFIG` prints the plan's counts without loading the matcher; the plan
is cached under `output_directory/plan-cache`, keyed by the catalogue, the routing
settings and the planning code. An explicit `candidate_pair_ids` list is a
diagnostic allowlist: every accepted listed pair becomes primary.

Not adopted (parked November 2026): matching only the cells each pair is assigned
(tile skipping). It halved GPU time but lost the fallback that later pairs give,
and survival, duplicates and accuracy were worse. A future design: one source scene
per GPU task; match its assigned cells to the next observation, find the failed or
drifted-out cells, and match only those to the following observation in the same
task.

## Matching

Every primary pair is matched over its whole overlap. Each independently scheduled production pair estimates coarse phase
correlation. EfficientLoFTR runs on north-up tiles whose non-overlapping source
cores prevent duplicate ownership. Endpoint validity, elapsed-time speed,
local vector consensus, and fold gates are applied in EPSG:3413 metres.

### Tile data and GPU execution

Production does not read input tiles from NPZ files. The frozen tile NPZs used
in performance experiments were temporary, immutable test fixtures. For a
native run, Rasterio reads each source COG scene into the four-scene RAM cache
on first use. North-up 512x512 tiles are then resampled from those in-memory
arrays and passed directly to the GPU:

```text
source COGs
    |
    v
whole scenes in the RAM cache
    |
    v
north-up 512x512 tile pairs in memory
    |
    v
fixed batches of four on the GPU
    |
    +--> recorded CNN + coarse-attention prefix (CUDA Graph replay)
    |
    +--> variable coarse-match selection + fine refinement (normal PyTorch)
    |
    v
existing metre-space filters, fields, and trajectories
    |
    v
output/resume NPZ pair products
```

Batching does not join tiles scientifically: EfficientLoFTR processes four
independent tile pairs in one GPU invocation, and the results are separated by
tile before the existing filters run. The final short batch is padded to four
internally and padded results are discarded.

A CUDA Graph is a recorded sequence of GPU operations. Replaying the fixed
prefix avoids asking Python to launch the same CNN and coarse-attention
operations for every batch. Match selection cannot be included because each
SAR tile produces a different number of matches; this dynamic suffix therefore
continues to run normally. No matches or attention tokens are padded to a fixed
capacity.

The production CUDA defaults are `tile_batch_size: 4`,
`prefix_cuda_graph: true`, and three warm-up batches. The one-time graph setup
is amortized over a long-lived pair worker. Outer FP32 inference with
matrix-multiply TF32 disabled, the reparameterized optimized checkpoint, SDPA,
skip-softmax, and the existing scientific filters remain unchanged. Batch 4
used 4.10 GB of reserved A100 memory in the frozen benchmark.

The phase-correlation domain is the direct overlap. Its coarse resolution is at
least 1 km/pixel and the translation is clipped by the configured maximum ice
speed. A phase response below `phase_correlation_minimum_response` evaluates
both the phase-shifted and zero-shift same-centre hypotheses. The result with
the most available fold-gated field cells wins, followed by spatial tile
support, agreeing matches, residual, and stable phase preference. All matcher
calls and both candidate-support counts are recorded. With
`phase_correlation_failure: same_center`, insufficient coarse support falls
back to zero initial shift; `error` instead fails the image pair.

Coarse routing places a 320 m/pixel window (163.84 km) on each 35.84 km fine
tile by default (`coarse_window_stride: 1`). With `coarse_window_stride: 3`,
one coarse window serves the 3 x 3 fine tiles nearest its centre: each tile
takes the local median of that window's matches within its own support radius,
which stays inside the window core. A tile without local support in a window
that did measure motion is retried with its own centred window, because an
off-centre tile tolerates less shift error. On 30 random March 2020 primary
pairs this used 3.1x fewer coarse matcher tiles and ran 1.5x faster per pair
(shared GPU), with 0.2% fewer field nodes and a refined-tile share of 0.430
versus 0.427. Field differences (median 13 m at common nodes) were about twice
those from a small change of the per-tile shift (34 km instead of 35.84 km
support radius, 5 m), so the setting changes results slightly and is opt-in.
Stride 1 reproduces earlier products exactly and keeps their configuration
identity.

The coarse pass is required. With it switched off (October 2026 test), field
nodes fell 6.8-7.7% on 39 August 2026 pairs and 4.2% on 30 random March 2020
pairs; 20-23 of the August pairs lost over 3%, some their whole field. Losses
grow with elapsed time (August: 2% within 24 h, 13% at 24-48 h, 69% beyond
48 h), where one pair-wide translation cannot place every tile, although it
saves 35-57% of matcher tiles.

Fine tile cores do not overlap by default, so each location is matched in one
tile per pair. Most failed tiles are partial target windows: a textured source
facing a mostly invalid target returns no confident matches (October 2026 tile
census, March 2020). Three opt-in settings address this, independent of tile size:

- `tile_grid_offsets: [[0, 0], [0.5, 0.5]]` adds a grid shifted by half a core
  diagonally. Coarse windows follow the primary grid; the extra tiles take
  their shift from the pooled coarse matches, so no coarse matching is added.
- `masked_rematch_target_valid_fraction: 0.5` re-matches a tile whose target
  window is under 50% valid with the source blanked where the aligned target
  is invalid, and keeps both results.
- Overlapping matches keep one match per `layout_dedup_m` (160 m) source bin
  before the field, so duplicates cannot inflate the agreement rule. The
  displacement difference between co-located matches is recorded as
  `layout_colocated_difference_*` (median 15 m, 0.3-0.5% over 1 km).
  `layout_field_merge: primary_first` keeps every primary-grid node and fills
  only its gaps; folds created by added nodes remove added nodes first.

On 30 random and 20 Chukchi/ESS stress primary pairs (A100) the first two
settings added 4.6% and 24.5% field nodes (3.7% and 17.9% after the
missing-node fallback) at 1.65-1.9x matcher tiles and 1.6x pair time, and
recovered 48% of ORB-consistent no-measurement endings on the stress pairs.
In the preceding experiment IABP buoy medians at common nodes were unchanged
(44 versus 45 m). Changing
the tile size alone (384 or 640 px) changed nodes by -2.4 to +2.9%, while the
layered layout gained 18.6-20.3% on the stress pairs at all three sizes.

`tile_sampling: pair_cache` resamples each scene once per 1024-pixel lattice
block and crops tiles from the blocks; target windows snap to the pixel lattice
(at most half a pixel). It cut sampling time by 30-48% and layered pair time
from 1.6x to 1.3x, but moves about 2% of nodes (median 32 m at common nodes)
because the matcher sees slightly shifted windows, so it is a separate opt-in.
The primary-first merge re-estimates only the nodes the primary field lacks
(identical values; under 0.1% of nodes differ through fold handling), which cut
layered field time by 14-34%. Masking partial coarse windows was tested and not
adopted: it raised the share of refined coarse windows (S 16% to 33%) but not
field nodes (-0.5 to +0.9%), at 60-90% more coarse matcher tiles. Matcher
timings above are A100; on newer GPUs sampling and field estimation take a
larger share, which favours `pair_cache` and fewer, larger tiles.

Inference tiles cover the source footprint intersected with the target
footprint buffered by the elapsed-time physical displacement limit. A validity
gate avoids inference only where source-core and target support cannot contain
a physically reachable endpoint pair. Optional OSI SAF filtering prefers
`ice_conc_unfiltered` and skips only when a complete sample grid is below the
configured SIC threshold on both dates. Missing, stale, partial, or one-date
evidence keeps the tile. SIC paths and SHA256 checksums and every gate count
are recorded with the completed pair.

Residual field recovery may rerun a tile when a large routing residual and
target-boundary pressure agree.

Field sampling first uses the containing triangle returned by the directed
lookup. If that triangle is unusable, an exact vertex or shared edge can use
another existing incident triangle under the same maximum-edge and orientation
requirements. The fallback preserves successful samples and does not extend
support into holes or extrapolate beyond the field.

## Pair products and resume

`limosat prepare` freezes the catalogue and candidate pair plan in a small SQLite
run database (and writes `scene_checksums` once, in parallel). Pair workers
(`limosat pairs CONFIG --batch-index I --batch-count N`) read the plan from it and
write atomic NPZ data and JSON completion markers. `pair_workers` controls CPU
concurrency within one process; CUDA configurations require `pair_workers: 1` and
one process per batch. A marker binds the product to its matching fingerprint, the
image-pair identity and the checksums of both scene files and their OSI SAF SIC
files. Completed products are verified and never overwritten; when two writers race
on one pair, the first published product wins and the other adopts it.

### Sharing pair products between runs

A pair product does not depend on the run that made it, only on what changes
matching: the matching settings, the model checkpoint and EfficientLoFTR source,
the land raster, the two scene files and their SIC files, and the matching code.
`limosat/pair_identity.py` hashes exactly these (file contents, never paths) into
the matching fingerprint. Every configuration field is classified there as
matching or excluded, and a unit test fails when a new field is left
unclassified. Run IDs, paths, pair selection and all composition settings are
excluded, so changing them never invalidates matched pairs. The production
fingerprint (`1cecfae0...`) is pinned by `tests/unit/test_production_fingerprint.py`.

Set `pair_store_directory` to share products: they are then stored under
`<pair_store_directory>/<matching fingerprint>/primary/` and any run with the same
fingerprint reuses them, on any cluster, provided its scene files are
byte-identical. Without it products stay in `<pair_product_directory>/primary/`.

Code changes invalidate stored products only through `MATCHING_VERSION` in
`limosat/pair_identity.py`. A unit test records a digest of the matching modules;
when it fails, bump `MATCHING_VERSION` if the change alters matching output and
record the new digest either way. The producer code hash stays in each marker as
provenance.

On Olivia the store lives under `/cluster/work`, which Slurm jobs can write, and
is archived to NIRD (`/nird/datapeak/NS2993K/seachu/eloftr-pair-store`) with
`scripts/sync_pair_store_nird.sh archive`, run on the svc01 transfer node; `restore`
copies it back after a work-area cleanup.

## Running

`limosat run CONFIG` runs prepare, every primary pair and composition on one
machine. On Olivia:

```bash
python3 scripts/prepare_chain_olivia.py --run-id RUN --run-root ROOT \
  --start 2020-01-01 --stop 2020-05-01 --sic-root SIC --land-raster LAND.npz --asi-root ASI
RUN_ROOT=ROOT DATA_BINDS=... scripts/submit_chain_olivia.sh
```

The prepare script writes the production configuration (`run_config` in the
script). The chain is prepare (CPU: plan, run database, scene checksums) ->
primary-pairs (GPU array, `PROCESSES_PER_GPU` matcher processes per GPU sharing it
through MPS) -> primary-compose (CPU). `STOP_AFTER=primary-pairs` stops after
matching; `--start-at primary-compose` later composes with the then-current
checkout. Stage submissions are logged in `control/chain.log`.

For a restart under a changed configuration choose a new `run_id` and
database/output path; reusing a `run_id` with a different resolved configuration is
rejected. With a shared pair store, the new run reuses every pair whose matching
fingerprint is unchanged. Configuration files written before the November 2026
clean-up still load when their removed settings describe the kept method
(`config.py` `_REMOVED_SETTINGS`).
