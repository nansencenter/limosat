# ALIKED workflow refactor

Branch: `aliked-refactor`

## Goal

Make the selected ALIKED workflow readable as six scientific steps without
changing its measured output:

1. extract and cache spatially distributed ALIKED features;
2. identify physically reachable source/target tile pairs;
3. match each selected pair with direct five-layer LightGlue;
4. reject displacement vectors above the configured speed;
5. estimate drift on the regular output grid from agreeing nearby matches; and
6. remove output points that produce folded triangles.

Buoy evaluation, ORB comparison, plotting, runtime sweeps, alternative
matchers, and full per-call audits are experiments that consume this workflow;
they are not part of the workflow itself.

## Selected behaviour to preserve

- EPSG:3413 analysis coordinates, stored in metres;
- 512-pixel north-up tiles with 32-pixel margins at 80 m/pixel;
- up to 1,024 `aliked-n16` features per tile, detection threshold 0.2;
- direct ALIKED-LightGlue with five layers, depth confidence 0.95, width
  confidence 0.99, and match threshold 0.10;
- maximum speed 30,000 m/day;
- regular 4,000 m output grid;
- nearest 12 matches within 6,000 m, requiring eight matches agreeing within
  1,000 m; and
- iterative removal of points involved in folded triangles.

CPU target batching is not selected. MNN target-tile ranking remains an
optional wide-search policy and must do no work when the physical candidate
count is already within its limit.

## Package structure

```text
limosat/learned_drift/
    __init__.py       public names only
    config.py         one explicit scientific config
    types.py          tile features, matches, field, and timings
    imagery.py        north-up sampling and coordinate conversion
    features.py       tile layout, ALIKED extraction, and cache
    matching.py       physical routing and direct LightGlue
    field.py          regular-grid consensus and topology
    pipeline.py       pair and sequence orchestration
```

This follows the existing LiMOSAT preference for named processing components
without forcing learned matching through the ORB-specific `KeypointDetector`
or `Matcher` classes. There are no detector registries, plugin systems, base
classes, or custom exception hierarchies.

The core uses arrays/tensors with units in their names. DataFrames, CSV files,
JSON manifests, plots, buoy truth, and ORB fields stay at experiment and CLI
boundaries.

## Current file inventory

### Move into the core package

| Current file | Responsibility after refactor |
|---|---|
| `experiments/aliked_matchers.py` | selected direct LightGlue adapter and optional MNN tile ranking in `matching.py` |
| `experiments/run_aliked_dense_pair.py` | core feature, matching, and field functions move out; file becomes a thin compatibility CLI |
| `experiments/run_aliked_selected_sequence.py` | pair/sequence loop moves to `pipeline.py`; file becomes a thin compatibility CLI |
| `experiments/refine_aliked_dense_topology.py` | fold detection and rejection move to `field.py` |
| `experiments/compare_aliked_orb_northup.py` | reusable north-up image sampling moves to `imagery.py`; comparison remains experimental |

### Keep as validation or evaluation

- `validate_aliked_controlled_warp.py`
- `compare_aliked_orb_northup.py`
- `plot_fair_orb_aliked_matches.py`
- `summarize_aliked_orb_northup.py`
- `summarize_fair_orb_aliked_benchmark.py`
- `analyze_aliked_buoy_repeat_qc.py`
- `analyze_aliked_local_grid_coverage.py`
- `compare_icesat2_aliked_fields.py`
- the corresponding focused tests

These files may import the package but the package must never import them.

### Preserve as frozen provenance

- `experiments/configs/aliked_lightglue_confirmation_20260822.json`
- `experiments/configs/fair_aliked_cuda_handoff_20260820.json`
- `experiments/configs/fair_orb_aliked_runtime_20260819.json`

### Review for deletion after parity and output review

- `pilot_learned_sar_features.py`
- `replay_aliked_dense_consensus.py`
- `replay_aliked_pattern_variants.py`
- superseded runtime shell scripts whose exact commands are already captured in
  frozen manifests;
- the rejected CPU batching implementation; and
- generated `results/` content that already has an authoritative Kingston copy.

`replay_aliked_candidate_policies.py` and `replay_aliked_dense_sequence.py`
remain until the selected material-point trajectory behaviour has a direct
replacement and parity test.

## Public API

```python
tracker = ALIKEDDrift(config, device="cuda", cache_dir=cache_dir)

features0 = tracker.extract(image0)
features1 = tracker.extract(image1)
matches = tracker.match(features0, features1, elapsed_hours=21.4)
field = tracker.estimate_field(matches, domain)
```

`track_pair` provides the same four stages as a convenience call and returns
the intermediate matches, field, and stage timings. `track_sequence` detects
each unique image once and applies `track_pair` to adjacent images.

## Persistence contract

The learned workflow now has a separate `LearnedDriftStore`. It does not use
the production ORB `Keypoints` and `Templates` tables because ALIKED redetects
features in each image; an ALIKED match is an immutable pair observation, not
an updated template descriptor.

```python
store = LearnedDriftStore(
    database_path="run.sqlite",
    zarr_path="run.zarr",
    run_name="aliked_2020",
    config=config,
)
pair = ImagePair(
    source_image_id=740,
    target_image_id=849,
    source_path=source_path,
    target_path=target_path,
    elapsed_hours=21.413,
)

result = store.load_pair(pair)
if result is None:
    result = tracker.track_images(
        pair.source_path, pair.target_path, pair.elapsed_hours
    )
    store.save_pair(pair, result)
```

SQLite stores the run config and its SHA-256, stable source/target image IDs,
paths, UTC times when supplied, elapsed hours, sequential prior, processing
times, counts, storage location, and one of three states: `writing`, `failed`,
or `complete`. A run name cannot be reopened with a different scientific
config. The exact-pair key also includes elapsed time, prior displacement, and
prior uncertainty so changing the sequential search state cannot silently
reuse the wrong result.

Each completed pair has one zipped Zarr archive containing:

- source/target match coordinates in EPSG:3413 metres, feature/tile IDs, and
  LightGlue scores;
- every regular-grid coordinate, displacement, availability flag, support
  count/radius, and maximum consensus residual; and
- indices rejected by the fold-removal stage.

The archive is first written under a temporary name, closed, and atomically
renamed. Only then is its SQLite row marked `complete`. A stopped or failed
write is therefore never returned by `load_pair`; rerunning the same pair
replaces it safely. One zip per pair also avoids the tens of small files that
made an ordinary Zarr group slow on Kingston and would burden a parallel
filesystem.

ALIKED feature tensors remain in the existing `.pt` feature cache because they
are large, reproducible intermediates rather than scientific output. Virtual
material-point trajectories are also not duplicated yet: they can be rebuilt
deterministically from the stored adjacent-pair fields. A later sequence
product can persist those derived trajectories if repeated arbitrary-window
queries show that reconstruction is a real cost.

The real 2020 pair 740→849 round-trips exactly with 45,715 matches, 8,548 grid
nodes, and 7,391 available nodes. On Kingston the finalized store took 0.342 s
to save and 0.037 s to load and occupied 1,314,724 bytes, compared with
7,155,143 bytes for its match and field CSV files. The first ordinary
directory-Zarr prototype took 10.7 s to save because of small-file metadata;
it is superseded by the zipped-pair layout. The authoritative validation is:

`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/aliked_refactor_persistence_v2_20260823`

The persistence interface is currently single-writer per run. Separate pair
archives make later multi-worker calculation feasible, but concurrent SQLite
claiming should be added and tested on the actual HPC filesystem before a
parallel production launch.

## Validation boundary

Configuration is validated once when constructed. A pair validates only its
CRS, positive elapsed time, and non-empty physical overlap. Internal functions
assume their typed inputs are internally consistent. Empty feature or match
sets return typed empty results; unexpected model errors are not silently
converted into missing data.

## Refactor gates

1. Small synthetic tests preserve tile layout, physical candidate selection,
   match filtering, regular-grid consensus, and fold rejection.
2. The batch-size-one 2020 pair 740→849 preserves match support, all 7,391
   fold-free nodes, 15/15 buoy availability, and vector differences within the
   existing numerical parity tolerance.
3. N-ICE pair 6801→6901 preserves 2,707 fold-free nodes, 6/6 buoy availability,
   and topology after rejection.
4. Only after these gates pass are compatibility implementations and
   superseded files removed.

Algorithm changes and speed optimizations are deliberately excluded from the
refactor.

## Implementation status: 2026-08-23

The selected workflow now exists under `limosat/learned_drift/` with a
76-line pair runner at `experiments/run_learned_drift_pair.py`. Processing and
persistence are 1,939 lines across nine small files, compared with 3,143 lines
across the four experiment files that previously contained the matcher, pair loop,
sequence loop, and topology logic. Validation and historical experiment code
are not counted and have not been deleted.

The package dependency direction is now one way: the runner and tests import
`limosat.learned_drift`; no package file imports `experiments`. The old scripts
remain available for frozen-result reproduction and alternative-method
audits, but they are no longer the recommended entry point for a selected
pair.

### Refactor gates

| Gate | Existing result | Refactored result | Numerical comparison |
|---|---:|---:|---|
| 2020 pair 740→849 matches | 45,715 | 45,715 | identical feature/tile IDs and coordinates |
| 2020 pair 740→849 fold-free nodes | 7,391 | 7,391 | identical availability and support; maximum vector difference `1.4e-11` m |
| N-ICE pair 6801→6901 matches | 20,678 | 20,678 | identical matched coordinates |
| N-ICE pair 6801→6901 fold-free nodes | 2,707 | 2,707 | identical availability, support, seven rejected nodes, and topology; maximum vector difference `2.9e-11` m |
| March sequence 10245→10341 matches / nodes | 24,932 / 4,189 | 24,932 / 4,189 | identical coordinates, support, and availability; first-pair prior fallback |
| March sequence 10341→10352 matches / nodes | 25,857 / 4,258 | 25,857 / 4,258 | identical coordinates, support, and availability; maximum vector difference `4.1e-12` m |

The N-ICE pair-local tile identifiers differ because the frozen reference
assigned IDs over a four-image union domain. This is bookkeeping only: the
matched projected coordinates and all scientific outputs are identical.

The measured refactor runs took 84.59 s and 77.90 s for matching the 2020 and
N-ICE pairs respectively. These are consistent with the existing 82.99 s and
80.34 s measurements, but they were not interleaved timing repetitions and do
not establish a matcher speed change. Array-based field estimation and
topology were faster in both runs; this should receive a separate repeated
timing check before being reported as a performance result.

Focused verification is 42 tests passing across the new core, existing ALIKED
pair and sequence functions, and the operational-baseline harness. Exposing
the real package path beneath the existing lightweight test stubs also fixes
the earlier test-harness import failure for unstubbed `limosat` modules.

The three-image sequence gate uses the frozen three-layer run solely to test
orchestration with identical matcher parameters. The first pair correctly uses
the full physics window. The second pair receives exactly the frozen prior,
`(2521.500310992539, -996.010649034501)` m, computed only from the immediately
preceding fold-free field. The initial gate exposed and fixed an adjacency
slice error before any matching occurred. The corresponding invariant is now
covered by the focused test suite.

### Cleanup boundary

No legacy experiment or generated result has been removed. The first
refactored 2020 output was superseded after correcting float32 rounding of
tile-centre coordinates; the authoritative output is:

`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/aliked_refactor_pair_740_849_v2_20260823`

The superseded `aliked_refactor_pair_740_849_v1_20260823` directory can be
removed after review. The N-ICE gate was written to
`/private/tmp/limosat_refactor_6801_6901_top8_v1` because approval for a new
Kingston output timed out; its frozen reference remains authoritative.

The sequence gate is now complete. The next concrete cleanup is to make the
old pair and sequence scripts explicit compatibility/experiment entry points,
then remove duplicated selected computations only where their audit and buoy
evaluation outputs already have a package-backed replacement. Alternative
matchers, controlled warps, buoy evaluation, and plotting remain experiments
around the core. The rejected batching code should be removed only after this
branch is committed, so its exact implementation remains recoverable in Git
history.
