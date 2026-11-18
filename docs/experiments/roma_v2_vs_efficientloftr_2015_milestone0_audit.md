# Milestone 0 audit: RoMa v2 versus EfficientLoFTR on N-ICE2015

Status: **reviewed; Milestone 1 complete with no scientific inference**

Audit date: 2026-09-03

Execution host: Vega

Campaign: `docs/experiments/roma_v2_vs_efficientloftr_2015.md`

Milestone 0 executed no experiment. This audit used only repository inspection,
metadata inspection, checksums, compact table validation, and read-only remote
source-ref inspection. No data or weights were transferred, no repository was
cloned, no environment was created or modified, no GPU inference was run, no
production matcher code was changed, and nothing was committed or staged.

### Milestone 1 status update (2026-09-03)

Milestone 1 was subsequently authorized under run ID
`20260903T143939Z-roma-v2-vs-eloftr`. Vega preflight passed; the exact inputs,
references, truth fixture, pinned sources, EfficientLoFTR checkpoint, and RoMa
v2.0.1 weight were staged and verified. The CPU evaluation environment was
created successfully. No matcher was instantiated and no scientific inference
was run.

Environment setup stopped before the RoMa environment was created. The first
EfficientLoFTR solve selected conda-forge PyTorch 2.6/CUDA 12.6 through the
`kornia` dependency, producing an incomplete 9,917,416,742-byte prefix before
the separately pinned PyTorch 2.8 CUDA-wheel transaction failed during a
network resume. The prefix already exceeded the approved 8 GiB per-environment
ceiling, so it was retained unchanged for review and Milestone 1 did not
continue. The authoritative evidence is under
`/Data/sat/seachu/results/limosat/roma-v2-vs-efficientloftr/20260903T143939Z-roma-v2-vs-eloftr/record/`.

The user reviewed this stop and authorized recovery on 2026-09-03. The
per-matcher-environment ceiling is amended from 8 GiB to 12 GiB and the total
environment allowance from 24 GiB to 28 GiB. The evaluator ceiling is 2 GiB.
These are storage safeguards only; the experiment-output cap remains 20 GiB.
Recovery may remove and recreate only the incomplete campaign-created
EfficientLoFTR prefix. Repeated downloads of the pinned PyTorch 2.8 wheel stack
then failed at the same 425 MB point on the NVIDIA cuDNN mirror, including when
requested through the official PyTorch index. The environment lock therefore
uses complete conda-forge CUDA builds: PyTorch 2.6/CUDA 12.6 for EfficientLoFTR,
and PyTorch 2.8/torchvision 0.23/CUDA 12.9 for RoMa. This changes no model,
weight, geometry, or scientific parameter; the accepted later CUDA parity gate
still applies.

Recovery completed successfully. Final environment sizes are 739,919,069 bytes
for evaluation, 9,157,947,860 bytes for EfficientLoFTR, and 11,381,124,189
bytes for RoMa, totalling 21,278,991,118 bytes (19.81 GiB). Import-only
preflights passed on the A100 with PyTorch 2.6.0/CUDA 12.6 for EfficientLoFTR
and RoMa 2.0.1, PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.9, and
fused-local-corr 0.3.211 for RoMa. The staged data and both model assets passed
their final SHA-256 checks, and `/Data/sat` retained 465 GiB free.

The preflight also established that the existing production EfficientLoFTR
loader imports `create_meshgrid` from the stale `kornia.geometry` location.
The pinned source imports successfully with an experiment-only alias from
`kornia.utils`; Milestone 2 must keep that adjustment outside production
matcher code.

## Experiment-first summary

The pilot remains focused on one scientific question: whether official RoMa v2
produces a larger fraction of coherent, valid two-pair 4 km trajectories than
the controlled EfficientLoFTR baseline on the fixed
`6775 -> 6801 -> 6901` chain.

The accepted comparison is:

- RoMa direct-grid is the primary new method; sampled RoMa is secondary.
- The initial-grid denominator is fixed before matching, and a method cannot
  remove a failed or unsupported point from it.
- The primary result is the post-topology end-to-end valid fraction, with the
  unthresholded physical/domain-valid result always shown.
- Overlap/cycle arms are exploratory and predeclared; buoy truth cannot select
  a preferred threshold.
- EfficientLoFTR must first reproduce the controlled reference inside the fixed
  parity tolerances.
- Pair-2 routes, geometry, masks, physical gates, and downstream evaluation are
  identical across methods.

RoMa commit provenance, licensing, isolated environments, transfers, and
checksums remain recorded for reproducibility, but are routine setup rather than
scientific decision points. Milestone 1 should stay mechanical and stop only on
an actual integrity, compatibility, storage, or GPU failure.

## 1. Audited instructions and source state

The complete campaign document was read at SHA-256
`8c859b1771388c6797504e704d846a1227a6d213c4b262a9bc6cfdf5ca3431be`.

The configured Research Workbench checkout was read from:

```text
/Users/seachu/Documents/Codex/2026-09-03/how-m/outputs/research-workbench
```

The complete files read were:

| File | SHA-256 | Status |
| --- | --- | --- |
| `LAB_PROTOCOL.md` | `56ed1e32801da91a057f34c842037b61eeac1ac7030499c6ae7dc32240ecf6f5` | untracked draft |
| `profiles/vega.toml` | `259084506609991c2b0c973a0e3a5c1c8e68a0206218b83cc47241ddb57da2dc` | untracked draft |

The workbench has no commits yet. Its Vega profile is therefore provisional,
not a versioned machine contract. It records a direct Ubuntu 24.04 host at
`vega.ad.nersc.no`, one NVIDIA A100-PCIE-40GB, driver `580.178.04`, and 499 GB
free under `/Data/sat` when observed on 2026-09-03. The project, result, cache,
and environment roots are marked `proposed`, so all must be checked directly on
Vega before any creation or transfer.

The Limosat checkout is on
`archive/arctic-tracking-research-20260831` at the campaign commit
`939c6d42c3f6d9d9b512c53db4a865fa263aba29`, tracking the corresponding origin
branch. The commit exists locally. The following unrelated changes were present
and were not touched:

```text
 M experiments/summarize_osisaf_routing_confirmations.py
 M tests/unit/test_summarize_osisaf_routing_confirmations.py
?? docs/experiments/
?? experiments/configs/osisaf405_antarctic_routing_20260831.json
?? experiments/osisaf405_drift.py
?? experiments/run_efficientloftr_antarctic_routing.py
?? tests/unit/test_osisaf405_drift.py
```

The campaign document itself and this audit are within the currently untracked
`docs/experiments/` directory. Preservation rule for later milestones: do not
reset, clean, stage, copy wholesale, or build a remote source tree from this
working directory. If approved, prepare a separate worktree from the exact
campaign commit, copy only the reviewed campaign/audit and named experiment
files into it, and record its patch or source hash. Do not commit unless the
user separately requests a commit.

## 2. Local evidence audit

### 2.1 N-ICE2015 rasters and EfficientLoFTR checkpoint

All four files exist at the campaign paths. Measured byte sizes and SHA-256
values exactly match the campaign evidence.

| Item | Bytes | SHA-256 | Result |
| --- | ---: | --- | --- |
| scene 6775 | 33,535,016 | `749006d5c2b63ba64a14cd7fefa70ef5eeffba97374d4312c8997814a9c5c312` | pass |
| scene 6801 | 29,193,725 | `4537d99cf2f61abc653f03c9bb97e8f791fb35ffd50307e1e29b82e5ce0cde5c` | pass |
| scene 6901 | 32,214,444 | `9d563d3dd174b6f27ab584eaa99d51a77b638cfd2e8cdc4e59626b4a74cd14bf` | pass |
| `eloftr_outdoor.ckpt` | 192,797,577 | `0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878` | pass |

Raster metadata also agrees with the contract: dimensions are respectively
`(4994, 5189)`, `(4994, 5213)`, and `(4994, 5189)` in `(rows, columns)` order;
each has two byte/`uint8` bands and GCP geolocation. Scene 6775 contains 441
GCPs. Milestone 1 must record the same metadata for all three staged copies,
including the band-2 `< 2` validity convention; a file hash alone does not
replace that semantic check.

The three scenes total 94,943,185 bytes. The four fixed binary inputs total
287,740,762 bytes.

### 2.2 Controlled EfficientLoFTR references

Each reference directory contains exactly the six expected non-AppleDouble
files. Every measured file checksum matches the campaign.

| Pair | File | Bytes | SHA-256 | Result |
| --- | --- | ---: | --- | --- |
| 6775-6801 | `buoy_results.csv` | 5,299 | `9e537bbb2fb478fc197669aff83715be6674e94bdab9fe50a7aed678d35d7ee5` | pass |
| 6775-6801 | `field_4km.csv` | 460,514 | `619aca77c92da2ca6f7e44220b50325d4be9cde53c86334ce5fe480fa3e58952` | pass |
| 6775-6801 | `matches.npz` | 1,174,286 | `e067a2df9aa4be105fcb293fa8fe0d84ebbc5c7927786d6cae6bd952fd3d4442` | pass |
| 6775-6801 | `run_manifest.json` | 538 | `2ca9a6af384bcfe074986583ce17dcc156f1e346942962165e594231b0629c2d` | pass |
| 6775-6801 | `summary.json` | 2,276 | `e7b753dabd9f0250498f723fc37fc5c9e0546304226109b9a878b83a032242fa` | pass |
| 6775-6801 | `tiles.csv` | 54,986 | `b4c3daa54ebc2f1d61e479ed3be6ff4b119007d8cc6629c67f7e510bbcdd9b49` | pass |
| 6801-6901 | `buoy_results.csv` | 5,230 | `d66d2906fd5fe18c97c1fbade155ca5777fe38f9d6966fdfe9cf6f315a5c169c` | pass |
| 6801-6901 | `field_4km.csv` | 579,137 | `c8312e69e7deae00b9f3be0e12ec125da8e4f2ac65f4ffc88581375a36291723` | pass |
| 6801-6901 | `matches.npz` | 998,510 | `e9ec4757e64bdc566b93d13a99726f2a2ae8105535874df82fc72ed9e7f71f25` | pass |
| 6801-6901 | `run_manifest.json` | 538 | `c210448d9aa545d373576e81804ed2d0fa5a0bbb13855a8d3112475652a92715` | pass |
| 6801-6901 | `summary.json` | 2,326 | `9e63f98d4d4d372c8fb463676f802463fe0256c9a7e78f922617d92450f545b6` | pass |
| 6801-6901 | `tiles.csv` | 65,663 | `b229db334828842f65854f28a5a47f8937ba9b159dd00302b18cd54c208495e2` | pass |

The 12 reference files total 3,349,303 bytes. Structural checks passed:

- 112 and 141 tile rows; the compound keys `(tile_id, row, column,
  source_center_x_m, source_center_y_m)` are unique in each pair;
- the pair-1 route median is exactly
  `(1138.8927347788638, -7149.500772510293) m`;
- summing per-tile physics-valid counts gives 68,910 and 58,077;
- the match archives contain 68,910 and 58,077 float64 `(n, 2)` projected
  source/target coordinates, float32 scores, and integer IDs;
- the field tables contain 6,992 and 9,258 unique
  `(grid_row, grid_column, source_x, source_y)` rows;
- their post-fold available counts are 2,388 and 2,754; and
- recomputing the current field-content hash gives
  `7885aa38533b52b9d72fd86e5f3192524ff7e0e6d3716bebe2eff5adddf7a36c`
  and `fd3be3d07f1cded9234bffc45749e77e1806b5fe87df9e14aa95e928b54a7344`,
  exactly matching the summaries.

The six buoy IDs are identical across the pairs: `SIMBA_2015d`,
`SNOW_2015b`, `SNOW_2015d`, `SNOW_2015e`, `WAVE_2015e`, and `WAVE_2015f`.
Elapsed times in the files match the fixed 14.893611111111111 h and
32.83166666666666 h.

### 2.3 Current reference metrics

The following values were read and independently cross-checked against table
row counts or calculations where possible. They are evidence for a later Vega
reproduction; no output will be altered to make a run agree with them.

| Metric | 6775 -> 6801 | 6801 -> 6901 | Audit |
| --- | ---: | ---: | --- |
| physics-valid matches | 68,910 | 58,077 | table sum/NPZ pass |
| grid nodes | 6,992 | 9,258 | CSV rows pass |
| available before fold removal | 2,392 | 2,772 | summary present |
| available after fold removal | 2,388 | 2,754 | CSV count pass |
| coverage | 34.1533180778032% | 29.7472456254051% | derived count pass |
| fold removals | 4 | 18 | count difference pass |
| buoys available | 6/6 | 5/6 | CSV count pass |
| buoy median error | 99.44165961737369 m | 91.62368685860534 m | CSV pass |
| buoy p90 error | 125.5567788525536 m | 193.56162006424938 m | CSV pass |
| buoy maximum error | 134.8867648227026 m | 226.17205343048957 m | CSV pass |

The controlled manifests identify MPS runs, not CUDA runs. They confirm the
official optimized model, reparameterization, and checkpoint byte size, but do
not contain checkpoint SHA-256, source/target IDs, raster paths, elapsed times,
or the full reproducible environment. Their `pair_identity_sha256` summary
fields are descriptive labels rather than 64-character hexadecimal hashes.
They are therefore comparison evidence, not complete run manifests.

The current `experiments/run_efficientloftr_sequence.py` cannot consume these
compact controlled manifests through `--reference-pair-run-dir`, because its
loader expects image IDs, paths, and elapsed time in each manifest. It also has
no CLI option to load the frozen per-tile routes, although its internal
`track_pair` function accepts route overrides. A small experiment-only runner
must provide the fixed pair specification and perform the keyed route join. No
change to the production matcher modules is proposed.

### 2.4 External source versions

- Limosat is locally verified at
  `939c6d42c3f6d9d9b512c53db4a865fa263aba29`.
- EfficientLoFTR commit
  `ffd4a4644064354468eb1f0c7a3e732233cb732f` is addressable in the official
  repository, but no local official checkout was found. Its legacy full
  requirements are not suitable as an unquestioned modern environment lock;
  the existing adapter contains explicit compatibility shims.
- A direct read-only `git ls-remote` on 2026-09-03 resolved RoMa tag `v2.0.1`
  to the lightweight-tag commit
  `95c9968145c8906b7b59383258e9f73b02853d89`. At that full commit,
  `pyproject.toml` reports package `2.0.1`, `precise` is 800 x 800 low
  resolution plus 1280 x 1280 high resolution with bidirectional matching,
  interpolation uses `align_corners=False`, and the constructor requests
  `releases/download/v2.0.1/romav2.0.1.pt`. The documented match/sample return
  values and normalized-coordinate convention agree with the campaign.

The RoMa repository code is MIT-licensed, while its incorporated DINOv3
component has a separate custom license. The user reviewed this issue on
2026-09-03 and accepted the license as suitable for this pilot.

## 3. Proposed fixed denominator

### 3.1 Stable grid identity

Regenerate the 4 km lattice from geometry at EPSG:3413 origin `(0, 0)`, rather
than using the EfficientLoFTR field as an input. The existing 6,992 pair-1
field keys are a parity check only. For every candidate point store integer
`grid_ix = x_m / 4000` and `grid_iy = y_m / 4000`, and define:

```text
grid_id = "epsg3413-4km:{grid_ix}:{grid_iy}"
```

Coordinates must be exact float64 multiples of 4,000 m. Sort canonically by
`grid_iy`, then `grid_ix`, serialize floats with six decimal places, use `0/1`
for booleans, and SHA-256 the final CSV bytes. The grid file and hash are frozen
before any full-pair RoMa output is opened.

Tile ownership uses the existing anchored 35,840 m tile cores and the same
half-open source-pixel convention as `source_core_mask`: lower core boundaries
included and upper boundaries excluded. A point with zero or multiple owners
is geometrically excluded and the condition is recorded; there is no nearest
tile fallback.

### 3.2 Matcher-independent support sets

The speed radii are fixed at:

```text
R1 = 30,000 * 14.893611111111111 / 24 = 18,617.0138888889 m
R2 = 30,000 * 32.83166666666666  / 24 = 41,039.5833333333 m
```

For each tile, build source and target support masks using the existing
north-up sampler, band-2 `< 2` validity rule, and fixed 16-pixel erosion. Use
the frozen route row joined exactly by tile ID, row, column, and source centre.

For an initial point `p0`, define `E1(p0)` as the set of pair-1 target pixel
centres that are within `R1` of `p0`, inside the pair-1 target domain and frozen
target window, and valid in the eroded target mask.

Define `S2` before matching as the union of pair-2 source positions that:

1. lie in exactly one frozen pair-2 source core and its eroded source-valid
   mask; and
2. have at least one eroded target-valid pixel centre inside the corresponding
   frozen pair-2 target window, target domain, and radius `R2`.

An initial grid ID enters the denominator exactly when it has unique pair-1
tile ownership, lies in the pair-1 source domain and eroded source-valid mask,
and `E1(p0)` has a non-empty intersection with `S2`. This existential test
removes only points for which the fixed geometry makes a two-pair estimate
impossible before either matcher is used.

At evaluation time, each method's predicted pair-1 endpoint `p1` is tested
against the already-frozen `S2`. If `p1` is outside `S2`, pair 2 is missing for
that method; the original `grid_id` remains in the denominator. The method's
outcome never creates, removes, or substitutes an initial ID. A valid numerator
case requires physical/domain/support validity for both pairs and survival of
the fixed spatial/topological coherence rule for both pairs.

The denominator table will contain at least:

```text
schema_version, grid_id, grid_ix, grid_iy, source_x_m, source_y_m,
pair1_tile_id, pair1_tile_row, pair1_tile_column,
pair1_unique_owner, pair1_source_domain, pair1_source_support,
pair1_reachable_target_count, pair2_potential_support,
in_primary_denominator, exclusion_reason
```

Exclusion reasons are single, ordered categories so counts add to the candidate
population: `outside_pair1_domain`, `tile_owner_missing`,
`tile_owner_nonunique`, `pair1_source_invalid`, `pair1_no_reachable_target`,
or `no_pair2_potential_support`.

## 4. Proposed quality and sensitivity contract

The frozen primary RoMa direct-grid arm has no learned-score threshold. It uses
only the fixed domain, mask/support, speed, unique ownership, and fixed-point
fold/topology rules. Its pre-topology result is always reported beside the
post-topology primary result.

RoMa quality diagnostics are a predeclared exploratory 3 x 4 grid:

- minimum forward overlap: none, `0.05`, or `0.10`;
- maximum forward-backward cycle error in original 512-pixel tile coordinates:
  none, `1`, `2`, or `4` pixels (80, 160, or 320 m).

All 12 arms are reported before and after the same fixed topology rejection.
No arm is selected as a new primary after viewing survival or buoy error. The
`0.05` overlap value has an official-setting precedent; the cycle values are
fixed spatial-error scales relative to the 80 m analysis pixel.

Precision matrices are retained and summarized through eigenvalues and the
maximum-axis standard-deviation equivalent after applying the official pixel
coordinate mapping. They do not reject points in this pilot because their
calibration and units have not yet been validated for these SAR tiles. Adding a
precision gate would require a campaign amendment approved before full results.

For sampled RoMa, use seeds `20260903`, `20260904`, and `20260905`. Run both
uniform and overlap-weighted downstream estimation for every seed using the
same cached dense prediction and physical gates. These six arms are secondary
and exploratory; no seed or weighting is selected using buoy results. RoMa
overlap is never numerically compared to the normalized EfficientLoFTR score.

Buoy tables remain sealed until the fixed population, all 12 direct-grid arms,
all six sampled arms, integrity checks, and method outputs are complete and
hashed. Every buoy metric retains six cases per pair in its denominator, with
missing predictions explicit.

## 5. Proposed EfficientLoFTR parity gates

These tolerances separate immutable-input parity from expected backend
floating-point variation. Any failure stops the campaign; references are not
rewritten and tolerances are not widened after inspecting RoMa or buoy results.

| Quantity | Proposed hard tolerance |
| --- | --- |
| input/reference/config hashes and byte sizes | exact |
| source/target IDs, elapsed times, CRS, raster shape/dtype/bands/mask rule | exact |
| route row count, compound key set, source centres, routing vectors and routing source labels | exact CSV values; computed target centre `atol=1e-9 m`, `rtol=0` |
| fixed-grid `grid_id` set and denominator CSV hash | exact |
| source tiles, skipped/matched tiles and skip-reason counts | exact |
| physics-valid match count | exact |
| grid-node count and available-ID sets before/after fold rejection | exact |
| fold-rejected ID set, topology triangle count, post-rejection folded count | exact |
| buoy expected/available/correct ID sets | exact |
| coverage and other count-derived fractions | `atol=1e-12`, `rtol=0` |
| available-node displacement against MPS reference, keyed by grid ID | median vector difference `<=0.10 m`; p99 `<=0.50 m`; maximum `<=1.00 m` |
| buoy predicted vector components and median/p90/maximum errors | absolute difference `<=1.00 m` |
| topology area-ratio p01/median/p99 | absolute difference `<=5e-4` |
| CSV persistence round trip | maximum absolute displacement difference `<=1e-6 m` |
| runtime, setup time and peak VRAM | recorded, not parity-gated |

Exact discrete parity is intentionally strict because a changed accepted match,
grid ID, or fold changes the scientific population. CUDA-versus-MPS floating
differences may pass only inside the preapproved metric bounds above. A count
or ID mismatch stops before RoMa pair inference and produces a diagnostic diff.

The previously reported conditional EfficientLoFTR survival must be recomputed
as `complete IDs / pair-1 accepted IDs` using stable IDs. The reference claim is
1,827 / 2,388 = 76.51%. A different numerator definition or ID mapping is
reported as a definition mismatch and stops comparison until reviewed.

## 6. Environment strategy

Use three isolated prefixes under the profile's proposed environment root; do
not modify `/Home/seachu` environments or the production Limosat environment:

1. `roma-v2-pilot-eval-py311`: CPU-only geospatial preparation, denominator,
   evaluation, figures, and tests; no matcher package.
2. `roma-v2-pilot-eloftr-py311`: CUDA PyTorch plus the minimum runtime packages
   needed by the pinned official EfficientLoFTR checkout and existing Limosat
   adapter. Do not install the entire legacy training requirements file.
3. `roma-v2-pilot-romav2-py312`: CUDA PyTorch/torchvision compatible with
   `torchvision>=0.23`, the pinned RoMa source installed from a local wheel
   build with no unreviewed dependency substitution, and the Linux
   fused-local-correlation dependency required by the pinned metadata.

Each environment gets a reviewed YAML/requirements lock, `conda list
--explicit`, `pip freeze`, Python/PyTorch/CUDA import record, and SHA-256 of the
specification. Matcher outputs cross the environment boundary only through
versioned files. The CPU evaluator remains independent of both matcher imports.

Milestone 1 may run import/version/CUDA visibility checks, but must not
instantiate RoMa (its constructor loads weights), call either matcher, sample a
warp, or perform scientific inference. Before solving the RoMa environment,
check that a compatible compiler/CUDA toolkit exists for
`fused-local-corr`; do not add a system toolkit or silently omit the kernel.

## 7. Storage estimate and limits

Known fixed staged files are 291,090,065 bytes (277.605 MiB): three rasters,
the EfficientLoFTR checkpoint, and both six-file reference directories.

| Class | Proposed allowance | Notes |
| --- | ---: | --- |
| staged fixed inputs | 0.30 GB | known 291,090,065 bytes plus small fixture/manifests |
| three source checkouts | 1.0 GiB | measure after clone; stop above allowance |
| EfficientLoFTR and RoMa model caches | 2.5 GiB | EfficientLoFTR is known; record RoMa asset size before download |
| three isolated environments | 28 GiB | separate from campaign output cap; 2 GiB evaluator and 12 GiB per matcher ceiling |
| Milestone 1 record/logs | 0.10 GiB | compact text and inventories only |
| one later campaign run | 20 GiB | campaign cap, including record/results/resume/scratch |

Conservative total reservation is 51.9 GiB. The profile's 499 GB observation
is ample but volatile; Milestone 1 requires at least 100 GiB free on the actual
filesystem before creating anything.

The dominant later-run risk is the precise dense cache. At 1,280 x 1,280,
two 2-channel warps, two 1-channel overlaps, and two 4-channel precision maps
are 14 float32 channels. For the current 185 matchable reference tiles, the
uncompressed upper estimate is:

```text
185 * 1280 * 1280 * 14 * 4 = 16,973,824,000 bytes = 15.808 GiB
```

Reserve at most 17.5 GiB for resume shards, 1.5 GiB for records/results, and
1.0 GiB for scratch. Stop before full inference if an observed one-tile shard
projects the run above 20 GiB. Compression may reduce use but is not credited
in the pre-run estimate. Full dense arrays are retained only as bounded resume
state and the named representative diagnostics; compact per-grid and per-seed
products are authoritative results.

## 8. Exact proposed files

This Milestone 0 changes only this audit file. After explicit approval, later
milestones would add the following files in a separate reviewed worktree:

```text
docs/experiments/roma_v2_vs_efficientloftr_2015.md
docs/experiments/roma_v2_vs_efficientloftr_2015_milestone0_audit.md
experiments/configs/roma_v2_vs_efficientloftr_2015_v1.json
experiments/environments/roma_v2_pilot_eval.yml
experiments/environments/roma_v2_pilot_eloftr.yml
experiments/environments/roma_v2_pilot_romav2.yml
experiments/prepare_roma_v2_nice2015_fixture.py
experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.csv
experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.provenance.json
experiments/run_roma_v2_nice2015_pilot.py
experiments/evaluate_roma_v2_nice2015_pilot.py
experiments/render_roma_v2_nice2015_pilot.py
tests/unit/test_roma_v2_nice2015_pilot.py
```

The truth fixture has exactly 12 rows and only provenance/identity and immutable
truth fields copied from the two verified `buoy_results.csv` files:

```text
transition_id, buoy_id, within_dataset_split, may_tune, report_primary,
source_fixture_image_id, target_fixture_image_id, source_image_id,
target_image_id, source_image_time, target_image_time, elapsed_hours,
source_x, source_y, target_x, target_y, truth_dx_m, truth_dy_m,
truth_distance_m, truth_speed_km_per_day, analysis_crs
```

The provenance JSON records both source paths and hashes, the ordered columns,
row count, extraction-command/source hash, and output hash. Prediction,
availability, candidate, proposal, support, and error columns are excluded.

No edits are proposed to `limosat/learned_drift/`,
`experiments/run_efficientloftr_sequence.py`, or any other production matcher
code. If the experiment-only runner cannot reuse those functions without a
production edit, stop and request a reviewed campaign amendment.

On Vega, Milestone 1 would create only:

```text
/Data/sat/seachu/projects/limosat
/Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464
/Data/sat/seachu/projects/external/RoMaV2-v2.0.1
/Data/sat/seachu/data/limosat/nice2015/roma-v2-pilot/
/Data/sat/seachu/cache/models/efficientloftr/
/Data/sat/seachu/cache/models/romav2/
/Data/sat/seachu/envs/roma-v2-pilot-{eval-py311,eloftr-py311,romav2-py312}/
/Data/sat/seachu/results/limosat/roma-v2-vs-efficientloftr/<RUN_ID>/{record,results,resume,scratch}/
```

Existing paths are never overwritten or repurposed. An existing verified
equivalent is recorded and proposed for reuse; an existing incompatible path is
a stop condition.

## 9. Exact proposed Milestone 1 commands

These commands are a reviewed proposal only. **Do not run them until Milestone
1 is explicitly approved.** `vega` denotes the existing SSH configuration for
`seachu@vega.ad.nersc.no`.

### 9.1 Read-only Vega preflight

```bash
ssh vega 'set -eu
hostname
uname -a
df -BG /Data/sat
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/Home/seachu/miniforge3/condabin/mamba --version
git --version
command -v sha256sum
command -v rsync
command -v nvcc || true
for p in \
  /Data/sat/seachu/projects/limosat \
  /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464 \
  /Data/sat/seachu/projects/external/RoMaV2-v2.0.1 \
  /Data/sat/seachu/data/limosat/nice2015/roma-v2-pilot \
  /Data/sat/seachu/cache/models/efficientloftr \
  /Data/sat/seachu/cache/models/romav2; do
  if test -e "$p"; then du -sh "$p"; fi
done'
```

Stop here if the resolved roots differ from the profile without an approved
mapping, free space is below 100 GiB, the A100/driver is not as expected, an
existing target is incompatible, or the compiler/toolkit requirement cannot be
met without system changes.

### 9.2 Local reviewed worktree and truth fixture

```bash
cd /Users/seachu/projects/limosat
test "$(git rev-parse HEAD)" = 939c6d42c3f6d9d9b512c53db4a865fa263aba29
git worktree add --detach /Users/seachu/projects/limosat-roma-v2-pilot 939c6d42c3f6d9d9b512c53db4a865fa263aba29
git -C /Users/seachu/projects/limosat-roma-v2-pilot switch -c codex/roma-v2-nice2015-pilot
mkdir -p /Users/seachu/projects/limosat-roma-v2-pilot/docs/experiments
cp docs/experiments/roma_v2_vs_efficientloftr_2015.md /Users/seachu/projects/limosat-roma-v2-pilot/docs/experiments/
cp docs/experiments/roma_v2_vs_efficientloftr_2015_milestone0_audit.md /Users/seachu/projects/limosat-roma-v2-pilot/docs/experiments/
/opt/homebrew/Cellar/micromamba/2.3.3_3/envs/limosat_scaling/bin/python \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/prepare_roma_v2_nice2015_fixture.py \
  --pair-reference /Users/seachu/projects/limosat/results/efficientloftr_reference_pair_6775_6801_20260903 \
  --pair-reference /Users/seachu/projects/limosat/results/efficientloftr_reference_pair_6801_6901_20260903 \
  --output /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.csv \
  --provenance-output /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.provenance.json
```

The worktree creation and fixture script/file additions require separate local
change approval. Review the resulting diff and fixture hash before any staging.
No commit is implied.

### 9.3 Run identity and local source manifest

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-roma-v2-vs-eloftr"
LOCAL_RECORD="/private/tmp/${RUN_ID}-m1-record"
mkdir -p "$LOCAL_RECORD"
cd /Users/seachu/projects/limosat
shasum -a 256 \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150428T154520_20150428T154620_005686_0074B8_CD78.tiff \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150429T063857_20150429T063957_005695_0074F1_9411.tiff \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150430T152851_20150430T152951_005715_007562_59BC.tiff \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt \
  results/efficientloftr_reference_pair_6775_6801_20260903/{buoy_results.csv,field_4km.csv,matches.npz,run_manifest.json,summary.json,tiles.csv} \
  results/efficientloftr_reference_pair_6801_6901_20260903/{buoy_results.csv,field_4km.csv,matches.npz,run_manifest.json,summary.json,tiles.csv} \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.csv \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.provenance.json \
  > "$LOCAL_RECORD/source-sha256.txt"
TRUTH_SHA="$(shasum -a 256 /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.csv | cut -d' ' -f1)"
PROVENANCE_SHA="$(shasum -a 256 /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.provenance.json | cut -d' ' -f1)"
{
  printf '%s  %s\n' 749006d5c2b63ba64a14cd7fefa70ef5eeffba97374d4312c8997814a9c5c312 scenes/6775.tiff
  printf '%s  %s\n' 4537d99cf2f61abc653f03c9bb97e8f791fb35ffd50307e1e29b82e5ce0cde5c scenes/6801.tiff
  printf '%s  %s\n' 9d563d3dd174b6f27ab584eaa99d51a77b638cfd2e8cdc4e59626b4a74cd14bf scenes/6901.tiff
  printf '%s  %s\n' 9e537bbb2fb478fc197669aff83715be6674e94bdab9fe50a7aed678d35d7ee5 references/efficientloftr_reference_pair_6775_6801_20260903/buoy_results.csv
  printf '%s  %s\n' 619aca77c92da2ca6f7e44220b50325d4be9cde53c86334ce5fe480fa3e58952 references/efficientloftr_reference_pair_6775_6801_20260903/field_4km.csv
  printf '%s  %s\n' e067a2df9aa4be105fcb293fa8fe0d84ebbc5c7927786d6cae6bd952fd3d4442 references/efficientloftr_reference_pair_6775_6801_20260903/matches.npz
  printf '%s  %s\n' 2ca9a6af384bcfe074986583ce17dcc156f1e346942962165e594231b0629c2d references/efficientloftr_reference_pair_6775_6801_20260903/run_manifest.json
  printf '%s  %s\n' e7b753dabd9f0250498f723fc37fc5c9e0546304226109b9a878b83a032242fa references/efficientloftr_reference_pair_6775_6801_20260903/summary.json
  printf '%s  %s\n' b4c3daa54ebc2f1d61e479ed3be6ff4b119007d8cc6629c67f7e510bbcdd9b49 references/efficientloftr_reference_pair_6775_6801_20260903/tiles.csv
  printf '%s  %s\n' d66d2906fd5fe18c97c1fbade155ca5777fe38f9d6966fdfe9cf6f315a5c169c references/efficientloftr_reference_pair_6801_6901_20260903/buoy_results.csv
  printf '%s  %s\n' c8312e69e7deae00b9f3be0e12ec125da8e4f2ac65f4ffc88581375a36291723 references/efficientloftr_reference_pair_6801_6901_20260903/field_4km.csv
  printf '%s  %s\n' e9ec4757e64bdc566b93d13a99726f2a2ae8105535874df82fc72ed9e7f71f25 references/efficientloftr_reference_pair_6801_6901_20260903/matches.npz
  printf '%s  %s\n' c210448d9aa545d373576e81804ed2d0fa5a0bbb13855a8d3112475652a92715 references/efficientloftr_reference_pair_6801_6901_20260903/run_manifest.json
  printf '%s  %s\n' 9e63f98d4d4d372c8fb463676f802463fe0256c9a7e78f922617d92450f545b6 references/efficientloftr_reference_pair_6801_6901_20260903/summary.json
  printf '%s  %s\n' b229db334828842f65854f28a5a47f8937ba9b159dd00302b18cd54c208495e2 references/efficientloftr_reference_pair_6801_6901_20260903/tiles.csv
  printf '%s  %s\n' "$TRUTH_SHA" truth/roma_v2_nice2015_buoy_truth_v1.csv
  printf '%s  %s\n' "$PROVENANCE_SHA" truth/roma_v2_nice2015_buoy_truth_v1.provenance.json
} > "$LOCAL_RECORD/staged-sha256.expected"
```

Record byte sizes with `stat`, the exact worktree status/patch hash, and the
run ID in the initial manifest before the first non-dry-run transfer.

### 9.4 Temporary Vega destination and copy dry runs

```bash
RESULT_ROOT=/Data/sat/seachu/results/limosat/roma-v2-vs-efficientloftr
RUN_DIR="$RESULT_ROOT/$RUN_ID"
DATA_PARENT=/Data/sat/seachu/data/limosat/nice2015
DATA_TMP="$DATA_PARENT/.roma-v2-pilot.incoming-$RUN_ID"
DATA_FINAL="$DATA_PARENT/roma-v2-pilot"
ssh vega "set -eu
test ! -e '$RUN_DIR'
test ! -e '$DATA_TMP'
test ! -e '$DATA_FINAL'
mkdir -p '$RUN_DIR'/record '$RUN_DIR'/results '$RUN_DIR'/resume '$RUN_DIR'/scratch
mkdir -p '$DATA_TMP'/scenes '$DATA_TMP'/references '$DATA_TMP'/truth
printf '%s\n' planned > '$RUN_DIR'/record/status.txt"
rsync -avhn --checksum --exclude='._*' --exclude='.DS_Store' --exclude='*.partial' \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150428T154520_20150428T154620_005686_0074B8_CD78.tiff \
  vega:"$DATA_TMP/scenes/6775.tiff"
rsync -avhn --checksum --exclude='._*' --exclude='.DS_Store' --exclude='*.partial' \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150429T063857_20150429T063957_005695_0074F1_9411.tiff \
  vega:"$DATA_TMP/scenes/6801.tiff"
rsync -avhn --checksum --exclude='._*' --exclude='.DS_Store' --exclude='*.partial' \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150430T152851_20150430T152951_005715_007562_59BC.tiff \
  vega:"$DATA_TMP/scenes/6901.tiff"
rsync -avhn --checksum --exclude='._*' --exclude='.DS_Store' --exclude='*.partial' \
  results/efficientloftr_reference_pair_6775_6801_20260903 \
  results/efficientloftr_reference_pair_6801_6901_20260903 \
  vega:"$DATA_TMP/references/"
rsync -avhn --checksum \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.csv \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/fixtures/roma_v2_nice2015_buoy_truth_v1.provenance.json \
  vega:"$DATA_TMP/truth/"
rsync -avhn --checksum \
  "$LOCAL_RECORD/source-sha256.txt" \
  "$LOCAL_RECORD/staged-sha256.expected" \
  vega:"$RUN_DIR/record/"
```

The EfficientLoFTR checkpoint is copied separately and is not duplicated under
the data root:

```bash
ELOFTR_CACHE=/Data/sat/seachu/cache/models/efficientloftr
ELOFTR_TMP="$ELOFTR_CACHE/.eloftr_outdoor.ckpt.incoming-$RUN_ID"
ELOFTR_FINAL="$ELOFTR_CACHE/eloftr_outdoor.ckpt"
ssh vega "set -eu; mkdir -p '$ELOFTR_CACHE'; test ! -e '$ELOFTR_TMP'; test ! -e '$ELOFTR_FINAL'"
rsync -avhn --checksum \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt \
  vega:"$ELOFTR_TMP"
# After reviewing the dry run:
rsync -avh --checksum \
  /Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt \
  vega:"$ELOFTR_TMP"
ssh vega "set -eu
test \"\$(stat -c %s '$ELOFTR_TMP')\" = 192797577
test \"\$(sha256sum '$ELOFTR_TMP' | cut -d' ' -f1)\" = 0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878
mv '$ELOFTR_TMP' '$ELOFTR_FINAL'"
```

After the dry-run logs are reviewed, repeat exactly with `-avh` instead of
`-avhn`, capture every command/exit code in `record/`, verify the expected file
count and all hashes at the temporary destination, then promote with:

```bash
ssh vega "set -eu
test \"\$(find '$DATA_TMP' -type f | wc -l)\" = 17
cd '$DATA_TMP'
sha256sum -c '$RUN_DIR/record/staged-sha256.expected'
test ! -e '$DATA_FINAL'
mv '$DATA_TMP' '$DATA_FINAL'"
```

No source is removed or changed.

### 9.5 Pinned source and environment staging

```bash
ssh vega 'set -eu
mkdir -p /Data/sat/seachu/projects/external
git clone --no-checkout git@github.com:nansencenter/limosat.git /Data/sat/seachu/projects/limosat.incoming
git -C /Data/sat/seachu/projects/limosat.incoming checkout --detach 939c6d42c3f6d9d9b512c53db4a865fa263aba29
test "$(git -C /Data/sat/seachu/projects/limosat.incoming rev-parse HEAD)" = 939c6d42c3f6d9d9b512c53db4a865fa263aba29
mv /Data/sat/seachu/projects/limosat.incoming /Data/sat/seachu/projects/limosat
git clone --no-checkout https://github.com/zju3dv/EfficientLoFTR.git /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464.incoming
git -C /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464.incoming checkout --detach ffd4a4644064354468eb1f0c7a3e732233cb732f
test "$(git -C /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464.incoming rev-parse HEAD)" = ffd4a4644064354468eb1f0c7a3e732233cb732f
mv /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464.incoming /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464
test "$(git ls-remote https://github.com/Parskatt/RoMaV2.git refs/tags/v2.0.1 | cut -f1)" = 95c9968145c8906b7b59383258e9f73b02853d89
git clone --no-checkout https://github.com/Parskatt/RoMaV2.git /Data/sat/seachu/projects/external/RoMaV2-v2.0.1.incoming
git -C /Data/sat/seachu/projects/external/RoMaV2-v2.0.1.incoming checkout --detach 95c9968145c8906b7b59383258e9f73b02853d89
test "$(git -C /Data/sat/seachu/projects/external/RoMaV2-v2.0.1.incoming rev-parse HEAD)" = 95c9968145c8906b7b59383258e9f73b02853d89
mv /Data/sat/seachu/projects/external/RoMaV2-v2.0.1.incoming /Data/sat/seachu/projects/external/RoMaV2-v2.0.1'
```

Run those commands only after the read-only existence checks confirm none of
the final targets exists. If a final target exists, audit it and either reuse it
by exact identity or stop; do not delete or overwrite it.

Transfer the three reviewed environment specifications into the immutable run
record, rather than changing the clean Limosat checkout:

```bash
ssh vega "mkdir -p '$RUN_DIR/record/environment_specs'"
rsync -avh --checksum \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/environments/roma_v2_pilot_eval.yml \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/environments/roma_v2_pilot_eloftr.yml \
  /Users/seachu/projects/limosat-roma-v2-pilot/experiments/environments/roma_v2_pilot_romav2.yml \
  vega:"$RUN_DIR/record/environment_specs/"
```

After their transferred hashes are checked:

```bash
ssh vega "set -eu
MAMBA=/Home/seachu/miniforge3/condabin/mamba
\$MAMBA env create --prefix /Data/sat/seachu/envs/roma-v2-pilot-eval-py311 --file '$RUN_DIR/record/environment_specs/roma_v2_pilot_eval.yml'
\$MAMBA env create --prefix /Data/sat/seachu/envs/roma-v2-pilot-eloftr-py311 --file '$RUN_DIR/record/environment_specs/roma_v2_pilot_eloftr.yml'
\$MAMBA env create --prefix /Data/sat/seachu/envs/roma-v2-pilot-romav2-py312 --file '$RUN_DIR/record/environment_specs/roma_v2_pilot_romav2.yml'
/Data/sat/seachu/envs/roma-v2-pilot-romav2-py312/bin/python -m pip install --no-deps /Data/sat/seachu/projects/external/RoMaV2-v2.0.1
for e in eval-py311 eloftr-py311 romav2-py312; do
  \$MAMBA list --prefix "/Data/sat/seachu/envs/roma-v2-pilot-\$e" --explicit > '$RUN_DIR/record/'"\$e"'-conda-explicit.txt'
  "/Data/sat/seachu/envs/roma-v2-pilot-\$e/bin/python" -m pip freeze > '$RUN_DIR/record/'"\$e"'-pip-freeze.txt'
done"
```

Download the RoMa asset once to a temporary file under
`/Data/sat/seachu/cache/models/romav2/`, record its source, byte size, and
SHA-256, verify a second read against that manifest, and atomically promote it
to `hub/checkpoints/romav2.0.1.pt`. The release does not need an independently
published checksum to proceed; the first verified download becomes the frozen
campaign checksum. Do not instantiate RoMa before this check.

```bash
ssh vega "set -eu
ROMA_CACHE=/Data/sat/seachu/cache/models/romav2/hub/checkpoints
ROMA_TMP=\"\$ROMA_CACHE/.romav2.0.1.pt.incoming-$RUN_ID\"
ROMA_FINAL=\"\$ROMA_CACHE/romav2.0.1.pt\"
mkdir -p \"\$ROMA_CACHE\"
test ! -e \"\$ROMA_TMP\"
test ! -e \"\$ROMA_FINAL\"
curl --fail --location --output \"\$ROMA_TMP\" https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt
stat -c '%s %n' \"\$ROMA_TMP\" > '$RUN_DIR/record/romav2-weight-size.txt'
cd \"\$ROMA_CACHE\"
sha256sum \"\$(basename \"\$ROMA_TMP\")\" > '$RUN_DIR/record/romav2-weight-sha256.txt'
sha256sum -c '$RUN_DIR/record/romav2-weight-sha256.txt'
mv \"\$ROMA_TMP\" \"\$ROMA_FINAL\""
```

### 9.6 Import/version/GPU preflight only

```bash
ssh vega 'set -eu
export TORCH_HOME=/Data/sat/seachu/cache/models/romav2
/Data/sat/seachu/envs/roma-v2-pilot-eval-py311/bin/python -c "import numpy, pandas, pyproj, rasterio, scipy, shapely; print(\"eval imports ok\")"
/Data/sat/seachu/envs/roma-v2-pilot-eloftr-py311/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
/Data/sat/seachu/envs/roma-v2-pilot-romav2-py312/bin/python -c "import importlib.metadata as m, torch; import romav2; print(m.version(\"romav2\"), torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
test "$(git -C /Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464 rev-parse HEAD)" = ffd4a4644064354468eb1f0c7a3e732233cb732f
test "$(git -C /Data/sat/seachu/projects/external/RoMaV2-v2.0.1 rev-parse HEAD)" = 95c9968145c8906b7b59383258e9f73b02853d89'
```

This ends Milestone 1. No model constructor, matcher call, warp sampling, or
scientific pair processing is included.

## 10. Review outcome and remaining observations

The user reviewed this plan on 2026-09-03. The scientific comparison,
denominator rule, parity tolerances, exploratory threshold grid, sampled seeds,
and weighting arms are accepted as written. The following setup choices are
also accepted:

- official RoMaV2 `v2.0.1` at full commit
  `95c9968145c8906b7b59383258e9f73b02853d89`;
- the RoMa/DINOv3 licensing terms for this pilot; and
- the three isolated environments, subsequently amended to a 28 GiB total
  allowance with 12 GiB per matcher and 2 GiB for evaluation.

No experiment-design decision remains unresolved at Milestone 0. Milestone 1
must still observe and record the actual Vega paths, package resolution, RoMa
weight size/SHA-256, and environment sizes. These are routine provenance facts,
not new approval gates unless they reveal an integrity or compatibility problem.
The isolated local implementation worktree is
`/private/tmp/limosat-roma-v2-pilot` on branch
`codex/roma-v2-nice2015-pilot`. No commit is authorized.

## 11. Milestone 1 stopping conditions

Stop and preserve compact evidence if any of the following occurs:

- `/Data/sat` has less than 100 GiB free, total environment use exceeds 24 GiB,
  or projected campaign output exceeds 20 GiB;
- an existing target cannot be proven suitable for reuse;
- an input/reference checksum, raster semantic, file count, or frozen route key
  differs from the locally verified evidence;
- a pinned source cannot be checked out, or RoMa's installed API/precise setting
  materially differs from the audited `v2.0.1` source;
- the downloaded RoMa asset cannot be read back with a stable SHA-256, or a
  later copy differs from the frozen first-download checksum;
- either isolated matcher environment fails its import/CUDA preflight, or
  fixing compatibility would require system or production-environment changes;
- EfficientLoFTR fails the accepted discrete or floating parity gates; or
- source preparation would touch unrelated work, change protected geometry or
  evaluation rules, or require a production matcher edit.

On a stop, do not retry blindly, substitute a different commit/weight/backend,
widen a tolerance, or proceed to model inference. Record all checks, exact
failure evidence, retained temporary paths and sizes, then request the smallest
specific decision or correction.

## 12. Milestone gate

Milestone 0 has been reviewed and its experiment/setup choices accepted.
Milestone 1 was authorized, encountered the original environment-size gate,
and completed after the user approved the minimal recovery and storage
amendment. All amended limits and import/GPU checks passed. Stop here for human
review before Milestone 2 implementation or any model construction, matcher
call, warp sampling, or scientific inference.
