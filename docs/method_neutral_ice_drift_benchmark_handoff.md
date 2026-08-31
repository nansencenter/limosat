# Method-neutral ice-drift benchmark: implementation handoff

Status: implementation not started  
Created: 2026-08-26  
Target repository: `nansencenter/arktalas_ice_drift_experiments`  
Initial deployment target: Olivia  

## 1. Objective

Build a small, reproducible assessment repository that can compare production
LiMOSAT ORB, ALIKED, EfficientLoFTR, and future drift methods without embedding
any one method or its software environment.

The framework must evaluate the quantities that matter for the intended Arctic
archive product:

- displacement accuracy against buoys;
- spatial coverage and valid measurement density;
- persistent Lagrangian trajectory behaviour, including gaps and reconnection;
- deformation accuracy and physical consistency;
- runtime and resource use; and
- reproducibility from a frozen data and run manifest.

The first physical deformation benchmark will use the N-ICE2015 buoy array. The
first broader method comparison will use production-calibrated Sentinel-1 data
and the existing stratified Arctic buoy/SAR cases. Antarctica is a later dataset,
not a requirement for the first implementation.

## 2. Repository decision

Use the existing `arktalas_ice_drift_experiments` repository. Do not create a
fourth repository and do not add this framework to LiMOSAT.

The local checkout at
`/Users/seachu/projects/arktalas_ice_drift_experiments` is currently on
`devlimosat`, which embeds an obsolete copy of LiMOSAT. Start a clean branch
named `codex/method-neutral-benchmark` from `origin/main`, not from
`devlimosat`. There is no requirement to preserve the repository's obsolete
notebooks, utilities, embedded LiMOSAT copy, environment or CI structure. Inspect
them for useful scientific logic and data provenance, then replace or remove
them when the new benchmark demonstrably supersedes them. Keep cleanup explicit
and reviewable rather than carrying legacy structure into the new design.

Repository responsibilities are:

| Repository | Responsibility |
| --- | --- |
| `limosat` and learned-method branches/repos | Generate displacement fields and trajectories |
| `arktalas_ice_drift_experiments` | Frozen datasets, schemas, method adapters, metrics, comparisons and reports |
| `arktalas-deployment` | Operational image catalogues, preprocessing and archive production |

The benchmark must read method outputs from disk. It must not import PyTorch,
LightGlue, EfficientLoFTR, or LiMOSAT internals into its core evaluator.

## 3. Design principles

1. **Evidence before tuning.** Every experiment must test a named hypothesis or
   compare frozen methods on a frozen benchmark.
2. **Separate measurement from evaluation.** Buoy positions must not influence
   operational feature detection, matching, candidate selection or field
   construction. Explicit buoy-seeded probes are allowed only as a separately
   labelled diagnostic.
3. **Missing results count.** Accuracy is reported with availability and coverage;
   methods are not compared only on their successful points.
4. **Common geometry.** Coordinates are float64 EPSG:3413 metres, velocities are
   metres per second, time intervals are seconds, and strain rates are calculated
   in inverse seconds and reported in inverse days.
5. **No hidden interpolation.** The sampling and interpolation used for validation
   are part of the benchmark definition and are identical across methods.
6. **No primary smoothing.** N-ICE validation uses displacement over the exact SAR
   pair interval. Smoothed time series may be secondary visualizations only.
7. **Compact core.** Prefer functions and small data classes over registries,
   plugin frameworks, or deep inheritance. Method adapters are thin converters.
8. **Portable manifests.** Tracked dataset definitions contain stable scene IDs
   and paths relative to a configured data root, never workstation or cluster
   absolute paths.
9. **Reproducible output.** Every result records code revisions, method config,
   image preparation, model/checkpoint hash, device, runtime and input checksums.
10. **Local proof before HPC.** All schemas, metrics and submission scripts must
    pass local synthetic and tiny-fixture tests before the Olivia handoff.

## 4. Intended repository layout

Keep the initial structure small:

```text
src/arktalas_benchmark/
    schema.py              canonical tables and validation
    datasets.py            manifests, scene roots and buoy loading
    sampling.py            common field sampling with explicit support rules
    drift.py               buoy displacement and trajectory metrics
    deformation.py         triangle geometry and strain metrics
    report.py              compact machine-readable and Markdown summaries
    cli.py                 validate, evaluate and report commands
    adapters/
        limosat.py         production ORB SQLite/output converter
        learned_field.py   canonical ALIKED/EfficientLoFTR field converter
benchmarks/
    nice2015_winter_v1/
        benchmark.yaml
        scenes.csv
        pairs.csv
        buoy_array.csv
        README.md
    arctic_buoy_sar_v1/
        benchmark.yaml
        scenes.csv
        pairs.csv
        buoy_splits.csv
        README.md
data/
    nice2015/
        buoy_positions.parquet
        PROVENANCE.md
        source_checksums.txt
scripts/
    prepare_nice2015.py
    evaluate_method.py
hpc/olivia/
    README.md
    site.example.yaml
    preflight.sh
    evaluate_method.sbatch
tests/
    fixtures/
    test_schema.py
    test_sampling.py
    test_drift.py
    test_deformation.py
    test_adapters.py
pyproject.toml
environment.yaml
```

The target structure may replace the existing repository layout. Delete obsolete
files once their useful logic, provenance and data have been accounted for. Keep
scientific implementation and legacy cleanup in separate commits where practical
so both remain easy to review.

## 5. Canonical method outputs

Version every schema. Use Parquet for tables and JSON or YAML for small manifests.
Large method outputs and reports live on cluster work storage, not in Git.

### 5.1 Pair displacement field

Required columns:

| Column | Type and unit |
| --- | --- |
| `pair_id` | string, stable within the benchmark |
| `source_scene_id`, `target_scene_id` | string |
| `source_time_utc`, `target_time_utc` | timezone-aware UTC timestamp |
| `crs_epsg` | integer; initially exactly 3413 |
| `x_m`, `y_m` | float64 source position in metres |
| `dx_m`, `dy_m` | float64 displacement to the target time |
| `available` | boolean |
| `support_count` | nullable integer |
| `confidence` | nullable float with method-specific meaning documented separately |

Optional method diagnostics may be retained in a separate table, but evaluation
must not require them. Never silently replace an unavailable vector with zero.

### 5.2 Lagrangian trajectories

Required columns:

| Column | Type and meaning |
| --- | --- |
| `trajectory_id` | stable string within a run |
| `scene_id`, `time_utc` | observation identity and UTC time |
| `crs_epsg`, `x_m`, `y_m` | float64 position |
| `state` | `observed`, `observed_skip_edge`, `dormant`, `reconnected`, or `terminated` |
| `source_pair_id` | nullable pair that produced the state |

The schema supports traditional LiMOSAT feature identities and trajectories
advected through learned displacement fields. Reports must distinguish an actual
observation from a propagated or missing state.

### 5.3 Run manifest

Record at least:

- schema and benchmark versions;
- method name and declared architecture;
- repository URL, Git revision and dirty-state flag;
- resolved method configuration and its hash;
- model/checkpoint path and SHA256 where applicable;
- preprocessing recipe and input-scene checksums;
- device, host, thread count, peak memory if available, and stage runtimes;
- exact command, start/end UTC times, completion status and output checksums.

### 5.4 Scientific product definition for public release

The learned-method trajectory product should be described as **SAR-derived
advected Lagrangian sea-ice trajectories**. It is mathematically Lagrangian,
but it is not a record of one image feature being directly recognized in every
scene. Keep that distinction explicit in the product title, abstract, metadata,
and user guide:

- production ORB LiMOSAT provides directly matched feature trajectories; and
- ALIKED and EfficientLoFTR provide virtual material-point trajectories advected
  through independently measured pairwise SAR displacement fields.

For an accepted displacement field from image time `t_k` to `t_(k+1)`, define
the discrete flow map

```text
F_k(x) = x + d_k(x)
X_(k+1) = F_k(X_k)
```

where `d_k(X_k)` is evaluated using the declared field-sampling rule. A complete
trajectory is the composition of the accepted pair maps. A position is emitted
only when the field has valid local support and passes the topology rules. An
unsupported point becomes dormant without receiving an interpolated or
constant-velocity coordinate. An observed skip edge is an independent SAR
measurement, not temporal interpolation.

The release lineage has four levels:

| Level | Product | Scientific role |
| --- | --- | --- |
| L0 | Pair correspondences and matcher diagnostics | Evidence supporting field construction |
| L1 | Pairwise displacement fields | Fundamental method-neutral SAR measurement |
| L2 | Advected material-point trajectories | Composition of supported pair measurements |
| L3 | Deformation and Lagrangian-cell products | Quantities derived from L1 or L2 with declared geometry |

L0 can be too large for the compact public archive, but its retention policy,
checksums, summary diagnostics and reproduction command must be declared. L1 and
L2 are separate release products: users must be able to recompute or replace the
trajectory composition without rerunning the matcher. L3 must identify whether
it was calculated directly from a pair field or from a composed trajectory.

The current `v1` evaluator schema and Olivia smoke test must remain frozen. Before
public release, introduce a versioned, additive trajectory-schema revision that
separates temporal state from measurement basis. At minimum it should add:

- `position_basis`: `seed_grid`, `direct_feature_match`,
  `field_advected_adjacent`, `field_advected_skip`, or `predicted_diagnostic`;
- `position_uncertainty_m`: nullable radial position uncertainty; and
- `uncertainty_method`: the named calculation or `not_available`.

The existing `state` column continues to describe temporal availability and
reconnection. It must not be used by itself to imply that a learned-method point
was directly recognized in the image. Predicted diagnostic positions are not
part of the primary scientific trajectory product.

Uncertainty must accumulate along the selected path. A suitable formal model is

```text
Sigma_(k+1) = J_F Sigma_k J_F^T + Q_k
```

where `J_F` is the local flow-map Jacobian and `Q_k` describes pair-field,
geolocation and sampling uncertainty. A reproducible bootstrap or conservative
scalar approximation is acceptable initially, but the method and calibration
must be recorded. Reconnection does not silently reset uncertainty. If no
defensible estimate is available, store missing uncertainty rather than false
precision.

#### Field-advection risks and controls

Hutter et al. (2018) show that trajectories integrated through gridded
satellite drift can accumulate position error and produce spurious deformation
where contributing fields have inconsistent timing or mean drift. They also
show that interpolation and the chosen deformation operator can materially
change deformation scaling. Apply the following controls:

1. Construct every released pair field from one explicit source time and one
   explicit target time. Do not mosaic vectors from different intervals into a
   deformation field without a separately validated temporal-normalization
   method.
2. Retain tile and acquisition provenance so seams can be audited. The current
   EfficientLoFTR tiling uses the same image pair and interval throughout a
   field, which avoids one important composite-field failure mode but does not
   remove seam or routing bias.
3. Do not interpolate across unsupported regions or blindly average distinct
   local motion modes. Report support radius, match count, residual dispersion,
   folds and evidence of multimodal motion near leads and floe boundaries.
4. Keep direct skip fields and composed adjacent fields as separate
   measurements. Compare closure and deformation before any fusion, and flag
   conflicts rather than averaging them silently.
5. Validate displacement and deformation separately. Small endpoint differences
   can become large strain-rate differences after spatial differentiation,
   especially at short time intervals.

#### Public-release gates

A learned trajectory archive is ready for scientific release only when all of
the following are reported, including failures and missing support:

1. one-step displacement accuracy and availability against exact-time,
   method-independent buoy observations;
2. cumulative and endpoint trajectory error, duration, survival, dormancy and
   reconnection on held-out sequences;
3. forward/reverse, direct/composed and image-omission closure;
4. fold/topology rates and N-ICE divergence, shear and total-deformation skill
   across declared spatial and temporal scales;
5. calibrated or explicitly unavailable trajectory uncertainty;
6. coverage and nearest-neighbour spacing on both full and exact-common support;
7. acquisition-time, CRS, preprocessing, matcher, field, path and code
   provenance sufficient to reproduce every released level; and
8. runtime, storage, restart and deterministic-resume evidence at archive scale.

ICESat-2 and CryoSat-2 remain independent structural validation and
interpretation sources. They should not substitute for buoy displacement or
N-ICE deformation truth, and alignment-sensitivity results must accompany their
reported relationships.

## 6. Data fixtures

### 6.1 N-ICE2015

The redistribution terms have been confirmed by the project owner. Commit a
compact canonical derived buoy table together with provenance; do not commit the
raw ZIP archives.

Authoritative local source material is currently in:

```text
/Users/seachu/Downloads/6ed9a8ca-95b0-43be-bedf-8176bf56da80_attachments
```

Dataset DOI:
`https://doi.org/10.21334/npolar.2015.6ed9a8ca`

The tracked legacy validation file on `origin/main` contains approximately
30,500 positions from 17 buoy names and is useful as a regression reference.
Do not treat it as the canonical source without comparison to the official JSON
attachments and the buoy tables supplied with the dataset.

Canonical buoy columns:

- `buoy_id` and instrument type;
- `time_utc` as a timezone-aware UTC timestamp;
- float64 `longitude_deg`, `latitude_deg`, `x_m`, `y_m`;
- `crs_epsg=3413` for projected coordinates;
- source quality flag and an explicit benchmark QC status;
- expedition leg/array and colocated-cluster identifier;
- whether the buoy is included in each published deformation protocol;
- source DOI and source filename.

Preparation requirements:

1. Read the official JSON LineStrings without modifying the raw files.
2. Validate coordinate ranges, monotonic time per buoy, duplicate timestamps,
   finite values and projection round trips.
3. Preserve source quality separately from benchmark exclusions.
4. Reconstruct the winter array selection from the published buoy tables and
   papers; store the mapping explicitly rather than burying it in code.
5. Compare row counts, buoy coverage and time ranges with the existing full
   validation GeoJSON and explain differences in `PROVENANCE.md`.
6. Write deterministic Parquet and source checksums. Re-running preparation must
   reproduce its content hash, apart from documented library metadata.

The initial N-ICE interval is the winter comparison used by Itkin (2025), roughly
15 January to 18 February 2015 and the approximately 20 km inner buoy ring. Add
the spring array only after the winter implementation is verified.

### 6.2 Arctic buoy/Sentinel-1 benchmark

Reuse the established 2020 Arctic design rather than selecting a convenient new
case after seeing method results. Its source plan is:

```text
/Users/seachu/projects/limosat/docs/arctic_tracking_next_experiment_plan.md
```

Preserve the within-dataset split rule: whole buoy identities remain within one
fold, and N-ICE is reported as its own regime rather than being pooled as the
2020 transfer holdout. The benchmark manifest should identify production-
calibrated Sentinel-1 assets by scene ID and checksum. It may include a week-plus
challenging sequence and same-pass overlaps, but those selections must be frozen
before comparing methods.

## 7. Drift and trajectory evaluation

For a SAR pair at `t0` and `t1`:

1. Interpolate each QC buoy in EPSG:3413 to the exact two acquisition times.
2. Do not extrapolate. Record the bracketing observation gaps at both endpoints.
3. Sample the generic method field at the buoy's `t0` position using the common,
   declared sampling rule.
4. Report availability before calculating successful-case accuracy.
5. Calculate `dx` and `dy` error, endpoint vector error, direction error and speed
   error. Cluster uncertainty by buoy and acquisition pass.

The sampling rule must be tested and frozen. Start with linear interpolation over
a Delaunay triangulation of available field nodes, reject points outside the
convex hull, and apply a declared maximum support-distance rule. Compare it with
nearest-node sampling on development data only to ensure that the evaluator is
not manufacturing an apparent method advantage.

Trajectory reports include:

- number of initial and persistent material points;
- observation count and elapsed duration distribution;
- survival by acquisition index and by elapsed time;
- spatial coverage and nearest-neighbour spacing;
- gap, skip-edge, dormancy and reconnection rates;
- endpoint and cumulative buoy error where truth exists;
- forward/reverse and direct/skip cycle closure;
- runtime per image, image pair and accepted measurement.

Compare methods on exact-common cases as well as each method's full support. The
former isolates accuracy; the latter exposes the coverage trade-off.

## 8. N-ICE deformation validation

Implement the published buoy-triangle calculation as an independent reference
operator. Maintain two named protocols if the source papers differ:

- `itkin2017_all_valid_triangles` for the original N-ICE array analysis; and
- `itkin2025_inner_ring_delaunay` for direct comparison with SAR deformation.

The first operational benchmark is the 2025 inner-ring protocol.

For each eligible SAR pair:

1. Interpolate the selected buoy positions to exact `t0` and `t1`; never
   extrapolate.
2. Use one scientifically selected buoy from each colocated cluster.
3. Construct the source-time Delaunay triangulation in EPSG:3413.
4. Preserve counter-clockwise vertex order and reject triangles with any source
   or target internal angle below 15 degrees.
5. Calculate observed vertex velocities in m/s from displacement divided by the
   exact interval in seconds.
6. Calculate `du/dx`, `du/dy`, `dv/dx`, and `dv/dy` with Green's theorem.
7. Calculate signed divergence, maximal shear and total deformation. Store s^-1
   and report day^-1.
8. Sample the satellite displacement field at the exact same three source
   vertices, then run the identical triangle operator. Do not compare differently
   scaled arbitrary raster cells with buoy triangles.
9. Independently compare finite area change `log(A1/A0)/dt` and a documented
   shape-change measure.
10. Retain signed divergence separately from non-negative shear and total
    deformation.

Primary deformation reporting:

- divergence bias, MAE, correlation and sign agreement;
- shear and total-deformation MAE and log-ratio;
- principal strain-orientation circular error where resolvable;
- high-deformation event precision/recall with thresholds frozen on development
  data or taken from the literature;
- triangle availability and missingness;
- metrics stratified by triangle length scale, time gap and N-ICE storm period;
- scale-dependent deformation distributions and power-law slope as a secondary
  structural diagnostic.

Estimate the resolvable deformation floor by perturbing buoy coordinates under a
documented GPS-error model. Measurements below that floor should be marked
unresolved rather than counted as exact zeros. Keep Monte Carlo draws and random
seeds reproducible.

Reference tests must include:

- uniform translation gives zero deformation;
- rigid rotation gives zero divergence and shear;
- known pure divergence and pure shear are recovered;
- results are invariant to translation, valid vertex reordering and time-unit
  conversion;
- degenerate and sub-15-degree triangles are rejected;
- finite area change agrees with infinitesimal divergence for small deformation;
- missing field support remains missing.

## 9. Method adapters

Start with two adapters:

1. **Production ORB:** convert a representative LiMOSAT SQLite/output database
   into canonical pair fields and trajectories. Preserve observed versus
   interpolated states. Do not alter the database.
2. **Learned field:** convert the method-neutral displacement and propagated-grid
   outputs currently produced by ALIKED and EfficientLoFTR. Method-specific raw
   matches remain optional diagnostic artifacts.

Adapters must be deterministic and tested on tiny fixtures. Do not copy matcher
logic into the benchmark. If a source cannot express a required state, record it
as unknown rather than guessing.

## 10. Reports and decision table

Every benchmark run should create:

```text
run_manifest.json
metrics.json
pair_metrics.parquet
trajectory_metrics.parquet
deformation_metrics.parquet
report.md
figures/
```

The comparison report must show, at minimum:

| Dimension | Required summary |
| --- | --- |
| Buoy accuracy | median, P90 and catastrophic-error rate, plus availability |
| Coverage | valid area, nodes, spacing and fold/topology rejection |
| Trajectories | length/duration distributions, full-sequence survival, gaps and reconnection |
| Deformation | N-ICE divergence/shear/total metrics by scale and coverage |
| Consistency | cycle closure and topology diagnostics |
| Runtime | cold/warm wall time, extraction, matching, field/trajectory stages and hardware |

Do not collapse selection to one undocumented score. Present an explicit Pareto
comparison of accuracy, coverage, deformation validity and computation.

## 11. Olivia handoff without a remote agent

Assume that a human will clone, configure and submit the work on Olivia without
an agent. The repository must therefore provide copy-paste commands and fail-fast
preflight checks.

`hpc/olivia/README.md` must document:

1. cloning and checking out the exact benchmark branch/tag;
2. creating the evaluator environment from the repository file;
3. configuring project, work, image, method-output and result roots in one
   untracked site YAML;
4. verifying production-calibrated scene existence and checksums;
5. verifying method repository revisions and model checkpoints;
6. running a CPU-only synthetic test and a one-pair smoke test;
7. submitting a small diagnostic job before a full benchmark;
8. monitoring logs and recognizing successful completion;
9. resuming without overwriting completed immutable outputs; and
10. collecting the small manifests, metrics and reports for local review.

Follow Olivia storage roles: source and durable small configuration under the
project area, large active results under work storage, and no large write-heavy
run in the home directory. Do not claim Olivia verification until the user
returns the job logs and run manifest.

Keep the evaluator environment CPU-only. Each learned method should run in its
own GPU-compatible environment and export canonical files. This avoids forcing
ORB, ALIKED and EfficientLoFTR into one fragile environment.

The preflight script must check paths, revisions, available space, schema
versions, scene counts, checksums, output non-collision and a tiny read/write
operation. It must not download data or mutate method repositories.

## 12. Milestones and acceptance gates

Work one milestone at a time. Make a small commit after each successful milestone
in the fresh task, but do not push until the complete local handoff is reviewed.

### M0: clean foundation

- Create `codex/method-neutral-benchmark` from `origin/main`.
- Add concise package/environment scaffolding and update the README.
- Audit the existing contents, retain only useful source material, and remove
  obsolete structure in a separate reviewable commit.
- Gate: package imports and focused tests run locally.

### M1: N-ICE fixture and provenance

- Build the canonical table from official attachments.
- Freeze winter array/cluster membership and checksums.
- Compare with the legacy validation GeoJSON.
- Gate: deterministic preparation, CRS/time/QC tests, and an auditable summary.

### M2: schemas and synthetic evaluator

- Implement pair, trajectory and run-manifest validation.
- Implement common field sampling.
- Gate: a tiny synthetic method can pass the complete CLI and deliberately bad
  units, CRS, timestamps and missingness fail clearly.

### M3: drift and trajectory metrics

- Implement exact-time buoy interpolation and method-neutral metrics.
- Gate: synthetic translations recover known errors and no truth leaks into
  method outputs.

### M4: N-ICE deformation

- Implement and document both named triangle protocols.
- Gate: all physical invariants pass and the official winter buoy fixture yields
  plausible, auditable triangle counts and scales.

### M5: real method adapters

- Add production ORB and learned-field adapters.
- Gate: both create schema-valid files from small existing local outputs, with no
  imports of the method implementations.

### M6: frozen benchmark manifests and reporting

- Add N-ICE and Arctic buoy/SAR definitions and split diagnostics.
- Produce a local comparison from available small fixtures; do not infer GPU
  runtime from it.
- Gate: exact-common and full-support views agree on denominators and are
  reproducible from the run record.

### M7: Olivia package

- Add site template, preflight, small/full job scripts and operator README.
- Gate: shell syntax checks pass, all commands use configurable roots, a dry run
  resolves inputs without submitting, and the handoff lists the exact files the
  user should return after execution.

## 13. Scope control

Do not do the following during the initial implementation:

- tune ORB, ALIKED or EfficientLoFTR;
- add RoMa or another matcher merely to exercise extensibility;
- redesign LiMOSAT persistence;
- download a broad new SAR archive;
- use ICESat-2 or CryoSat-2 to choose the first framework thresholds;
- pool N-ICE and 2020 folds;
- add a web dashboard or database service;
- run a full local benchmark that duplicates the intended Olivia GPU work.

ICESat-2, CryoSat-2, AMSR2, Antarctic cases and additional methods should later
enter through new benchmark datasets or the same canonical method-output schema,
not through changes to the core scientific quantities.

## 14. Required final handoff from the fresh task

The fresh task should end with:

- a changed-file and commit summary;
- focused test results and the exact environment used;
- N-ICE source-versus-canonical audit counts;
- the canonical schema version and example files;
- a local synthetic end-to-end report;
- any small real-output adapter smoke results;
- exact Olivia setup, preflight, one-pair and full submission commands;
- expected output paths and files to bring back;
- unresolved scientific or operational decisions; and
- a summary of obsolete files removed and any remaining cleanup that is still
  justified.

## 15. Primary references

- N-ICE2015 buoy dataset: `https://doi.org/10.21334/npolar.2015.6ed9a8ca`
- Itkin et al. (2017), *Thin ice and storms: Sea ice deformation from buoy
  arrays deployed during N-ICE2015*:
  `https://doi.org/10.1002/2016JC012403`
- Itkin (2025), *Novel methods to study sea ice deformation using ship radar and
  satellite images during the N-ICE2015 expedition*:
  `https://doi.org/10.5194/tc-19-1135-2025`
- Hutter et al. (2018), *Scaling Properties of Arctic Sea Ice Deformation in a
  High-Resolution Viscous-Plastic Sea Ice Model and in Satellite Observations*:
  `https://doi.org/10.1002/2017JC013119`
- Existing broader tracking plan:
  `/Users/seachu/projects/limosat/docs/arctic_tracking_next_experiment_plan.md`
