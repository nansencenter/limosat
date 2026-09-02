# April 2020 week-01 GPU rerun

This procedure prepares a new native run from 2020-04-01T00:00:00Z through
2020-04-08T00:00:00Z. The Kingston
`april2020-week01-primary-v2` directory is read-only reference data: it contains
781 scene records and 604 completed primary pair fields selected at a 0.25
minimum overlap. It is not an input raster archive and must not be modified.
The local metadata audit found zero same-platform/absolute-orbit combinations
even before footprint filtering among pairs in the configured 1--96 hour time
window, so this safeguard is not expected to remove a pair in this week.

## Inputs to resolve on the GPU host

- the source catalogue whose recorded SHA256 is
  `12ba2654f82a7da78dff3724fd89f880583ea5347345009f2d7a57477c32f58d`;
- the corresponding 781 Sentinel-1 rasters;
- the EfficientLoFTR repository revision and checkpoint used for the rerun;
- daily OSI SAF northern-hemisphere concentration files covering 2020-03-31
  through 2020-04-08, preferably with `ice_conc_unfiltered`; and
- a new schema-v4 database, output directory, and run ID.

Do not reuse the completed field-replay database or an old native database.
First verify that the catalogue carries `platform` and `absolute_orbit` (or a
supported alias); standard Sentinel-1 product IDs are also parsed.
Before starting inference, verify that the GPU environment can open one SIC
file through Rasterio's NetCDF driver or its existing `netCDF4` runtime. The
project does not add `netCDF4` as a mandatory dependency.

## Conservative configuration

Keep the current 12-neighbour/8-agreeing, 6 km search, and 1 km agreement field
policy for this rerun. The adaptive consensus screen increased deformation
tails and has not passed the scientific decision gate.

```yaml
routing:
  initial: phase_correlation
  phase_correlation_failure: same_center
  phase_correlation_minimum_response: 0.05
  exclude_same_acquisition_pass: true
  require_orbit_metadata: true
  candidate_minimum_elapsed_hours: 1.0
  candidate_maximum_elapsed_hours: 96.0
  candidate_minimum_overlap_fraction: 0.05
  candidate_minimum_overlap_area_m2: 1024000000.0
  maximum_recovery_elapsed_hours: 96.0
  candidate_pair_ids: []

open_water:
  enabled: true
  sic_root: /path/to/osi-saf-sic
  threshold_percent: 15.0
  maximum_age_days: 1
  samples_per_axis: 5

trajectories:
  convergence_audit_radius_m: null

retain_pair_matches: true
```

The 15% threshold follows the
[standard operational ice-edge convention](https://nsidc.org/learn/parts-cryosphere/sea-ice/science-sea-ice), but
the implementation is deliberately stricter than a centre-point mask: all 25
samples on both dates must be finite and below the threshold. It is a compute
gate, not a declaration that parcel motion is known or missing. The 30 km/day
speed bound and field-consensus tolerances remain explicit experiment settings
rather than universal sea-ice constants.

## Full native rerun

On Olivia, use the source-controlled preparation and submission scripts instead
of an interactive Python heredoc. Preparation verifies the frozen source
catalogue checksum, selects exactly 781 images, verifies every raster, stages
only the nine required SIC files, and writes the reviewed JSON configuration.
It refuses to replace an existing run database:

```bash
cd /cluster/projects/nn9878k/seachu/limosat-efficientloftr
python3 scripts/prepare_april_week_olivia.py --download-sic
scripts/submit_april_week_olivia.sh
```

The submit wrapper records the method and official EfficientLoFTR revisions
plus the configuration, catalogue, and checkpoint checksums before submitting
a dependency chain: CPU preparation, primary GPU pair processing, CPU primary
composition, recovery GPU pair processing, and CPU final
composition/finalization. CPU jobs run on the Olivia accelerator partition
with zero GPUs so the GPU is released while millions of trajectory rows are
written. Every stage uses the same container, configuration, and source
revision.

`GPU_WORKERS=1` is the default. A later run can set `GPU_WORKERS=N` to create
deterministic primary and recovery Slurm arrays without concurrent SQLite
writes. Re-running only the submit wrapper resumes verified pair products and
already imported fields. Site settings can be overridden with `GPU_ACCOUNT`,
`CPU_ACCOUNT`, `WALL_TIME`, `CPU_WALL_TIME`, `MEMORY`, `CPU_MEMORY`, `RUN_ID`,
and `RUN_ROOT`; `--dry-run` prints all five `sbatch` commands without submitting
them.

For this assessment rerun, retained matches allow later field-consensus and
post-processing tests without repeating GPU inference. They are stored inside
the authoritative SQLite database after coordinator import. Intermediate
pair-product files remain under the run's `work/` directory for worker-level
resume and do not need to be downloaded. Download
`global-trajectory-catalogue-v1.parquet` and
`assessment-summary-v1.json` first for routine analysis. Preserve or archive
the SQLite database when raw matches, pair fields, deformation, and full audit
state are required.

Primary pair fields are independent and resumable. Recovery begins only after
primary global composition identifies measured loss. Eligibility depends on
the 96-hour source-to-target interval, not the number of unrelated catalogue
images between them. Every candidate recovery pair within that interval is
scheduled only when it targets genuine measured loss. Recovery never creates
deformation cells.

The 5% threshold retains 99.35% of production ORB continuation events on the
shared April catalogue, while the 1,024 km2 threshold requires the area of 64
nominal 4 km field cells. These are fixed rerun settings rather than overlap
strata. Review candidate, primary, per-target, phase-comparison, and absolute
overlap counts in the dry plan before starting inference.

Convergence logic is available only as an optional audit. If a radius is later
selected in metres, use a separate trajectory-composition run or replay and
compare persistent events first; do not merge parcel identities during this
GPU baseline.
