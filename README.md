# LiMOSAT

LiMOSAT produces EfficientLoFTR sea-ice displacement fields and observed
Lagrangian trajectories from chronological satellite image catalogues. The
coordinate system is float64 EPSG:3413 metres; time is explicit UTC; velocity
limits use metres per day.

This branch has one method (below). It does not contain the ORB pipeline, the
ALIKED research code, continuation (recovery) passes or the per-target pair
selection used before November 2026; see the git history for those.

## Install

```bash
mamba env create -f environment.yaml
mamba activate limosat
pip install -e .
```

The official EfficientLoFTR source checkout and optimized outdoor checkpoint
are external scientific inputs. Record their local paths in the run
configuration; their checksums enter the matching fingerprint.

## Method

1. **Planning** (`planning.py`). Candidate pairs span 1-96 h, overlap at least 5 %
   of the smaller footprint and 1,024 km2, and are not from one acquisition pass.
   Primary pairs are chosen by *next observation*: each 4 km cell of a source
   image goes to the next later image that images it; a pair is kept when it is
   the next observation for at least 25 cells, and cells whose next observation is
   more than 30 h away also go to the following one as a backup. Land and AMSR2
   open-water cells are dropped first (`plan_surface_cells`).
2. **Matching** (`pairs.py`). Every tile of each primary pair's overlap is matched
   with EfficientLoFTR, routed by phase correlation and a coarse 320 m pass, and
   turned into a 4 km displacement field. Each result is stored as an immutable
   pair product keyed by the matching fingerprint (`pair_identity.py`), so runs
   with the same matching settings share them.
3. **Composition** (`compose.py`, `trajectory.py`, `field.py`). Trajectories are
   chained through the primary pair fields in UTC order, with gap filling and
   shear point estimates; duplicate trajectories can optionally be merged.
   Output is Parquet.

## Run

Copy `config.defaults.yaml`, set the catalogue, output, model repository and
checkpoint paths, then run everything on one machine:

```bash
limosat run config.yaml      # = prepare, pairs, compose-parquet
limosat status config.yaml
```

or the same stages separately, for example on a scheduler:

```bash
limosat plan config.yaml                                     # counts only, no model
limosat prepare config.yaml                                  # register catalogue and plan
limosat pairs config.yaml --batch-index I --batch-count N    # one deterministic batch
limosat compose-parquet config.yaml OUTPUT_DIRECTORY
```

Pair workers write immutable NPZ data plus JSON completion markers and resume
completed pairs without recomputing them. CUDA runs use one process per batch and
`pair_workers: 1`. Production on Olivia uses `scripts/prepare_chain_olivia.py`
(which writes the production configuration) and `scripts/submit_chain_olivia.sh`
(prepare -> primary-pairs GPU array -> primary-compose).

A composition is published as one CF NetCDF trajectory file with
`python -m limosat.publish build --source COMPOSITION_DIRECTORY --output-dir OUT --release-id ID`
(needs the `publish` extra); see [published trajectories](docs/published_trajectories.md).

See [operations](docs/operations.md) for planning, matching, pair sharing and
running, [composition](docs/composition.md) for trajectories and QC,
[schemas](docs/schemas.md) for the stored products and
[published trajectories](docs/published_trajectories.md) for the public NetCDF release.

## Scientific semantics

- Pair fields are independently measured EfficientLoFTR products.
- Unsupported field nodes have `available = false` and NaN displacement; they are
  never encoded as zero displacement.
- Trajectories are virtual material points advected only through supported,
  orientation-preserving fields; no temporal prediction is included. A gap is
  simply an image without a row.
- Sampling preserves successful triangle lookups. At an exact field vertex or
  shared triangle edge, a rejected lookup can use an existing incident triangle
  that passes the same edge-length and orientation gates.

## License and citation

LiMOSAT is MIT licensed. Cite Chua and Korosov (2025), *limosat - Lagrangian
Ice Motion from Satellites*, <https://doi.org/10.5281/zenodo.15111936>.
