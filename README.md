# LiMOSAT

LiMOSAT produces EfficientLoFTR sea-ice displacement fields, observed
Lagrangian trajectories, and deformation products from chronological satellite
image catalogues. The operational coordinate system is float64 EPSG:3413
metres. Time is explicit UTC; velocity limits use metres per day; deformation
rates use inverse seconds.

This branch intentionally has one matching method. It does not contain the old
ORB detector/descriptor/template pipeline or the frozen ALIKED research code.

## Install

```bash
mamba env create -f environment.yaml
mamba activate limosat
pip install -e .
```

The official EfficientLoFTR source checkout and optimized outdoor checkpoint
are external scientific inputs. Record their local paths in the run
configuration; the checkpoint SHA256 is written to the manifest.

## Run

Copy `config.defaults.yaml`, set the catalogue, output, model repository, and
checkpoint paths, then run:

```bash
limosat run config.yaml
limosat status config.yaml
```

The run command uses component labels only to plan compute, resumes completed
image pairs without overwriting them, composes one global parcel catalogue,
schedules non-consecutive recovery pairs only after measured trajectory loss,
writes deformation from primary pair fields, and emits
`run-manifest-v4.json`. Set `retain_pair_matches: true` for assessment runs to
keep the selected post-gate, pre-consensus EfficientLoFTR matches as one
compressed, checksummed SQLite record per completed image pair.

Candidate image pairs are registered before inference. By default they span
1--96 hours, overlap at least 5% of the smaller footprint, and have at least
1,024 km2 of direct overlap (the area of 64 nominal 4 km field cells). The most
recent eligible source is selected independently for every fixed 4 km
planning cell. The union of those cell choices defines the target's primary
image pairs without a per-target cap. Primary pair fields are independent and
`pair_workers` controls local concurrency.

`limosat run` is the portable single-machine interface. Internally it uses the
same explicit stages available to batch systems:

```bash
limosat prepare config.yaml
limosat pairs config.yaml --kind primary
limosat compose config.yaml --phase primary
limosat pairs config.yaml --kind recovery
limosat compose config.yaml --phase final
```

Pair workers write immutable NPZ data plus JSON completion markers under
`pair_product_directory`; they never write SQLite. Each CPU composition stage
validates and imports those products, then streams trajectory rows into the
global database. `--batch-index I --batch-count N` partitions pair work
deterministically for scheduler arrays. These files are resumable work
products, not additional deliverables; the finalized SQLite and Parquet files
remain the products to transfer.

Production catalogues should provide platform and absolute orbit metadata.
Pairs from the same Sentinel-1 platform and absolute orbit are excluded. The
matcher estimates coarse phase correlation for every independent production
pair. A response below 0.05 runs both phase-shifted and same-centre hypotheses;
the normal field and fold gates select the better truth-free result. Too little
common raster support falls back to same-centre rather than dropping the pair.
Optional OSI SAF filtering skips a tile only when complete SIC samples on both
dates are below 15%.

Inspect the frozen plan without loading the model with `limosat plan
config.yaml`. See [the April-week GPU rerun procedure](docs/gpu-april-week-rerun.md)
before inference.

After a native run is complete, package the compact trajectory product and
assessment checksums with:

```bash
limosat finalize config.yaml
```

This writes `global-trajectory-catalogue-v1.parquet` and
`assessment-summary-v1.json`. SQLite remains the authoritative resumable run
and contains fields, deformation, provenance, and optional retained matches.
Parquet finalization imports PyArrow only when requested; PyArrow is not a core
runtime dependency.

The public Python entry point is direct:

```python
from limosat import LiMOSATRun, load_config

summary = LiMOSATRun(load_config("config.yaml")).execute()
```

See [operations](docs/operations.md) for catalogue and recovery behavior,
[schemas](docs/schemas.md) for the SQLite and manifest contracts, and the
[production hardening and experiment plan](docs/implementation-plan.md) for
pending scientific and operational decision gates.

Completed production CSV fields can also be composed without imagery or GPU
inference. `scripts/replay_global_catalogue_fields.py` creates a new schema-v4
SQLite catalogue and checksummed field-replay provenance file;
`scripts/render_global_catalogue.py` creates the static distributions and
thin-trail pan-Arctic animation. Replay products are analysis outputs outside
Git, not substitutes for a native run manifest.

## Scientific semantics

- Pair fields are independently measured EfficientLoFTR products.
- Unsupported field nodes have `available = false` and `dx_m = dy_m = NULL`
  in SQLite; they are never encoded as zero displacement.
- Trajectories are virtual material points advected only through supported,
  orientation-preserving fields. Dormant points have no coordinate.
- Reappearance requires a measured non-consecutive recovery pair; no temporal
  prediction is included in the primary trajectory product.
- Sparse targeted recovery fields reconnect trajectories but are not emitted as
  standalone deformation products.

## License and citation

LiMOSAT is MIT licensed. Cite Chua and Korosov (2025), *limosat - Lagrangian
Ice Motion from Satellites*, <https://doi.org/10.5281/zenodo.15111936>.
