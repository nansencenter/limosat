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
`run-manifest-v2.json`.

Candidate image pairs are registered before inference. By default they span
1--96 hours and at least 25% footprint overlap. The most recent source
acquisition for each target defines the primary pair; equal-time alternatives
remain primary so field quality can resolve them. Primary pair fields are
independent and `pair_workers` controls local concurrency.

The public Python entry point is direct:

```python
from limosat import LiMOSATRun, load_config

summary = LiMOSATRun(load_config("config.yaml")).execute()
```

See [operations](docs/operations.md) for catalogue and recovery behavior and
[schemas](docs/schemas.md) for the SQLite and manifest contracts.

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
