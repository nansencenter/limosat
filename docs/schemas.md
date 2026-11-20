# Product schemas

## Pair product schema 3

Pair workers publish one compressed NPZ data file and one JSON completion marker
per primary image pair, under `<pair_store_directory>/<matching fingerprint>/primary/`
(or `<pair_product_directory>/primary/`), named by the SHA256 of the pair ID. The
self-checksummed marker is written last and is the completion boundary. It records
the matching version, fingerprint and identity (`pair_identity.py`), the SHA256 of
both scenes and their OSI SAF SIC files, the producing run, configuration and code
hashes (provenance only), the image-pair identity and times, the field checksum,
the data-file SHA256 and size, an array-content checksum, counts, diagnostics and
ancillary-input checksums. For compatibility with stored products every marker also
records `kind: "primary"`, `targeted: false` and `targeted_positions_sha256: null`.

The NPZ holds the displacement field (grid indices, float64 EPSG:3413 source
positions and displacements, availability, agreeing and candidate match counts,
support radius, maximum residual), the retained raw matches (source and target
positions, score, tile indices) and the fold-rejected indices. A worker never
replaces a marked product; schema 2 (run-bound) products are not read.

## Run database schema 6

A small SQLite file written by `limosat prepare` and read by pair workers.
`runs` stores the resolved configuration JSON and SHA256, a checksum of all LiMOSAT
Python source, the checkpoint SHA256 and the status; workers reject changed code,
model or configuration under an existing `run_id`. `images` stores image identity,
platform, absolute orbit, path, UTC time, size and SHA256. `candidate_pairs` stores
the deterministic plan (ordinal, candidate/primary selection, UTC times, elapsed
seconds, overlap fraction and area, intervening images) and `planning_counts` the
planning counts. Earlier databases are rejected; there is no migration.

## Primary composition (Parquet, version 1)

`compose-parquet` writes `trajectory-points-v1.parquet` (trajectory ID, image, UTC
time, state `created` or `observed`, position basis `seed_grid` or
`primary_pair_field`, x/y in EPSG:3413 metres, source pair, agreeing matches,
support radius, maximum residual), `trajectory-extensions-qc-v1.parquet` (one row
per measured source-to-target link with its QC attributes; `pair_kind` is always
`primary`), `trajectory-merges-v1.parquet` when merging is on, and
`composition-manifest-v1.json` last. The manifest records the configuration SHA256,
the matching fingerprint, the producer code hashes, a checksum of the ordered pair
product set, counts and file checksums. Parquet files carry the schema name, run
ID, configuration SHA256, CRS, units and time zone as metadata.
