#!/usr/bin/env python3
"""Copy one run's schema 2 primary pair products into a shared store as schema 3.

Schema 2 products are tied to the run that made them. This republishes them under
the matching fingerprint of a destination run (one with ``pair_store_directory``
set), so that run and any later run with the same fingerprint reuse them. The
source run is only read.

Every product is first verified as its own run loads it (marker and data checksums,
run ID, configuration hash), then saved through the destination store, which records
the destination's scene and SIC checksums and keeps the source run, configuration
and code hashes as provenance. Both runs must have the same matching fingerprint,
and ``--matching-version`` must name the current ``MATCHING_VERSION``: it states
that the source run's code matched pairs the way the current version does (check
the source commit against the version history before passing it).

Usage: migrate_pair_products.py SOURCE_CONFIG DESTINATION_CONFIG --matching-version N
       [--workers K] [--dry-run]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limosat.catalog import ImagePair, load_catalogue  # noqa: E402
from limosat.config import load_config  # noqa: E402
from limosat.pair_identity import (  # noqa: E402
    MATCHING_VERSION,
    identity_sha256,
    matching_identity,
    write_scene_checksums,
)
from limosat.pair_products import LEGACY_PAIR_PRODUCT_SCHEMA_VERSION, PairProductStore  # noqa: E402
from limosat.planning import run_plan  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("source_config", type=Path)
    result.add_argument("destination_config", type=Path)
    result.add_argument("--matching-version", type=int, required=True)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--dry-run", action="store_true", help="verify and count only")
    return result


_STATE = {}


def _init(source_config, destination_config, dry_run):
    source = load_config(source_config)
    destination = load_config(destination_config)
    _STATE.update(
        source=source,
        destination=destination,
        source_store=PairProductStore(source),
        destination_store=None if dry_run else PairProductStore(destination),
        images={image.image_id: image for image in load_catalogue(destination.catalogue, destination.analysis_epsg).records},
    )


def _migrate(item):
    pair_id, source_pair = item
    source, store = _STATE["source"], _STATE["source_store"]
    product = store.load(source_pair, "primary", False, require_current_implementation=False)
    if product is None:
        return pair_id, "missing"
    marker = json.loads(store._paths(pair_id, "primary")[1].read_text(encoding="utf-8"))
    if marker.get("pair_product_schema_version") != LEGACY_PAIR_PRODUCT_SCHEMA_VERSION:
        return pair_id, "not_legacy"
    destination = _STATE["destination_store"]
    if destination is None:
        return pair_id, "verified"
    images = _STATE["images"]
    if source_pair.source.image_id not in images or source_pair.target.image_id not in images:
        return pair_id, "not_in_destination_catalogue"
    # Use the destination catalogue's records (paths and checksums of the same scenes).
    pair = ImagePair(images[source_pair.source.image_id], images[source_pair.target.image_id])
    if destination.has_product(pair_id, "primary"):
        return pair_id, "already_in_store"
    saved = destination.save(
        pair,
        "primary",
        False,
        product.result,
        provenance={
            "producer_run_id": marker["run_id"],
            "producer_config_sha256": marker["config_sha256"],
            "implementation_sha256": marker["implementation_sha256"],
            "migrated_from": {
                "schema_version": LEGACY_PAIR_PRODUCT_SCHEMA_VERSION,
                "marker_content_sha256": marker["marker_content_sha256"],
                "data_sha256": marker["data_sha256"],
                "content_sha256": marker["content_sha256"],
            },
        },
    )
    if saved.content_sha256 != marker["content_sha256"]:
        return pair_id, "content_changed"
    return pair_id, "migrated"


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.matching_version != MATCHING_VERSION:
        raise SystemExit(f"--matching-version {arguments.matching_version} is not the current {MATCHING_VERSION}")
    source = load_config(arguments.source_config)
    destination = load_config(arguments.destination_config)
    if not destination.pair_store_directory:
        raise SystemExit("destination configuration has no pair_store_directory")
    source_identity, destination_identity = matching_identity(source), matching_identity(destination)
    if source_identity != destination_identity:
        raise SystemExit(
            "matching fingerprints differ:\n"
            + json.dumps({"source": source_identity, "destination": destination_identity}, indent=1, sort_keys=True)
        )
    if destination.scene_checksums and not arguments.dry_run:
        # Checksum the destination's scenes once, in parallel, before the workers need
        # them (the destination's prepare stage then finds them done).
        print(json.dumps(write_scene_checksums(
            load_catalogue(destination.catalogue, destination.analysis_epsg).records,
            destination.scene_checksums,
            arguments.workers,
        )), file=sys.stderr, flush=True)
    pairs = [
        (item.pair.pair_id, item.pair)
        for item in run_plan(source, load_catalogue(source.catalogue, source.analysis_epsg)).pairs
        if item.selection == "primary"
    ]
    counts = {}
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        initializer=_init,
        initargs=(str(arguments.source_config), str(arguments.destination_config), arguments.dry_run),
    ) as pool:
        for number, (_pair_id, outcome) in enumerate(pool.map(_migrate, pairs, chunksize=16), 1):
            counts[outcome] = counts.get(outcome, 0) + 1
            if number % 1000 == 0:
                print(json.dumps({"done": number, "of": len(pairs), **counts}), file=sys.stderr, flush=True)
    report = {
        "source_run_id": source.run_id,
        "destination_run_id": destination.run_id,
        "matching_sha256": identity_sha256(destination_identity),
        "store": str(Path(destination.pair_store_directory) / identity_sha256(destination_identity) / "primary"),
        "primary_pairs": len(pairs),
        "outcomes": counts,
        "dry_run": arguments.dry_run,
    }
    print(json.dumps(report, indent=2))
    failures = {key: value for key, value in counts.items() if key not in {"migrated", "verified", "already_in_store"}}
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
