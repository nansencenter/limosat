import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

import limosat
from limosat import PairProductStore
from limosat.config import (
    FieldConfig,
    MatcherConfig,
    OpenWaterConfig,
    RoutingConfig,
    RunConfig,
    SurfaceConfig,
    TrajectoryConfig,
)
from limosat.pair_identity import (
    EXCLUDED_SETTINGS,
    MATCHING_SETTINGS,
    MATCHING_VERSION,
    SceneChecksums,
    matching_sha256,
    write_scene_checksums,
)

from test_staged_execution import catalogue, config, result

# Digest of the modules that decide a pair product's content. When this test fails,
# decide whether the change alters matching output: if it does, bump MATCHING_VERSION
# in limosat/pair_identity.py (stored pairs are then rematched); either way, record the
# new digest below.
MATCHING_MODULES = (
    "efficientloftr.py", "field.py", "imagery.py", "models.py", "pairs.py",
    "routing.py", "surface.py", "tile_gates.py",
)
MATCHING_MODULES_SHA256 = "581573616280fc0c3b39407ce590f792a6c166f1741a5097477bd5c7af8ad4c4"


def test_every_configuration_setting_is_classified_for_pair_identity():
    sections = {
        "matcher": MatcherConfig, "field": FieldConfig, "routing": RoutingConfig,
        "open_water": OpenWaterConfig, "surface": SurfaceConfig,
    }
    for name, cls in sections.items():
        names = {item.name for item in fields(cls)}
        matching = set(MATCHING_SETTINGS[name])
        excluded = set(EXCLUDED_SETTINGS[name])
        assert not matching & excluded, name
        assert matching | excluded == names, f"unclassified {name} settings: {names - matching - excluded}"
    run_fields = {item.name for item in fields(RunConfig)} - set(sections) - {"analysis_epsg"}
    assert run_fields == set(EXCLUDED_SETTINGS["run"])
    assert {item.name for item in fields(TrajectoryConfig)}  # composition only, excluded whole


def test_matching_fingerprint_ignores_run_identity_paths_and_composition(tmp_path):
    base = config(tmp_path)
    same = replace(
        base,
        run_id="another-run",
        database=str(tmp_path / "other.sqlite"),
        output_directory=str(tmp_path / "other"),
        pair_product_directory=str(tmp_path / "other-pairs"),
        pair_store_directory=str(tmp_path / "store"),
        pair_workers=3,
        trajectories=TrajectoryConfig(new_point_exclusion_radius_m=3_000.0),
        routing=replace(base.routing, primary_maximum_pairs_per_target=4),
        field=replace(base.field, missing_node_fallback=not base.field.missing_node_fallback),
        matcher=replace(base.matcher, tile_batch_size=8, prefix_cuda_graph=False),
        surface=SurfaceConfig(point_sic_root="/somewhere"),
    )
    assert matching_sha256(same) == matching_sha256(base)
    for changed in (
        replace(base, matcher=replace(base.matcher, tile_size_px=384)),
        replace(base, field=replace(base.field, minimum_agreeing_matches=4)),
        replace(base, routing=replace(base.routing, coarse_matching=True)),
    ):
        assert matching_sha256(changed) != matching_sha256(base)


def test_runs_with_one_matching_fingerprint_share_primary_pairs(tmp_path):
    store = tmp_path / "store"
    first = replace(config(tmp_path), pair_store_directory=str(store))
    second = replace(
        first,
        run_id="second-run",
        database=str(tmp_path / "second.sqlite"),
        pair_product_directory=str(tmp_path / "second-pairs"),
        trajectories=TrajectoryConfig(new_point_exclusion_radius_m=3_000.0),
    )
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    saved = PairProductStore(first).save(pair, "primary", False, result(pair))
    assert saved.path.parent == store / matching_sha256(first) / "primary"

    loaded = PairProductStore(second).load(pair, "primary", False)
    assert loaded is not None and loaded.sha256 == saved.sha256
    metadata = json.loads(saved.path.with_suffix(".json").read_text())
    assert metadata["producer_run_id"] == first.run_id
    assert metadata["matching_identity"]["matching_version"] == MATCHING_VERSION

    # Other matching settings use another part of the store: nothing to reuse.
    other = replace(second, field=replace(second.field, agreement_distance_m=50.0))
    assert PairProductStore(other).load(pair, "primary", False) is None
    # Recovery products stay with their run.
    assert PairProductStore(second).kind_root("recovery") == second.pair_products / "recovery"


def test_shared_pairs_are_refused_when_a_scene_file_differs(tmp_path):
    store = tmp_path / "store"
    cfg = replace(config(tmp_path), pair_store_directory=str(store))
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    PairProductStore(cfg).save(pair, "primary", False, result(pair))
    with pair.target.path.open("ab") as stream:
        stream.write(b"another version of this scene")
    with pytest.raises(ValueError, match="scene_sha256"):
        PairProductStore(cfg).load(pair, "primary", False)


def test_scene_checksum_file_is_written_once_and_refuses_changed_scenes(tmp_path):
    images = catalogue(tmp_path).chronological()
    path = tmp_path / "control" / "scene-checksums.json"
    report = write_scene_checksums(images, path, workers=2)
    assert report == {"scene_checksums": str(path), "images": 3, "computed": 3}
    assert write_scene_checksums(images, path, workers=2)["computed"] == 0
    checksums = SceneChecksums(path)
    assert checksums.sha256(images[0]) == hashlib.sha256(images[0].path.read_bytes()).hexdigest()
    with images[0].path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed since its checksum"):
        checksums.sha256(images[0])


def test_matching_modules_digest_is_recorded():
    package = Path(limosat.__file__).resolve().parent
    digest = hashlib.sha256()
    for name in MATCHING_MODULES:
        digest.update(name.encode("utf-8"))
        digest.update((package / name).read_bytes())
    assert digest.hexdigest() == MATCHING_MODULES_SHA256, (
        "matching code changed: bump MATCHING_VERSION in limosat/pair_identity.py if pair "
        f"products change, then record {digest.hexdigest()} in this test"
    )


def test_compose_mixes_producer_versions_only_under_a_shared_fingerprint():
    from limosat.compose import _check_producers

    _check_producers({"a" * 64, "b" * 64}, {3})
    _check_producers({"a" * 64}, {2})
    with pytest.raises(ValueError, match="multiple producer implementations"):
        _check_producers({"a" * 64, "b" * 64}, {2, 3})
