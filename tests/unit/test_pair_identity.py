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
    OPTIONAL_MATCHING_SETTINGS,
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
MATCHING_MODULES_SHA256 = "bae6c2eade756000ea1cb68ea51936829a1b764681d9ff39525595e6865cb284"


def test_every_configuration_setting_is_classified_for_pair_identity():
    sections = {
        "matcher": MatcherConfig, "field": FieldConfig, "routing": RoutingConfig,
        "open_water": OpenWaterConfig, "surface": SurfaceConfig,
    }
    for name, cls in sections.items():
        names = {item.name for item in fields(cls)}
        matching = set(MATCHING_SETTINGS[name])
        excluded = set(EXCLUDED_SETTINGS[name])
        matching |= set(OPTIONAL_MATCHING_SETTINGS.get(name, {}))
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
        routing=replace(base.routing, next_observation_backup_after_hours=30.0),
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
    saved = PairProductStore(first).save(pair, result(pair))
    assert saved.path.parent == store / matching_sha256(first) / "primary"

    loaded = PairProductStore(second).load(pair)
    assert loaded is not None and loaded.sha256 == saved.sha256
    metadata = json.loads(saved.path.with_suffix(".json").read_text())
    assert metadata["producer_run_id"] == first.run_id
    assert metadata["matching_identity"]["matching_version"] == MATCHING_VERSION

    # Other matching settings use another part of the store: nothing to reuse.
    other = replace(second, field=replace(second.field, agreement_distance_m=50.0))
    assert PairProductStore(other).load(pair) is None


def test_shared_pairs_are_refused_when_a_scene_file_differs(tmp_path):
    store = tmp_path / "store"
    cfg = replace(config(tmp_path), pair_store_directory=str(store))
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    PairProductStore(cfg).save(pair, result(pair))
    with pair.target.path.open("ab") as stream:
        stream.write(b"another version of this scene")
    with pytest.raises(ValueError, match="scene_sha256"):
        PairProductStore(cfg).load(pair)


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


def test_optional_matcher_settings_change_the_fingerprint_only_when_set(tmp_path):
    from dataclasses import replace
    base = config(tmp_path)
    same = replace(base, matcher=replace(base.matcher, tile_contrast="scene", coarse_confidence_threshold=None))
    assert matching_sha256(same) == matching_sha256(base)
    for matcher in (replace(base.matcher, tile_contrast="tile_percentile"),
                    replace(base.matcher, coarse_confidence_threshold=20.0)):
        assert matching_sha256(replace(base, matcher=matcher)) != matching_sha256(base)


def test_tile_percentile_stretch_keeps_invalid_pixels_and_spreads_valid_ones():
    import numpy as np
    from limosat.efficientloftr import tile_percentile_stretch
    tile = np.zeros((64, 64), np.uint8)
    tile[:, 16:] = np.tile(np.arange(100, 148, dtype=np.uint8), (64, 1))
    out = tile_percentile_stretch(tile)
    assert (out[:, :16] == 0).all() and (out[:, 16:] >= 1).all()
    assert out[:, 16:].max() == 255 and out[:, 16:].min() == 1


def test_strong_first_consensus_keeps_strong_nodes_and_fills_from_weak_matches():
    import numpy as np
    from shapely.geometry import box
    from limosat import FieldConfig, MotionMatches
    from limosat.field import estimate_queries, _strong_first_estimates
    rng = np.random.default_rng(0)
    # node A: 10 strong matches moving 1 km east plus 14 weak random ones nearer to it;
    # node B (20 km away): only weak matches, 10 of them agreeing on 2 km north.
    a_strong = np.c_[rng.uniform(-2000, 2000, 10), rng.uniform(-2000, 2000, 10)]
    a_weak = np.c_[rng.uniform(-500, 500, 14), rng.uniform(-500, 500, 14)]
    b_weak = np.c_[20000 + rng.uniform(-1500, 1500, 10), rng.uniform(-1500, 1500, 10)]
    source = np.vstack([a_strong, a_weak, b_weak])
    shift = np.vstack([np.tile([1000.0, 0.0], (10, 1)), rng.uniform(-8000, 8000, (14, 2)), np.tile([0.0, 2000.0], (10, 1))])
    score = np.r_[np.full(10, 0.8), np.full(14, 0.2), np.full(10, 0.2)]
    tiles = np.zeros(len(source), dtype=np.int32)
    matches = MotionMatches(source, source + shift, score, tiles, tiles)
    query = np.array([[0.0, 0.0], [20000.0, 0.0]])
    plain = estimate_queries(matches, query, FieldConfig())
    first = _strong_first_estimates(matches, query, FieldConfig(strong_match_score=0.5))
    assert not plain["available"][0]          # weak matches crowd out the strong ones
    assert first["available"].all()
    assert np.allclose(first["displacement_m"][0], [1000.0, 0.0])
    assert np.allclose(first["displacement_m"][1], [0.0, 2000.0])


def test_fallback_nodes_can_require_stricter_agreement():
    import numpy as np
    from limosat import FieldConfig, MotionMatches
    from limosat.field import _strong_first_estimates
    rng = np.random.default_rng(1)
    # 9 weak matches agree within 1 km but not within 500 m: the default fallback accepts
    # the node, a fallback needing 8 within 500 m does not.
    source = np.c_[rng.uniform(-1000, 1000, 9), rng.uniform(-1000, 1000, 9)]
    shift = np.c_[np.linspace(0, 1800, 9), np.zeros(9)]
    tiles = np.zeros(9, dtype=np.int32)
    matches = MotionMatches(source, source + shift, np.full(9, 0.2), tiles, tiles)
    query = np.array([[0.0, 0.0]])
    loose = _strong_first_estimates(matches, query, FieldConfig(strong_match_score=0.5, fallback_neighbour_count=24))
    strict = _strong_first_estimates(matches, query, FieldConfig(strong_match_score=0.5, fallback_neighbour_count=24,
                                                                 fallback_agreement_distance_m=500.0))
    assert loose["available"][0] and not strict["available"][0]
