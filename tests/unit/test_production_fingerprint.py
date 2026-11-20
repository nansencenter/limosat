"""The production matching fingerprint must not change: the shared pair store
(about 52,000 Jan-Apr 2020 primary pairs) is keyed by it."""

from limosat.config import (
    MatcherConfig,
    OpenWaterConfig,
    RoutingConfig,
    RunConfig,
    SurfaceConfig,
)
from limosat.pair_identity import MATCHING_VERSION, identity_sha256, matching_settings

PRODUCTION_MATCHING_SHA256 = "1cecfae0dd5b2763a06d8dda44004738c09a46bd73fbc7f0f718a26425f8c3b5"
# Content checksums of the production run-wide inputs (official EfficientLoFTR source
# ffd4a464, eloftr_outdoor.ckpt, OSM land raster land_osm_3413_1km.npz).
PRODUCTION_INPUTS = {
    "model_checkpoint_sha256": "0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878",
    "model_source_sha256": "7a0c7e5c518fe4102205de5d1555fd3050917b26d139704b9425e9503d2910f9",
    "land_raster_sha256": "0dbee8f27710b5ef659fff32400aeb5288e72ae721d81ceb2bd418f68b89c852",
}


def production_config() -> RunConfig:
    """Matching-relevant values as written by scripts/prepare_chain_olivia.py."""
    return RunConfig(
        "production", "catalogue.json", "limosat.sqlite", "outputs",
        matcher=MatcherConfig(
            device="cuda", maximum_speed_m_per_day=43_200.0, scene_mask_invalid_from=3,
        ),
        routing=RoutingConfig(require_orbit_metadata=True),
        open_water=OpenWaterConfig(enabled=True, sic_root="/osisaf"),
        surface=SurfaceConfig(
            land_raster="/land_osm_3413_1km.npz", land_minimum_inland_km=1.0,
            point_sic_root="/asi",
        ),
    )


def test_production_matching_fingerprint_is_unchanged():
    identity = {
        "matching_version": MATCHING_VERSION,
        "settings": matching_settings(production_config()),
        "inputs": PRODUCTION_INPUTS,
    }
    assert identity_sha256(identity) == PRODUCTION_MATCHING_SHA256


def test_prepare_chain_olivia_writes_the_production_matching_settings(tmp_path):
    from limosat.config import load_config
    from test_prepare_chain_olivia import production_config_file

    prepared = load_config(production_config_file(tmp_path))
    assert matching_settings(prepared) == matching_settings(production_config())
