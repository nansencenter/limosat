import importlib.util
import json
from pathlib import Path

import pytest

from limosat.config import load_config


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_april_week_olivia.py"
SPEC = importlib.util.spec_from_file_location("prepare_april_week_olivia", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def _source_catalogue(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "S1A_in_window",
                        "geometry": None,
                        "properties": {
                            "datetime": "2020-04-01T01:02:03Z",
                            "filename": "S1A_in_window.tiff",
                        },
                    },
                    {
                        "type": "Feature",
                        "id": "S1A_after_window",
                        "geometry": None,
                        "properties": {
                            "datetime": "2020-04-08T00:00:00Z",
                            "filename": "S1A_after_window.tiff",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return source


def test_preparation_selects_window_and_rewrites_raster_path(tmp_path):
    source = _source_catalogue(tmp_path)
    scene_root = tmp_path / "scenes"
    raster = scene_root / "2020" / "04" / "S1A_in_window.tiff"
    raster.parent.mkdir(parents=True)
    raster.write_bytes(b"raster")
    destination = tmp_path / "run" / "control" / "catalogue.json"

    summary = PREPARE.prepare_catalogue(
        source,
        scene_root,
        destination,
        expected_source_sha256=PREPARE.file_sha256(source),
        expected_images=1,
    )

    result = json.loads(destination.read_text(encoding="utf-8"))
    assert summary["catalogue_images"] == 1
    assert [feature["id"] for feature in result["features"]] == ["S1A_in_window"]
    assert result["features"][0]["properties"]["filepath"] == str(raster)
    assert not list(destination.parent.glob("*.tmp.*"))


def test_preparation_rejects_missing_selected_raster(tmp_path):
    source = _source_catalogue(tmp_path)
    with pytest.raises(FileNotFoundError, match="raster paths are missing"):
        PREPARE.prepare_catalogue(
            source,
            tmp_path / "missing-scenes",
            tmp_path / "catalogue.json",
            expected_source_sha256=PREPARE.file_sha256(source),
            expected_images=1,
        )


def test_existing_sic_fixture_is_checksummed_without_network(tmp_path):
    for day in PREPARE.SIC_DATES:
        (tmp_path / PREPARE.sic_filename(day)).write_bytes(day.encode("ascii"))

    checksums = PREPARE.prepare_sic(tmp_path, download=False)

    assert len(checksums) == 9
    assert (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").count("\n") == 9


def test_generated_configuration_enables_reviewed_gpu_policy(tmp_path):
    config = PREPARE.run_config(
        "run",
        tmp_path / "catalogue.json",
        tmp_path / "run",
        tmp_path / "EfficientLoFTR",
        tmp_path / "checkpoint.ckpt",
        tmp_path / "sic",
    )

    assert config["analysis_epsg"] == 3413
    assert config["retain_pair_matches"] is True
    assert config["routing"]["candidate_minimum_overlap_fraction"] == 0.05
    assert config["routing"]["candidate_minimum_overlap_area_m2"] == 1_024_000_000.0
    assert config["routing"]["require_orbit_metadata"] is True
    assert config["routing"]["maximum_recovery_elapsed_hours"] == 96.0
    assert config["open_water"]["enabled"] is True

    config_path = tmp_path / "config.json"
    PREPARE.write_json_atomic(config_path, config)
    loaded = load_config(config_path)
    assert loaded.analysis_epsg == 3413
    assert loaded.routing.require_orbit_metadata is True
    assert loaded.open_water.enabled is True
