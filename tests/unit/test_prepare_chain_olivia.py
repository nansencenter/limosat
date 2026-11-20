import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from limosat.config import load_config

SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_chain_olivia.py"
SPEC = importlib.util.spec_from_file_location("prepare_chain_olivia", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)

START = datetime(2020, 4, 1, tzinfo=timezone.utc)
STOP = datetime(2020, 4, 8, tzinfo=timezone.utc)


def _source_catalogue(tmp_path):
    source = tmp_path / "source.json"
    features = [
        {"type": "Feature", "id": name, "geometry": None,
         "properties": {"datetime": time, "filename": name + ".tiff"}}
        for name, time in (("S1A_in_window", "2020-04-01T01:02:03Z"),
                           ("S1A_after_window", "2020-04-08T00:00:00Z"))
    ]
    source.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return source


def test_preparation_selects_window_and_rewrites_raster_path(tmp_path):
    source = _source_catalogue(tmp_path)
    raster = tmp_path / "scenes" / "2020" / "04" / "S1A_in_window.tiff"
    raster.parent.mkdir(parents=True)
    raster.write_bytes(b"raster")
    destination = tmp_path / "run" / "control" / "catalogue.json"
    summary = PREPARE.prepare_catalogue(
        source, tmp_path / "scenes", destination, START, STOP,
        expected_source_sha256=PREPARE.file_sha256(source), expected_images=1,
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
            source, tmp_path / "missing-scenes", tmp_path / "catalogue.json", START, STOP,
            expected_source_sha256=PREPARE.file_sha256(source),
        )


def test_existing_sic_files_are_checksummed_without_network(tmp_path):
    days = tuple(PREPARE.sic_days(START, STOP))
    assert days[0] == "20200331" and days[-1] == "20200408"
    for day in days:
        (tmp_path / PREPARE.sic_filename(day)).write_bytes(day.encode("ascii"))
    checksums = PREPARE.prepare_sic(tmp_path, days)
    assert len(checksums) == 9
    assert (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").count("\n") == 9


def production_config_file(tmp_path, *extra):
    arguments = PREPARE.parser().parse_args([
        "--run-id", "run", "--run-root", str(tmp_path / "run"),
        "--start", "2020-01-01", "--stop", "2020-05-01",
        "--sic-root", "/osisaf", "--land-raster", "/land_osm_3413_1km.npz", "--asi-root", "/asi",
        *extra,
    ])
    path = tmp_path / "config.json"
    PREPARE.write_json_atomic(path, PREPARE.run_config(arguments, tmp_path / "catalogue.json", tmp_path / "run"))
    return path


def test_prepared_configuration_is_the_chosen_method(tmp_path):
    config = load_config(production_config_file(tmp_path))
    assert config.routing.plan_surface_cells is True
    assert config.routing.next_observation_backup_after_hours == 30.0
    assert config.routing.next_observation_minimum_cells == 25
    assert config.routing.require_orbit_metadata is True
    assert config.field.gap_filling and config.field.shear_point_estimates
    assert config.trajectories.new_point_exclusion_radius_m == 3_000.0
    assert config.trajectories.merge_radius_m is None
    assert config.open_water.enabled and config.surface.land_raster
    assert config.pair_store_directory
    merged = load_config(production_config_file(
        tmp_path, "--merge-radius-m", "250", "--merge-maximum-younger-age-hours", "48"))
    assert merged.trajectories.merge_radius_m == 250.0
    assert merged.trajectories.merge_maximum_younger_age_hours == 48.0
