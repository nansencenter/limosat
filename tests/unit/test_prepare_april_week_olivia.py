import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from limosat.config import load_config


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_april_week_olivia.py"
SUBMIT_SCRIPT = Path(__file__).parents[2] / "scripts" / "submit_april_week_olivia.sh"
JOB_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_limosat_olivia.sbatch"
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
    assert config["pair_product_directory"] == str(
        tmp_path / "run" / "work" / "pair-products"
    )
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


def test_generated_diagnostic_configuration_bounds_primary_and_disables_recovery(
    tmp_path,
):
    config = PREPARE.run_config(
        "diagnostic",
        tmp_path / "catalogue.json",
        tmp_path / "run",
        tmp_path / "EfficientLoFTR",
        tmp_path / "checkpoint.ckpt",
        tmp_path / "sic",
        primary_maximum_pairs_per_target=2,
        targeted_recovery=False,
    )
    config_path = tmp_path / "config.json"
    PREPARE.write_json_atomic(config_path, config)

    loaded = load_config(config_path)

    assert loaded.routing.primary_maximum_pairs_per_target == 2
    assert loaded.routing.targeted_recovery is False


def test_olivia_submission_uses_gpu_workers_and_cpu_composition_barriers():
    submit = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    job = JOB_SCRIPT.read_text(encoding="utf-8")

    ordered_stages = (
        "prepare cpu",
        "primary-pairs gpu",
        "primary-compose cpu",
        "recovery-pairs gpu",
        "final-compose cpu",
    )
    assert [submit.index(stage) for stage in ordered_stages] == sorted(
        submit.index(stage) for stage in ordered_stages
    )
    assert "--dependency=\"afterok:$dependency\"" in submit
    assert "--gpus-per-node=0" in submit
    assert '--array="0-$((gpu_workers - 1))"' in submit
    assert "LIMOSAT_CONFIG_SHA256" in submit
    assert "LIMOSAT_CATALOGUE_SHA256" in submit
    assert 'check_sha256 "$LIMOSAT_CATALOGUE"' in job
    assert 'for day in days:' in job
    assert 'fields.append(load_sic_field(path))' in job
    assert 'import pyarrow' in job
    assert 'replace(config.matcher, device="cpu")' in job
    assert "python -m limosat pairs" in job
    assert "python -m limosat compose" in job
    assert "python -m limosat ingest" not in job
    assert "python -m limosat run" not in job
    assert 'if [[ "$recovery_enabled" == "1" ]]' in submit


def test_olivia_submission_omits_disabled_recovery_stage(tmp_path):
    method_root = tmp_path / "method"
    official_repository = tmp_path / "official"
    for repository in (method_root, official_repository):
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        marker = repository / "marker"
        marker.write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "marker"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=LiMOSAT test",
                "-c",
                "user.email=limosat-test@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            check=True,
        )

    run_root = tmp_path / "run"
    control = run_root / "control"
    control.mkdir(parents=True)
    config = control / "config.json"
    config.write_text(
        json.dumps({"routing": {"targeted_recovery": False}}), encoding="utf-8"
    )
    catalogue = control / "catalogue.json"
    catalogue.write_text("{}\n", encoding="utf-8")
    scene_root = tmp_path / "scenes"
    sic_root = tmp_path / "sic"
    scene_root.mkdir()
    sic_root.mkdir()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    apptainer = fake_bin / "apptainer"
    apptainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    apptainer.chmod(0o755)

    resources = {
        "CHECKPOINT": tmp_path / "checkpoint.ckpt",
        "CONTAINER": tmp_path / "container.sif",
        "OVERLAY": tmp_path / "overlay.img",
        "READY_MARKER": tmp_path / "overlay.ready",
        "CPU_AUDIT": tmp_path / "cpu-audit.json",
    }
    for path in resources.values():
        path.write_bytes(b"fixture\n")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
            "METHOD_ROOT": str(method_root),
            "ELOFTR_REPO": str(official_repository),
            "RUN_ID": "diagnostic",
            "RUN_ROOT": str(run_root),
            "CONFIG": str(config),
            "CATALOGUE": str(catalogue),
            "SCENE_ROOT": str(scene_root),
            "SIC_ROOT": str(sic_root),
            **{name: str(path) for name, path in resources.items()},
        }
    )
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "eloftr-primary-pairs" in result.stdout
    assert "eloftr-final-compose" in result.stdout
    assert "eloftr-recovery-pairs" not in result.stdout
    assert sum(line.startswith("sbatch ") for line in result.stdout.splitlines()) == 4
