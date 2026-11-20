import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from limosat import RunStages
from limosat.cli import main
from limosat.store import DATABASE_SCHEMA_VERSION, RunStore

from test_staged_execution import Processor, catalogue, config


def _config_file(tmp_path, cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    return path


def test_cli_status_reports_the_registered_plan_without_loading_the_model(tmp_path, capsys):
    cfg = config(tmp_path)
    RunStages(cfg, catalogue(tmp_path), Processor()).prepare()

    assert main(["status", str(_config_file(tmp_path, cfg))]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["run"]["schema_version"] == DATABASE_SCHEMA_VERSION
    assert output["run"]["status"] == "running"
    assert output["candidate_pairs"] == {"candidate": 1, "primary": 2}
    assert output["pair_products"] == 0


def test_cli_status_does_not_create_missing_database(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(FileNotFoundError, match="run prepare first"):
        main(["status", str(_config_file(tmp_path, cfg))])
    assert not Path(cfg.database).exists()


def test_cli_plan_counts_primary_pairs(tmp_path, capsys):
    cfg = config(tmp_path)
    catalogue(tmp_path)
    assert main(["plan", str(_config_file(tmp_path, cfg))]) == 0
    output = json.loads(capsys.readouterr().out)
    assert (output["candidate_pairs"], output["primary_pairs"]) == (3, 2)
    assert output["planning_counts"]["next_observation_pairs"] == 2


def test_registered_plan_is_immutable(tmp_path):
    cfg = config(tmp_path)
    store = RunStore(cfg)
    store.register_planning_counts({"accepted_candidate_pairs": 3})
    with pytest.raises(ValueError, match="planning counts changed"):
        store.register_planning_counts({"accepted_candidate_pairs": 4})


def test_legacy_database_is_rejected_without_migration(tmp_path):
    legacy = tmp_path / "legacy.sqlite"
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY)")

    with pytest.raises(ValueError, match=f"new schema-v{DATABASE_SCHEMA_VERSION} database"):
        RunStore(replace(config(tmp_path), database=str(legacy)))
