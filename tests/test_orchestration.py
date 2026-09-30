import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_data import bad_day, normal_day, write_days
from src.pipeline import run_pipeline
from src.storage import load_manifest


def setup_project(tmp_path: Path) -> Path:
    shutil.copy("config.yaml", tmp_path / "config.yaml")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    return tmp_path


def test_six_day_run_quarantines_bad_files_and_is_idempotent(tmp_path: Path) -> None:
    project = setup_project(tmp_path)
    write_days(project / "data" / "raw")

    first_run = run_pipeline(project)
    second_run = run_pipeline(project)

    assert [outcome.status for outcome in first_run].count("processed") == 4
    assert [outcome.status for outcome in first_run].count("quarantined") == 2
    assert second_run == []
    assert len(list((project / "data" / "processed").glob("*.parquet"))) == 4
    assert len(list((project / "data" / "quarantine").glob("*.csv"))) == 2
    assert len(load_manifest(project / "state" / "manifest.json")["files"]) == 6


def test_corrected_same_name_with_new_hash_is_reprocessed(tmp_path: Path) -> None:
    project = setup_project(tmp_path)
    raw_path = project / "data" / "raw" / "user_activity_2026-09-24.csv"
    normal_day("2026-09-24").to_csv(raw_path, index=False)

    first_run = run_pipeline(project)
    corrected = normal_day("2026-09-24")
    corrected.loc[0, "sessions_last_30d"] = 30
    corrected.to_csv(raw_path, index=False)
    second_run = run_pipeline(project)

    manifest = load_manifest(project / "state" / "manifest.json")
    assert first_run[0].status == "processed"
    assert second_run[0].status == "processed"
    assert len(manifest["files"]) == 2
    assert len({entry["sha256"] for entry in manifest["files"]}) == 2


def test_newest_same_date_candidate_is_the_only_one_processed(tmp_path: Path) -> None:
    project = setup_project(tmp_path)
    raw_dir = project / "data" / "raw"
    older = raw_dir / "user_activity_2026-09-24_old.csv"
    newer = raw_dir / "user_activity_2026-09-24_corrected.csv"
    normal_day("2026-09-24").to_csv(older, index=False)
    corrected = normal_day("2026-09-24")
    corrected.loc[0, "sessions_last_30d"] = 99
    corrected.to_csv(newer, index=False)
    os.utime(older, (1, 1))

    outcomes = run_pipeline(project)
    manifest = load_manifest(project / "state" / "manifest.json")

    assert len(outcomes) == 1
    assert outcomes[0].path == newer
    assert {entry["status"] for entry in manifest["files"]} == {"passed", "superseded"}
    assert 99 in pd.read_parquet(project / "data" / "processed" / "user_activity_2026-09-24.parquet")["sessions_last_30d"].values


def test_quarantined_file_never_reaches_training(tmp_path: Path, monkeypatch) -> None:
    project = setup_project(tmp_path)
    bad_day().to_csv(project / "data" / "raw" / "user_activity_2026-09-27.csv", index=False)
    calls = []
    monkeypatch.setattr("src.pipeline.train_and_evaluate", lambda *args: calls.append(args))

    outcomes = run_pipeline(project)

    assert [outcome.status for outcome in outcomes] == ["quarantined"]
    assert calls == []


def test_small_missing_key_rates_are_repaired_before_training(tmp_path: Path) -> None:
    project = setup_project(tmp_path)
    for file_date in ("2026-09-24", "2026-09-25"):
        frame = normal_day(file_date)
        frame.loc[:29, "user_id"] = None
        frame.loc[30:59, "plan_type"] = None
        frame.to_csv(project / "data" / "raw" / f"user_activity_{file_date}.csv", index=False)

    outcomes = run_pipeline(project)
    manifest = load_manifest(project / "state" / "manifest.json")

    assert [outcome.status for outcome in outcomes] == ["processed", "processed"]
    assert all((project / "models" / f"metrics_{day}.json").exists() for day in ("2026-09-24", "2026-09-25"))
    assert all(entry["cleaning_actions"]["rows_missing_user_or_plan_dropped"] == 60 for entry in manifest["files"])
    for path in (project / "data" / "processed").glob("*.parquet"):
        cleaned = pd.read_parquet(path)
        assert cleaned[["user_id", "plan_type"]].notna().all().all()


def test_training_failure_does_not_mark_file_complete(tmp_path: Path, monkeypatch) -> None:
    project = setup_project(tmp_path)
    raw_path = project / "data" / "raw" / "user_activity_2026-09-24.csv"
    normal_day("2026-09-24").to_csv(raw_path, index=False)

    def fail_training(*args):
        raise RuntimeError("training failed")

    monkeypatch.setattr("src.pipeline.train_and_evaluate", fail_training)
    with pytest.raises(RuntimeError, match="training failed"):
        run_pipeline(project)

    assert raw_path.exists()
    assert load_manifest(project / "state" / "manifest.json")["files"] == []
