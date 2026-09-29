import shutil
import os
from pathlib import Path

import pandas as pd

from scripts.generate_data import normal_day, write_days
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
