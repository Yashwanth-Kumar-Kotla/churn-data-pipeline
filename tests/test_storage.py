import json
from pathlib import Path

import pandas as pd

from src.storage import (
    hash_is_recorded,
    load_manifest,
    quarantine_raw_file,
    record_manifest,
    sha256_file,
    write_parquet,
    write_report,
)


def test_parquet_write_and_manifest_are_readable(tmp_path: Path) -> None:
    frame = pd.DataFrame({"user_id": [1, 2], "churned": [0, 1]})
    parquet_path = tmp_path / "processed" / "user_activity_2026-09-24.parquet"
    manifest_path = tmp_path / "state" / "manifest.json"

    write_parquet(frame, parquet_path)
    record_manifest(
        manifest_path,
        {"filename": "user_activity_2026-09-24.csv", "sha256": "abc", "status": "passed"},
    )

    assert pd.read_parquet(parquet_path).equals(frame)
    assert hash_is_recorded(load_manifest(manifest_path), "abc")


def test_quarantine_preserves_raw_bytes_and_report(tmp_path: Path) -> None:
    raw_path = tmp_path / "user_activity_2026-09-27.csv"
    raw_path.write_bytes(b"user_id\n1\n")
    original_hash = sha256_file(raw_path)
    report_path = tmp_path / "reports" / "user_activity_2026-09-27.json"

    write_report(report_path, {"passed": False, "reason": "missing column"})
    quarantined = quarantine_raw_file(raw_path, tmp_path / "quarantine", original_hash)

    assert not raw_path.exists()
    assert sha256_file(quarantined) == original_hash
    assert json.loads(report_path.read_text())["passed"] is False


def test_identical_hash_is_skipped_by_manifest(tmp_path: Path) -> None:
    raw_path = tmp_path / "user_activity_2026-09-24.csv"
    raw_path.write_text("user_id\n1\n")
    manifest_path = tmp_path / "manifest.json"
    file_hash = sha256_file(raw_path)
    record_manifest(manifest_path, {"filename": raw_path.name, "sha256": file_hash, "status": "passed"})

    assert hash_is_recorded(load_manifest(manifest_path), file_hash)

