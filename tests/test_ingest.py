from pathlib import Path

import pytest

from src.ingest import discover_raw_files, file_date_from_path, read_raw_csv


def test_discovery_sorts_files_by_date(tmp_path: Path) -> None:
    for name in ("user_activity_2026-09-29.csv", "user_activity_2026-09-24.csv"):
        (tmp_path / name).write_text("user_id\n1\n")

    files = discover_raw_files(tmp_path)

    assert [item.file_date.isoformat() for item in files] == ["2026-09-24", "2026-09-29"]


def test_invalid_filename_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid raw filename"):
        file_date_from_path(tmp_path / "activity.csv")


def test_reader_removes_bom_and_header_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "user_activity_2026-09-24.csv"
    path.write_text("\ufeff user_id , date \n1,2026-09-24\n")

    frame = read_raw_csv(path)

    assert frame.columns.tolist() == ["user_id", "date"]

