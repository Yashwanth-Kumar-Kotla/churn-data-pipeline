"""Find and read dated raw CSV files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(r"^user_activity_(\d{4}-\d{2}-\d{2})(?:_.+)?\.csv$")


@dataclass(frozen=True)
class RawFile:
    path: Path
    file_date: date


def file_date_from_path(path: Path) -> date:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"invalid raw filename: {path.name}")
    return date.fromisoformat(match.group(1))


def read_raw_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    return frame
