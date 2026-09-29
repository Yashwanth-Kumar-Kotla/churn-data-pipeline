"""Prepare and repair daily churn data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


NUMERIC_COLUMNS = (
    "user_id",
    "sessions_last_30d",
    "avg_session_minutes",
    "days_since_last_login",
    "support_tickets",
    "churned",
)
IMPUTED_COLUMNS = (
    "sessions_last_30d",
    "avg_session_minutes",
    "days_since_last_login",
    "support_tickets",
)


@dataclass
class CleanResult:
    frame: pd.DataFrame
    actions: dict[str, int]


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared.columns = prepared.columns.str.strip()
    for column in NUMERIC_COLUMNS:
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if "plan_type" in prepared:
        prepared["plan_type"] = prepared["plan_type"].astype("string").str.strip().str.lower()
    if "date" in prepared:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date
    return prepared


def clean_frame(frame: pd.DataFrame, config: dict[str, Any]) -> CleanResult:
    cleaned = prepare_frame(frame)
    actions: dict[str, int] = {}

    key_columns = [column for column in ("user_id", "plan_type") if column in cleaned]
    missing_key = cleaned[key_columns].isna().any(axis=1)
    actions["rows_missing_user_or_plan_dropped"] = int(missing_key.sum())
    cleaned = cleaned.loc[~missing_key].copy()

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    actions["exact_duplicates_dropped"] = duplicate_rows

    ranges = config["value_ranges"]
    for column, (minimum, maximum) in ranges.items():
        if column not in cleaned:
            continue
        invalid = cleaned[column].notna() & ~cleaned[column].between(minimum, maximum)
        actions[f"{column}_set_to_null"] = int(invalid.sum())
        cleaned.loc[invalid, column] = pd.NA

    null_limit = config["quality_thresholds"]["maximum_null_rate"]
    for column in IMPUTED_COLUMNS:
        if column not in cleaned:
            continue
        null_rate = cleaned[column].isna().mean()
        if 0 < null_rate <= null_limit:
            missing = int(cleaned[column].isna().sum())
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            actions[f"{column}_imputed"] = missing
        else:
            actions[f"{column}_imputed"] = 0

    for column in ("user_id", "sessions_last_30d", "days_since_last_login", "support_tickets", "churned"):
        if column in cleaned and cleaned[column].notna().all():
            cleaned[column] = cleaned[column].astype("int64")
    if "avg_session_minutes" in cleaned:
        cleaned["avg_session_minutes"] = cleaned["avg_session_minutes"].astype("float64")
    return CleanResult(frame=cleaned, actions=actions)
