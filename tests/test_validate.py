from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.generate_data import bad_day, normal_day
from src.config import load_config
from src.validate import validate_frame


def checks_by_name(report):
    return {check.name: check for check in report.checks}


def valid_frame() -> pd.DataFrame:
    rows = 1_000
    return pd.DataFrame(
        {
            "user_id": np.arange(1, rows + 1),
            "date": ["2026-09-24"] * rows,
            "sessions_last_30d": [12] * rows,
            "avg_session_minutes": [18.0] * rows,
            "days_since_last_login": [10] * rows,
            "support_tickets": [1] * rows,
            "plan_type": ["free"] * rows,
            "churned": [1 if row % 4 == 0 else 0 for row in range(rows)],
        }
    )


@pytest.mark.parametrize(
    ("change", "check_name"),
    [
        (lambda frame: frame.drop(columns="support_tickets"), "required_columns"),
        (lambda frame: frame.assign(sessions_last_30d=[None] * 51 + [12] * 949), "null_rate:sessions_last_30d"),
        (lambda frame: frame.assign(days_since_last_login=[-1] * 11 + [10] * 989), "out_of_range_rate:days_since_last_login"),
        (lambda frame: frame.assign(plan_type=["enterprise"] * 6 + ["free"] * 994), "unknown_plan_type_rate"),
        (lambda frame: frame.iloc[:999], "minimum_row_count"),
        (lambda frame: frame.assign(churned=[2] + [0] * 999), "valid_churn_labels"),
        (lambda frame: frame.assign(date=["2026-09-23"] * 1_000), "date_matches_filename"),
    ],
)
def test_each_error_check_can_fail_independently(change, check_name: str) -> None:
    report = validate_frame(change(valid_frame()), date(2026, 9, 24), load_config())

    assert not report.passed
    assert not checks_by_name(report)[check_name].passed


def test_generated_clean_day_passes() -> None:
    frame = normal_day("2026-09-24")

    report = validate_frame(frame, date(2026, 9, 24), load_config())

    assert report.passed


def test_generated_bad_day_reports_multiple_failures() -> None:
    frame = bad_day()

    report = validate_frame(frame, date(2026, 9, 27), load_config())
    checks = checks_by_name(report)

    assert not report.passed
    assert not checks["required_columns"].passed
    assert not checks["null_rate:sessions_last_30d"].passed
    assert not checks["out_of_range_rate:days_since_last_login"].passed
    assert not checks["unknown_plan_type_rate"].passed


def test_drift_is_a_warning_not_a_failure() -> None:
    reference = normal_day("2026-09-24")
    current = normal_day("2026-09-26", drifted=True)

    report = validate_frame(current, date(2026, 9, 26), load_config(), reference, date(2026, 9, 24))
    checks = checks_by_name(report)

    assert report.passed
    assert checks["mean_shift:sessions_last_30d"].severity == "warning"
    assert not checks["mean_shift:sessions_last_30d"].passed
