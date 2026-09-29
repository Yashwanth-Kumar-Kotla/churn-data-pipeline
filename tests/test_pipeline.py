from datetime import date
import pandas as pd

from src.config import load_config
from src.pipeline import training_gate
from src.validate import validate_frame


def passing_report():
    rows = 1_000
    frame = pd.DataFrame(
        {
            "user_id": range(rows),
            "date": ["2026-09-24"] * rows,
            "sessions_last_30d": [12] * rows,
            "avg_session_minutes": [18.0] * rows,
            "days_since_last_login": [10] * rows,
            "support_tickets": [1] * rows,
            "plan_type": ["free"] * rows,
            "churned": [1 if value % 4 == 0 else 0 for value in range(rows)],
        }
    )
    return validate_frame(frame, date(2026, 9, 24), load_config())


def test_gate_allows_passing_history() -> None:
    decision = training_gate(passing_report(), 3_000, 100, load_config())
    assert decision.allowed


def test_gate_rejects_failed_validation() -> None:
    config = load_config()
    report = passing_report()
    failed_report = report.__class__(
        report.file_date,
        [*report.checks, report.checks[0].__class__("forced_failure", "error", 1, 0, False, "test")],
    )
    decision = training_gate(failed_report, 3_000, 100, config)

    assert not decision.allowed
    assert decision.reason == "validation has error-severity failures"


def test_insufficient_history_skips_training() -> None:
    decision = training_gate(passing_report(), 999, 100, load_config())

    assert not decision.allowed
    assert decision.reason == "processed history has too few rows"
