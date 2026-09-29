from datetime import date
from unittest.mock import Mock

import pandas as pd

from src.config import load_config
from src.pipeline import run_training_if_allowed, training_gate
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


def test_training_runs_only_after_gate_allows_it() -> None:
    config = load_config()
    trainer = Mock()
    decision = training_gate(passing_report(), True, 3_000, 100, config)

    run_training_if_allowed(decision, trainer)

    assert decision.allowed
    trainer.assert_called_once()


def test_failed_validation_never_calls_training() -> None:
    config = load_config()
    trainer = Mock()
    report = passing_report()
    failed_report = report.__class__(
        report.file_date,
        [*report.checks, report.checks[0].__class__("forced_failure", "error", 1, 0, False, "test")],
    )
    decision = training_gate(failed_report, True, 3_000, 100, config)

    run_training_if_allowed(decision, trainer)

    assert not decision.allowed
    assert decision.reason == "validation has error-severity failures"
    trainer.assert_not_called()


def test_insufficient_history_skips_training() -> None:
    decision = training_gate(passing_report(), True, 999, 100, load_config())

    assert not decision.allowed
    assert decision.reason == "processed history has too few rows"


def test_storage_failure_skips_training() -> None:
    decision = training_gate(passing_report(), False, 3_000, 100, load_config())

    assert not decision.allowed
    assert decision.reason == "processed data was not stored"
