import pandas as pd

from src.clean import clean_frame
from src.config import load_config


def test_cleaning_normalizes_repairs_and_imputes() -> None:
    frame = pd.DataFrame({"user_id": range(100), "date": ["2026-09-24"] * 100})
    frame["sessions_last_30d"] = 12
    frame["avg_session_minutes"] = 20.0
    frame["days_since_last_login"] = 4
    frame["support_tickets"] = 1
    frame["plan_type"] = "basic"
    frame["churned"] = 0
    frame.loc[0, "avg_session_minutes"] = None
    frame.loc[1, "days_since_last_login"] = -5
    frame.loc[2, "plan_type"] = " Pro "
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)

    result = clean_frame(frame, load_config())

    assert len(result.frame) == 100
    assert result.frame.loc[2, "plan_type"] == "pro"
    assert result.frame["days_since_last_login"].isna().sum() == 0
    assert result.frame["avg_session_minutes"].isna().sum() == 0
    assert result.actions["exact_duplicates_dropped"] == 1
    assert result.actions["days_since_last_login_set_to_null"] == 1


def test_cleaning_does_not_impute_large_null_rate() -> None:
    frame = pd.DataFrame(
        {
            "user_id": range(10),
            "sessions_last_30d": [None] * 6 + [10, 11, 12, 13],
            "avg_session_minutes": [20.0] * 10,
        }
    )

    result = clean_frame(frame, load_config())

    assert result.frame["sessions_last_30d"].isna().sum() == 6
    assert result.actions["sessions_last_30d_imputed"] == 0
