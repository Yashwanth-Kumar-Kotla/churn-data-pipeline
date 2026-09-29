"""Generate deterministic daily churn activity files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NORMAL_DATES = ("2026-09-24", "2026-09-25", "2026-09-26", "2026-09-29")
ALL_DATES = (*NORMAL_DATES[:3], "2026-09-27", "2026-09-28", NORMAL_DATES[3])
PLAN_TYPES = np.array(["free", "basic", "pro"])


def seed_for(file_date: str) -> int:
    return int(file_date.replace("-", ""))


def churn_probability(frame: pd.DataFrame) -> np.ndarray:
    plan_effect = frame["plan_type"].map({"free": 0.12, "basic": 0.0, "pro": -0.10})
    log_odds = (
        -3.0
        + 0.075 * frame["days_since_last_login"]
        + 0.24 * frame["support_tickets"]
        - 0.060 * frame["sessions_last_30d"]
        - 0.012 * (frame["avg_session_minutes"] - 18)
        + plan_effect
    )
    return 1 / (1 + np.exp(-log_odds))


def normal_day(file_date: str, drifted: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed_for(file_date))
    rows = 2_000
    sessions_mean = 16 if drifted else 12
    minutes_mean = 22 if drifted else 18
    frame = pd.DataFrame(
        {
            "user_id": rng.choice(np.arange(1, 2_501), size=rows, replace=False),
            "date": file_date,
            "sessions_last_30d": rng.poisson(sessions_mean, rows),
            "avg_session_minutes": np.clip(rng.normal(minutes_mean, 6, rows), 2, 90),
            "days_since_last_login": rng.integers(0, 60, rows),
            "support_tickets": rng.poisson(1, rows),
            "plan_type": rng.choice(PLAN_TYPES, size=rows, p=[0.5, 0.3, 0.2]),
        }
    )
    frame["churned"] = rng.binomial(1, churn_probability(frame))

    null_rows = rng.choice(frame.index, size=40, replace=False)
    frame.loc[null_rows, "avg_session_minutes"] = np.nan

    messy_plan_rows = rng.choice(frame.index, size=20, replace=False)
    for row in messy_plan_rows:
        value = frame.at[row, "plan_type"]
        frame.at[row, "plan_type"] = value.title() if row % 2 else f" {value} "

    duplicates = frame.iloc[rng.choice(frame.index, size=15, replace=False)].copy()
    return pd.concat([frame, duplicates], ignore_index=True)


def bad_day() -> pd.DataFrame:
    frame = normal_day("2026-09-27")
    rng = np.random.default_rng(seed_for("2026-09-27") + 1)
    null_rows = rng.choice(frame.index, size=705, replace=False)
    invalid_rows = rng.choice(frame.index, size=50, replace=False)
    frame.loc[null_rows, "sessions_last_30d"] = np.nan
    frame.loc[invalid_rows, "days_since_last_login"] = -5
    normalized_plan = frame["plan_type"].astype(str).str.strip().str.lower()
    frame.loc[normalized_plan.eq("pro"), "plan_type"] = "PRO_v2"
    return frame.drop(columns="support_tickets")


def outage_day() -> pd.DataFrame:
    return normal_day("2026-09-28").iloc[:60].reset_index(drop=True)


def write_days(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    days = {
        "2026-09-24": normal_day("2026-09-24"),
        "2026-09-25": normal_day("2026-09-25"),
        "2026-09-26": normal_day("2026-09-26", drifted=True),
        "2026-09-27": bad_day(),
        "2026-09-28": outage_day(),
        "2026-09-29": normal_day("2026-09-29"),
    }
    written = []
    for file_date in ALL_DATES:
        path = output_dir / f"user_activity_{file_date}.csv"
        days[file_date].to_csv(path, index=False)
        written.append(path)
    return written


def signal_summary(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame.dropna().drop_duplicates().copy()
    features = [
        "sessions_last_30d",
        "avg_session_minutes",
        "days_since_last_login",
        "support_tickets",
        "plan_type",
    ]
    numeric_features = features[:-1]
    model = Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric_features),
                        ("plan", OneHotEncoder(handle_unknown="ignore"), ["plan_type"]),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    baseline = DummyClassifier(strategy="prior")
    split_index = int(len(clean) * 0.75)
    train, test = clean.iloc[:split_index], clean.iloc[split_index:]
    model.fit(train[features], train["churned"])
    baseline.fit(train[features], train["churned"])
    model_scores = model.predict_proba(test[features])[:, 1]
    baseline_scores = baseline.predict_proba(test[features])[:, 1]
    return {
        "churn_rate": clean["churned"].mean(),
        "logistic_roc_auc": roc_auc_score(test["churned"], model_scores),
        "dummy_roc_auc": roc_auc_score(test["churned"], baseline_scores),
        "logistic_pr_auc": average_precision_score(test["churned"], model_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate six deterministic daily churn CSV files.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    written = write_days(args.output_dir)
    summary = signal_summary(normal_day("2026-09-24"))
    print(f"wrote {len(written)} files to {args.output_dir}")
    print(
        "day 1 signal check: "
        f"churn={summary['churn_rate']:.1%}, "
        f"logistic ROC AUC={summary['logistic_roc_auc']:.3f}, "
        f"dummy ROC AUC={summary['dummy_roc_auc']:.3f}, "
        f"logistic PR AUC={summary['logistic_pr_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
