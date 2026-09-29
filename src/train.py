"""Train and compare the churn model on validated history."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.storage import atomic_json, atomic_write


NUMERIC_FEATURES = [
    "sessions_last_30d",
    "avg_session_minutes",
    "days_since_last_login",
    "support_tickets",
]
CATEGORICAL_FEATURES = ["plan_type"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@dataclass
class TrainingResult:
    trained: bool
    reason: str
    promoted: bool | None
    metrics: dict[str, Any]


def build_model(seed: int) -> Pipeline:
    preparation = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [("prepare", preparation), ("model", LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed))]
    )


def split_history(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    dates = pd.to_datetime(frame["date"]).dt.date
    unique_dates = sorted(dates.dropna().unique())
    if len(unique_dates) >= 2:
        holdout_date = unique_dates[-1]
        test = frame.loc[dates == holdout_date].copy()
        train = frame.loc[dates < holdout_date].copy()
    else:
        groups = frame["user_id"]
        if groups.nunique() < 2:
            return None
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_index, test_index = next(splitter.split(frame, groups=groups))
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
    if train.empty or test.empty or train["churned"].nunique() < 2 or test["churned"].nunique() < 2:
        return None
    return train, test


def model_metrics(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    scores = model.predict_proba(features)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def should_promote(challenger_roc_auc: float, champion_roc_auc: float, margin: float) -> bool:
    return challenger_roc_auc >= champion_roc_auc + margin


def atomic_joblib(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, lambda temporary_path: joblib.dump(payload, temporary_path))


def train_and_evaluate(
    history: pd.DataFrame,
    artifact_dir: Path,
    config: dict[str, Any],
) -> TrainingResult:
    seed = config["training_thresholds"]["random_seed"]
    split = split_history(history, seed)
    if split is None:
        return TrainingResult(False, "holdout needs both classes and enough groups", None, {})
    train, test = split
    challenger = build_model(seed)
    challenger.fit(train[FEATURES], train["churned"])
    challenger_scores = model_metrics(challenger, test[FEATURES], test["churned"])

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(train[FEATURES], train["churned"])
    baseline_scores = model_metrics(baseline, test[FEATURES], test["churned"])

    champion_path = artifact_dir / "champion.joblib"
    champion_scores: dict[str, float] | None = None
    promoted = True
    if champion_path.exists():
        champion = joblib.load(champion_path)["model"]
        champion_scores = model_metrics(champion, test[FEATURES], test["churned"])
        promoted = should_promote(
            challenger_scores["roc_auc"],
            champion_scores["roc_auc"],
            config["training_thresholds"]["promotion_margin_roc_auc"],
        )

    latest_date = str(pd.to_datetime(test["date"]).dt.date.max())
    metadata = {
        "feature_list": FEATURES,
        "training_dates": sorted(str(value) for value in pd.to_datetime(train["date"]).dt.date.unique()),
        "holdout_dates": sorted(str(value) for value in pd.to_datetime(test["date"]).dt.date.unique()),
        "training_rows": len(train),
        "holdout_rows": len(test),
        "training_positive_labels": int(train["churned"].sum()),
        "holdout_positive_labels": int(test["churned"].sum()),
        "random_seed": seed,
        "library_versions": {
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    metrics = {
        "challenger": challenger_scores,
        "dummy_baseline": baseline_scores,
        "champion": champion_scores,
        "promotion_margin_roc_auc": config["training_thresholds"]["promotion_margin_roc_auc"],
        "promoted": promoted,
        **metadata,
    }
    candidate_path = artifact_dir / f"challenger_{latest_date}.joblib"
    atomic_joblib(candidate_path, {"model": challenger, "metadata": metadata})
    if promoted:
        atomic_joblib(champion_path, {"model": challenger, "metadata": metadata})
    atomic_json(artifact_dir / f"metrics_{latest_date}.json", metrics)
    return TrainingResult(True, "trained and evaluated", promoted, metrics)
