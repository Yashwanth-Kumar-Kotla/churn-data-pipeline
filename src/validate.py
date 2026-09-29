"""Validate daily churn data before it is stored or trained on."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.clean import NUMERIC_COLUMNS, prepare_frame


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: str
    observed: Any
    threshold: Any
    passed: bool
    stage: str


@dataclass
class ValidationReport:
    file_date: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed or check.severity == "warning" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_date": self.file_date,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def at_most(
    name: str,
    observed: float,
    limit: float,
    severity: str = "error",
    stage: str = "pre_imputation",
) -> CheckResult:
    return CheckResult(name, severity, observed, limit, observed <= limit, stage)


def psi(reference: pd.Series, current: pd.Series) -> float | None:
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_hist, _ = np.histogram(reference, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    reference_share = np.maximum(reference_hist / reference_hist.sum(), epsilon)
    current_share = np.maximum(current_hist / current_hist.sum(), epsilon)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def validate_frame(
    raw_frame: pd.DataFrame,
    file_date: date,
    config: dict[str, Any],
    reference_frame: pd.DataFrame | None = None,
    reference_date: date | None = None,
) -> ValidationReport:
    checks: list[CheckResult] = []
    required_columns = config["required_columns"]
    raw_columns = [str(column).strip() for column in raw_frame.columns]
    missing = sorted(set(required_columns) - set(raw_columns))
    checks.append(CheckResult("required_columns", "error", missing, required_columns, not missing, "raw_schema"))

    extra = sorted(set(raw_columns) - set(required_columns))
    checks.append(CheckResult("extra_columns", "warning", extra, [], not extra, "raw_schema"))
    frame = prepare_frame(raw_frame)
    quality = config["quality_thresholds"]
    rows = len(frame)
    checks.append(CheckResult(
        "minimum_row_count", "error", rows, quality["minimum_rows"],
        rows >= quality["minimum_rows"], "pre_imputation",
    ))

    for column in required_columns:
        if column not in frame:
            continue
        null_rate = float(frame[column].isna().mean()) if rows else 1.0
        checks.append(at_most(f"null_rate:{column}", null_rate, quality["maximum_null_rate"]))

    duplicate_rate = float(frame.duplicated().mean()) if rows else 0.0
    checks.append(at_most("duplicate_rate", duplicate_rate, quality["maximum_duplicate_rate"]))

    for column, (minimum, maximum) in config["value_ranges"].items():
        if column not in frame:
            continue
        values = frame[column]
        invalid_rate = float((values.notna() & ~values.between(minimum, maximum)).mean()) if rows else 0.0
        checks.append(at_most(
            f"out_of_range_rate:{column}", invalid_rate, quality["maximum_out_of_range_rate"]
        ))

    if "plan_type" in frame:
        unknown_rate = float((~frame["plan_type"].isin(config["allowed_plan_types"]) & frame["plan_type"].notna()).mean())
        checks.append(at_most("unknown_plan_type_rate", unknown_rate, quality["maximum_unknown_plan_rate"]))

    if "churned" in frame:
        invalid_labels = int((~frame["churned"].isin([0, 1])).sum())
        checks.append(at_most("valid_churn_labels", invalid_labels, 0))
        valid_labels = frame.loc[frame["churned"].isin([0, 1]), "churned"]
        churn_rate = float(valid_labels.mean()) if not valid_labels.empty else None
        checks.append(
            CheckResult(
                "plausible_churn_rate",
                "error",
                churn_rate,
                [quality["minimum_churn_rate"], quality["maximum_churn_rate"]],
                churn_rate is not None and quality["minimum_churn_rate"] <= churn_rate <= quality["maximum_churn_rate"],
                "pre_imputation",
            )
        )

    if "date" in frame:
        mismatch_rate = float((frame["date"] != file_date).mean()) if rows else 1.0
        checks.append(at_most("date_matches_filename", mismatch_rate, 0.0))

    numeric_features = [column for column in NUMERIC_COLUMNS if column not in {"user_id", "churned"} and column in frame]
    for column in numeric_features:
        variance = float(frame[column].dropna().var()) if frame[column].notna().sum() > 1 else 0.0
        checks.append(CheckResult(
            f"zero_variance:{column}", "warning", variance, 0.0, variance > 0.0, "pre_imputation"
        ))

    if "user_id" in frame:
        feature_columns = [column for column in required_columns if column not in {"user_id", "date"} and column in frame]
        conflicts = 0
        for _, group in frame.dropna(subset=["user_id"]).groupby("user_id"):
            if len(group) > 1 and group[feature_columns].drop_duplicates().shape[0] > 1:
                conflicts += 1
        checks.append(at_most("conflicting_user_ids", conflicts, 0, "warning"))

    if reference_frame is not None:
        checks.extend(drift_checks(frame, reference_frame, config, file_date, reference_date))
    return ValidationReport(file_date=file_date.isoformat(), checks=checks)


def drift_checks(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    config: dict[str, Any],
    file_date: date,
    reference_date: date | None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    thresholds = config["drift_thresholds"]
    minimum_rows = thresholds["minimum_rows_for_comparison"]
    if len(frame) < minimum_rows or len(reference_frame) < minimum_rows:
        return [CheckResult(
            "drift_reference", "warning", "skipped: insufficient rows", minimum_rows,
            True, "pre_imputation",
        )]

    reference = prepare_frame(reference_frame)
    numeric_columns = set(config["value_ranges"]) & set(frame) & set(reference)
    for column in sorted(numeric_columns):
        reference_std = float(reference[column].std())
        if reference_std == 0 or np.isnan(reference_std):
            checks.append(CheckResult(
                f"mean_shift:{column}", "warning", "skipped: zero reference variance",
                thresholds["mean_shift_standard_deviations"], True, "pre_imputation",
            ))
        else:
            shift = abs(float(frame[column].mean()) - float(reference[column].mean())) / reference_std
            checks.append(at_most(
                f"mean_shift:{column}", shift, thresholds["mean_shift_standard_deviations"], "warning"
            ))
        value = psi(reference[column], frame[column])
        if value is None:
            checks.append(CheckResult(f"psi:{column}", "warning", "skipped", thresholds["psi"], True, "pre_imputation"))
        else:
            checks.append(at_most(f"psi:{column}", value, thresholds["psi"], "warning"))

    if "plan_type" in frame and "plan_type" in reference:
        categories = set(frame["plan_type"].dropna()) | set(reference["plan_type"].dropna())
        shift = max(
            (
                abs(float((frame["plan_type"] == category).mean()) - float((reference["plan_type"] == category).mean()))
                for category in categories
            ),
            default=0.0,
        )
        checks.append(at_most("plan_type_proportion_shift", shift, thresholds["category_proportion_points"], "warning"))

    row_change = abs(len(frame) - len(reference)) / len(reference)
    checks.append(at_most("row_count_shift", row_change, thresholds["row_count_change"], "warning"))
    if "churned" in frame and "churned" in reference:
        churn_shift = abs(float(frame["churned"].mean()) - float(reference["churned"].mean()))
        checks.append(at_most("churn_rate_shift", churn_shift, thresholds["churn_rate_change"], "warning"))
    if reference_date is not None:
        gap = (file_date - reference_date).days
        checks.append(at_most("date_gap", gap, 1, "warning"))
    return checks
