"""Run the churn data pipeline."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.clean import clean_frame
from src.config import load_config
from src.ingest import RawFile, file_date_from_path, read_raw_csv
from src.storage import (
    hash_is_recorded,
    load_manifest,
    quarantine_raw_file,
    record_manifest,
    sha256_file,
    write_parquet,
    write_report,
)
from src.train import train_and_evaluate
from src.validate import CheckResult, ValidationReport, validate_frame


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


def training_gate(
    report: ValidationReport,
    history_rows: int,
    history_positive_labels: int,
    config: dict[str, Any],
) -> GateDecision:
    if not report.passed:
        return GateDecision(False, "validation has error-severity failures")
    thresholds = config["training_thresholds"]
    if history_rows < thresholds["minimum_history_rows"]:
        return GateDecision(False, "processed history has too few rows")
    if history_positive_labels < thresholds["minimum_positive_labels"]:
        return GateDecision(False, "processed history has too few positive labels")
    return GateDecision(True, "validation passed and processed history is sufficient")


@dataclass(frozen=True)
class FileOutcome:
    path: Path
    status: str
    training_reason: str | None = None


def configure_logging(base_dir: Path) -> logging.Logger:
    logger = logging.getLogger("churn_pipeline")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log_path = base_dir / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def failure_report(file_date: date | None, name: str, observed: str) -> ValidationReport:
    display_date = file_date.isoformat() if file_date else "unknown"
    return ValidationReport(
        file_date=display_date,
        checks=[CheckResult(name, "error", observed, "readable CSV with rows", False, "ingest")],
    )


def report_path(base_dir: Path, raw_file: RawFile, file_hash: str) -> Path:
    return base_dir / "logs" / "reports" / f"{raw_file.path.stem}_{file_hash[:12]}.json"


def processed_path(base_dir: Path, file_date: date) -> Path:
    return base_dir / "data" / "processed" / f"user_activity_{file_date}.parquet"


def processed_history(base_dir: Path) -> pd.DataFrame:
    paths = sorted((base_dir / "data" / "processed").glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def reference_before(base_dir: Path, current_date: date) -> tuple[pd.DataFrame | None, date | None]:
    candidates: list[tuple[date, Path]] = []
    for path in (base_dir / "data" / "processed").glob("*.parquet"):
        try:
            candidate_date = file_date_from_path(path.with_suffix(".csv"))
        except ValueError:
            continue
        if candidate_date < current_date:
            candidates.append((candidate_date, path))
    if not candidates:
        return None, None
    reference_date, reference_path = max(candidates)
    return pd.read_parquet(reference_path), reference_date


def quarantine_with_report(
    raw_file: RawFile,
    file_hash: str,
    report: ValidationReport,
    base_dir: Path,
    reason: str,
) -> FileOutcome:
    saved_report_path = report_path(base_dir, raw_file, file_hash)
    write_report(saved_report_path, report.to_dict())
    quarantined_path = quarantine_raw_file(raw_file.path, base_dir / "data" / "quarantine", file_hash)
    record_manifest(
        base_dir / "state" / "manifest.json",
        {
            "filename": raw_file.path.name,
            "sha256": file_hash,
            "file_date": raw_file.file_date.isoformat(),
            "status": "quarantined",
            "reason": reason,
            "report_path": str(saved_report_path),
            "quarantine_path": str(quarantined_path),
        },
    )
    return FileOutcome(raw_file.path, "quarantined")


def process_file(
    raw_file: RawFile,
    file_hash: str,
    base_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> FileOutcome:
    try:
        raw_frame = read_raw_csv(raw_file.path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        report = failure_report(raw_file.file_date, "unreadable_file", str(error))
        logger.info("quarantining %s because it cannot be read", raw_file.path.name)
        return quarantine_with_report(raw_file, file_hash, report, base_dir, "unreadable_file")

    if raw_frame.empty:
        report = failure_report(raw_file.file_date, "no_rows", 0)
        logger.info("quarantining %s because it has no rows", raw_file.path.name)
        return quarantine_with_report(raw_file, file_hash, report, base_dir, "no_rows")

    reference_frame, reference_date = reference_before(base_dir, raw_file.file_date)
    report = validate_frame(raw_frame, raw_file.file_date, config, reference_frame, reference_date)
    if not report.passed:
        failed = [check.name for check in report.checks if check.severity == "error" and not check.passed]
        logger.info("quarantining %s, failed checks: %s", raw_file.path.name, ", ".join(failed))
        return quarantine_with_report(raw_file, file_hash, report, base_dir, "validation_failed")

    cleaned = clean_frame(raw_frame, config)
    if cleaned.frame.empty:
        raise ValueError(f"cleaning removed every row in {raw_file.path.name}")
    destination = processed_path(base_dir, raw_file.file_date)
    write_parquet(cleaned.frame, destination)
    saved_report_path = report_path(base_dir, raw_file, file_hash)
    report_payload = report.to_dict()
    report_payload["cleaning_actions"] = cleaned.actions
    write_report(saved_report_path, report_payload)
    manifest_entry = {
        "filename": raw_file.path.name,
        "sha256": file_hash,
        "file_date": raw_file.file_date.isoformat(),
        "status": "passed",
        "report_path": str(saved_report_path),
        "processed_path": str(destination),
        "cleaning_actions": cleaned.actions,
    }
    history = processed_history(base_dir)
    decision = training_gate(report, len(history), int(history["churned"].sum()), config)
    warnings = [check.name for check in report.checks if check.severity == "warning" and not check.passed]
    if warnings:
        logger.info("%s passed with warnings: %s", raw_file.path.name, ", ".join(warnings))
    if decision.allowed:
        training = train_and_evaluate(history, base_dir / "models", config)
        if training.trained:
            challenger = training.metrics["challenger"]["roc_auc"]
            champion = training.metrics["champion"]
            comparison = "first champion" if champion is None else f"champion {champion['roc_auc']:.3f}"
            promotion = "promoted" if training.promoted else "kept champion"
            logger.info(
                "processed %s, challenger ROC AUC %.3f vs %s, %s",
                raw_file.path.name,
                challenger,
                comparison,
                promotion,
            )
        else:
            logger.info("processed %s, training skipped: %s", raw_file.path.name, training.reason)
        record_manifest(base_dir / "state" / "manifest.json", manifest_entry)
        return FileOutcome(raw_file.path, "processed", training.reason)
    logger.info("processed %s, training skipped: %s", raw_file.path.name, decision.reason)
    record_manifest(base_dir / "state" / "manifest.json", manifest_entry)
    return FileOutcome(raw_file.path, "processed", decision.reason)


def selected_files(base_dir: Path, logger: logging.Logger) -> tuple[list[tuple[RawFile, str]], list[Path]]:
    raw_dir = base_dir / "data" / "raw"
    valid: list[RawFile] = []
    invalid: list[Path] = []
    for path in raw_dir.glob("*.csv"):
        try:
            valid.append(RawFile(path, file_date_from_path(path)))
        except ValueError:
            invalid.append(path)
    manifest = load_manifest(base_dir / "state" / "manifest.json")
    hashes = {item.path: sha256_file(item.path) for item in valid}
    by_date: dict[date, list[RawFile]] = defaultdict(list)
    for item in valid:
        by_date[item.file_date].append(item)
    selected: list[tuple[RawFile, str]] = []
    for file_date, candidates in by_date.items():
        newest = max(candidates, key=lambda item: (item.path.stat().st_mtime_ns, item.path.name))
        for candidate in candidates:
            if candidate == newest:
                continue
            file_hash = hashes[candidate.path]
            if hash_is_recorded(manifest, file_hash):
                continue
            logger.warning("skipping older same-date candidate %s", candidate.path.name)
            record_manifest(
                base_dir / "state" / "manifest.json",
                {
                    "filename": candidate.path.name,
                    "sha256": file_hash,
                    "file_date": candidate.file_date.isoformat(),
                    "status": "superseded",
                },
            )
        if not hash_is_recorded(manifest, hashes[newest.path]):
            selected.append((newest, hashes[newest.path]))
    return sorted(selected, key=lambda item: (item[0].file_date, item[0].path.name)), invalid


def run_pipeline(base_dir: Path) -> list[FileOutcome]:
    logger = configure_logging(base_dir)
    config = load_config(base_dir / "config.yaml")
    selected, invalid = selected_files(base_dir, logger)
    outcomes: list[FileOutcome] = []
    for path in invalid:
        file_hash = sha256_file(path)
        raw_file = RawFile(path, date.min)
        report = failure_report(None, "invalid_filename", path.name)
        outcomes.append(quarantine_with_report(raw_file, file_hash, report, base_dir, "invalid_filename"))
    if not selected and not invalid:
        logger.info("nothing to process")
        return outcomes
    for raw_file, file_hash in selected:
        outcomes.append(process_file(raw_file, file_hash, base_dir, config, logger))
    logger.info("run complete: %s processed, %s quarantined", sum(item.status == "processed" for item in outcomes), sum(item.status == "quarantined" for item in outcomes))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the churn data pipeline.")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("project_dir", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_pipeline(args.project_dir.resolve())


if __name__ == "__main__":
    main()
