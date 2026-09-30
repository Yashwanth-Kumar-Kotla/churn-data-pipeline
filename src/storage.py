"""Persist pipeline outputs without exposing partial files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        writer(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    def write_json(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    atomic_write(path, write_json)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    atomic_write(path, lambda temporary_path: frame.to_parquet(temporary_path, index=False))


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {"files": []}
    with path.open() as handle:
        return json.load(handle)


def hash_is_recorded(manifest: dict[str, list[dict[str, Any]]], file_hash: str) -> bool:
    return any(entry["sha256"] == file_hash for entry in manifest["files"])


def record_manifest(path: Path, entry: dict[str, Any]) -> None:
    manifest = load_manifest(path)
    manifest["files"].append(entry)
    atomic_json(path, manifest)


def write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_json(path, report)


def quarantine_raw_file(raw_path: Path, quarantine_dir: Path, file_hash: str) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / f"{raw_path.stem}_{file_hash[:12]}{raw_path.suffix}"
    shutil.move(str(raw_path), destination)
    return destination
