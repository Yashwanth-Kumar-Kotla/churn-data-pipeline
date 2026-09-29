"""Persist pipeline outputs without exposing partial files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


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

