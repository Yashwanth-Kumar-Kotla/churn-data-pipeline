"""Load the pipeline configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path = Path("config.yaml")) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)

