from pathlib import Path

import pandas as pd

from scripts.generate_data import normal_day
from src.clean import clean_frame
from src.config import load_config
from src.train import should_promote, train_and_evaluate


def clean_history(dates: list[str]) -> pd.DataFrame:
    config = load_config()
    frames = [clean_frame(normal_day(file_date, drifted=file_date == "2026-09-26"), config).frame for file_date in dates]
    return pd.concat(frames, ignore_index=True)


def test_challenger_does_not_promote_when_it_loses() -> None:
    assert not should_promote(0.75, 0.76, 0.01)
    assert should_promote(0.77, 0.76, 0.01)


def test_first_training_run_writes_champion_and_metrics(tmp_path: Path) -> None:
    result = train_and_evaluate(clean_history(["2026-09-24", "2026-09-25"]), tmp_path, load_config())

    assert result.trained
    assert result.promoted is True
    assert (tmp_path / "champion.joblib").exists()
    assert (tmp_path / "metrics_2026-09-25.json").exists()
    assert result.metrics["challenger"]["roc_auc"] > result.metrics["dummy_baseline"]["roc_auc"]


def test_champion_and_challenger_are_scored_on_the_new_holdout(tmp_path: Path) -> None:
    config = load_config()
    train_and_evaluate(clean_history(["2026-09-24", "2026-09-25"]), tmp_path, config)
    result = train_and_evaluate(clean_history(["2026-09-24", "2026-09-25", "2026-09-26"]), tmp_path, config)

    assert result.trained
    assert result.metrics["champion"] is not None
    assert result.metrics["holdout_dates"] == ["2026-09-26"]
    assert result.metrics["holdout_rows"] == 2_000

