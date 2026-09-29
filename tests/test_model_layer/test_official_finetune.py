import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FINETUNE = ROOT / "third_party" / "Kronos" / "finetune"


def load_module(name: str, path: Path):
    sys.path.insert(0, str(FINETUNE))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(FINETUNE))


def test_epoch_sampling_is_unique_and_reproducible():
    dataset = load_module("kronos_finetune_dataset", FINETUNE / "dataset.py")

    first = dataset.sample_without_replacement(1_000, 200, seed=100)
    repeated = dataset.sample_without_replacement(1_000, 200, seed=100)
    next_epoch = dataset.sample_without_replacement(1_000, 200, seed=101)

    assert len(first) == len(set(first)) == 200
    assert first == repeated
    assert first != next_epoch


def test_early_stopping_keeps_best_and_honors_patience():
    utils = load_module(
        "kronos_finetune_training_utils",
        FINETUNE / "utils" / "training_utils.py",
    )
    best, stale = float("inf"), 0
    decisions = []
    for loss in (3.0, 2.0, 2.1, 2.2, 2.3):
        improved, best, stale, should_stop = utils.update_early_stopping(
            best, stale, loss, min_delta=0.0, patience=3
        )
        decisions.append((improved, should_stop))

    assert best == 2.0
    assert decisions[-1] == (False, True)


def test_dataset_window_matches_horizon_and_filters_target_dates(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=12, name="datetime")
    close = np.arange(10.0, 22.0)
    frame = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "vol": np.full(len(dates), 1_000.0),
        "amt": close * 1_000.0,
        "limit_up": np.zeros(len(dates)),
        "limit_down": np.zeros(len(dates)),
    }, index=dates)
    for split in ("train", "val"):
        with (tmp_path / f"{split}_data.pkl").open("wb") as handle:
            pickle.dump({"SH600000": frame}, handle)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "split_target_ranges": {
            "train": [str(dates[5].date()), str(dates[9].date())],
            "val": [str(dates[6].date()), str(dates[10].date())],
        }
    }), encoding="utf-8")

    monkeypatch.setenv("KRONOS_DATASET_PATH", str(tmp_path))
    monkeypatch.setenv("KRONOS_LOOKBACK", "3")
    monkeypatch.setenv("KRONOS_PRED_LEN", "2")
    monkeypatch.setenv("KRONOS_TRAIN_SAMPLES", "100")
    monkeypatch.setenv("KRONOS_VAL_SAMPLES", "100")
    monkeypatch.setenv("KRONOS_TOKENIZER_EDGE_PRESSURE", "0")
    dataset = load_module("kronos_target_bounded_dataset", FINETUNE / "dataset.py")
    train = dataset.QlibDataset("train")

    assert train.window == 5
    assert [start for _, start in train.indices] == [2, 3, 4, 5]
    sample, _ = train[0]
    assert sample.shape == (5, 6)


def test_preprocessor_keeps_context_before_validation_targets():
    dates = pd.bdate_range("2024-01-01", periods=12, name="datetime")
    close = np.arange(10.0, 22.0)
    frame = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "vol": np.full(len(dates), 1_000.0),
        "amt": close * 1_000.0,
        "limit_up": np.zeros(len(dates)),
        "limit_down": np.zeros(len(dates)),
    }, index=dates)
    dataset = load_module("kronos_context_slice_dataset", FINETUNE / "dataset.py")
    val_data = dataset.slice_frame_with_context(
        frame,
        [str(dates[5].date()), str(dates[9].date())],
        lookback=3,
        horizon=2,
    )

    assert val_data is not None
    assert val_data.index[0] == dates[2]
    assert val_data.index[-1] == dates[9]


def test_training_metadata_allows_test_range_in_addition_to_required_splits():
    runner = load_module(
        "kronos_official_finetune_runner",
        ROOT / "scripts" / "run_official_finetune.py",
    )
    runner.validate_dataset_metadata({
        "adjustment": "hfq",
        "limit_features": ["limit_up", "limit_down"],
        "first_listing_day_policy": "drop exact metadata listing date",
        "split_context_policy": "lookback buffer; forecast targets strictly inside split_target_ranges",
        "split_target_ranges": {
            "train": ["2010-01-01", "2022-12-31"],
            "val": ["2023-01-01", "2024-12-31"],
            "test": ["2025-01-01", "2025-09-30"],
        },
        "lookback": 250,
        "horizon": 5,
    }, lookback=250, horizon=5)
