import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_monthly_kronos_dataset import build_monthly_dataset


def make_symbol_frame(offset: float) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", "2024-07-31")
    close = np.linspace(10 + offset, 20 + offset, len(index))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "vol": np.full(len(index), 1_000_000.0),
            "amt": close * 1_000_000.0,
            "limit_up": np.zeros(len(index), dtype=np.float32),
            "limit_down": np.zeros(len(index), dtype=np.float32),
        },
        index=index.rename("datetime"),
    )


def write_source(path: Path) -> None:
    path.mkdir()
    (path / "metadata.json").write_text(
        json.dumps({
            "adjustment": "hfq",
            "limit_features": ["limit_up", "limit_down"],
            "first_listing_day_policy": "drop exact metadata listing date",
        }),
        encoding="utf-8",
    )
    frames = {f"00000{i}": make_symbol_frame(float(i)) for i in range(1, 4)}
    for name, date_slice in {
        "train": ("2024-01-01", "2024-03-31"),
        "val": ("2024-04-01", "2024-05-31"),
        "test": ("2024-06-01", "2024-07-31"),
    }.items():
        split = {symbol: frame.loc[start:end] for symbol, frame in frames.items() for start, end in [date_slice]}
        with (path / f"{name}_data.pkl").open("wb") as handle:
            pickle.dump(split, handle)


def test_monthly_dataset_uses_three_month_window_without_prediction_month_leakage(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "rolling"
    write_source(source)

    metadata = build_monthly_dataset(
        source=source,
        output=output,
        target_month="2024-07",
        train_months=3,
        validation_days=6,
        lookback=10,
        horizon=5,
    )

    assert metadata["train_target_start"] == "2024-04-01"
    assert metadata["validation_target_end"] == "2024-06-30"
    assert metadata["split_target_ranges"] == {
        "train": ["2024-04-01", metadata["train_target_end"]],
        "val": [metadata["validation_target_start"], "2024-06-30"],
    }
    assert metadata["split_context_policy"].startswith("lookback buffer")
    assert metadata["symbol_count"] == 3
    assert metadata["train_windows"] > 0
    assert metadata["val_windows"] > 0

    with (output / "train_data.pkl").open("rb") as handle:
        train_data = pickle.load(handle)
    with (output / "val_data.pkl").open("rb") as handle:
        val_data = pickle.load(handle)

    assert all(frame.index.max() < pd.Timestamp("2024-07-01") for frame in train_data.values())
    assert all(frame.index.max() < pd.Timestamp("2024-07-01") for frame in val_data.values())
    assert all(frame.index.min() < pd.Timestamp(metadata["train_target_start"]) for frame in train_data.values())
