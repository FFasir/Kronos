import pandas as pd

from src.common.config import load_config
from src.data_layer.storage import StorageLayer


def _frame(date: str, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "volume": [1.0],
        },
        index=pd.to_datetime([date]),
    )


def test_configured_index_does_not_merge_same_code_stock_delta(monkeypatch, tmp_path):
    storage = StorageLayer(load_config())
    storage.h5_path = tmp_path / "missing.h5"
    monkeypatch.setattr(storage, "_load_csv_raw_data", lambda symbol: _frame("2026-07-10", 8198.0))
    monkeypatch.setattr(storage, "_load_sqlite_daily", lambda symbol: _frame("2026-07-15", 5.11))

    result = storage.load_raw_data("000852")

    assert list(result["close"]) == [8198.0]


def test_regular_stock_still_merges_sqlite_delta(monkeypatch, tmp_path):
    storage = StorageLayer(load_config())
    storage.h5_path = tmp_path / "missing.h5"
    monkeypatch.setattr(storage, "_load_csv_raw_data", lambda symbol: _frame("2026-07-10", 10.0))
    monkeypatch.setattr(storage, "_load_sqlite_daily", lambda symbol: _frame("2026-07-15", 10.5))

    result = storage.load_raw_data("000001")

    assert list(result["close"]) == [10.0, 10.5]
