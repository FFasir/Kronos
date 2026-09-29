import pandas as pd

from src.data_layer.downloader import DataDownloader


def test_csi1000_uses_shanghai_index_symbol(monkeypatch):
    seen = {}

    def fake_download(symbol):
        seen["symbol"] = symbol
        return pd.DataFrame([{
            "date": "2026-07-15",
            "open": 7959.37,
            "high": 7979.08,
            "low": 7783.22,
            "close": 7817.83,
            "volume": 25843327100,
        }])

    monkeypatch.setattr("src.data_layer.downloader.ak.stock_zh_index_daily", fake_download)
    downloader = DataDownloader()

    result = downloader._download_index("000852", "中证1000", "20260701", "20260716")

    assert seen["symbol"] == "sh000852"
    assert result.iloc[-1]["close"] == 7817.83
