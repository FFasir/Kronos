import pandas as pd

from src.data_layer.a_share_downloader import AShareDailyDownloader


def test_prepare_snapshot_filters_to_catalog_and_invalid_prices(monkeypatch):
    downloader = object.__new__(AShareDailyDownloader)
    monkeypatch.setattr(
        AShareDailyDownloader,
        "_catalog",
        staticmethod(lambda: {"000001": "平安银行", "600000": "浦发银行"}),
    )
    raw = pd.DataFrame([
        {"代码": "000001", "名称": "平安银行", "今开": 10, "最高": 11, "最低": 9, "最新价": 10.5, "成交量": 100, "成交额": 1000},
        {"代码": "600000", "名称": "浦发银行", "今开": 10, "最高": 9, "最低": 11, "最新价": 10, "成交量": 100, "成交额": 1000},
        {"代码": "00700", "名称": "非A股", "今开": 10, "最高": 11, "最低": 9, "最新价": 10, "成交量": 100, "成交额": 1000},
    ])

    result = downloader.prepare_snapshot(raw, "2026-07-15")

    assert result["code"].tolist() == ["000001"]
    assert result.iloc[0]["date"] == "2026-07-15"
    assert result.iloc[0]["name"] == "平安银行"
