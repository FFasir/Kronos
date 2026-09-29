from src.service_layer.api_selection import _with_liquidity
from src.service_layer.models import PredictionResult
from src.strategy_layer.stock_selector import StockSelector


class FakeConfig:
    def get(self, *keys, default=None):
        return default


def prediction(code: str, score: float = 8.0) -> PredictionResult:
    return PredictionResult(
        symbol=code,
        name=code,
        pred_date="2026-07-21",
        pred_close_t5=10.2,
        pred_return=0.02,
        last_close=10.0,
        confidence=0.8,
        score=score,
    )


def test_liquidity_enrichment_uses_one_bulk_query(monkeypatch):
    class FakeStorage:
        calls = 0

        def __init__(self, config):
            pass

        def execute_sql(self, query, params=()):
            FakeStorage.calls += 1
            assert "recent_dates" in query
            return [{
                "code": "000001", "date": "2026-07-16", "close": 10.0,
                "volume": 1_000_000.0, "amount": 10_000_000.0,
            }]

        def load_raw_data(self, symbol):
            raise AssertionError("selection must not open one file per stock")

    monkeypatch.setattr("src.data_layer.storage.StorageLayer", FakeStorage)

    result = _with_liquidity([prediction("000001")])

    assert FakeStorage.calls == 1
    assert result[0]["daily_amount"] == 10_000_000.0
    assert result[0]["suspended"] is False


def test_unknown_industry_does_not_truncate_pool_to_three():
    selector = StockSelector(FakeConfig())
    candidates = [prediction(f"{index:06d}", score=10 - index / 100).model_dump() for index in range(10)]

    selected = selector.select(candidates)

    assert len(selected) == 10
