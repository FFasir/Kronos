from src.strategy_layer.stock_selector import StockSelector
from src.strategy_layer.signal_generator import SignalGenerator


class ConfigStub:
    def get(self, *keys, default=None):
        values = {
            ("strategy", "stock_selection_prob_up"): 0.65,
            ("strategy", "stock_selection_score"): 6.0,
            ("strategy", "stock_selection_min_amount"): 5_000_000,
            ("strategy", "max_per_industry"): 1,
            ("strategy", "stock_selection_top_k"): 10,
        }
        return values.get(keys, default)


def test_select_filters_risk_and_ranks_candidates():
    selector = StockSelector(ConfigStub())
    selected = selector.select([
        {"symbol": "000001", "name": "A", "confidence": 0.8, "score": 8.0, "pred_return": 0.03, "amount": 10_000_000, "industry": "bank"},
        {"symbol": "000002", "name": "B", "confidence": 0.7, "score": 7.0, "pred_return": 0.02, "amount": 10_000_000, "industry": "bank"},
        {"symbol": "000003", "name": "ST C", "confidence": 0.9, "score": 9.0, "pred_return": 0.04, "amount": 10_000_000},
        {"symbol": "000004", "name": "D", "confidence": 0.9, "score": 9.0, "pred_return": 0.04, "amount": 100},
    ])

    assert [item["symbol"] for item in selected] == ["000001"]
    assert selected[0]["rank"] == 1
    assert selected[0]["selected"] is True


def test_signal_positions_follow_prediction_confidence():
    selector_config = ConfigStub()
    generator = SignalGenerator(selector_config)
    generator.min_positions = 2
    generator.max_single_position = 0.8
    generator.buy_threshold = 0.005

    signals = generator.generate([
        {"symbol": "000001", "pred_return": 0.03, "pred_close_t5": 10, "confidence": 0.8},
        {"symbol": "000002", "pred_return": 0.03, "pred_close_t5": 10, "confidence": 0.2},
    ], "2026-07-15")

    positions = {item["symbol"]: item["position_pct"] for item in signals}
    assert positions["000001"] == 0.8
    assert positions["000002"] == 0.2
