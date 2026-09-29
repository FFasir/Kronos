from src.strategy_layer.signal_generator import SignalGenerator


class FakeConfig:
    def get(self, *keys, default=None):
        values = {
            ("strategy", "buy_threshold"): 0.005,
            ("strategy", "min_positions"): 2,
            ("strategy", "max_single_position"): 0.30,
        }
        return values.get(tuple(keys), default)


def test_single_buy_signal_uses_position_cap():
    generator = SignalGenerator(FakeConfig())

    signals = generator.generate(
        [{
            "symbol": "000001",
            "pred_return": 0.02,
            "pred_close_t5": 10.2,
            "confidence": 0.8,
        }],
        "2026-07-15",
    )

    assert signals[0]["signal_type"] == "BUY"
    assert signals[0]["position_pct"] == 0.30
