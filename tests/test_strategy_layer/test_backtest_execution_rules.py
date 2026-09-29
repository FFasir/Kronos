from decimal import Decimal

import pandas as pd

from src.strategy_layer.backtest_engine import BacktestEngine


class FakeConfig:
    seed = 42

    def get(self, *keys, default=None):
        values = {
            ("strategy", "paper_mode"): True,
            ("strategy", "paper_symbol"): "512100",
            ("strategy", "max_single_position"): 1.0,
            ("strategy", "risk_controls_enabled"): False,
            ("strategy", "exit_at_close"): True,
            ("backtest", "default_capital"): 1000000.0,
            ("backtest", "commission_rate"): 0.0003,
            ("backtest", "stamp_tax_rate"): 0.001,
            ("backtest", "slippage_rate"): 0.001,
        }
        return values.get(tuple(keys), default)


def prices(**overrides):
    row = {
        "open": Decimal("10"),
        "high": Decimal("10.5"),
        "low": Decimal("9.5"),
        "close": Decimal("10.2"),
        "volume": Decimal("100000"),
        "limit_up": False,
        "limit_down": False,
    }
    row.update(overrides)
    return row


def test_limit_up_blocks_buy_and_limit_down_blocks_sell():
    engine = BacktestEngine(FakeConfig())
    signal = {"symbol": "000001", "position_pct": 0.3, "reason": "test"}
    engine._execute_buy(signal, {"000001": prices(limit_up=True)}, "2026-07-01")
    assert "000001" not in engine._positions

    engine._execute_buy(signal, {"000001": prices()}, "2026-07-02")
    assert "000001" in engine._positions
    engine._execute_sell(signal, {"000001": prices(limit_down=True)}, "2026-07-03")
    assert "000001" in engine._positions


def test_etf_exit_uses_close_and_has_no_stamp_tax():
    engine = BacktestEngine(FakeConfig())
    signal = {"symbol": "512100", "position_pct": 1.0, "reason": "test"}
    market = {"512100": prices(open=Decimal("2.00"), close=Decimal("2.10"))}
    engine._execute_buy(signal, market, "2026-07-01")
    engine._execute_sell(signal, market, "2026-07-08", price_field="close")
    sell = engine._trades[-1]
    assert sell["price"] == 2.10
    assert sell["stamp_tax"] == 0.0


def test_benchmark_curve_is_scaled_to_initial_capital():
    engine = BacktestEngine(FakeConfig())
    dates = pd.to_datetime(["2026-07-01", "2026-07-02"])
    frame = pd.DataFrame({"close": [100.0, 101.0]}, index=dates)
    engine._build_benchmark(dates, {"000852": frame})
    assert engine._benchmark_equity[0]["value"] == 1000000.0
    assert engine._benchmark_equity[1]["value"] == 1010000.0
