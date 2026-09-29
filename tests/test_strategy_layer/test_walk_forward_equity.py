from types import SimpleNamespace

import pandas as pd

from scripts.run_paper_walk_forward import (
    build_daily_equity,
    simulate_non_overlapping_trades,
)


def test_daily_equity_reconciles_with_trade_compounding():
    dates = pd.bdate_range("2025-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "open": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [9.5, 10.5, 11.5, 12.5, 13.5, 15.0],
        },
        index=dates,
    )
    predictions = pd.DataFrame(
        [{
            "origin_date": dates[0].strftime("%Y-%m-%d"),
            "entry_date": dates[1].strftime("%Y-%m-%d"),
            "exit_date": dates[5].strftime("%Y-%m-%d"),
            "pred_return_t5": 0.10,
            "execution_entry_open": 10.0,
            "execution_exit_close": 15.0,
        }]
    )
    args = SimpleNamespace(
        buy_threshold=0.005,
        commission_rate=0.0003,
        stamp_tax_rate=0.0,
        slippage_rate=0.001,
    )

    trades, trade_metrics = simulate_non_overlapping_trades(predictions, args)
    daily, daily_metrics = build_daily_equity(
        trades, prices, "2025-01", "2025-01", args
    )

    assert len(trades) == 1
    assert len(daily) == 6
    assert daily["in_market"].sum() == 5
    assert abs(daily["equity"].iloc[-1] - (1 + trade_metrics["net_compounded_return"])) < 1e-12
    assert daily_metrics["in_market_days"] == 5
    assert daily_metrics["max_drawdown"] <= 0


def test_daily_equity_stays_in_cash_without_trades():
    dates = pd.bdate_range("2025-01-02", periods=4)
    prices = pd.DataFrame(
        {"open": [10.0] * 4, "close": [10.0, 9.0, 8.0, 7.0]}, index=dates
    )
    args = SimpleNamespace(commission_rate=0.0003, stamp_tax_rate=0.0, slippage_rate=0.001)

    daily, metrics = build_daily_equity(
        pd.DataFrame(), prices, "2025-01", "2025-01", args
    )

    assert (daily["equity"] == 1.0).all()
    assert metrics["max_drawdown"] == 0.0
    assert metrics["in_market_ratio"] == 0.0
