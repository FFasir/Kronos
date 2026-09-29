#!/usr/bin/env python3
"""Evaluate simple ETF timing baselines with the same execution assumptions.

Signals are known only after a trading day's close and are executed at the next
trading day's open.  This makes the outputs comparable with the Kronos ETF
walk-forward experiment without using same-day close information for fills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Costs:
    commission_rate: float
    stamp_tax_rate: float
    slippage_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF buy-and-hold, moving-average, and momentum baselines")
    parser.add_argument("--data", default="data/raw/research/512100_etf_daily.csv")
    parser.add_argument("--start", default=None, help="inclusive YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="inclusive YYYY-MM-DD")
    parser.add_argument("--fast-window", type=int, default=20)
    parser.add_argument("--slow-window", type=int, default=60)
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--output-dir", default="data/outputs/etf_baselines")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prices(path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = ["date", "open", "close"]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    df = df[required].dropna().sort_values("date").drop_duplicates("date").set_index("date")
    if (df[["open", "close"]] <= 0).any().any():
        raise ValueError("open and close prices must be positive")
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    if len(df) < 2:
        raise ValueError("the requested date range has fewer than two trading days")
    return df


def build_signals(prices: pd.DataFrame, fast_window: int, slow_window: int, momentum_window: int) -> dict[str, pd.Series]:
    if fast_window <= 0 or slow_window <= fast_window or momentum_window <= 0:
        raise ValueError("require positive windows and slow-window > fast-window")
    close = prices["close"]
    return {
        "buy_and_hold": pd.Series(True, index=prices.index),
        "ma_cross": close.rolling(fast_window).mean() > close.rolling(slow_window).mean(),
        "momentum": close.pct_change(momentum_window) > 0,
    }


def execute_strategy(name: str, prices: pd.DataFrame, signal: pd.Series, costs: Costs) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cash = 1.0
    units = 0.0
    entry: dict | None = None
    trades: list[dict] = []
    equity_rows: list[dict] = []

    def buy(date: pd.Timestamp, price: float) -> None:
        nonlocal cash, units, entry
        capital = cash
        units = capital / (price * (1 + costs.commission_rate + costs.slippage_rate))
        cash = 0.0
        entry = {
            "entry_date": date.strftime("%Y-%m-%d"),
            "entry_open": price,
            "entry_capital": capital,
        }

    def sell(date: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal cash, units, entry
        if entry is None:
            raise RuntimeError("attempted to sell without an entry")
        gross_proceeds = units * price
        cash = gross_proceeds * (1 - costs.commission_rate - costs.stamp_tax_rate - costs.slippage_rate)
        trades.append(
            {
                **entry,
                "exit_date": date.strftime("%Y-%m-%d"),
                "exit_price": price,
                "exit_reason": reason,
                "gross_return": price / entry["entry_open"] - 1,
                "net_return": cash / entry["entry_capital"] - 1,
            }
        )
        units = 0.0
        entry = None

    for position, (date, row) in enumerate(prices.iterrows()):
        # The first buy-and-hold entry is the defined start-of-period benchmark.
        desired = name == "buy_and_hold" if position == 0 else bool(signal.iloc[position - 1])
        if units == 0 and desired:
            buy(date, float(row.open))
        elif units > 0 and not desired:
            sell(date, float(row.open), "signal_exit")
        equity = cash if units == 0 else units * float(row.close)
        equity_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": float(row.close),
                "signal_after_close": bool(signal.iloc[position]),
                "in_position_after_open": units > 0,
                "equity_before_final_liquidation": equity,
            }
        )

    if units > 0:
        final_date = prices.index[-1]
        sell(final_date, float(prices["close"].iloc[-1]), "end_of_sample")
        equity_rows[-1]["equity_before_final_liquidation"] = cash

    equity_df = pd.DataFrame(equity_rows)
    equity_df["equity"] = equity_df["equity_before_final_liquidation"]
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)
    trade_df = pd.DataFrame(trades)
    drawdown = equity_df["equity"] / equity_df["equity"].cummax() - 1
    volatility = float(equity_df["daily_return"].std(ddof=0))
    periods = max(len(equity_df) - 1, 1)
    annualized_return = float(equity_df["equity"].iloc[-1] ** (252 / periods) - 1)
    metrics = {
        "strategy": name,
        "net_return": float(equity_df["equity"].iloc[-1] - 1),
        "annualized_return": annualized_return,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility * np.sqrt(252),
        "sharpe_zero_rf": float(equity_df["daily_return"].mean() / volatility * np.sqrt(252)) if volatility else 0.0,
        "round_trip_count": len(trade_df),
        "in_market_days": int(equity_df["in_position_after_open"].sum()),
    }
    return trade_df, equity_df, metrics


def main() -> None:
    args = parse_args()
    data_path = (PROJECT_ROOT / args.data).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    all_prices = load_prices(data_path, None, None)
    prices = all_prices
    if args.start:
        prices = prices.loc[prices.index >= pd.Timestamp(args.start)]
    if args.end:
        prices = prices.loc[prices.index <= pd.Timestamp(args.end)]
    if len(prices) < 2:
        raise ValueError("the requested date range has fewer than two trading days")
    costs = Costs(args.commission_rate, args.stamp_tax_rate, args.slippage_rate)
    all_signals = build_signals(all_prices, args.fast_window, args.slow_window, args.momentum_window)
    signals = {name: values.reindex(prices.index) for name, values in all_signals.items()}
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, signal in signals.items():
        trades, equity, metrics = execute_strategy(name, prices, signal, costs)
        trades.to_csv(output_dir / f"{name}_trades.csv", index=False)
        equity.to_csv(output_dir / f"{name}_equity.csv", index=False)
        summary_rows.append(metrics)
    summary = pd.DataFrame(summary_rows).sort_values("strategy")
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    metadata = {
        "input": str(data_path),
        "input_sha256": sha256(data_path),
        "date_range": [prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")],
        "signal_history_range": [all_prices.index.min().strftime("%Y-%m-%d"), all_prices.index.max().strftime("%Y-%m-%d")],
        "rows": len(prices),
        "costs": asdict(costs),
        "signals": {
            "ma_cross": {"fast_window": args.fast_window, "slow_window": args.slow_window},
            "momentum": {"window": args.momentum_window},
        },
        "execution": "signals are observed after close and executed next trading day at open; final positions are liquidated at the final close",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
