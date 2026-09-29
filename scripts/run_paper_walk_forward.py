#!/usr/bin/env python3
"""Run the paper's CSI 1000 single-index walk-forward timing experiment.

This intentionally does not use the official ``qlib_test.py`` top-K equity
backtest.  It selects inference parameters from the preceding calendar month,
then evaluates one CSI 1000 index signal at a time on the following month.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Params:
    temperature: float
    top_p: float
    lookback: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSI 1000 Kronos walk-forward timing")
    parser.add_argument("--data", default="data/raw/000852_中证1000_daily.csv")
    parser.add_argument(
        "--execution-data",
        default=None,
        help="Optional tradable OHLCV CSV. Signals use --data; entries and exits use this file.",
    )
    parser.add_argument("--official-repo", default="third_party/Kronos")
    parser.add_argument(
        "--tokenizer-path",
        default="models/official_finetune/csi1000_tokenizer/checkpoints/best_model",
    )
    parser.add_argument(
        "--model-path",
        default="models/official_finetune/csi1000_predictor/checkpoints/best_model",
    )
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2025-09")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--selection-metric",
        choices=["net_return", "spearman"],
        default="net_return",
        help="Validation metric used to select each month's inference parameters.",
    )
    parser.add_argument("--buy-threshold", type=float, default=0.005)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument(
        "--stamp-tax-rate",
        type=float,
        default=0.0,
        help="ETF trades are exempt from China's stock stamp duty; default is zero.",
    )
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--output-dir", default="data/outputs/paper_walk_forward")
    parser.add_argument(
        "--frozen-manifest",
        default=None,
        help="JSON experiment manifest that verifies inputs and overrides mutable run settings.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_frozen_manifest(args: argparse.Namespace) -> None:
    manifest_path = (PROJECT_ROOT / args.frozen_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_final_holdout":
        raise ValueError("frozen manifest has an unexpected status")

    model = manifest["model"]
    data = manifest["data"]
    settings = manifest["inference_and_execution"]
    holdout = manifest["final_holdout_period"]
    expected_hashes = {
        PROJECT_ROOT / model["tokenizer_path"] / "model.safetensors": model["tokenizer_sha256"],
        PROJECT_ROOT / model["predictor_path"] / "model.safetensors": model["predictor_sha256"],
        PROJECT_ROOT / data["signal_data"]: data["signal_data_sha256"],
        PROJECT_ROOT / data["execution_data"]: data["execution_data_sha256"],
        PROJECT_ROOT / settings["walk_forward_script"]: settings["walk_forward_script_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input checksum mismatch: {path}")

    args.data = data["signal_data"]
    args.execution_data = data["execution_data"]
    args.tokenizer_path = model["tokenizer_path"]
    args.model_path = model["predictor_path"]
    args.start_month = holdout["start_month"]
    args.end_month = holdout["end_month"]
    args.output_dir = holdout["output_dir"]
    for name in ("samples", "batch_size", "seed", "selection_metric", "buy_threshold", "commission_rate", "stamp_tax_rate", "slippage_rate"):
        setattr(args, name, settings[name])
    print(f"Frozen manifest verified: {manifest['experiment_id']}", flush=True)


def load_ohlcv_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    df = df[required].dropna().sort_values("date").drop_duplicates("date")
    df = df.set_index("date")
    if (df[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise ValueError("OHLCV must be strictly positive for Kronos tokenization")
    return df


def attach_execution_prices(predictions: pd.DataFrame, execution_df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach actual tradable entry/exit prices without changing model targets."""
    if execution_df is None or predictions.empty:
        predictions["execution_entry_open"] = predictions.get("entry_open")
        predictions["execution_exit_close"] = predictions.get("actual_close_t5")
        return predictions

    execution_open = execution_df[["open"]].copy()
    execution_open.index = execution_open.index.strftime("%Y-%m-%d")
    execution_open = execution_open.rename(columns={"open": "execution_entry_open"})
    execution_close = execution_df[["close"]].copy()
    execution_close.index = execution_close.index.strftime("%Y-%m-%d")
    execution_close = execution_close.rename(columns={"close": "execution_exit_close"})
    output = predictions.merge(execution_open, how="left", left_on="entry_date", right_index=True)
    output = output.merge(execution_close, how="left", left_on="exit_date", right_index=True)
    return output


def month_range(start_month: str, end_month: str) -> list[pd.Period]:
    start, end = pd.Period(start_month, freq="M"), pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end month must not precede start month")
    return list(pd.period_range(start, end, freq="M"))


def make_predictor(args: argparse.Namespace):
    repo = (PROJECT_ROOT / args.official_repo).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"official Kronos source not found: {repo}")
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosPredictor, KronosTokenizer

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = KronosTokenizer.from_pretrained(str((PROJECT_ROOT / args.tokenizer_path).resolve()))
    model = Kronos.from_pretrained(str((PROJECT_ROOT / args.model_path).resolve()))
    return KronosPredictor(model, tokenizer, device=device, max_context=512, clip=5)


def eligible_origins(df: pd.DataFrame, period: pd.Period, lookback: int, target_end: pd.Timestamp) -> list[int]:
    indices: list[int] = []
    for pos, timestamp in enumerate(df.index):
        if timestamp.to_period("M") != period:
            continue
        target_pos = pos + 5
        if pos - lookback + 1 < 0 or target_pos >= len(df):
            continue
        if df.index[target_pos] > target_end:
            continue
        indices.append(pos)
    return indices


def predict_origins(
    predictor,
    df: pd.DataFrame,
    origins: list[int],
    params: Params,
    samples: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for batch_start in range(0, len(origins), batch_size):
        batch = origins[batch_start:batch_start + batch_size]
        histories = [df.iloc[pos - params.lookback + 1:pos + 1].copy() for pos in batch]
        # The official predictor's timestamp helper uses the pandas ``.dt`` accessor,
        # so timestamps must be Series rather than DatetimeIndex instances.
        history_times = [pd.Series(history.index) for history in histories]
        future_times = [pd.Series(df.index[pos + 1:pos + 6]) for pos in batch]
        predictions = predictor.predict_batch(
            histories,
            history_times,
            future_times,
            pred_len=5,
            T=params.temperature,
            top_k=0,
            top_p=params.top_p,
            sample_count=samples,
            verbose=False,
        )
        for pos, prediction in zip(batch, predictions):
            last_close = float(df["close"].iloc[pos])
            rows.append(
                {
                    "origin_date": df.index[pos].strftime("%Y-%m-%d"),
                    "entry_date": df.index[pos + 1].strftime("%Y-%m-%d"),
                    "exit_date": df.index[pos + 5].strftime("%Y-%m-%d"),
                    "last_close": last_close,
                    "pred_close_t5": float(prediction["close"].iloc[-1]),
                    "actual_close_t5": float(df["close"].iloc[pos + 5]),
                    "entry_open": float(df["open"].iloc[pos + 1]),
                    "actual_return_t5": float(df["close"].iloc[pos + 5] / last_close - 1),
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["pred_return_t5"] = output["pred_close_t5"] / output["last_close"] - 1
    return output


def choose_params(
    predictor,
    signal_df: pd.DataFrame,
    execution_df: pd.DataFrame | None,
    validation_month: pd.Period,
    args: argparse.Namespace,
) -> tuple[Params, pd.DataFrame]:
    validation_end = validation_month.end_time.normalize()
    candidates: list[dict] = []
    for number, values in enumerate(itertools.product([0.3, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.9], [30, 60, 90])):
        params = Params(*values)
        origins = eligible_origins(signal_df, validation_month, params.lookback, validation_end)
        predictions = predict_origins(
            predictor, signal_df, origins, params, args.samples, args.batch_size, args.seed
        )
        predictions = attach_execution_prices(predictions, execution_df)
        if len(predictions) < 5:
            spearman = float("nan")
            net_return = float("nan")
        else:
            spearman = float(spearmanr(predictions["pred_close_t5"], predictions["actual_close_t5"]).statistic)
            _, validation_metrics = simulate_non_overlapping_trades(predictions, args)
            net_return = float(validation_metrics["net_compounded_return"])
        candidates.append(
            {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "lookback": params.lookback,
                "spearman": spearman,
                "net_return": net_return,
                "n_predictions": len(predictions),
                "n_execution_prices": int(predictions["execution_exit_close"].notna().sum()),
            }
        )
    grid = pd.DataFrame(candidates).sort_values(args.selection_metric, ascending=False, na_position="last")
    winner = grid.iloc[0]
    return Params(float(winner.temperature), float(winner.top_p), int(winner.lookback)), grid


def simulate_non_overlapping_trades(predictions: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    equity = 1.0
    next_available = ""
    trades: list[dict] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    for row in predictions.sort_values("origin_date").itertuples(index=False):
        if row.origin_date < next_available or row.pred_return_t5 <= args.buy_threshold:
            continue
        if pd.isna(row.execution_entry_open) or pd.isna(row.execution_exit_close):
            continue
        gross = row.execution_exit_close / row.execution_entry_open - 1
        net_factor = (row.execution_exit_close / row.execution_entry_open) * (1 - args.commission_rate - args.stamp_tax_rate - args.slippage_rate) / (1 + args.commission_rate + args.slippage_rate)
        net = net_factor - 1
        equity *= net_factor
        next_available = row.exit_date
        gross_returns.append(gross)
        net_returns.append(net)
        trades.append({
            "signal_date": row.origin_date,
            "entry_date": row.entry_date,
            "exit_date": row.exit_date,
            "pred_return_t5": row.pred_return_t5,
            "execution_entry_open": row.execution_entry_open,
            "execution_exit_close": row.execution_exit_close,
            "actual_return_from_entry": gross,
            "net_return_after_cost": net,
            "equity_after_trade": equity,
        })
    trade_df = pd.DataFrame(trades)
    metrics = {
        "trade_count": len(trades),
        "gross_compounded_return": float(np.prod([1 + item for item in gross_returns]) - 1) if gross_returns else 0.0,
        "net_compounded_return": equity - 1,
        "win_rate_before_cost": float(np.mean(np.array(gross_returns) > 0)) if gross_returns else 0.0,
        "mean_net_trade_return": float(np.mean(net_returns)) if net_returns else 0.0,
        "cost_assumption": {
            "commission_rate": args.commission_rate,
            "stamp_tax_rate": args.stamp_tax_rate,
            "slippage_rate": args.slippage_rate,
        },
    }
    return trade_df, metrics


def build_daily_equity(
    trades: pd.DataFrame,
    execution_df: pd.DataFrame,
    start_month: str,
    end_month: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    """Mark the long/cash strategy to market on every execution trading day."""
    start = pd.Timestamp(f"{start_month}-01")
    end = pd.Period(end_month, freq="M").end_time.normalize()
    prices = execution_df.loc[(execution_df.index >= start) & (execution_df.index <= end)]
    if prices.empty:
        raise ValueError("execution data has no prices in the requested test range")

    entries = {}
    exits = {}
    if not trades.empty:
        entries = {row.entry_date: row for row in trades.itertuples(index=False)}
        exits = {row.exit_date: row for row in trades.itertuples(index=False)}
        if len(entries) != len(trades) or len(exits) != len(trades):
            raise ValueError("overlapping trades share an entry or exit date")

    buy_cost = args.commission_rate + args.slippage_rate
    sell_cost = args.commission_rate + args.stamp_tax_rate + args.slippage_rate
    cash = 1.0
    units = 0.0
    active_entry = None
    rows = []

    for date, price_row in prices.iterrows():
        date_text = date.strftime("%Y-%m-%d")
        if date_text in entries:
            if units != 0.0:
                raise ValueError(f"attempted to enter while already invested: {date_text}")
            trade = entries[date_text]
            entry_open = float(price_row.open)
            if not math.isclose(entry_open, float(trade.execution_entry_open), rel_tol=1e-9):
                raise ValueError(f"entry price mismatch on {date_text}")
            units = cash / (entry_open * (1.0 + buy_cost))
            cash = 0.0
            active_entry = trade.entry_date

        exited = False
        if date_text in exits:
            if units == 0.0:
                raise ValueError(f"attempted to exit without a position: {date_text}")
            trade = exits[date_text]
            exit_close = float(price_row.close)
            if not math.isclose(exit_close, float(trade.execution_exit_close), rel_tol=1e-9):
                raise ValueError(f"exit price mismatch on {date_text}")
            cash = units * exit_close * (1.0 - sell_cost)
            units = 0.0
            active_entry = None
            exited = True

        equity = cash if units == 0.0 else units * float(price_row.close)
        rows.append({
            "date": date_text,
            "close": float(price_row.close),
            "equity": equity,
            "in_market": bool(units > 0.0 or exited),
            "active_entry_date": active_entry,
        })

    daily = pd.DataFrame(rows)
    daily["daily_return"] = daily["equity"].div(
        daily["equity"].shift(fill_value=1.0)
    ).sub(1.0)
    daily["benchmark_equity"] = daily["close"] / daily["close"].iloc[0]
    daily["benchmark_daily_return"] = daily["benchmark_equity"].pct_change().fillna(0.0)
    expected_final = 1.0 if trades.empty else float(trades["equity_after_trade"].iloc[-1])
    if not math.isclose(float(daily["equity"].iloc[-1]), expected_final, rel_tol=1e-9):
        raise RuntimeError("daily equity does not reconcile with compounded trade equity")

    drawdown = daily["equity"] / daily["equity"].cummax() - 1.0
    benchmark_drawdown = daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1.0
    volatility = float(daily["daily_return"].std(ddof=0))
    benchmark_volatility = float(daily["benchmark_daily_return"].std(ddof=0))
    periods = max(len(daily) - 1, 1)
    annualized_return = float(daily["equity"].iloc[-1] ** (252 / periods) - 1.0)
    benchmark_annualized_return = float(
        daily["benchmark_equity"].iloc[-1] ** (252 / periods) - 1.0
    )
    metrics = {
        "trading_days": int(len(daily)),
        "annualized_return": annualized_return,
        "annualized_volatility": volatility * math.sqrt(252),
        "sharpe_zero_rf": (
            float(daily["daily_return"].mean() / volatility * math.sqrt(252))
            if volatility > 0
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
        "in_market_days": int(daily["in_market"].sum()),
        "in_market_ratio": float(daily["in_market"].mean()),
        "benchmark_annualized_return": benchmark_annualized_return,
        "benchmark_annualized_volatility": benchmark_volatility * math.sqrt(252),
        "benchmark_max_drawdown": float(benchmark_drawdown.min()),
        "total_relative_excess_return": float(
            daily["equity"].iloc[-1] / daily["benchmark_equity"].iloc[-1] - 1.0
        ),
    }
    return daily, metrics


def main() -> None:
    args = parse_args()
    if args.frozen_manifest:
        apply_frozen_manifest(args)
    data_path = (PROJECT_ROOT / args.data).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_df = load_ohlcv_data(data_path)
    execution_df = None
    if args.execution_data:
        execution_df = load_ohlcv_data((PROJECT_ROOT / args.execution_data).resolve())
    predictor = make_predictor(args)
    all_test_predictions: list[pd.DataFrame] = []
    monthly_rows: list[dict] = []

    for test_month in month_range(args.start_month, args.end_month):
        validation_month = test_month - 1
        print(f"Selecting parameters for {test_month} from {validation_month} ...", flush=True)
        params, grid = choose_params(predictor, signal_df, execution_df, validation_month, args)
        grid.to_csv(output_dir / f"{test_month}_grid.csv", index=False)
        test_end = test_month.end_time.normalize()
        origins = eligible_origins(signal_df, test_month, params.lookback, test_end)
        print(f"Predicting {test_month}: {len(origins)} eligible signals with {params}", flush=True)
        predictions = predict_origins(predictor, signal_df, origins, params, args.samples, args.batch_size, args.seed)
        predictions = attach_execution_prices(predictions, execution_df)
        predictions["test_month"] = str(test_month)
        predictions["temperature"] = params.temperature
        predictions["top_p"] = params.top_p
        predictions["lookback"] = params.lookback
        all_test_predictions.append(predictions)
        monthly_rows.append({
            "test_month": str(test_month),
            "validation_month": str(validation_month),
            "temperature": params.temperature,
            "top_p": params.top_p,
            "lookback": params.lookback,
            "selection_metric": args.selection_metric,
            "validation_selection_score": float(grid.iloc[0][args.selection_metric]),
            "validation_spearman": float(grid.iloc[0].spearman),
            "validation_net_return": float(grid.iloc[0].net_return),
            "validation_predictions": int(grid.iloc[0].n_predictions),
            "test_predictions": len(predictions),
        })

    prediction_df = pd.concat(all_test_predictions, ignore_index=True)
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(output_dir / "monthly_params.csv", index=False)
    trades, metrics = simulate_non_overlapping_trades(prediction_df, args)
    trades.to_csv(output_dir / "trades.csv", index=False)
    benchmark_df = execution_df if execution_df is not None else signal_df
    daily_equity, daily_metrics = build_daily_equity(
        trades, benchmark_df, args.start_month, args.end_month, args
    )
    daily_equity.to_csv(output_dir / "daily_equity.csv", index=False)
    metrics.update(daily_metrics)
    benchmark = benchmark_df.loc[(benchmark_df.index >= pd.Timestamp(f"{args.start_month}-01")) & (benchmark_df.index <= pd.Period(args.end_month, freq="M").end_time.normalize()), "close"]
    if benchmark.empty:
        raise ValueError("execution data has no prices in the requested test range")
    metrics["benchmark_buy_and_hold_return"] = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
    metrics["prediction_spearman"] = float(spearmanr(prediction_df["pred_close_t5"], prediction_df["actual_close_t5"]).statistic)
    metrics["directional_accuracy"] = float(np.mean((prediction_df["pred_return_t5"] > 0) == (prediction_df["actual_return_t5"] > 0)))
    metrics["signal_data_range"] = [signal_df.index.min().strftime("%Y-%m-%d"), signal_df.index.max().strftime("%Y-%m-%d")]
    metrics["execution_data_range"] = [benchmark_df.index.min().strftime("%Y-%m-%d"), benchmark_df.index.max().strftime("%Y-%m-%d")]
    metrics["execution_data"] = str((PROJECT_ROOT / args.execution_data).resolve()) if args.execution_data else None
    metrics["test_range"] = [args.start_month, args.end_month]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
