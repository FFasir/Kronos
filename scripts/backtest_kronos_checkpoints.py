#!/usr/bin/env python3
"""Run reproducible sparse-rebalance Qlib backtests for Kronos checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Kronos checkpoints with a Qlib TopK backtest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default="", help="Comma-separated model names; default is all")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--backtest-only", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_data(dataset_path: Path) -> dict[str, pd.DataFrame]:
    with (dataset_path / "test_data.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    result = {}
    required = ["open", "high", "low", "close", "volume", "amount"]
    for symbol, frame in raw.items():
        clean = frame.rename(columns={"vol": "volume", "amt": "amount"})
        clean = clean[required].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]
        if not clean.empty:
            result[symbol] = clean
    return result


def trading_calendar(data: dict[str, pd.DataFrame], start: str, end: str) -> pd.DatetimeIndex:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = {date for frame in data.values() for date in frame.loc[start_ts:end_ts].index}
    return pd.DatetimeIndex(sorted(dates))


def rebalance_dates(calendar: pd.DatetimeIndex, horizon: int) -> list[pd.Timestamp]:
    # Leave a full forecast horizon inside the sample and trade on the next day.
    return list(calendar[: max(0, len(calendar) - horizon) : horizon])


def make_predictor(repo: Path, tokenizer_path: Path, model_path: Path, device: str):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(model_path))
    return KronosPredictor(model, tokenizer, device=device, max_context=512, clip=5)


def existing_dates(path: Path) -> set[pd.Timestamp]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    values = pd.read_csv(path, usecols=["datetime"])["datetime"].drop_duplicates()
    return set(pd.to_datetime(values))


def append_signals(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def generate_signals(
    predictor,
    data: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    dates: list[pd.Timestamp],
    output_path: Path,
    settings: dict,
) -> dict:
    completed = existing_dates(output_path)
    symbols = sorted(data)
    total_predictions = 0
    started = time.time()

    for date_number, origin_date in enumerate(dates, start=1):
        if origin_date in completed:
            continue
        calendar_position = calendar.get_loc(origin_date)
        future_dates = calendar[calendar_position + 1 : calendar_position + settings["horizon"] + 1]
        rows = []
        batch_frames, batch_symbols, batch_closes = [], [], []

        def flush_batch() -> None:
            nonlocal total_predictions
            if not batch_frames:
                return
            predictions = predictor.predict_batch(
                batch_frames,
                [pd.Series(frame.index) for frame in batch_frames],
                [pd.Series(future_dates) for _ in batch_frames],
                pred_len=settings["horizon"],
                T=settings["temperature"],
                top_k=0,
                top_p=settings["top_p"],
                sample_count=settings["sample_count"],
                verbose=False,
            )
            for symbol, last_close, prediction in zip(batch_symbols, batch_closes, predictions):
                score = float(prediction["close"].iloc[-1]) / last_close - 1.0
                if np.isfinite(score):
                    rows.append({
                        "datetime": origin_date.strftime("%Y-%m-%d"),
                        "instrument": symbol,
                        "score": score,
                    })
            total_predictions += len(predictions)
            batch_frames.clear()
            batch_symbols.clear()
            batch_closes.clear()

        for symbol in symbols:
            frame = data[symbol]
            if origin_date not in frame.index:
                continue
            position = frame.index.get_loc(origin_date)
            if not isinstance(position, (int, np.integer)) or position < settings["lookback"] - 1:
                continue
            history = frame.iloc[position - settings["lookback"] + 1 : position + 1]
            last_close = float(history["close"].iloc[-1])
            if not np.isfinite(last_close) or last_close <= 0:
                continue
            batch_frames.append(history)
            batch_symbols.append(symbol)
            batch_closes.append(last_close)
            if len(batch_frames) >= settings["batch_size"]:
                flush_batch()
        flush_batch()
        append_signals(output_path, rows)
        elapsed = time.time() - started
        print(
            f"[{date_number}/{len(dates)}] {origin_date.date()} signals={len(rows)} "
            f"new_predictions={total_predictions} elapsed={elapsed / 60:.1f}m",
            flush=True,
        )

    signal_count = sum(1 for _ in output_path.open("rb")) - 1
    return {
        "signal_file": str(output_path),
        "signal_count": signal_count,
        "rebalance_dates": len(existing_dates(output_path)),
        "new_predictions": total_predictions,
        "inference_seconds": time.time() - started,
    }


def performance_metrics(report: pd.DataFrame, annual_days: int = 252) -> dict:
    gross = report["return"].fillna(0.0)
    cost = report["cost"].fillna(0.0)
    net = gross - cost
    equity = (1.0 + net).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (annual_days / len(net)) - 1.0)
    annual_volatility = float(net.std(ddof=1) * np.sqrt(annual_days))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else None
    drawdown = equity / equity.cummax() - 1.0
    benchmark = report["bench"].fillna(0.0)
    benchmark_total_return = float((1.0 + benchmark).prod() - 1.0)
    return {
        "trading_days": int(len(report)),
        "total_return": total_return,
        "benchmark_total_return": benchmark_total_return,
        "total_excess_return": float(
            (1.0 + total_return) / (1.0 + benchmark_total_return) - 1.0
        ),
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe_zero_rf": None if sharpe is None else float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "total_cost_return": float(cost.sum()),
        "mean_daily_turnover": float(report["turnover"].fillna(0.0).mean()),
        "final_account_value": float(report["account"].iloc[-1]),
    }


def run_backtest(signal_path: Path, output_dir: Path, model_name: str, settings: dict) -> dict:
    import qlib
    from qlib.backtest import backtest, executor
    from qlib.config import REG_CN
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D

    qlib.init(provider_uri=str(resolve_path(settings["qlib_data_path"])), region=REG_CN)
    benchmark_frame = D.features(
        D.instruments("all_a_share"),
        ["$close/Ref($close,1)-1"],
        start_time=settings["start_date"],
        end_time=settings["end_date"],
        freq="day",
    )
    benchmark_series = (
        benchmark_frame.iloc[:, 0]
        .replace([np.inf, -np.inf], np.nan)
        .groupby(level="datetime")
        .mean()
        .fillna(0.0)
        .sort_index()
    )
    provider_calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    requested_end = pd.Timestamp(settings["end_date"])
    effective_end = (
        provider_calendar[-2]
        if requested_end >= provider_calendar[-1]
        else requested_end
    )
    signals = pd.read_csv(signal_path, parse_dates=["datetime"])
    signal_series = signals.set_index(["instrument", "datetime"])["score"].sort_index()
    strategy = TopkDropoutStrategy(
        topk=settings["topk"],
        n_drop=settings["n_drop"],
        hold_thresh=settings["hold_thresh"],
        signal=signal_series,
        only_tradable=False,
        # The Exchange below enforces directional limits. Keeping this False
        # permits the legal cases: selling at limit-up and buying at limit-down.
        forbid_all_trade_at_limit=False,
    )
    exchange_kwargs = {
        "codes": D.instruments("all_a_share"),
        "freq": "day",
        "limit_threshold": ("$limit_up > 0.5", "$limit_down > 0.5"),
        "deal_price": "open",
        "open_cost": settings["open_cost"],
        "close_cost": settings["close_cost"],
        "min_cost": settings["min_cost"],
    }
    portfolio, indicators = backtest(
        start_time=settings["start_date"],
        end_time=effective_end,
        strategy=strategy,
        executor=executor.SimulatorExecutor(
            time_per_step="day", generate_portfolio_metrics=True, delay_execution=True
        ),
        benchmark=benchmark_series,
        account=settings["account"],
        exchange_kwargs=exchange_kwargs,
    )
    report, positions = portfolio["1day"]
    report_path = output_dir / f"{model_name}_daily_report.csv"
    report.to_csv(report_path)
    with (output_dir / f"{model_name}_positions.pkl").open("wb") as handle:
        pickle.dump(positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metrics = performance_metrics(report)
    metrics.update({
        "model": model_name,
        "requested_end_date": settings["end_date"],
        "effective_end_date": effective_end.strftime("%Y-%m-%d"),
        "signal_count": int(len(signals)),
        "signal_dates": int(signals["datetime"].nunique()),
        "report_path": str(report_path),
        "exchange_limit_threshold": list(exchange_kwargs["limit_threshold"]),
        "benchmark": "dynamic all-A equal-weight return from the same HFQ provider",
        "strategy_forbid_all_trade_at_limit": False,
        "indicator_frequencies": sorted(indicators),
    })
    return metrics


def main() -> None:
    args = parse_args()
    if args.inference_only and args.backtest_only:
        raise ValueError("--inference-only and --backtest-only are mutually exclusive")
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["settings"]
    settings["qlib_data_path"] = config["qlib_data_path"]
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {name for name in args.models.split(",") if name}
    models = [model for model in config["models"] if not selected or model["name"] in selected]
    if selected.difference(model["name"] for model in models):
        raise ValueError(f"unknown model names: {sorted(selected.difference(model['name'] for model in models))}")

    data = None
    calendar = None
    dates = None
    if not args.backtest_only:
        data = load_test_data(resolve_path(config["dataset_path"]))
        calendar = trading_calendar(data, settings["start_date"], settings["end_date"])
        dates = rebalance_dates(calendar, settings["horizon"])

    summaries = {}
    manifest = {
        "experiment_id": config["experiment_id"],
        "research_status": config["research_status"],
        "config": str(config_path),
        "dataset_path": str(resolve_path(config["dataset_path"])),
        "qlib_data_path": str(resolve_path(config["qlib_data_path"])),
        "settings": settings,
        "execution_policy": {
            "signal_at_close": True,
            "delay_execution": True,
            "deal_price": "open",
            "rebalance_every_trading_days": settings["horizon"],
            "limit_up_buy_blocked": True,
            "limit_down_sell_blocked": True,
            "limit_up_sell_allowed": True,
            "limit_down_buy_allowed": True,
        },
        "models": {},
    }

    for model_config in models:
        name = model_config["name"]
        tokenizer_path = resolve_path(model_config["tokenizer_path"])
        model_path = resolve_path(model_config["model_path"])
        tokenizer_weights = tokenizer_path / "model.safetensors"
        model_weights = model_path / "model.safetensors"
        if not tokenizer_weights.is_file() or not model_weights.is_file():
            raise FileNotFoundError(f"missing checkpoint for {name}")
        signal_path = output_dir / f"{name}_signals.csv"
        model_record = {
            "tokenizer_path": str(tokenizer_path),
            "model_path": str(model_path),
            "tokenizer_sha256": sha256(tokenizer_weights),
            "predictor_sha256": sha256(model_weights),
        }

        if not args.backtest_only:
            np.random.seed(settings["seed"])
            torch.manual_seed(settings["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(settings["seed"])
            print(f"Loading {name}", flush=True)
            predictor = make_predictor(
                resolve_path(config.get("official_repo", "third_party/Kronos")),
                tokenizer_path,
                model_path,
                args.device,
            )
            model_record["inference"] = generate_signals(
                predictor, data, calendar, dates, signal_path, settings
            )
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not args.inference_only:
            print(f"Backtesting {name}", flush=True)
            metrics = run_backtest(signal_path, output_dir, name, settings)
            summaries[name] = metrics
            model_record["backtest"] = metrics
            (output_dir / f"{name}_summary.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        manifest["models"][name] = model_record
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if summaries:
        comparison = pd.DataFrame(summaries.values()).sort_values("total_return", ascending=False)
        comparison.to_csv(output_dir / "comparison.csv", index=False)
        print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
