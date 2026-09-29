#!/usr/bin/env python3
"""Generate and persist 20-day OHLC paths from a monthly Kronos checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_layer.probability_calibration import (
    apply_probability_calibration,
    normal_probability_up,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monthly Kronos path prediction")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--official-repo", default="third_party/Kronos")
    parser.add_argument("--lookback", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", help="Comma-separated A-share codes")
    parser.add_argument("--as-of-date", help="Ignore market data after this date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="data/outputs/monthly_predictions")
    parser.add_argument("--no-database", action="store_true")
    parser.add_argument("--calibration", help="Optional probability calibration JSON")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def make_predictor(repo: Path, tokenizer_path: Path, predictor_path: Path, device: str):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(predictor_path))
    return KronosPredictor(model, tokenizer, device=device, max_context=512, clip=5)


def prepare_frame(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    result = frame[columns].iloc[-lookback:].copy()
    if "amount" in frame:
        result["amount"] = pd.to_numeric(frame["amount"].iloc[-lookback:], errors="coerce")
    else:
        result["amount"] = np.nan
    result["amount"] = result["amount"].where(
        result["amount"] > 0,
        result["close"] * result["volume"],
    )
    if len(result) != lookback or result.isna().any().any() or (result <= 0).any().any():
        raise ValueError("insufficient or invalid OHLCVA history")
    return result


def probability_up(pred_close: float, last_close: float, pred_std: float) -> float:
    return normal_probability_up(pred_close, last_close, pred_std)


def persist(storage, model_rows: list[tuple], path_rows: list[tuple], as_of_date: str, version: str) -> None:
    storage.execute_write(
        "DELETE FROM price_prediction WHERE as_of_date=? AND model_version=?",
        (as_of_date, version),
    )
    storage.execute_many(
        """INSERT INTO model_prediction
           (code, date, pred_date, pred_close, pred_return, last_close,
            prediction_std, score, confidence, temperature, top_p, lookback, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(code, date, pred_date) DO UPDATE SET
             pred_close=excluded.pred_close, pred_return=excluded.pred_return,
             last_close=excluded.last_close, prediction_std=excluded.prediction_std,
             score=excluded.score, confidence=excluded.confidence,
             temperature=excluded.temperature,
             top_p=excluded.top_p, lookback=excluded.lookback,
             model_version=excluded.model_version""",
        model_rows,
    )
    storage.execute_many(
        """INSERT INTO price_prediction
           (code, created_at, pred_horizon, pred_price, pred_lower, pred_upper,
            model_version, as_of_date, pred_date, pred_open, pred_high,
            pred_low, pred_close, prediction_std)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        path_rows,
    )


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("monthly probability and intervals require at least two samples")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    calibration_path = resolve_path(args.calibration) if args.calibration else None
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path else {"enabled": False, "method": "identity"}
    )

    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(load_config())
    storage.init_database()
    symbols = (
        [item.strip().zfill(6) for item in args.symbols.split(",") if item.strip()]
        if args.symbols else storage.list_stock_symbols()
    )
    if args.limit:
        symbols = symbols[:args.limit]

    predictor = make_predictor(
        resolve_path(args.official_repo),
        resolve_path(args.tokenizer_path),
        resolve_path(args.predictor_path),
        args.device,
    )
    pending = []
    failures = {}
    for symbol in symbols:
        try:
            raw = storage.load_raw_data(symbol)
            if args.as_of_date:
                raw = raw.loc[:pd.Timestamp(args.as_of_date)]
            pending.append((symbol, prepare_frame(raw, args.lookback)))
        except Exception as exc:
            failures[symbol] = str(exc)

    if pending:
        batch_as_of = max(pd.Timestamp(frame.index[-1]) for _, frame in pending)
        current, stale = [], []
        for item in pending:
            if pd.Timestamp(item[1].index[-1]) == batch_as_of:
                current.append(item)
            else:
                stale.append(item)
        pending = current
        for symbol, frame in stale:
            failures[symbol] = f"stale market data ending {pd.Timestamp(frame.index[-1]).date()}"

    generated_at = datetime.now(timezone.utc).isoformat()
    model_rows = []
    path_rows = []
    prediction_records = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        frames = [item[1] for item in batch]
        future_dates = [
            pd.Series(pd.bdate_range(frame.index[-1] + pd.offsets.BDay(1), periods=args.horizon))
            for frame in frames
        ]
        try:
            forecasts, forecast_stds = predictor.predict_batch(
                frames,
                [pd.Series(frame.index) for frame in frames],
                future_dates,
                pred_len=args.horizon,
                T=args.temperature,
                top_k=0,
                top_p=args.top_p,
                sample_count=args.samples,
                verbose=False,
                return_std=True,
            )
        except Exception as exc:
            for symbol, _ in batch:
                failures[symbol] = str(exc)
            continue

        for (symbol, frame), forecast, forecast_std, dates in zip(
            batch, forecasts, forecast_stds, future_dates
        ):
            last_close = float(frame["close"].iloc[-1])
            as_of_date = pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d")
            final_close = float(forecast["close"].iloc[-1])
            final_std = float(forecast_std["close"].iloc[-1])
            final_date = pd.Timestamp(dates.iloc[-1]).strftime("%Y-%m-%d")
            expected_return = final_close / last_close - 1
            raw_confidence = probability_up(final_close, last_close, final_std)
            confidence = float(apply_probability_calibration([raw_confidence], calibration)[0])
            score = float(np.clip(5 + expected_return * 100, 0, 10))
            model_rows.append((
                symbol, as_of_date, final_date, final_close, expected_return,
                last_close, final_std, score, confidence, args.temperature, args.top_p,
                args.lookback, args.model_version,
            ))
            prediction_records.append({
                "symbol": symbol,
                "as_of_date": as_of_date,
                "pred_date": final_date,
                "pred_close": final_close,
                "pred_return": expected_return,
                "raw_prob_up": raw_confidence,
                "prob_up": confidence,
                "score": score,
            })
            for horizon_index, (_, row), (_, std_row) in zip(
                range(1, args.horizon + 1), forecast.iterrows(), forecast_std.iterrows()
            ):
                close = float(row["close"])
                close_std = float(std_row["close"])
                pred_date = pd.Timestamp(dates.iloc[horizon_index - 1]).strftime("%Y-%m-%d")
                path_rows.append((
                    symbol, generated_at, horizon_index, close,
                    max(0.0, close - 1.96 * close_std), close + 1.96 * close_std,
                    args.model_version, as_of_date, pred_date,
                    float(row["open"]), float(row["high"]), float(row["low"]),
                    close, close_std,
                ))
        print(f"Predicted {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)

    if not model_rows:
        raise RuntimeError(f"no monthly predictions succeeded; failures={len(failures)}")
    if not args.no_database:
        as_of_dates = {row[1] for row in model_rows}
        if len(as_of_dates) != 1:
            raise ValueError(f"prediction batch has inconsistent market dates: {sorted(as_of_dates)}")
        persist(storage, model_rows, path_rows, next(iter(as_of_dates)), args.model_version)

    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": generated_at,
        "model_version": args.model_version,
        "requested_symbols": len(symbols),
        "successful_symbols": len(model_rows),
        "failed_symbols": len(failures),
        "lookback": args.lookback,
        "horizon": args.horizon,
        "sample_count": args.samples,
        "calibration_path": str(calibration_path) if calibration_path else None,
        "calibration_enabled": bool(calibration.get("enabled")),
        "calibration_method": calibration.get("method", "identity"),
        "failures": failures,
    }
    (output / f"{args.model_version}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(prediction_records).to_csv(
        output / f"{args.model_version}_predictions.csv", index=False
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
