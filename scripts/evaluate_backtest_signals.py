#!/usr/bin/env python3
"""Evaluate cached Kronos signals against realized horizon returns."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure direction and cross-sectional ranking of cached backtest signals"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="", help="Comma-separated model names; default is all")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_closes(dataset_path: Path) -> dict[str, pd.Series]:
    with (dataset_path / "test_data.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    closes = {}
    for symbol, frame in raw.items():
        series = frame["close"].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if not series.empty:
            closes[symbol] = series.astype(float)
    return closes


def trading_calendar(closes: dict[str, pd.Series]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted({date for series in closes.values() for date in series.index}))


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def attach_actual_returns(
    signals: pd.DataFrame,
    closes: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    target_dates = {}
    for date in signals["datetime"].drop_duplicates():
        position = calendar.get_loc(date)
        if position + horizon < len(calendar):
            target_dates[date] = calendar[position + horizon]

    rows = []
    for row in signals.itertuples(index=False):
        target_date = target_dates.get(row.datetime)
        series = closes.get(row.instrument)
        if target_date is None or series is None:
            continue
        if row.datetime not in series.index or target_date not in series.index:
            continue
        origin_close = float(series.loc[row.datetime])
        target_close = float(series.loc[target_date])
        if origin_close <= 0 or not np.isfinite(origin_close + target_close):
            continue
        rows.append({
            "datetime": row.datetime,
            "target_date": target_date,
            "instrument": row.instrument,
            "score": float(row.score),
            "actual_return": target_close / origin_close - 1.0,
        })
    return pd.DataFrame(rows)


def direction_recall(predicted_up: pd.Series, actual_up: pd.Series, outcome: bool) -> float | None:
    mask = actual_up == outcome
    if not mask.any():
        return None
    return float((predicted_up[mask] == actual_up[mask]).mean())


def evaluate(frame: pd.DataFrame, topk: int) -> dict:
    daily_rows = []
    for date, group in frame.groupby("datetime", sort=True):
        ranked = group.nlargest(min(topk, len(group)), "score")
        daily_rows.append({
            "datetime": date.strftime("%Y-%m-%d"),
            "count": int(len(group)),
            "return_ic": safe_spearman(group["score"], group["actual_return"]),
            "topk_mean_return": float(ranked["actual_return"].mean()),
            "universe_mean_return": float(group["actual_return"].mean()),
            "topk_positive_ratio": float((ranked["actual_return"] > 0).mean()),
        })
    daily = pd.DataFrame(daily_rows)
    valid_ic = daily["return_ic"].dropna()
    predicted_up = frame["score"] > 0
    actual_up = frame["actual_return"] > 0
    up_recall = direction_recall(predicted_up, actual_up, True)
    down_recall = direction_recall(predicted_up, actual_up, False)
    balanced_accuracy = None
    if up_recall is not None and down_recall is not None:
        balanced_accuracy = (up_recall + down_recall) / 2.0
    return {
        "matched_predictions": int(len(frame)),
        "evaluation_dates": int(len(daily)),
        "mean_daily_return_ic": float(valid_ic.mean()),
        "median_daily_return_ic": float(valid_ic.median()),
        "positive_daily_ic_ratio": float((valid_ic > 0).mean()),
        "global_return_ic": safe_spearman(frame["score"], frame["actual_return"]),
        "directional_accuracy": float((predicted_up == actual_up).mean()),
        "always_up_accuracy": float(actual_up.mean()),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "balanced_directional_accuracy": balanced_accuracy,
        "mean_topk_horizon_return": float(daily["topk_mean_return"].mean()),
        "mean_universe_horizon_return": float(daily["universe_mean_return"].mean()),
        "mean_topk_excess_horizon_return": float(
            (daily["topk_mean_return"] - daily["universe_mean_return"]).mean()
        ),
        "mean_topk_positive_ratio": float(daily["topk_positive_ratio"].mean()),
        "daily_metrics": daily_rows,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    settings = config["settings"]
    output_dir = resolve_path(config["output_dir"])
    closes = load_closes(resolve_path(config["dataset_path"]))
    calendar = trading_calendar(closes)
    comparison = []
    selected = {name for name in args.models.split(",") if name}
    known = {model["name"] for model in config["models"]}
    if selected - known:
        raise ValueError(f"unknown model names: {sorted(selected - known)}")

    for model in config["models"]:
        name = model["name"]
        if selected and name not in selected:
            continue
        signal_path = output_dir / f"{name}_signals.csv"
        if not signal_path.is_file():
            print(f"Skipping missing signals: {name}", flush=True)
            continue
        signals = pd.read_csv(signal_path, parse_dates=["datetime"])
        expected_dates = len(signals["datetime"].drop_duplicates())
        evaluated = attach_actual_returns(signals, closes, calendar, settings["horizon"])
        summary = evaluate(evaluated, settings["topk"])
        summary.update({
            "model": name,
            "signal_dates": expected_dates,
            "horizon": settings["horizon"],
            "topk": settings["topk"],
        })
        detail_path = output_dir / f"{name}_signal_metrics.json"
        detail_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        comparison.append({key: value for key, value in summary.items() if key != "daily_metrics"})

    result = pd.DataFrame(comparison)
    if not result.empty:
        result = result.sort_values("mean_daily_return_ic", ascending=False)
    result.to_csv(output_dir / "signal_metrics_comparison.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
