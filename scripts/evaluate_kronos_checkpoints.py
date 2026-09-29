#!/usr/bin/env python3
"""Compare Kronos checkpoints on deterministic held-out return forecasts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kronos checkpoints on held-out returns")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_data(dataset_path: Path) -> dict[str, pd.DataFrame]:
    splits = []
    for split in ("train", "val", "test"):
        with (dataset_path / f"{split}_data.pkl").open("rb") as handle:
            splits.append(pickle.load(handle))

    symbols = sorted(set().union(*(split.keys() for split in splits)))
    combined = {}
    for symbol in symbols:
        frames = [split[symbol] for split in splits if symbol in split and not split[symbol].empty]
        if not frames:
            continue
        frame = pd.concat(frames).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.rename(columns={"vol": "volume", "amt": "amount"})
        required = ["open", "high", "low", "close", "volume", "amount"]
        frame = frame[required].replace([np.inf, -np.inf], np.nan).dropna()
        if not frame.empty and (frame > 0).all().all():
            combined[symbol] = frame
    return combined


def evaluation_dates(data: dict[str, pd.DataFrame], start: str, end: str, per_month: int) -> list[pd.Timestamp]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    all_dates = sorted({date for frame in data.values() for date in frame.loc[start_ts:end_ts].index})
    calendar = pd.DatetimeIndex(all_dates)
    selected = []
    for _, dates in pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")):
        month_dates = list(dates.index)
        if len(month_dates) < 10:
            continue
        positions = np.linspace(4, len(month_dates) - 6, per_month, dtype=int)
        selected.extend(month_dates[position] for position in sorted(set(positions)))
    return selected


def select_symbols(
    data: dict[str, pd.DataFrame],
    dates: list[pd.Timestamp],
    lookback: int,
    horizon: int,
    count: int,
    seed: int,
) -> list[str]:
    eligible = []
    global_calendar = pd.DatetimeIndex(sorted({date for frame in data.values() for date in frame.index}))
    target_dates = {}
    for date in dates:
        position = global_calendar.get_loc(date)
        target_dates[date] = global_calendar[position + horizon]

    for symbol, frame in data.items():
        valid = True
        for date in dates:
            if date not in frame.index or target_dates[date] not in frame.index:
                valid = False
                break
            if frame.index.get_loc(date) < lookback - 1:
                valid = False
                break
        if valid:
            eligible.append(symbol)
    eligible.sort(key=lambda symbol: hashlib.sha256(f"{seed}:{symbol}".encode()).hexdigest())
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} symbols satisfy the evaluation contract; requested {count}")
    return sorted(eligible[:count])


def build_tasks(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: list[pd.Timestamp],
    lookback: int,
    horizon: int,
) -> list[dict]:
    global_calendar = pd.DatetimeIndex(sorted({date for frame in data.values() for date in frame.index}))
    tasks = []
    for origin_date in dates:
        calendar_position = global_calendar.get_loc(origin_date)
        future_dates = global_calendar[calendar_position + 1:calendar_position + horizon + 1]
        target_date = future_dates[-1]
        for symbol in symbols:
            frame = data[symbol]
            origin_position = frame.index.get_loc(origin_date)
            history = frame.iloc[origin_position - lookback + 1:origin_position + 1].copy()
            last_close = float(frame.loc[origin_date, "close"])
            actual_close = float(frame.loc[target_date, "close"])
            tasks.append({
                "symbol": symbol,
                "origin_date": origin_date,
                "target_date": target_date,
                "future_dates": future_dates,
                "history": history,
                "last_close": last_close,
                "actual_close_t5": actual_close,
                "actual_return_t5": actual_close / last_close - 1,
            })
    return tasks


def make_predictor(repo: Path, tokenizer_path: Path, model_path: Path, device: str):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(model_path))
    return KronosPredictor(model, tokenizer, device=device, max_context=512, clip=5)


def predict_tasks(predictor, tasks: list[dict], settings: dict) -> pd.DataFrame:
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])

    rows = []
    batch_size = settings["batch_size"]
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        predictions = predictor.predict_batch(
            [task["history"] for task in batch],
            [pd.Series(task["history"].index) for task in batch],
            [pd.Series(task["future_dates"]) for task in batch],
            pred_len=settings["horizon"],
            T=settings["temperature"],
            top_k=0,
            top_p=settings["top_p"],
            sample_count=settings["sample_count"],
            verbose=False,
        )
        for task, prediction in zip(batch, predictions):
            pred_close = float(prediction["close"].iloc[-1])
            rows.append({
                "symbol": task["symbol"],
                "origin_date": task["origin_date"].strftime("%Y-%m-%d"),
                "target_date": task["target_date"].strftime("%Y-%m-%d"),
                "last_close": task["last_close"],
                "pred_close_t5": pred_close,
                "actual_close_t5": task["actual_close_t5"],
                "pred_return_t5": pred_close / task["last_close"] - 1,
                "actual_return_t5": task["actual_return_t5"],
            })
        print(f"Predicted {min(start + batch_size, len(tasks))}/{len(tasks)}", flush=True)
    return pd.DataFrame(rows)


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def metrics(predictions: pd.DataFrame) -> dict:
    daily_ics = []
    for date, group in predictions.groupby("origin_date"):
        daily_ics.append({
            "origin_date": date,
            "return_ic": safe_spearman(group["pred_return_t5"], group["actual_return_t5"]),
            "price_spearman": safe_spearman(group["pred_close_t5"], group["actual_close_t5"]),
            "count": len(group),
        })
    valid_ics = [row["return_ic"] for row in daily_ics if row["return_ic"] is not None]
    errors = predictions["pred_return_t5"] - predictions["actual_return_t5"]
    actual_up = predictions["actual_return_t5"] > 0
    predicted_up = predictions["pred_return_t5"] > 0
    return {
        "prediction_count": len(predictions),
        "evaluation_dates": len(daily_ics),
        "mean_daily_return_ic": float(np.mean(valid_ics)),
        "median_daily_return_ic": float(np.median(valid_ics)),
        "positive_daily_ic_ratio": float(np.mean(np.array(valid_ics) > 0)),
        "global_return_ic": safe_spearman(predictions["pred_return_t5"], predictions["actual_return_t5"]),
        "directional_accuracy": float(np.mean(predicted_up == actual_up)),
        "always_up_accuracy": float(np.mean(actual_up)),
        "return_mae": float(np.mean(np.abs(errors))),
        "zero_return_mae": float(np.mean(np.abs(predictions["actual_return_t5"]))),
        "return_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_predicted_return": float(predictions["pred_return_t5"].mean()),
        "mean_actual_return": float(predictions["actual_return_t5"].mean()),
        "daily_metrics": daily_ics,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["settings"]
    dataset_path = resolve_path(config["dataset_path"])
    repo = resolve_path(config.get("official_repo", "third_party/Kronos"))
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_split_data(dataset_path)
    dates = evaluation_dates(data, settings["start_date"], settings["end_date"], settings["dates_per_month"])
    symbols = select_symbols(
        data, dates, settings["lookback"], settings["horizon"],
        settings["symbol_count"], settings["seed"],
    )
    tasks = build_tasks(data, symbols, dates, settings["lookback"], settings["horizon"])
    contract = {
        "config": str(config_path),
        "research_status": config["research_status"],
        "dataset_path": str(dataset_path),
        "symbols": symbols,
        "dates": [date.strftime("%Y-%m-%d") for date in dates],
        "settings": settings,
        "task_count": len(tasks),
    }
    (output_dir / "evaluation_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries = {}
    for model_config in config["models"]:
        name = model_config["name"]
        tokenizer_path = resolve_path(model_config["tokenizer_path"])
        model_path = resolve_path(model_config["model_path"])
        required = [tokenizer_path / "model.safetensors", model_path / "model.safetensors"]
        if not all(path.is_file() for path in required):
            if model_config.get("optional", False):
                print(f"Skipping optional model with missing checkpoint: {name}", flush=True)
                continue
            raise FileNotFoundError(f"checkpoint is missing for {name}: {required}")

        print(f"Evaluating {name}", flush=True)
        predictor = make_predictor(repo, tokenizer_path, model_path, args.device)
        predictions = predict_tasks(predictor, tasks, settings)
        predictions.insert(0, "model", name)
        predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        summary = metrics(predictions)
        summary["tokenizer_sha256"] = sha256(required[0])
        summary["predictor_sha256"] = sha256(required[1])
        summaries[name] = summary
        (output_dir / f"{name}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = []
    for name, summary in summaries.items():
        comparison.append({key: value for key, value in summary.items() if key != "daily_metrics"} | {"model": name})
    comparison_df = pd.DataFrame(comparison)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("mean_daily_return_ic", ascending=False)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    print(comparison_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
