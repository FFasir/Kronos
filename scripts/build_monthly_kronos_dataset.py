#!/usr/bin/env python3
"""Build leakage-safe 3-month/1-month datasets for monthly Kronos fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_COLUMNS = ["open", "high", "low", "close", "vol", "amt"]
LIMIT_COLUMNS = ["limit_up", "limit_down"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one monthly Kronos rolling dataset")
    parser.add_argument("--source", default="data/official_processed_datasets_all_a_share")
    parser.add_argument("--target-month", required=True, help="Prediction month, for example 2025-10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--validation-days", type=int, default=22)
    parser.add_argument("--lookback", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--force", action="store_true")
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


def load_source(source: Path) -> tuple[dict[str, pd.DataFrame], list[Path], int, dict]:
    paths = [source / f"{split}_data.pkl" for split in ("train", "val", "test")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"rolling source splits are missing: {missing}")
    source_metadata_path = source / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"rolling source metadata is missing: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("adjustment") != "hfq":
        raise ValueError("A-share rolling fine-tuning requires an HFQ source dataset")
    split_data = []
    for path in paths:
        with path.open("rb") as handle:
            split_data.append(pickle.load(handle))

    symbols = sorted(set().union(*(split.keys() for split in split_data)))
    combined: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames = [split[symbol] for split in split_data if symbol in split and not split[symbol].empty]
        if not frames:
            continue
        frame = pd.concat(frames).sort_index()
        frame.index = pd.to_datetime(frame.index)
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.rename(columns={"volume": "vol", "amount": "amt"})
        if not all(column in frame for column in REQUIRED_COLUMNS):
            continue
        if not all(column in frame for column in LIMIT_COLUMNS):
            continue
        frame = frame[REQUIRED_COLUMNS + LIMIT_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if not frame.empty and (frame[REQUIRED_COLUMNS] > 0).all().all():
            frame.index.name = "datetime"
            combined[symbol] = frame
    return combined, paths + [source_metadata_path], len(symbols), source_metadata


def slice_for_target_dates(
    frame: pd.DataFrame,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    lookback: int,
    horizon: int,
) -> pd.DataFrame | None:
    target_positions = np.flatnonzero((frame.index >= target_start) & (frame.index <= target_end))
    if len(target_positions) < horizon:
        return None
    first_target = int(target_positions[0])
    last_target = int(target_positions[-1])
    if first_target < lookback:
        return None
    sliced = frame.iloc[first_target - lookback:last_target + 1].copy()
    if len(sliced) < lookback + horizon:
        return None
    return sliced


def build_monthly_dataset(
    source: Path,
    output: Path,
    target_month: str,
    train_months: int,
    validation_days: int,
    lookback: int,
    horizon: int,
    force: bool = False,
) -> dict:
    if train_months < 1 or validation_days < horizon + 1:
        raise ValueError("validation_days must be at least horizon + 1")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output}; pass --force to replace files")
    output.mkdir(parents=True, exist_ok=True)

    prediction_period = pd.Period(target_month, freq="M")
    prediction_start = prediction_period.start_time.normalize()
    train_start = prediction_start - pd.DateOffset(months=train_months)
    train_end = prediction_start - pd.Timedelta(days=1)

    data, source_paths, source_symbol_count, source_metadata = load_source(source)
    calendar = pd.DatetimeIndex(sorted({
        date
        for frame in data.values()
        for date in frame.index
        if train_start <= date <= train_end
    }))
    if len(calendar) <= validation_days + horizon:
        raise ValueError(f"insufficient trading dates in training window: {len(calendar)}")
    validation_dates = calendar[-validation_days:]
    validation_start = validation_dates[0]
    training_target_end = calendar[-validation_days - 1]

    train_data: dict[str, pd.DataFrame] = {}
    val_data: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        train_frame = slice_for_target_dates(
            frame, train_start, training_target_end, lookback, horizon
        )
        val_frame = slice_for_target_dates(
            frame, validation_start, train_end, lookback, horizon
        )
        if train_frame is not None and val_frame is not None:
            train_data[symbol] = train_frame
            val_data[symbol] = val_frame

    if not train_data:
        raise ValueError("no symbols satisfy the rolling dataset contract")

    window = lookback + horizon
    train_windows = sum(len(frame) - window + 1 for frame in train_data.values())
    val_windows = sum(len(frame) - window + 1 for frame in val_data.values())
    metadata = {
        "target_month": str(prediction_period),
        "prediction_start": prediction_start.strftime("%Y-%m-%d"),
        "train_months": train_months,
        "train_target_start": train_start.strftime("%Y-%m-%d"),
        "train_target_end": training_target_end.strftime("%Y-%m-%d"),
        "validation_target_start": validation_start.strftime("%Y-%m-%d"),
        "validation_target_end": train_end.strftime("%Y-%m-%d"),
        "lookback": lookback,
        "horizon": horizon,
        "adjustment": "hfq",
        "limit_features": LIMIT_COLUMNS,
        "first_listing_day_policy": source_metadata.get("first_listing_day_policy"),
        "split_context_policy": "lookback buffer; forecast targets strictly inside split_target_ranges",
        "split_target_ranges": {
            "train": [
                train_start.strftime("%Y-%m-%d"),
                training_target_end.strftime("%Y-%m-%d"),
            ],
            "val": [
                validation_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
            ],
        },
        "symbol_count": len(train_data),
        "source_symbol_count": source_symbol_count,
        "excluded_symbol_count": source_symbol_count - len(train_data),
        "exclusion_reason": "insufficient valid history for lookback+horizon in this window",
        "train_windows": train_windows,
        "val_windows": val_windows,
        "source": str(source),
        "source_sha256": {path.name: sha256(path) for path in source_paths},
    }

    with (output / "train_data.pkl").open("wb") as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output / "val_data.pkl").open("wb") as handle:
        pickle.dump(val_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_monthly_dataset(
        source=resolve_path(args.source),
        output=resolve_path(args.output),
        target_month=args.target_month,
        train_months=args.train_months,
        validation_days=args.validation_days,
        lookback=args.lookback,
        horizon=args.horizon,
        force=args.force,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
