#!/usr/bin/env python3
"""Fit a leakage-safe probability calibrator from a rolling validation window."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_layer.probability_calibration import (
    apply_probability_calibration,
    fit_probability_calibrator,
    normal_probability_up,
    probability_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate monthly Kronos probabilities")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-repo", default="third_party/Kronos")
    parser.add_argument("--lookback", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=1024)
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
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


def context_frame(frame: pd.DataFrame, position: int, lookback: int) -> pd.DataFrame:
    result = frame.iloc[position - lookback + 1:position + 1].copy()
    result = result.rename(columns={"vol": "volume", "amt": "amount"})
    return result[["open", "high", "low", "close", "volume", "amount"]]


def eligible_windows(data: dict, metadata: dict, lookback: int, horizon: int) -> list[tuple]:
    validation_start = pd.Timestamp(metadata["validation_target_start"])
    validation_end = pd.Timestamp(metadata["validation_target_end"])
    windows = []
    for symbol in sorted(data):
        frame = data[symbol].sort_index()
        for position in range(lookback - 1, len(frame) - horizon):
            target_date = pd.Timestamp(frame.index[position + horizon])
            if validation_start <= target_date <= validation_end:
                windows.append((symbol, position, target_date))
    return windows


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("calibration requires at least two forecast samples")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = resolve_path(args.dataset_path)
    with (dataset_path / "val_data.pkl").open("rb") as handle:
        data = pickle.load(handle)
    metadata = json.loads((dataset_path / "metadata.json").read_text(encoding="utf-8"))
    candidates = eligible_windows(data, metadata, args.lookback, args.horizon)
    if len(candidates) < 50:
        raise RuntimeError(f"too few eligible calibration windows: {len(candidates)}")
    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(
        len(candidates), size=min(args.max_windows, len(candidates)), replace=False
    )
    selected = sorted(
        (candidates[int(index)] for index in selected_indices), key=lambda item: (item[2], item[0])
    )

    predictor = make_predictor(
        resolve_path(args.official_repo),
        resolve_path(args.tokenizer_path),
        resolve_path(args.predictor_path),
        args.device,
    )
    records = []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        frames = []
        x_timestamps = []
        y_timestamps = []
        actuals = []
        for symbol, position, target_date in batch:
            source = data[symbol].sort_index()
            frame = context_frame(source, position, args.lookback)
            frames.append(frame)
            x_timestamps.append(pd.Series(frame.index))
            y_timestamps.append(pd.Series(source.index[position + 1:position + args.horizon + 1]))
            actuals.append((
                symbol,
                target_date,
                float(source["close"].iloc[position]),
                float(source["close"].iloc[position + args.horizon]),
            ))
        try:
            forecasts, forecast_stds = predictor.predict_batch(
                frames,
                x_timestamps,
                y_timestamps,
                pred_len=args.horizon,
                T=args.temperature,
                top_k=0,
                top_p=args.top_p,
                sample_count=args.samples,
                verbose=False,
                return_std=True,
            )
        except Exception as exc:
            print(f"Calibration batch {start} failed: {exc}", flush=True)
            continue
        for actual, forecast, forecast_std in zip(actuals, forecasts, forecast_stds):
            symbol, target_date, last_close, actual_close = actual
            pred_close = float(forecast["close"].iloc[-1])
            pred_std = float(forecast_std["close"].iloc[-1])
            records.append({
                "symbol": symbol,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "raw_prob_up": normal_probability_up(pred_close, last_close, pred_std),
                "pred_return": pred_close / last_close - 1,
                "actual_return": actual_close / last_close - 1,
                "actual_up": int(actual_close > last_close),
            })
        print(f"Calibrated inference {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    frame = pd.DataFrame(records).sort_values(["target_date", "symbol"])
    if len(frame) < 50:
        raise RuntimeError(f"only {len(frame)} calibration predictions succeeded")
    artifact = fit_probability_calibrator(
        frame["raw_prob_up"], frame["actual_up"], frame["target_date"], args.fit_fraction
    )
    calibrated = apply_probability_calibration(frame["raw_prob_up"], artifact)
    frame["calibrated_prob_up"] = calibrated
    artifact.update({
        "dataset_path": str(dataset_path),
        "predictor_path": str(resolve_path(args.predictor_path)),
        "observation_count": len(frame),
        "settings": {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "samples": args.samples,
            "seed": args.seed,
        },
        "all_validation_calibrated_metrics": probability_metrics(
            calibrated, frame["actual_up"]
        ),
    })
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
