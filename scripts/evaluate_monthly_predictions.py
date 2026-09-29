#!/usr/bin/env python3
"""Evaluate a realized monthly prediction CSV and backfill actual outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate realized monthly Kronos predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument("--no-database", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions)
    predictions = pd.read_csv(predictions_path, dtype={"symbol": str})
    required = {"symbol", "as_of_date", "pred_date", "pred_close", "pred_return"}
    if not required <= set(predictions):
        raise ValueError(f"prediction CSV is missing columns: {sorted(required - set(predictions))}")

    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(load_config())
    realized = []
    for row in predictions.itertuples(index=False):
        try:
            frame = storage.load_raw_data(str(row.symbol).zfill(6))
            target_date = pd.Timestamp(row.pred_date)
            if target_date not in frame.index:
                continue
            actual_close = float(frame.loc[target_date, "close"])
            last_close = float(row.pred_close) / (1 + float(row.pred_return))
            realized.append({
                "symbol": str(row.symbol).zfill(6),
                "as_of_date": str(row.as_of_date),
                "pred_date": str(row.pred_date),
                "pred_return": float(row.pred_return),
                "actual_close": actual_close,
                "actual_return": actual_close / last_close - 1,
            })
        except Exception:
            continue
    if not realized:
        raise RuntimeError("none of the prediction targets has realized market data")

    frame = pd.DataFrame(realized)
    errors = frame["pred_return"] - frame["actual_return"]
    return_ic = float(spearmanr(frame["pred_return"], frame["actual_return"]).statistic)
    metrics = {
        "prediction_file": str(predictions_path),
        "prediction_count": len(predictions),
        "realized_count": len(frame),
        "realized_ratio": len(frame) / len(predictions),
        "return_ic": return_ic if np.isfinite(return_ic) else None,
        "directional_accuracy": float(np.mean(
            (frame["pred_return"] > 0) == (frame["actual_return"] > 0)
        )),
        "always_up_accuracy": float(np.mean(frame["actual_return"] > 0)),
        "always_down_accuracy": float(np.mean(frame["actual_return"] <= 0)),
        "return_mae": float(np.mean(np.abs(errors))),
        "zero_return_mae": float(np.mean(np.abs(frame["actual_return"]))),
        "return_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_predicted_return": float(frame["pred_return"].mean()),
        "mean_actual_return": float(frame["actual_return"].mean()),
    }
    metrics["majority_direction_accuracy"] = max(
        metrics["always_up_accuracy"], metrics["always_down_accuracy"]
    )
    if not args.no_database:
        storage.execute_many(
            """UPDATE model_prediction SET actual_close=?, actual_return=?
               WHERE code=? AND date=? AND pred_date=?""",
            [
                (
                    row["actual_close"], row["actual_return"], row["symbol"],
                    row["as_of_date"], row["pred_date"],
                )
                for row in realized
            ],
        )

    output = resolve_path(args.output) if args.output else predictions_path.with_name(
        f"{predictions_path.stem}_realized.json"
    )
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
