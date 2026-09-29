#!/usr/bin/env python3
"""Build per-test-month rolling fine-tune datasets from official split pickles.

Test month ``m`` gets a dataset whose forecast targets lie before ``m``:
- default: trailing 12 calendar months [m-12 months, end of m-1]
- ``--window-months all``: expanding window from 2010-01-01 to the end
- ``--target-end YYYY-MM-DD``: explicit training target end (overrides m-1 end)

The source dataset (HFQ or QFQ official pickles) is selected via ``--dataset-src``;
the adjustment label in emitted metadata mirrors the source metadata.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOOKBACK, HORIZON = 250, 5
MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
          "2025-06", "2025-07", "2025-08", "2025-09"]


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def slice_frame(frame: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame | None:
    """Same context policy as finetune/dataset.py slice_frame_with_context."""
    idx = pd.DatetimeIndex(frame.index)
    mask = (idx >= t0) & (idx <= t1)
    target_pos = np.flatnonzero(mask)
    if len(target_pos) < HORIZON:
        return None
    ctx = max(0, int(target_pos[0]) - LOOKBACK)
    out = frame.iloc[ctx:int(target_pos[-1]) + 1]
    if len(out) < LOOKBACK + HORIZON:
        return None
    return out


def merged_symbols(val: dict, test: dict) -> dict:
    merged: dict[str, pd.DataFrame] = {}
    for sym in sorted(set(val) | set(test)):
        parts = [d[sym] for d in (val, test) if sym in d]
        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        for col in ("limit_up", "limit_down"):
            if col not in df.columns:
                df[col] = np.nan
        merged[sym] = df
    return merged


def month_bounds(month: str, window_months: str, target_end: str | None):
    y, mo = int(month[:4]), int(month[5:7])
    month_start = pd.Timestamp(f"{y}-{mo:02d}-01")
    if window_months == "all":
        start = pd.Timestamp("2010-01-01")
    else:
        start = (month_start - pd.DateOffset(months=int(window_months))).normalize()
    end = (pd.Timestamp(target_end) if target_end else
           (month_start - pd.DateOffset(days=1)).normalize())
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default=None, help="Single month YYYY-MM; default builds all MONTHS")
    parser.add_argument("--dataset-src", default=str(ROOT / "data" / "official_processed_datasets_csi1000_pts"))
    parser.add_argument("--out-root", default=str(ROOT / "data" / "rolling_datasets"))
    parser.add_argument("--window-months", default="12", help="12 | all")
    parser.add_argument("--target-end", default=None, help="YYYY-MM-DD explicit training target end")
    parser.add_argument("--tag", default=None, help="output folder name; default RL-{month}")
    args = parser.parse_args()

    src = Path(args.dataset_src)
    out_root = Path(args.out_root)
    src_meta_path = src / "metadata.json"
    if not (src / "val_data.pkl").is_file() or not (src / "test_data.pkl").is_file():
        raise FileNotFoundError(f"source pickles missing in {src}")
    adjustment = "unknown"
    if src_meta_path.is_file():
        adjustment = json.loads(src_meta_path.read_text(encoding="utf-8")).get("adjustment", "unknown")

    months = [args.month] if args.month else MONTHS
    print(f"Loading source pickles from {src} ...")
    val = load_pkl(src / "val_data.pkl")
    test = load_pkl(src / "test_data.pkl")
    all_frames = merged_symbols(val, test)
    print(f"Merged {len(all_frames)} symbols, adjustment={adjustment}")

    for month in months:
        start, end = month_bounds(month, args.window_months, args.target_end)
        frames: dict[str, pd.DataFrame] = {}
        for sym, df in all_frames.items():
            sliced = slice_frame(df, start, end)
            if sliced is not None:
                frames[sym] = sliced
        tag = args.tag or f"RL-{month}"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_data.pkl", "wb") as handle:
            pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_dir / "val_data.pkl", "wb") as handle:
            pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "adjustment": adjustment,
            "limit_features": ["limit_up", "limit_down"],
            "first_listing_day_policy": "drop first listing day (rolling archive)",
            "split_context_policy": "rolling lookback buffer; forecast targets strictly inside split_target_ranges",
            "split_target_ranges": {
                "train": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
                "val": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
            },
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "target_month": month,
            "window_months": args.window_months,
            "explicit_target_end": args.target_end,
            "source_dataset": str(src),
            "symbol_count": len(frames),
            "note": "Rolling fine-tune dataset; predictor-only warm-start retrain; val mirrors train "
                    "for logging only (early stopping disabled).",
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{tag}: {len(frames)} symbols, targets {start.date()}..{end.date()} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
