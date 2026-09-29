#!/usr/bin/env python3
"""Evaluate the trained direction head out-of-sample on 2025-01..09.

Signals: prob(up) = sigmoid(head(context at origin)) with a 250-bar history.
Strategy: long (full) when prob(up) > threshold, entry next open, exit 5 sessions
later at close, non-overlapping, commission 0.03% + slippage 0.1%.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "Kronos"))
sys.path.insert(0, str(ROOT / "third_party" / "Kronos" / "finetune"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import Config  # noqa: E402
from train_direction import DirectionHead  # noqa: E402
import run_paper_walk_forward as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lookback", type=int, default=250)
    ap.add_argument("--start-month", default="2025-01")
    ap.add_argument("--end-month", default="2025-09")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = Config()
    device = args.device
    from model import Kronos, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(cfg.pretrained_tokenizer_path))
    tokenizer.eval().to(device)
    model = Kronos.from_pretrained(str(cfg.pretrained_predictor_path)).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    head = DirectionHead(model.d_model).to(device)
    head.load_state_dict(ck["head"])
    model.eval(); head.eval()
    print("checkpoint loaded, val acc (in-sample) =", ck.get("best_val_acc"))

    sf = W.load_ohlcv_data((ROOT / "data" / "raw" / "000852_中证1000_daily.csv").resolve())
    if "amount" not in sf.columns:
        sf["amount"] = sf["volume"] * sf[["open", "high", "low", "close"]].mean(axis=1)

    def time_features(idx: pd.DatetimeIndex) -> np.ndarray:
        return np.stack([
            idx.minute.to_numpy(), idx.hour.to_numpy(), idx.weekday.to_numpy(),
            idx.day.to_numpy(), idx.month.to_numpy(),
        ], axis=1).astype(np.float32)

    close_idx = 3
    lb = args.lookback
    rows = []
    for month in W.month_range(args.start_month, args.end_month):
        end = month.end_time.normalize()
        for pos, ts in enumerate(sf.index):
            if ts.to_period("M") != month:
                continue
            if pos - lb + 1 < 0 or pos + 5 >= len(sf):
                continue
            if sf.index[pos + 5] > end:
                continue
            hist = sf.iloc[pos - lb + 1:pos + 1].copy()
            x = hist[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
            x_mean, x_std = x.mean(axis=0), x.std(axis=0)
            x = (x - x_mean) / (x_std + 1e-5)
            x = np.clip(x, -5, 5)
            stamp = time_features(pd.DatetimeIndex(hist.index))
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                st = torch.from_numpy(stamp).unsqueeze(0).to(device)
                tok0, tok1 = tokenizer.encode(xt, half=True)
                _, ctx = model.decode_s1(tok0, tok1, st)
                feat = ctx[:, -1, :]  # origin position
                logit = head(feat)
                prob = float(torch.sigmoid(logit).item())
            last_close = float(sf["close"].iloc[pos])
            rows.append({
                "origin_date": ts.strftime("%Y-%m-%d"),
                "entry_date": sf.index[pos + 1].strftime("%Y-%m-%d"),
                "exit_date": sf.index[pos + 5].strftime("%Y-%m-%d"),
                "last_close": last_close,
                "actual_close_t5": float(sf["close"].iloc[pos + 5]),
                "entry_open": float(sf["open"].iloc[pos + 1]),
                "prob_up": prob,
                "actual_return_t5": float(sf["close"].iloc[pos + 5] / last_close - 1),
            })
    pdf = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "outputs" / "direction_head_walkforward_2025_9m"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(out_dir / "signals.csv", index=False)
    print(f"{len(pdf)} signals")

    acc = float(np.mean((pdf["prob_up"] > 0.5) == (pdf["actual_close_t5"] > pdf["last_close"])))
    base_rate = float(np.mean(pdf["actual_close_t5"] > pdf["last_close"]))
    print(f"direction acc(prob>0.5)={acc:.4f}  base_rate={base_rate:.4f}")

    results = {"n": len(pdf), "dir_acc_0.5": acc, "base_rate": base_rate}
    for thr in (0.45, 0.5, 0.55, 0.6):
        eq, next_avail, trades = 1.0, "", 0
        for r in pdf.sort_values("origin_date").itertuples(index=False):
            if r.origin_date < next_avail or r.prob_up <= thr:
                continue
            gross = r.actual_close_t5 / r.entry_open - 1
            nf = (r.actual_close_t5 / r.entry_open) * (1 - 0.0003 - 0.0 - 0.001) / (1 + 0.0003 + 0.001)
            eq *= nf
            next_avail = r.exit_date
            trades += 1
        results[f"net_thr{thr}"] = round(eq - 1, 6)
        results[f"trades_thr{thr}"] = trades
        print(f"threshold {thr}: net {eq - 1:+.2%}, trades {trades}")

    # regime splits
    hh = pd.to_datetime(pdf["origin_date"])
    for tag, mask in [("H1", hh < pd.Timestamp("2025-07-01")), ("H2", hh >= pd.Timestamp("2025-07-01"))]:
        s = pdf[mask]
        if len(s) >= 5:
            results[f"{tag}_dir_acc"] = float(np.mean((s["prob_up"] > 0.5) == (s["actual_close_t5"] > s["last_close"])))
            results[f"{tag}_base_rate"] = float(np.mean(s["actual_close_t5"] > s["last_close"]))
            results[f"{tag}_n"] = int(len(s))

    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
