# -*- coding: utf-8 -*-
"""固定推理参数（不做月度选参）跑 walk-forward，参数由命令行给出。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_paper_walk_forward as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--temperature", type=float, required=True)
ap.add_argument("--top-p", type=float, required=True)
ap.add_argument("--lookback", type=int, required=True)
ap.add_argument("--start-month", default="2026-01")
ap.add_argument("--end-month", default="2026-08")
ap.add_argument("--seed", type=int, default=100, help="采样随机种子（用于种子稳健性检验）")
ap.add_argument("--buy-threshold", type=float, default=0.005)
ap.add_argument("--output-dir", required=True)
a = ap.parse_args()

ns = argparse.Namespace(
    data="data/raw/000852_中证1000_daily.csv",
    execution_data=None,
    official_repo="third_party/Kronos",
    tokenizer_path="models/kronos_tokenizer",
    model_path="models/kronos_base",
    device="cuda",
    samples=10,
    batch_size=32,
    seed=a.seed,
    selection_metric="fixed",
    buy_threshold=a.buy_threshold,
    commission_rate=0.0003,
    stamp_tax_rate=0.0,
    slippage_rate=0.001,
    start_month=a.start_month,
    end_month=a.end_month,
)

OUT = ROOT / a.output_dir
OUT.mkdir(parents=True, exist_ok=True)

signal_df = W.load_ohlcv_data((ROOT / ns.data).resolve())
predictor = W.make_predictor(ns)
params = W.Params(a.temperature, a.top_p, a.lookback)
print(f"fixed params {params}, threshold {ns.buy_threshold}, range {ns.start_month}~{ns.end_month}", flush=True)

parts, monthly = [], []
for tm in W.month_range(ns.start_month, ns.end_month):
    end = tm.end_time.normalize()
    origins = W.eligible_origins(signal_df, tm, params.lookback, end)
    print(f"{tm}: {len(origins)} eligible signals", flush=True)
    preds = W.predict_origins(predictor, signal_df, origins, params, ns.samples, ns.batch_size, ns.seed)
    preds = W.attach_execution_prices(preds, None)
    preds["test_month"] = str(tm)
    parts.append(preds)
    monthly.append({"test_month": str(tm), "temperature": params.temperature,
                    "top_p": params.top_p, "lookback": params.lookback,
                    "selection_metric": "fixed", "test_predictions": len(preds)})

pdf = pd.concat(parts, ignore_index=True)
pdf.to_csv(OUT / "test_predictions.csv", index=False)
pd.DataFrame(monthly).to_csv(OUT / "monthly_params.csv", index=False)

trades, metrics = W.simulate_non_overlapping_trades(pdf, ns)
trades.to_csv(OUT / "trades.csv", index=False)
daily, dmetrics = W.build_daily_equity(trades, signal_df, ns.start_month, ns.end_month, ns)
daily.to_csv(OUT / "daily_equity.csv", index=False)
metrics.update(dmetrics)

bench = signal_df.loc[
    (signal_df.index >= pd.Timestamp(f"{ns.start_month}-01")) &
    (signal_df.index <= pd.Period(ns.end_month, freq="M").end_time.normalize()), "close"]
metrics["benchmark_buy_and_hold_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)
metrics["prediction_spearman"] = float(spearmanr(pdf["pred_close_t5"], pdf["actual_close_t5"]).statistic)
metrics["directional_accuracy"] = float(np.mean((pdf["pred_return_t5"] > 0) == (pdf["actual_return_t5"] > 0)))
metrics["test_range"] = [ns.start_month, ns.end_month]
metrics["fixed_params"] = {"temperature": params.temperature, "top_p": params.top_p, "lookback": params.lookback}
metrics["seed"] = a.seed

(OUT / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
                  if k in ("trade_count", "net_compounded_return", "gross_compounded_return",
                           "win_rate_before_cost", "total_relative_excess_return",
                           "prediction_spearman", "directional_accuracy",
                           "benchmark_buy_and_hold_return", "max_drawdown")},
                 ensure_ascii=False, indent=2))
print("artifacts ->", OUT)
