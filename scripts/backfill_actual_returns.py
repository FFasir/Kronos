#!/usr/bin/env python3
"""
Kronos — 回填实际收益（actual_return / actual_close）

对 model_prediction 中「目标日(pred_date)已到期」且 actual_return IS NULL 的行，
用本地 hfq 序列计算：
    actual_return = close(pred_date) / close(as_of=date) - 1
    actual_close  = close(pred_date)

- hfq 序列来自 storage.load_raw_data（daily_hfq_live 覆盖层优先，归档兜底），
  与预测输入同口径，比例天然正确；
- 幂等：只更新 actual_* IS NULL 的行；目标日未到期（序列未覆盖）的保持 NULL，
  之后随行情推进可重复运行补算；
- 用法: python scripts/backfill_actual_returns.py [--codes 600519,000001]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.data_layer.storage import StorageLayer

logger = get_logger("scripts.backfill_actual_returns")


def main() -> int:
    parser = argparse.ArgumentParser(description="回填实际收益")
    parser.add_argument("--codes", default=None, help="逗号分隔代码；默认处理全部")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = parser.parse_args()

    setup_logger(level="INFO", log_dir="logs/", console=True)
    config = load_config()
    storage = StorageLayer(config)
    storage.init_database()

    sql = (
        "SELECT code, date, pred_date, last_close FROM model_prediction "
        "WHERE actual_return IS NULL"
    )
    params: tuple = ()
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        placeholders = ",".join("?" for _ in codes)
        sql += f" AND code IN ({placeholders})"
        params = tuple(codes)
    rows = storage.execute_sql(sql + " ORDER BY code, date", params)

    pending: dict[str, list[tuple[str, str, float | None]]] = {}
    for r in rows:
        code = str(r["code"]).zfill(6)
        pending.setdefault(code, []).append(
            (str(r["date"]), str(r["pred_date"]), float(r["last_close"]) if r["last_close"] is not None else None)
        )
    logger.info("待回填代码 %d，共 %d 行", len(pending), len(rows))

    total_filled = 0
    total_kept_null = 0
    t0 = time.time()
    for i, (code, items) in enumerate(sorted(pending.items())):
        try:
            df = storage.load_raw_data(code)
        except Exception:
            df = None
        if df is None or len(df) == 0:
            total_kept_null += len(items)
            continue
        closes = df["close"]
        idx_map = {str(d)[:10]: float(closes.iloc[k]) for k, d in enumerate(df.index)}
        updates: list[tuple[float, float, str, str]] = []
        for as_of, pred_date, _last in items:
            close_asof = idx_map.get(as_of)
            close_pred = idx_map.get(pred_date)
            if close_asof and close_pred and close_asof > 0:
                actual_return = round(close_pred / close_asof - 1.0, 6)
                updates.append((close_pred, actual_return, as_of, pred_date))
            else:
                total_kept_null += 1
        if updates and not args.dry_run:
            storage.execute_many(
                """UPDATE model_prediction
                   SET actual_close = ?, actual_return = ?
                   WHERE code = ? AND date = ? AND pred_date = ? AND actual_return IS NULL""",
                [(c, r_, code, a, p) for (c, r_, a, p) in updates],
            )
        total_filled += len(updates)
        if (i + 1) % 400 == 0:
            logger.info("进度 %d/%d 已填 %d", i + 1, len(pending), total_filled)
    logger.info("完成：填 %d 行，暂缺(未到期/无数据) %d 行，用时 %.0fs",
                total_filled, total_kept_null, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
