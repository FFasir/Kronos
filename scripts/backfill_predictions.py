#!/usr/bin/env python3
"""
Kronos — 历史预测补档（每日一个 as-of 批次）

用法:
    # 指数信号标的（000852/512100 等配置标的），快（<2 分钟/月）
    python scripts/backfill_predictions.py --mode index --start 2026-07-20 --end 2026-09-03

    # 全 A 股每日预测（每交易日约 5-8 分钟，可断点续跑）
    python scripts/backfill_predictions.py --mode all --start 2026-07-20 --end 2026-09-03

设计:
  - point-in-time：对每个交易日 D，每只标的只用『截至 D』的行情现算 5 日预测，
    输入不含未来信息；
  - 结果以 date=D(as_of) 写入 model_prediction（幂等 upsert），重复运行自动跳过已完成日期；
  - 诚实边界：当前微调模型训练数据截止约 2026-09-01，早于该日的补档存在前视泄漏，
    仅供“演示/补齐历史界面”用，不代表当年真实预测水平。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.data_layer.storage import StorageLayer
from src.model_layer.model_factory import create_model
from src.model_layer.kronos_predictor import KronosPredictor

logger = get_logger("scripts.backfill_predictions")
STATE_FILE = "/tmp/kronos_backfill_state.json"


def _state() -> set[str]:
    try:
        return set(json.load(open(STATE_FILE, encoding="utf-8")))
    except Exception:
        return set()


def _mark_done(mode: str, day: str, symbols: list[str]) -> None:
    state = _state()
    key = _day_key(mode, day, symbols)
    if key not in state:
        state.add(key)
        json.dump(sorted(state), open(STATE_FILE, "w", encoding="utf-8"))


def _day_key(mode: str, day: str, symbols: list[str]) -> str:
    """状态按 mode+日期+标集合区分：不同标的集合互不误判完成。"""
    import hashlib
    sig = hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:10]
    return f"{mode}:{day}:{sig}"


def main() -> int:
    parser = argparse.ArgumentParser(description="历史预测按日补档")
    parser.add_argument("--mode", choices=["index", "all"], default="index")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", default=None,
                        help="index 模式下指定标的，逗号分隔（默认 config 的指数对）")
    parser.add_argument("--device", default="auto",
                        help="计算设备：auto/cpu/cuda（全 A 补档建议 cpu，避免 CUDA OOM）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已完成状态，强制重新处理（用于修复历史空跑标记）")
    args = parser.parse_args()

    setup_logger(level="INFO", log_dir="logs/", console=True)
    config = load_config()
    storage = StorageLayer(config)
    storage.init_database()

    # 交易日历（用 000852 行情日作为 A 股统一交易日参考）
    calendar_df = storage.load_raw_data("000852")
    if calendar_df is None or len(calendar_df) == 0:
        logger.error("无法用 000852 行情确定交易日历")
        return 1
    calendar_df.index = _as_datetime(calendar_df.index)
    days = [
        d for d in calendar_df.index
        if _parse(args.start) <= d <= _parse(args.end)
    ]
    if not days:
        logger.error("所选区间内没有交易日")
        print("NO-TRADING-DAYS", flush=True)
        return 1
    logger.info("交易日：%s ~ %s 共 %d 天", days[0].date(), days[-1].date(), len(days))
    print(f"DAYS {len(days)} {days[0].date()}..{days[-1].date()}", flush=True)

    # 标的清单
    if args.mode == "index":
        symbols = [str(s) for s in args.symbols.split(",")] if args.symbols else []
        if not symbols:
            symbols = [
                str(s) for s in (config.get("strategy", "prediction_symbols", default=None) or [])
            ]
        if not symbols:
            symbols = sorted({
                str(config.get("strategy", "paper_symbol", default="000852")),
                str(config.get("strategy", "execution_symbol", default="512100")),
            })
    else:
        symbols = storage.list_stock_symbols()

    # 一次性载入各标的完整行情（内存缓存），按日切片
    logger.info("加载 %d 个标的行情…", len(symbols))
    series_map: dict[str, "object"] = {}
    for i, sym in enumerate(symbols):
        try:
            df = storage.load_raw_data(sym)
            if df is None or len(df) == 0:
                continue
            df.index = _as_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            series_map[sym] = df
        except Exception as exc:
            logger.debug("加载 %s 失败: %s", sym, exc)
        if (i + 1) % 500 == 0:
            logger.info("行情加载 %d/%d", i + 1, len(symbols))
    logger.info("可用标的 %d/%d", len(series_map), len(symbols))

    logger.info("创建模型…")
    model, _path = create_model(config, device=args.device or "auto")
    predictor = KronosPredictor(model, config)
    lookback = int(getattr(predictor, "seq_len", None) or 250)

    done_days = _state()
    force = getattr(args, "force", False)
    for idx, day in enumerate(days):
        day_str = str(day.date())
        if not force and _day_key(args.mode, day_str, list(series_map.keys())) in done_days:
            logger.info("跳过已完成的 %s", day_str)
            continue

        # 为当日构建 ≤ day 的窗口集
        batch = {}
        for sym, df in series_map.items():
            hist = df[df.index <= day]
            if hist is not None and len(hist) >= lookback:
                batch[sym] = hist
        if not batch:
            logger.warning("%s：无可预测标的，标记完成", day_str)
            print(f"DAY {day_str} NO-BATCH", flush=True)
            _mark_done(args.mode, day_str, list(series_map.keys()))
            continue

        t0 = time.time()
        logger.info("[%s/%d] %s 预测 %d 只…", idx + 1, len(days), day_str, len(batch))
        print(f"DAY {day_str} batch={len(batch)}", flush=True)
        try:
            if len(batch) <= 8:
                # 小批量（指数/ETF）：逐只预测，避免原生批量对不同窗口长度报错
                results = []
                for sym, hist in batch.items():
                    try:
                        results.extend(predictor.predict_batch({sym: hist}))
                    except Exception as exc:
                        logger.error("%s %s 预测失败：%s", day_str, sym, exc)
                        print(f"DAY {day_str} {sym} PRED-ERR {exc}", flush=True)
            else:
                results = predictor.predict_batch(batch)
        except Exception as exc:
            logger.error("%s 批推理失败：%s", day_str, exc)
            print(f"DAY {day_str} BATCH-ERR {exc}", flush=True)
            time.sleep(5)
            continue

        rows = []
        for item in results:
            if item.get("error") or not item.get("pred_date") or not item.get("symbol"):
                continue
            try:
                pred_return = float(item["pred_return"])
                if pred_return != pred_return:  # NaN
                    continue
            except (TypeError, ValueError):
                continue
            rows.append((
                item["symbol"], day_str, str(item["pred_date"]),
                float(item["pred_close_t5"]), float(item.get("pred_return", 0.0)),
                float(item.get("last_close", 0.0) or 0.0),
                float(item.get("prediction_std", 0.0) or 0.0),
                float(item.get("score", 5.0) or 0.0),
                float(item.get("confidence", 0.5) or 0.0),
                float(item.get("temperature", predictor.default_T) or 0.0),
                float(item.get("top_p", predictor.default_top_p) or 0.0),
                int(item.get("lookback_window", lookback)),
                str(getattr(model, "model_version", "backfill")),
            ))
        if rows:
            storage.execute_many(
                """INSERT INTO model_prediction
                   (code, date, pred_date, pred_close, pred_return, last_close,
                    prediction_std, score, confidence,
                    temperature, top_p, lookback, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(code, date, pred_date) DO UPDATE SET
                     pred_close=excluded.pred_close, pred_return=excluded.pred_return,
                     last_close=excluded.last_close, prediction_std=excluded.prediction_std,
                     score=excluded.score, confidence=excluded.confidence,
                     temperature=excluded.temperature, top_p=excluded.top_p,
                     lookback=excluded.lookback, model_version=excluded.model_version""",
                rows,
            )
            print(f"DAY {day_str} WROTE {len(rows)}", flush=True)
            logger.info("%s 写入 %d 条，用时 %.1fs", day_str, len(rows), time.time() - t0)
            # 只有实际写入成功才标记完成（避免把空跑日误记）
            _mark_done(args.mode, day_str, list(series_map.keys()))
        else:
            print(f"DAY {day_str} ROWS-EMPTY", flush=True)
            logger.warning("%s 无有效结果，未标记完成（可重跑）", day_str)

    logger.info("补档完成：共 %d 个交易日", len(days))
    return 0


def _parse(s: str):
    from pandas import Timestamp
    return Timestamp(s)


def _as_datetime(idx):
    from pandas import to_datetime
    return to_datetime(idx)


if __name__ == "__main__":
    sys.exit(main())
