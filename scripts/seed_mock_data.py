#!/usr/bin/env python3
"""
Kronos 择时系统 — 离线 Mock 数据注入脚本

在没有真实行情数据 / Kronos 模型预测文件的情况下，向 SQLite 注入
确定性的模拟数据，让前端选股池、信号、预测K线、回测等页面可以
完整预览效果。

注入内容:
  1. stock_info   — 40 只知名 A 股个股（type=stock）
  2. stock_daily  — 每只股票近 40 个交易日 OHLCV（含停牌/低流动性样本）
  3. model_prediction — 最新一期预测（prob_up=confidence, score）
  4. trading_signal   — 混合 BUY/HOLD/CASH/SELL 信号

用法:
    python scripts/seed_mock_data.py [--days 40] [--symbols 40]

可重复执行（幂等）：先清空相关表再写入。
"""

from __future__ import annotations

import sys
import csv
import argparse
import random
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config, PROJECT_ROOT
from src.common.logger import get_logger

logger = get_logger(__name__)


# ── 40 只知名 A 股（代码, 名称, 所属行业）──────────────────────
STOCKS = [
    ("600519", "贵州茅台", "食品饮料"),
    ("000858", "五粮液", "食品饮料"),
    ("000333", "美的集团", "家用电器"),
    ("000651", "格力电器", "家用电器"),
    ("600036", "招商银行", "银行"),
    ("601398", "工商银行", "银行"),
    ("601318", "中国平安", "非银金融"),
    ("600030", "中信证券", "非银金融"),
    ("000002", "万科A", "房地产"),
    ("600048", "保利发展", "房地产"),
    ("002415", "海康威视", "电子"),
    ("002475", "立讯精密", "电子"),
    ("603501", "韦尔股份", "电子"),
    ("300059", "东方财富", "非银金融"),
    ("300750", "宁德时代", "电力设备"),
    ("002594", "比亚迪", "汽车"),
    ("601012", "隆基绿能", "电力设备"),
    ("300274", "阳光电源", "电力设备"),
    ("600276", "恒瑞医药", "医药生物"),
    ("300760", "迈瑞医疗", "医药生物"),
    ("000661", "长春高新", "医药生物"),
    ("603259", "药明康德", "医药生物"),
    ("600887", "伊利股份", "食品饮料"),
    ("601888", "中国中免", "社会服务"),
    ("600809", "山西汾酒", "食品饮料"),
    ("000568", "泸州老窖", "食品饮料"),
    ("002714", "牧原股份", "农林牧渔"),
    ("601899", "紫金矿业", "有色金属"),
    ("600028", "中国石化", "石油石化"),
    ("601088", "中国神华", "煤炭"),
    ("600900", "长江电力", "公用事业"),
    ("601668", "中国建筑", "建筑装饰"),
    ("600585", "海螺水泥", "建筑材料"),
    ("000063", "中兴通讯", "通信"),
    ("002230", "科大讯飞", "计算机"),
    ("600570", "恒生电子", "计算机"),
    ("300124", "汇川技术", "机械设备"),
    ("002050", "三花智控", "机械设备"),
    ("600031", "三一重工", "机械设备"),
    ("601615", "明阳智能", "电力设备"),
]


def _trading_days(n: int, end: date) -> list[str]:
    """生成 end 往前 n 个交易日（跳过周末）的日期字符串列表。"""
    days: list[str] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:  # 周一~周五
            days.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(days))


def _gen_ohlcv(rng: random.Random, base_price: float, days: list[str],
               suspended: bool, low_liquidity: bool) -> list[tuple]:
    """生成一只股票的 OHLCV 序列。"""
    rows: list[tuple] = []
    price = base_price
    for i, ds in enumerate(days):
        # 停牌股：最近 3 天 volume=0
        is_suspended_day = suspended and i >= len(days) - 3
        if is_suspended_day:
            rows.append((ds, price, price, price, price, 0.0, 0.0, 0.0))
            continue

        daily_ret = rng.gauss(0.0005, 0.018)
        open_price = price * (1 + rng.gauss(0, 0.004))
        close_price = price * (1 + daily_ret)
        high = max(open_price, close_price) * (1 + abs(rng.gauss(0.005, 0.004)))
        low = min(open_price, close_price) * (1 - abs(rng.gauss(0.005, 0.004)))

        # 低流动性股：volume/amount 很小
        volume = rng.uniform(20000, 80000) if low_liquidity else rng.uniform(5_000_000, 50_000_000)
        amount = close_price * volume

        rows.append((ds, round(open_price, 2), round(high, 2), round(low, 2),
                     round(close_price, 2), round(volume, 0), round(amount, 0),
                     round(rng.uniform(0.5, 5.0), 2)))
        price = close_price
    return rows


def main():
    parser = argparse.ArgumentParser(description="注入离线 mock 数据")
    parser.add_argument("--days", type=int, default=40, help="交易日数量")
    parser.add_argument("--symbols", type=int, default=40, help="股票数量")
    args = parser.parse_args()

    config = load_config()
    from src.data_layer.storage import StorageLayer
    storage = StorageLayer(config)

    # 用固定 seed 保证可复现
    rng = random.Random(20260831)

    # 今天作为预测日/信号日
    today = date.today()
    trading_days = _trading_days(args.days, today)
    latest_date = trading_days[-1]
    pred_date = (today + timedelta(days=5)).isoformat()  # 预测 T+5

    symbols = STOCKS[: args.symbols]

    # 设计各阶段样本：停牌 3 只、低流动性 5 只、其余正常
    suspended_codes = {symbols[i][0] for i in (5, 13, 21)}
    low_liq_codes = {symbols[i][0] for i in (8, 16, 24, 29, 33)}

    logger.info(f"注入 {len(symbols)} 只股票，{args.days} 个交易日 ...")

    # ── 1. stock_info ──────────────────────────────────────────
    info_rows = []
    for code, name, industry in symbols:
        info_rows.append((code, name, "stock", None, "2015-01-01"))
    storage.execute_write(
        """DELETE FROM stock_info WHERE type = 'stock'"""
    )
    for r in info_rows:
        storage.execute_write(
            """INSERT OR REPLACE INTO stock_info (code, name, type, track_index, list_date)
               VALUES (?, ?, ?, ?, ?)""", r,
        )

    # ── 2. stock_daily ─────────────────────────────────────────
    storage.execute_write("DELETE FROM stock_daily WHERE code IN "
                          "(" + ",".join("?" * len(symbols)) + ")",
                          tuple(s[0] for s in symbols))

    base_prices = {}
    daily_rows = []
    for idx, (code, name, industry) in enumerate(symbols):
        base = rng.uniform(8.0, 180.0)
        base_prices[code] = base
        ohlcv = _gen_ohlcv(rng, base, trading_days,
                           suspended=code in suspended_codes,
                           low_liquidity=code in low_liq_codes)
        for ds, o, h, l, c, v, amt, tvr in ohlcv:
            daily_rows.append((code, ds, o, h, l, c, v, amt, tvr))

    # 批量写入（分块）
    chunk = 500
    for i in range(0, len(daily_rows), chunk):
        storage.execute_many(
            """INSERT OR REPLACE INTO stock_daily
               (code, date, open, high, low, close, volume, amount, turnover)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            daily_rows[i:i + chunk],
        )

    # ── 3. model_prediction ────────────────────────────────────
    storage.execute_write("DELETE FROM model_prediction")

    pred_rows = []
    for idx, (code, name, industry) in enumerate(symbols):
        last_close = base_prices[code]
        # prob_up 分布 0.30 ~ 0.95，score 分布 1~9.5
        prob_up = round(rng.uniform(0.30, 0.95), 3)
        score = round(rng.uniform(1.0, 9.5), 2)
        pred_return = round(rng.gauss(0.004, 0.02), 4)
        pred_close = round(last_close * (1 + pred_return), 2)
        confidence = prob_up  # prob_up 字段映射到 confidence

        pred_rows.append((
            code, latest_date, pred_date, pred_close, pred_return,
            round(last_close, 2), 0.02, None, None,  # prediction_std, actual_close, actual_return
            score, confidence, 1.0, 0.95, 60, "mock-1.0",
        ))

    storage.execute_many(
        """INSERT OR REPLACE INTO model_prediction
           (code, date, pred_date, pred_close, pred_return, last_close,
            prediction_std, actual_close, actual_return, score, confidence,
            temperature, top_p, lookback, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        pred_rows,
    )

    # ── 4. trading_signal ──────────────────────────────────────
    storage.execute_write("DELETE FROM trading_signal")

    signal_rows = []
    for idx, (code, name, industry) in enumerate(symbols):
        prob_up = pred_rows[idx][10]  # confidence
        pred_return = pred_rows[idx][4]
        pred_close = pred_rows[idx][3]

        # 信号判定：高概率 → BUY，中 → HOLD，低 → CASH，极低 → SELL
        if prob_up >= 0.65:
            sig = "BUY"
            pos = round(rng.uniform(0.5, 0.9), 2)
            reason = "预测上涨概率高，建议买入"
        elif prob_up >= 0.50:
            sig = "HOLD"
            pos = round(rng.uniform(0.2, 0.5), 2)
            reason = "方向不确定，建议持有观望"
        elif prob_up >= 0.40:
            sig = "CASH"
            pos = 0.0
            reason = "预测偏弱，建议空仓"
        else:
            sig = "SELL"
            pos = 0.0
            reason = "预测下跌，建议卖出"

        signal_rows.append((
            code, latest_date, sig, pred_return, pred_close,
            round(prob_up, 3), pos, reason,
        ))

    storage.execute_many(
        """INSERT OR REPLACE INTO trading_signal
           (code, date, signal_type, pred_return, pred_close,
            confidence, position_pct, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        signal_rows,
    )

    # ── 5. price_prediction（预测K线）─────────────────────────
    storage.execute_write("DELETE FROM price_prediction")
    pp_rows = []
    for idx, (code, name, industry) in enumerate(symbols):
        last_close = base_prices[code]
        prob_up = pred_rows[idx][10]
        daily_ret = pred_rows[idx][4] / 5.0
        price = last_close
        for h in range(1, 21):
            price = last_close * (1 + daily_ret * h * (0.95 ** h))
            vol = last_close * 0.015 * (1 + 0.08 * h)
            pp_rows.append((
                code, f"{latest_date}T15:00:00", h,
                round(price, 2), round(price - vol * 2, 2), round(price + vol * 2, 2),
                "mock-1.0", latest_date, (today + timedelta(days=h)).isoformat(),
                round(price - vol * 0.3, 2), round(price + vol, 2),
                round(price - vol, 2), round(price, 2), 0.02,
            ))

    chunk = 2000
    for i in range(0, len(pp_rows), chunk):
        storage.execute_many(
            """INSERT OR REPLACE INTO price_prediction
               (code, created_at, pred_horizon, pred_price, pred_lower, pred_upper,
                model_version, as_of_date, pred_date, pred_open, pred_high, pred_low,
                pred_close, prediction_std)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            pp_rows[i:i + chunk],
        )

    logger.info(
        f"完成: {len(symbols)} 只股票 / {len(daily_rows)} 条行情 / "
        f"{len(pred_rows)} 条预测 / {len(signal_rows)} 条信号 / {len(pp_rows)} 条预测K线"
    )

    # ── 6. 中证1000成分股快照（signal 端点需要 ≥900 只成分股）────
    code_to_name = {code: name for code, name, _ in symbols}
    member_codes = list(code_to_name.keys())
    existing = set(member_codes)
    extra = []
    n = 600000
    while len(member_codes) + len(extra) < 1000:
        c = f"{n:06d}"
        if c not in existing:
            extra.append(c)
        n += 1

    research_dir = PROJECT_ROOT / "data" / "raw" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = research_dir / "csi1000_csindex_mock.csv"
    with open(snapshot_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "name", "snapshot_date"])
        for code in member_codes + extra:
            writer.writerow([code, code_to_name.get(code, f"成分股{code}"), today.isoformat()])

    logger.info(f"  已生成中证1000快照: {snapshot_path.name} ({len(member_codes) + len(extra)} 只成分股)")


if __name__ == "__main__":
    main()
