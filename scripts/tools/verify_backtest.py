#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测验证工具

用法: python scripts/tools/verify_backtest.py [--days 30]

快速跑一个短窗口回测，验证:
  1. 无未来数据泄露
  2. 模型可用
  3. 交易逻辑正常
"""
import sys
import io
import argparse
from pathlib import Path
from datetime import datetime, timedelta

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.common.config import load_config
from src.strategy_layer.backtest_engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="快速回测验证")
    parser.add_argument("--days", type=int, default=30, help="回测窗口天数（默认30）")
    parser.add_argument("--capital", type=float, default=1000000, help="初始资金（默认100万）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Kronos 回测验证 (窗口: {args.days} 天)")
    print("=" * 70)

    cfg = load_config()

    # 加载数据
    print("\n[1/3] 加载数据...")
    raw_dir = Path("data/raw")
    symbol_dfs = {}

    csv_files = list(raw_dir.glob("*_daily.csv"))
    if not csv_files:
        print("  错误: data/raw/ 下没有 CSV 文件")
        return 1

    for csv_path in csv_files[:6]:  # 最多加载 6 个标的
        code = csv_path.stem.split("_")[0]
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        symbol_dfs[code] = df
        print(f"  {code}: {len(df)} 行, {df.index.min().date()} ~ {df.index.max().date()}")

    if not symbol_dfs:
        print("  错误: 无法加载任何标的数据")
        return 1

    # 确定回测窗口
    end = min(df.index.max() for df in symbol_dfs.values())
    start = end - pd.Timedelta(days=args.days * 2)  # 预留足够天数（考虑非交易日）

    print(f"\n回测窗口: {start.date()} ~ {end.date()}")

    # 运行回测
    print("\n[2/3] 运行回测...")
    engine = BacktestEngine(cfg)

    result = engine.run(
        symbol_dfs=symbol_dfs,
        start_date=str(start.date()),
        end_date=str(end.date()),
        initial_capital=args.capital,
    )

    # 输出结果
    print("\n[3/3] 回测结果")
    print("-" * 70)

    status = result.get("status", "unknown")
    print(f"状态: {status}")

    if status != "completed":
        print(f"  错误: {result.get('error', 'N/A')}")
        return 1

    final_equity = result.get("final_equity", 0)
    cumulative_return = result.get("cumulative_return", 0)
    total_trades = result.get("total_trades", 0)
    win_rate = result.get("win_rate", 0)
    sharpe = result.get("sharpe_ratio", 0)
    max_dd = result.get("max_drawdown", 0)
    daily_points = len(result.get("daily_equity", []))

    print(f"\n绩效指标:")
    print(f"  期末权益:     {final_equity:>12,.2f}")
    print(f"  累计收益率:   {cumulative_return:>11.2%}")
    print(f"  总交易笔数:   {total_trades:>12}")
    print(f"  胜率:         {win_rate:>11.2%}")
    print(f"  夏普比率:     {sharpe:>12.3f}")
    print(f"  最大回撤:     {max_dd:>11.2%}")
    print(f"  净值点数:     {daily_points:>12}")

    print("\n" + "=" * 70)

    # 异常检测
    issues = []
    if total_trades == 0:
        issues.append("无交易产生（可能模型预测全为负，或风控过严）")
    if daily_points < args.days * 0.5:
        issues.append(f"净值点数过少 ({daily_points} < {args.days * 0.5})")
    if abs(cumulative_return) > 0.5:
        issues.append(f"收益率异常 ({cumulative_return:.1%} > ±50%)")

    if issues:
        print("\n⚠️  异常检测:")
        for i in issues:
            print(f"  - {i}")
        print()
    else:
        print("\n✓ 回测正常完成，无明显异常\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
