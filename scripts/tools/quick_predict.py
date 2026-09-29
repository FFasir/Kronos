#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速预测工具

用法: python scripts/tools/quick_predict.py

获取所有标的的最新预测信号（用于验证系统是否正常运行）
"""
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.common.config import load_config
from src.model_layer.kronos_predictor import KronosPredictor
from src.model_layer.model_factory import create_model
from src.strategy_layer.signal_generator import SignalGenerator

import pandas as pd


def load_csv_data(raw_dir: Path) -> dict:
    """从 data/raw/*.csv 加载各标的日线数据"""
    dfs = {}
    for csv_path in sorted(raw_dir.glob("*_daily.csv")):
        code = csv_path.stem.split("_")[0]
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        dfs[code] = df[["open", "high", "low", "close", "volume"]]
    return dfs


def main():
    print("=" * 70)
    print("Kronos 快速预测 - 最新信号")
    print("=" * 70)

    cfg = load_config()

    # 加载模型
    print("\n[1/3] 加载模型...")
    model, path_label = create_model(cfg)
    predictor = KronosPredictor(model, cfg)
    print(f"  模型: {path_label}")

    # 加载数据并预测（直接从 data/raw CSV，避免依赖 HDF5）
    print("\n[2/3] 预测所有标的...")
    symbol_dfs = load_csv_data(Path("data/raw"))
    if not symbol_dfs:
        print("  错误: data/raw/ 下没有 *_daily.csv 文件")
        return 1

    predictions = []
    for code, df in symbol_dfs.items():
        if len(df) < predictor.seq_len:
            print(f"  {code}: 数据不足 ({len(df)} < {predictor.seq_len})，跳过")
            continue
        try:
            pred = predictor.predict_single(df)
            pred["symbol"] = code
            predictions.append(pred)
        except Exception as e:
            print(f"  {code}: 预测失败 - {e}")
    print(f"  完成: {len(predictions)} 个标的")

    # 生成信号
    print("\n[3/3] 生成信号...")
    sig_gen = SignalGenerator(cfg)
    signals = sig_gen.generate(predictions=predictions, current_date=None, existing_positions={})

    # 按预测收益率排序显示
    predictions.sort(key=lambda x: x.get("pred_return", 0), reverse=True)

    print("\n" + "=" * 70)
    print(f"{'标的':<10} {'名称':<12} {'预测收益率':>10} {'信号':>8} {'置信度':>8}")
    print("-" * 70)

    for pred in predictions:
        symbol = pred.get("symbol", "N/A")
        # 从配置获取名称
        symbol_objs = cfg.get_all_symbols()
        name = next((s["name"] for s in symbol_objs if s["code"] == symbol), symbol)
        pred_ret = pred.get("pred_return", 0.0)
        conf = pred.get("confidence", 0.0)

        # 找对应信号
        sig = next((s for s in signals if s.get("symbol") == symbol), None)
        sig_type = str(sig["signal_type"]) if sig else "N/A"

        print(f"{symbol:<10} {name:<12} {pred_ret:>9.2%} {sig_type:>8} {conf:>7.2f}")

    print("=" * 70)

    # 统计
    buy_count = sum(1 for s in signals if str(s["signal_type"]) == "BUY")
    sell_count = sum(1 for s in signals if str(s["signal_type"]) == "SELL")
    hold_count = sum(1 for s in signals if str(s["signal_type"]) == "HOLD")
    cash_count = sum(1 for s in signals if str(s["signal_type"]) == "CASH")

    print(f"\n信号统计: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}, CASH={cash_count}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
