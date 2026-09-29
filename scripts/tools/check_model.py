#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型健康检查工具

用法: python scripts/tools/check_model.py

检查项:
  1. checkpoint 文件是否存在
  2. 模型是否可加载
  3. 单次预测是否产生合理输出（非 NaN、收益率在合理区间）
  4. 预测器是否正确初始化
"""
import sys
import io
from pathlib import Path

# 强制 UTF-8 输出（Windows 兼容）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import pandas as pd
import numpy as np
from src.common.config import load_config
from src.model_layer.model_factory import create_model, detect_model_path
from src.model_layer.kronos_predictor import KronosPredictor


def main():
    print("=" * 60)
    print("Kronos 模型健康检查")
    print("=" * 60)

    cfg = load_config()

    # 1. 检查模型路径
    print("\n[1/4] 检查模型路径...")
    path = detect_model_path(cfg)
    print(f"  检测到: {path}")

    if path == "fallback":
        ckpt = Path("models/checkpoints/fallback_best.pt")
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1024 / 1024
            mtime = ckpt.stat().st_mtime
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Checkpoint: {ckpt}")
            print(f"  大小: {size_mb:.1f} MB")
            print(f"  修改时间: {mtime_str}")
        else:
            print(f"  ❌ Checkpoint 不存在: {ckpt}")
            return 1
    else:
        print(f"  使用 Kronos (HuggingFace): {cfg.model.get('pretrained_path', 'N/A')}")

    # 2. 尝试加载模型
    print("\n[2/4] 加载模型...")
    try:
        model, path_label = create_model(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ 模型已加载 ({path_label})")
        print(f"  参数量: {n_params:,}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return 1

    # 3. 初始化预测器
    print("\n[3/4] 初始化预测器...")
    try:
        predictor = KronosPredictor(model, cfg)
        print(f"  ✓ KronosPredictor 已初始化")
        print(f"  seq_len={predictor.seq_len}, pred_len={predictor.pred_len}")
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return 1

    # 4. 单次预测测试
    print("\n[4/4] 单次预测测试...")
    try:
        # 加载一个真实 CSV 的最后 60 天
        csv_path = Path("data/raw/000300_沪深300_daily.csv")
        if not csv_path.exists():
            print(f"  ⚠️  测试数据不存在: {csv_path}")
            print(f"  跳过预测测试")
        else:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]]

            if len(df) < predictor.seq_len:
                print(f"  ⚠️  数据不足 ({len(df)} < {predictor.seq_len})")
            else:
                pred = predictor.predict_single(df)

                last_close = float(df["close"].iloc[-1])
                pred_close = pred["pred_close_t5"]
                pred_return = pred["pred_return"]
                confidence = pred.get("confidence", 0.0)

                print(f"  ✓ 预测完成")
                print(f"    末收盘价: {last_close:.2f}")
                print(f"    预测T+5收盘: {pred_close:.2f}")
                print(f"    预测收益率: {pred_return:.2%}")
                print(f"    置信度: {confidence:.2f}")

                # 合理性检查
                issues = []
                if not np.isfinite(pred_close) or not np.isfinite(pred_return):
                    issues.append("预测值包含 NaN/Inf")
                if abs(pred_return) > 0.5:
                    issues.append(f"预测收益率异常 ({pred_return:.1%} > ±50%)")
                if pred_close < 0:
                    issues.append("预测价格为负")

                if issues:
                    print("\n  ⚠️  异常检测:")
                    for i in issues:
                        print(f"    - {i}")
                    print("\n  → 模型可能未正确训练或归一化存在问题")
                else:
                    print("\n  ✓ 预测值在合理区间")

    except Exception as e:
        print(f"  ❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("健康检查完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
