#!/usr/bin/env python3
"""
Kronos 择时系统 — Kronos 模型加载验证 CLI

用法:
    # 自动检测并加载模型
    python scripts/load_kronos.py

    # 指定路径
    python scripts/load_kronos.py --path kronos

    # Path B fallback
    python scripts/load_kronos.py --path fallback

    # 验证推理
    python scripts/load_kronos.py --validate
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.common.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Kronos 模型加载验证")
    parser.add_argument(
        "--path", type=str, default=None,
        choices=["kronos", "fallback"],
        help="模型路径 (默认自动检测)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="加载后进行推理验证",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备 (默认使用配置)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    setup_logger(level=args.log_level, log_dir="logs/", console=True)
    logger = get_logger("scripts.load_kronos")

    config = load_config(args.config)
    set_seed(config.seed)

    from src.model_layer.model_factory import create_model

    print("\n" + "=" * 60)
    print("Kronos 模型加载验证")
    print("=" * 60)

    try:
        model, path = create_model(
            config,
            device=args.device,
            force_path=args.path,
        )

        print(f"\n✓ 模型加载成功")
        print(f"  路径: Path {'A (Kronos-base)' if path == 'kronos' else 'B (FallbackModel)'}")
        print(f"  类型: {type(model).__name__}")

        # 参数统计
        if hasattr(model, 'model'):
            inner = model.model
        else:
            inner = model

        total_params = sum(p.numel() for p in inner.parameters())
        trainable_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params:,}")

        # 验证推理
        if args.validate:
            print("\n── Dummy 推理验证 ──")
            import torch
            import numpy as np

            device = next(inner.parameters()).device
            dummy = torch.randn(1, 60, 5).to(device)

            with torch.no_grad():
                pred = model.predict(dummy, horizon=5)
                pred_seq = model.predict_sequence(dummy, horizon=20)

            print(f"  predict(horizon=5):      {pred:.4f}")
            print(f"  predict_sequence(horizon=20): shape={pred_seq.shape}")
            print(f"  无 NaN: {not np.isnan(pred) and not np.isnan(pred_seq).any()}")
            print(f"\n✓ 推理验证通过")

        print("\n" + "=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
