#!/usr/bin/env python3
"""
Kronos 择时系统 — 数据预处理 CLI

用法:
    # 完整预处理流水线（默认处理所有已下载标的）
    python scripts/preprocess.py

    # 仅处理指定标的
    python scripts/preprocess.py --symbols 000852,000300,512100

    # 指定训练集范围（用于 scaler 拟合）
    python scripts/preprocess.py --train-start 2010-01-01 --train-end 2022-12-31
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.data_layer.preprocessor import Preprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Kronos 择时系统 — 数据预处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的标的代码列表（默认处理全部）",
    )
    parser.add_argument(
        "--train-start", type=str, default=None,
        help="训练集起始日期（用于 scaler 拟合），默认使用 finetune_train_start",
    )
    parser.add_argument(
        "--train-end", type=str, default=None,
        help="训练集结束日期（用于 scaler 拟合），默认使用 finetune_train_end",
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

    # 初始化
    setup_logger(level=args.log_level, log_dir="logs/", console=True)
    logger = get_logger("scripts.preprocess")

    config = load_config(args.config)

    # 解析标的列表
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    # 执行预处理
    try:
        preprocessor = Preprocessor(config)
        h5_path = preprocessor.run(
            symbols=symbols,
            train_start=args.train_start,
            train_end=args.train_end,
        )

        print("\n" + "=" * 50)
        print(f"预处理完成!")
        print(f"HDF5 文件: {h5_path}")
        print(f"Scaler 参数: {config.data.get('processed_dir')}/scaler_params.json")
        print(f"质量报告:   {config.data.get('processed_dir')}/quality_report.json")
        print(f"异常日志:   {config.data.get('processed_dir')}/anomaly_log.json")
        print("=" * 50)

        sys.exit(0)

    except Exception as e:
        logger.error(f"预处理异常终止: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
