#!/usr/bin/env python3
"""
Kronos 择时系统 — 数据下载 CLI

用法:
    # 全量下载所有12个标的
    python scripts/download.py --mode full

    # 增量下载（仅下载上次更新后的新数据）
    python scripts/download.py --mode incremental

    # 单标的下载（调试用）
    python scripts/download.py --mode single --symbol 000852

    # 指定日期范围
    python scripts/download.py --mode full --start 2020-01-01 --end 2024-12-31

    # 下载特定标的
    python scripts/download.py --mode full --symbols 000852,000300,512100
"""

import sys
import argparse
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.data_layer.a_share_downloader import AShareDailyDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Kronos 择时系统 — 数据下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["full", "incremental", "single"], default="full",
        help="下载模式: full(全量) / incremental(增量) / single(单标的)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="单标的下载时的标的代码（如 000852），仅 --mode single 时有效",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的标的代码列表（如 000852,512100），覆盖配置",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="起始日期 YYYY-MM-DD（默认使用配置文件值）",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="结束日期 YYYY-MM-DD（默认今天）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（默认 config/system.yaml）",
    )
    parser.add_argument(
        "--snapshot-file", type=Path, default=None,
        help="导入本地全A股日线快照 CSV，不访问网络",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 初始化
    setup_logger(level=args.log_level, log_dir="logs/", console=True)
    logger = get_logger("scripts.download")

    config = load_config(args.config)

    # 解析标的列表
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.symbol:
        symbols = [args.symbol]

    # 单标的模式校验
    if args.mode == "single":
        if not symbols or len(symbols) != 1:
            logger.error("single 模式需要 --symbol 参数指定单个标的代码")
            sys.exit(2)

    # 执行下载
    downloader = AShareDailyDownloader(config)
    try:
        if args.snapshot_file:
            result = downloader.import_snapshot(args.snapshot_file, as_of=args.end)
        else:
            result = downloader.download(as_of=args.end, symbols=symbols)

        # 输出摘要
        print("\n" + "=" * 50)
        print(f"下载状态: {result['status']}")
        print(f"交易日期: {result['date']}")
        print(f"写入标的: {result['rows']}")
        if result.get("error"):
            print(f"错误: {result['error']}")
        print(f"数据目录: {config.data.get('raw_dir', 'data/raw/')}")
        print("=" * 50)

        # 退出码
        if result["status"] == "completed":
            sys.exit(0)
        else:
            sys.exit(2)

    except Exception as e:
        logger.error(f"下载异常终止: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
