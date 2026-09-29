#!/usr/bin/env python3
"""
Kronos 择时系统 — 数据库初始化 CLI

用法:
    python scripts/init_db.py

    执行后创建 data/kronos.db（SQLite），包含15张表。
    幂等操作：重复执行不会报错。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.data_layer.storage import StorageLayer


def main():
    # 简单参数解析
    import argparse
    parser = argparse.ArgumentParser(description="Kronos 择时系统 — 数据库初始化")
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
    logger = get_logger("scripts.init_db")

    config = load_config(args.config)

    storage = StorageLayer(config)
    db_path = storage.init_database()

    # 验证
    tables = storage.list_tables()
    print("\n" + "=" * 50)
    print(f"数据库初始化完成: {db_path}")
    print(f"已创建 {len(tables)} 张表:")
    for i, t in enumerate(tables, 1):
        print(f"  {i:2d}. {t}")
    print("=" * 50)

    if len(tables) >= 15:
        logger.info(f"全部 15 张表创建成功 ✓")
        sys.exit(0)
    else:
        logger.warning(f"预期 15 张表，实际 {len(tables)} 张")
        sys.exit(1)


if __name__ == "__main__":
    main()
