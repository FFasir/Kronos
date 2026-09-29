#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库状态查看工具

用法: python scripts/tools/db_status.py

显示 SQLite 数据库各表的记录数、最新日期等状态信息
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
from src.data_layer.storage import StorageLayer


def main():
    print("=" * 70)
    print("Kronos 数据库状态")
    print("=" * 70)

    cfg = load_config()
    storage = StorageLayer(cfg)

    # 关键表及其字段
    tables = {
        "model_prediction": {
            "name": "模型预测",
            "date_col": "pred_date",
        },
        "trading_signal": {
            "name": "交易信号",
            "date_col": "date",
        },
        "backtest_summary": {
            "name": "回测记录",
            "date_col": "created_at",
        },
        "sim_trade": {
            "name": "回测交易",
            "date_col": "trade_date",
        },
        "training_log": {
            "name": "训练日志",
            "date_col": "start_time",
        },
    }

    print(f"\n数据库: {cfg.data.get('db_path', 'data/kronos.db')}\n")
    print(f"{'表名':<20} {'中文名':<12} {'记录数':>10} {'最新日期':<20}")
    print("-" * 70)

    for table, info in tables.items():
        try:
            # 计数
            count_row = storage.execute_sql(f"SELECT COUNT(*) FROM {table}")
            count = count_row[0][0] if count_row else 0

            # 最新日期
            date_col = info["date_col"]
            date_row = storage.execute_sql(
                f"SELECT MAX({date_col}) FROM {table}"
            )
            latest = date_row[0][0] if date_row and date_row[0][0] else "N/A"

            # 格式化日期（截取前19字符，去掉毫秒）
            if latest != "N/A" and len(latest) > 19:
                latest = latest[:19]

            print(f"{table:<20} {info['name']:<12} {count:>10,} {latest:<20}")

        except Exception as e:
            print(f"{table:<20} {info['name']:<12} {'ERROR':>10} {str(e)[:20]:<20}")

    print("=" * 70)

    # HDF5 状态
    print("\nHDF5 存储:")
    hdf5_path = Path(cfg.data.get("hdf5_path", "data/market_data.h5"))
    if hdf5_path.exists():
        size_mb = hdf5_path.stat().st_size / 1024 / 1024
        print(f"  路径: {hdf5_path}")
        print(f"  大小: {size_mb:.1f} MB")
    else:
        print(f"  路径: {hdf5_path}")
        print(f"  状态: 不存在")

    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
