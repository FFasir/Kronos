#!/usr/bin/env python3
"""Refresh per-stock 后复权 (hfq) overlay CSVs from Sina.

These overlay files hold one self-consistent full-history hfq series per
symbol and take precedence over the frozen Eastmoney archive member in
``StorageLayer._load_csv_raw_data``, so arbitrary-stock predictions and stock
pages use market data through the latest trading day.

Usage:
  python scripts/refresh_stock_hfq.py --symbols 000001,600519
  python scripts/refresh_stock_hfq.py --recent           # 最近被预测/入选的个股
  python scripts/refresh_stock_hfq.py --symbols 000001 --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger

logger = get_logger(__name__)


def _recent_stock_codes(config: Config, days: int = 30) -> list[str]:
    """Stock codes referenced by recent prediction/selection activity."""
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(config)
    configured_codes = sorted({str(item["code"]) for item in config.get_all_symbols()})
    since = (date.today() - timedelta(days=days)).isoformat()
    placeholders = ",".join("?" for _ in configured_codes) or "''"
    params = tuple(
        [since] + configured_codes
        + [since] + configured_codes
        + [since] + configured_codes
    )
    rows = storage.execute_sql(
        f"""SELECT code FROM model_prediction WHERE date >= ? AND code NOT IN ({placeholders})
            UNION SELECT code FROM price_prediction WHERE substr(created_at,1,10) >= ? AND code NOT IN ({placeholders})
            UNION SELECT code FROM stock_selection WHERE date >= ? AND code NOT IN ({placeholders})
            ORDER BY code""",
        params,
    )
    return [str(row["code"]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", help="逗号分隔的股票代码")
    parser.add_argument("--recent", action="store_true", help="刷新最近被预测/入选的个股")
    parser.add_argument("--force", action="store_true", help="忽略每日防抖强制刷新")
    parser.add_argument("--interval", type=float, default=1.0, help="请求间隔秒数")
    args = parser.parse_args()

    config = Config()
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.recent:
        symbols = _recent_stock_codes(config)
    else:
        parser.error("need --symbols or --recent")

    from src.data_layer.stock_hfq_live import refresh_symbols

    logger.info("refreshing %d symbols: %s", len(symbols), ",".join(symbols))
    result = refresh_symbols(symbols, config=config, force=args.force,
                             request_interval=args.interval)
    print(f"total={result['total']} refreshed={result['refreshed']}")
    for item in result["results"]:
        print(item)
    if result["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
