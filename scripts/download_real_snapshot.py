#!/usr/bin/env python3
"""Download real A-share snapshot and replace mock data.

Data source: akshare ``stock_zh_a_spot`` (Sina) — confirmed reachable in this
network when the local Windows proxy (127.0.0.1:7897) is bypassed.

Produces:
  1. ``A股数据/股票列表.csv`` — full A-share code/name catalog
  2. ``stock_info`` + ``stock_daily`` — today's real post-close snapshot (~5500 rows)

Also clears the previously injected mock prediction/signal rows so the database
holds only real market data (predictions will be re-populated once the Kronos
model is downloaded).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Bypass the broken local proxy (Windows system proxy 127.0.0.1:7897); direct
# connections to the Sina/gtimg market feed are confirmed working.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.data_layer.a_share_downloader import AShareDailyDownloader  # noqa: E402
from src.data_layer.storage import StorageLayer  # noqa: E402


def main() -> None:
    config = load_config()
    storage = StorageLayer(config)
    storage.init_database()

    # 1. Fetch the full A-share snapshot from Sina.
    print("[1/4] fetching A-share snapshot from Sina ...", flush=True)
    df = ak.stock_zh_a_spot()
    print(f"      got {len(df)} rows", flush=True)

    # Extract the real trade date from the snapshot timestamp.
    trade_date = None
    if "时间戳" in df.columns:
        stamp = df["时间戳"].dropna()
        if not stamp.empty:
            trade_date = str(stamp.iloc[0])[:10]
    print(f"      trade_date = {trade_date}", flush=True)

    # 2. Write the catalog used by both the downloader and the stock list API.
    cat_dir = Path("A股数据")
    cat_dir.mkdir(exist_ok=True)
    catalog = df[["代码", "名称"]].copy()
    catalog.columns = ["股票代码", "股票名称"]
    catalog["股票代码"] = catalog["股票代码"].astype(str).str.extract(r"(\d{6})")
    catalog = catalog.dropna(subset=["股票代码"]).drop_duplicates("股票代码")
    catalog["所属行业"] = ""
    catalog["上市状态"] = "上市"
    catalog_path = cat_dir / "股票列表.csv"
    catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")
    print(f"[2/4] wrote catalog: {catalog_path} ({len(catalog)} stocks)", flush=True)

    # 3. Clear mock prediction/signal rows (and stale market rows) so the DB
    #    ends up holding only real data.
    print("[3/4] clearing mock data ...", flush=True)
    for table in ("model_prediction", "trading_signal", "price_prediction", "stock_selection"):
        storage.execute_write(f"DELETE FROM {table}")
    storage.execute_write("DELETE FROM stock_daily")
    storage.execute_write("DELETE FROM stock_info")

    # 4. Import the real snapshot through the existing validated path.
    print("[4/4] importing snapshot ...", flush=True)
    snap_path = cat_dir / "_snapshot_tmp.csv"
    df.to_csv(snap_path, index=False, encoding="utf-8-sig")
    downloader = AShareDailyDownloader(config)
    result = downloader.import_snapshot(snap_path, as_of=trade_date)
    snap_path.unlink(missing_ok=True)
    print(f"      result: {result['status']}, rows={result['rows']}, date={result['date']}",
          flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
