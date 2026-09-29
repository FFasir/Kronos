#!/usr/bin/env python3
"""Import local historical market data from the sibling ``data`` directory.

The supplied archives live one level above the code checkout, at
``F:/jinrong_analysis/data``.  This script:

  1. Extracts the six configured indices from ``指数_日_kline.zip`` into
     ``data/raw/research/{code}_daily.csv`` (the path the storage layer already
     reads via ``_load_csv_raw_data``).
  2. Points ``a_share_archive`` in ``config/system.yaml`` at the real
     ``daily_hfq.zip`` so individual-stock K-lines read straight from the
     archive without extracting ~6000 CSVs.

ETFs (512100 / 510300 / 510500 / 159915 / 588000 / 510050) are absent from the
local archives and are handled separately (see ``--etfs``).

Usage:
    python scripts/import_local_data.py
    python scripts/import_local_data.py --data-dir F:/jinrong_analysis/data
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 6 个配置的指数标的（与 config/system.yaml index_list 对齐）
INDEX_SYMBOLS = {
    "000852": "中证1000",
    "000300": "沪深300",
    "000905": "中证500",
    "399006": "创业板指",
    "000688": "科创50",
    "000016": "上证50",
}

# 6 个配置的 ETF 标的（本地归档缺失，需联网下载）
ETF_SYMBOLS = {
    "512100": "中证1000ETF",
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
    "510050": "上证50ETF",
}

_COLUMN_MAP = {
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
}


def import_indices(data_dir: Path, out_dir: Path) -> None:
    archive = data_dir / "指数_日_kline.zip"
    if not archive.is_file():
        print(f"  [skip] 指数归档不存在: {archive}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        available = set(z.namelist())
        for code, name in INDEX_SYMBOLS.items():
            member = f"{code}_日.csv"
            if member not in available:
                print(f"  [skip] {code} {name}: 归档中无 {member}")
                continue
            raw = pd.read_csv(z.open(member), encoding="utf-8-sig")
            df = raw.rename(columns=_COLUMN_MAP)
            keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df]
            df = df[keep]
            out = out_dir / f"{code}_daily.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  [ok] {code} {name}: {len(df)} 行 -> {out.name}")


def download_etfs(out_dir: Path) -> None:
    """Download ETF daily OHLCV with akshare (Sina source), bypassing the local proxy."""
    import os
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    import akshare as ak

    out_dir.mkdir(parents=True, exist_ok=True)
    for code, name in ETF_SYMBOLS.items():
        symbol = f"sh{code}" if code.startswith(("5", "6")) else f"sz{code}"
        try:
            df = ak.fund_etf_hist_sina(symbol=symbol)
            df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                    "low": "low", "close": "close", "volume": "volume"})
            keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df]
            df = df[keep]
            out = out_dir / f"{code}_etf_daily.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  [ok] {code} {name}: {len(df)} 行 -> {out.name}")
        except Exception as exc:
            print(f"  [fail] {code} {name}: {exc}")


def update_archive_config(data_dir: Path) -> None:
    """Point a_share_archive at the real daily_hfq.zip via a relative path."""
    yaml_path = PROJECT_ROOT / "config" / "system.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    relative = Path("..") / data_dir.name / "daily_hfq.zip"
    old = 'a_share_archive: "A股数据/daily_hfq.zip"'
    new = f'a_share_archive: "{relative.as_posix()}"'
    if old in text:
        text = text.replace(old, new)
        yaml_path.write_text(text, encoding="utf-8")
        print(f"  [ok] config a_share_archive -> {relative.as_posix()}")
    elif "daily_hfq.zip" in text:
        print("  [warn] a_share_archive 已是自定义值，未改动，请手动确认")
    else:
        print("  [warn] 未找到 a_share_archive 配置行，请手动确认")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import local market data archives")
    parser.add_argument("--data-dir", type=Path, default=Path(r"F:\jinrong_analysis\data"))
    parser.add_argument("--etfs", action="store_true", help="download missing ETF daily OHLCV")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"data 目录不存在: {data_dir}", file=sys.stderr)
        return 2

    research = PROJECT_ROOT / "data" / "raw" / "research"

    print("[1/3] 导入指数日线 ...")
    import_indices(data_dir, research)

    print("[2/3] 指向个股归档 ...")
    update_archive_config(data_dir)

    if args.etfs:
        print("[3/3] 下载 ETF 日线 ...")
        download_etfs(research)
    else:
        print("[3/3] ETF 归档本地缺失，跳过（可加 --etfs 联网下载）")

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
