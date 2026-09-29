#!/usr/bin/env python3
"""Download research inputs that are absent from the supplied A-share archive.

ETF prices are retrieved from AKShare/Eastmoney.  Historical CSI 1000 membership
is retrieved only from Tushare Pro's ``index_weight`` endpoint, because a current
constituent snapshot cannot establish historical membership or weights.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ETF OHLCV and CSI 1000 historical weights")
    parser.add_argument("--etf", action="store_true", help="download ETF daily OHLCV with AKShare")
    parser.add_argument(
        "--current-components-akshare",
        action="store_true",
        help="download the current CSI 1000 constituent snapshot with AKShare",
    )
    parser.add_argument(
        "--historical-components",
        action="store_true",
        help="download CSI 1000 historical constituent weights with Tushare Pro",
    )
    parser.add_argument("--etf-symbol", default="512100", help="six-digit ETF code")
    parser.add_argument(
        "--etf-source",
        choices=["auto", "eastmoney", "sina"],
        default="auto",
        help="ETF history source; auto falls back to Sina when Eastmoney is unavailable",
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-09-30")
    parser.add_argument(
        "--adjust",
        choices=["", "qfq", "hfq"],
        default="",
        help="ETF adjustment mode; empty means unadjusted exchange prices",
    )
    parser.add_argument("--output-dir", default="data/raw/research")
    return parser.parse_args()


def normalize_etf_frame(raw: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    if set(column_map).issubset(raw.columns):
        df = raw[list(column_map)].rename(columns=column_map)
    else:
        required = ["date", "open", "high", "low", "close", "volume", "amount"]
        missing = set(required).difference(raw.columns)
        if missing:
            raise RuntimeError(f"unexpected ETF columns; missing {sorted(missing)}")
        df = raw[required].copy()
    df["date"] = pd.to_datetime(df["date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna().sort_values("date").drop_duplicates("date")
    if df.empty or (df[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise RuntimeError("ETF output has empty or non-positive OHLCV values")
    return df


def download_etf_eastmoney(ak, symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    chunks = []
    failures = []
    for begin, finish in yearly_ranges(start, end):
        frame = None
        for attempt in range(3):
            try:
                frame = ak.fund_etf_hist_em(
                    symbol=symbol,
                    period="daily",
                    start_date=begin.strftime("%Y%m%d"),
                    end_date=finish.strftime("%Y%m%d"),
                    adjust=adjust,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    failures.append(f"{begin.year}: {exc}")
                else:
                    time.sleep(2 * (attempt + 1))
        if frame is not None and not frame.empty:
            chunks.append(frame)
    if failures:
        raise RuntimeError("ETF download failed for one or more years: " + "; ".join(failures))
    if not chunks:
        raise RuntimeError(f"ETF {symbol} returned no rows")
    return normalize_etf_frame(pd.concat(chunks, ignore_index=True))


def download_etf_sina(ak, symbol: str, start: str, end: str) -> pd.DataFrame:
    failures = []
    for market in ("sh", "sz"):
        try:
            df = normalize_etf_frame(ak.fund_etf_hist_sina(symbol=f"{market}{symbol}"))
            df = df.loc[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
            if not df.empty:
                return df
        except Exception as exc:
            failures.append(f"{market}: {exc}")
    raise RuntimeError("Sina ETF download returned no usable rows: " + "; ".join(failures))


def download_etf(symbol: str, start: str, end: str, adjust: str, source: str, output_dir: Path) -> Path:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare is required for ETF download") from exc

    if source == "sina":
        df = download_etf_sina(ak, symbol, start, end)
        resolved_source = "sina"
    else:
        try:
            df = download_etf_eastmoney(ak, symbol, start, end, adjust)
            resolved_source = "eastmoney"
        except Exception:
            if source != "auto":
                raise
            df = download_etf_sina(ak, symbol, start, end)
            resolved_source = "sina"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{symbol}_etf_daily.csv"
    df.to_csv(output, index=False, date_format="%Y-%m-%d")
    print(
        f"ETF source={resolved_source} rows={len(df)} range={df.date.min().date()}..{df.date.max().date()} "
        f"output={output}"
    )
    return output


def download_current_components_akshare(output_dir: Path) -> Path:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare is required for constituent snapshot download") from exc

    raw = ak.index_stock_cons_csindex(symbol="000852")
    if raw.empty:
        raise RuntimeError("AKShare returned no current CSI 1000 constituents")
    code_column = next(
        (column for column in ["成分券代码", "品种代码", "股票代码"] if column in raw.columns),
        None,
    )
    if code_column is None:
        raise RuntimeError(f"unexpected AKShare constituent columns: {list(raw.columns)}")
    snapshot = pd.DataFrame({"code": raw[code_column].astype(str).str.strip().str.split(".").str[0].str.zfill(6)})
    snapshot = snapshot[snapshot["code"].str.fullmatch(r"\d{6}")].drop_duplicates().sort_values("code")
    if snapshot.empty:
        raise RuntimeError("AKShare constituent response contained no six-digit A-share codes")
    snapshot["snapshot_date"] = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"csi1000_akshare_{snapshot['snapshot_date'].iloc[0].replace('-', '')}.csv"
    snapshot.to_csv(output, index=False)
    print(f"current CSI 1000 snapshot members={len(snapshot)} output={output}")
    return output


def yearly_ranges(start: str, end: str):
    begin = pd.Timestamp(start).normalize()
    finish = pd.Timestamp(end).normalize()
    while begin <= finish:
        year_end = min(begin + pd.offsets.YearEnd(0), finish)
        yield begin, year_end
        begin = year_end + pd.Timedelta(days=1)


def download_historical_components(start: str, end: str, output_dir: Path) -> Path:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "TUSHARE_TOKEN is not configured. Set it in the shell before requesting "
            "historical CSI 1000 weights; do not substitute a current constituent snapshot."
        )
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("install tushare before requesting historical components") from exc

    pro = ts.pro_api(token)
    chunks = []
    for begin, finish in yearly_ranges(start, end):
        frame = pro.index_weight(
            index_code="000852.SH",
            start_date=begin.strftime("%Y%m%d"),
            end_date=finish.strftime("%Y%m%d"),
        )
        if frame is not None and not frame.empty:
            chunks.append(frame)
    if not chunks:
        raise RuntimeError("Tushare returned no CSI 1000 historical weights")
    raw = pd.concat(chunks, ignore_index=True).drop_duplicates()
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"unexpected Tushare index_weight columns; missing {sorted(missing)}")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    raw["weight"] = pd.to_numeric(raw["weight"], errors="coerce")
    raw = raw.dropna(subset=["con_code", "trade_date", "weight"]).sort_values(["trade_date", "con_code"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "csi1000_index_weight_history.csv"
    raw.to_csv(output, index=False, date_format="%Y-%m-%d")
    print(
        f"weight rows={len(raw)} adjustment_dates={raw.trade_date.nunique()} "
        f"range={raw.trade_date.min().date()}..{raw.trade_date.max().date()} output={output}"
    )
    return output


def main() -> None:
    args = parse_args()
    if not args.etf and not args.historical_components and not args.current_components_akshare:
        raise SystemExit("select --etf, --historical-components, and/or --current-components-akshare")
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if args.etf:
        download_etf(args.etf_symbol, args.start, args.end, args.adjust, args.etf_source, output_dir)
    if args.current_components_akshare:
        download_current_components_akshare(output_dir)
    if args.historical_components:
        download_historical_components(args.start, args.end, output_dir)


if __name__ == "__main__":
    main()
