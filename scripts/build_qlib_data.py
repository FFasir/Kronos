#!/usr/bin/env python3
"""Build official Kronos Qlib daily data directly from an A-share ZIP archive.

The input archive is never extracted.  Each selected CSV is read from the ZIP,
validated, and written in Qlib's standard ``calendars/instruments/features``
binary layout.  The result is consumed by
``third_party/Kronos/finetune/qlib_data_preprocess.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.logger import get_logger, setup_logger
from src.data_layer.a_share_rules import (
    StockMetadata,
    detect_price_limits,
    load_stock_metadata,
)


RAW_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
}
PRICE_COLUMNS = ["open", "high", "low", "close"]
QLIB_FEATURES = [
    "open", "close", "high", "low", "volume", "vwap", "limit_up", "limit_down"
]


def read_symbols(path: Path) -> set[str]:
    table = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if table.empty:
        raise ValueError(f"symbol file is empty: {path}")
    return {
        str(value).strip().split(".")[0].zfill(6)
        for value in table.iloc[:, 0].dropna()
        if str(value).strip()
    }


def read_historical_components(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DatetimeIndex, list[set[str]]]:
    """Read Tushare ``index_weight`` snapshots as effective-date memberships."""
    table = pd.read_csv(path, dtype={"con_code": str}, encoding="utf-8-sig")
    required = {"con_code", "trade_date"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"historical components missing columns: {sorted(missing)}")
    table["trade_date"] = pd.to_datetime(
        table["trade_date"].astype(str).str.replace(
            r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True
        ),
        errors="coerce",
    )
    table["code"] = table["con_code"].str.strip().str.split(".").str[0].str.zfill(6)
    if "weight" in table.columns:
        table["weight"] = pd.to_numeric(table["weight"], errors="coerce")
    table = table.dropna(subset=["trade_date", "code"])
    table = table[(table["trade_date"] >= start) & (table["trade_date"] <= end)]
    if table.empty:
        raise ValueError("historical components contain no rows in the requested date range")
    grouped = table.groupby("trade_date", sort=True)["code"].agg(lambda values: set(values))
    snapshot_dates = pd.DatetimeIndex(grouped.index)
    memberships = grouped.tolist()
    if not all(memberships):
        raise ValueError("historical components include an empty adjustment snapshot")
    return snapshot_dates, memberships


def filter_historical_membership(
    frame: pd.DataFrame,
    code: str,
    snapshot_dates: pd.DatetimeIndex,
    memberships: list[set[str]],
) -> pd.DataFrame:
    """Keep a stock only after the latest available snapshot includes it."""
    locations = snapshot_dates.searchsorted(frame.index, side="right") - 1
    include = np.fromiter(
        (location >= 0 and code in memberships[location] for location in locations),
        dtype=bool,
        count=len(locations),
    )
    return frame.loc[include]


def archive_code(name: str) -> str:
    return Path(name).name.split("_")[0].zfill(6)


def qlib_symbol(code: str) -> str:
    if code.startswith(("60", "68", "69", "90")):
        return f"SH{code}"
    if code.startswith(("43", "83", "87", "88", "92")):
        return f"BJ{code}"
    return f"SZ{code}"


def prepare_frame(
    raw: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metadata: StockMetadata,
) -> pd.DataFrame | None:
    frame = raw.rename(columns=RAW_COLUMNS)
    required = {"date", *PRICE_COLUMNS, "volume", "pct_change"}
    if not required.issubset(frame.columns):
        return None

    keep = ["date", *PRICE_COLUMNS, "volume", "pct_change"]
    if "amount" in frame.columns:
        keep.append("amount")
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in keep[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", *PRICE_COLUMNS, "volume", "pct_change"])
    frame = frame[(frame[PRICE_COLUMNS] > 0).all(axis=1) & (frame["high"] >= frame["low"])]
    frame = frame[frame["volume"] >= 0]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["trading_day_number"] = np.arange(1, len(frame) + 1)
    first_day_dropped = 0
    if metadata.listing_date is not None:
        listing_day = frame["date"].eq(metadata.listing_date)
        first_day_dropped = int(listing_day.sum())
        frame = frame.loc[~listing_day]
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if frame.empty:
        return None

    if "amount" not in frame:
        frame["amount"] = np.nan
    # Raw turnover divided by volume is not on the same scale as HFQ prices.
    # Use an HFQ-consistent typical price; downstream code reconstructs amount.
    frame["vwap"] = frame[PRICE_COLUMNS].mean(axis=1)
    dates = pd.DatetimeIndex(frame["date"])
    limit_up, limit_down = detect_price_limits(
        dates,
        frame["pct_change"],
        metadata,
        frame["trading_day_number"],
        high_prices=frame["high"],
        low_prices=frame["low"],
        close_prices=frame["close"],
    )
    frame = frame.set_index("date")
    frame["limit_up"] = limit_up
    frame["limit_down"] = limit_down
    result = frame[QLIB_FEATURES]
    result.attrs["first_day_dropped"] = first_day_dropped
    result.attrs["limit_up_count"] = int(result["limit_up"].sum())
    result.attrs["limit_down_count"] = int(result["limit_down"].sum())
    return result


def load_calendar(path: Path | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex | None:
    if path is None or not path.is_file():
        return None
    table = pd.read_csv(path, encoding="utf-8-sig")
    if not {"日期", "是否交易"}.issubset(table.columns):
        return None
    dates = pd.to_datetime(table.loc[table["是否交易"] == "交易", "日期"], errors="coerce")
    dates = dates[(dates >= start) & (dates <= end)].dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def infer_calendar_from_archive(
    archive: Path, selected: set[str] | None, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Build a market calendar from selected ZIP members when no CSV was uploaded."""
    dates: set[pd.Timestamp] = set()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not entry.filename.lower().endswith(".csv"):
                continue
            if selected is not None and archive_code(entry.filename) not in selected:
                continue
            try:
                with source.open(entry) as handle:
                    raw_dates = pd.read_csv(handle, usecols=["日期"], encoding="utf-8-sig")["日期"]
                parsed = pd.to_datetime(raw_dates, errors="coerce")
                dates.update(day for day in parsed.dropna() if start <= day <= end)
            except Exception:
                continue
    return pd.DatetimeIndex(sorted(dates))


def write_feature(path: Path, values: pd.Series, calendar: pd.DatetimeIndex) -> None:
    valid = values.notna().to_numpy().nonzero()[0]
    if len(valid) == 0:
        return
    start_index, end_index = int(valid[0]), int(valid[-1])
    payload = np.concatenate((
        np.asarray([start_index], dtype="<f4"),
        values.iloc[start_index:end_index + 1].to_numpy(dtype="<f4"),
    ))
    payload.tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HFQ Qlib data without extracting the ZIP")
    parser.add_argument("--archive", type=Path, default=ROOT / "A股数据" / "daily_hfq.zip")
    parser.add_argument(
        "--stock-metadata",
        type=Path,
        nargs="+",
        default=[ROOT / "A股数据" / "股票列表.csv", ROOT / "A股数据" / "退市股票列表.csv"],
        help="current and delisted stock master CSV files",
    )
    parser.add_argument("--symbols-file", type=Path, help="CSV with CSI 1000 codes in its first column")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="use every valid stock CSV in the archive; incompatible with --symbols-file and --historical-components",
    )
    parser.add_argument(
        "--historical-components",
        type=Path,
        help="Tushare index_weight CSV; enables effective-date CSI 1000 membership filtering",
    )
    parser.add_argument("--output", type=Path, help="Qlib provider directory; default from system.yaml")
    parser.add_argument("--instrument", help="Qlib instrument name; default csi1000")
    parser.add_argument("--calendar-source", type=Path, default=ROOT / "A股数据" / "交易日历.csv")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-09-30")
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--min-val-rows", type=int, default=250)
    parser.add_argument("--max-symbols", type=int, help="limit conversion for a non-training smoke test")
    parser.add_argument("--config", help="system YAML path")
    args = parser.parse_args()

    setup_logger(log_dir="logs/", console=True)
    logger = get_logger("scripts.build_qlib_data")
    config = load_config(args.config)
    archive = args.archive.resolve()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    if "hfq" in archive.stem.lower():
        adjustment = "hfq"
    elif "qfq" in archive.stem.lower():
        adjustment = "qfq"
    else:
        raise ValueError(
            f"A-share training requires an HFQ or QFQ archive; filename does not identify it: {archive.name}"
        )
    metadata_paths = [path.resolve() for path in args.stock_metadata]
    missing_metadata = [str(path) for path in metadata_paths if not path.is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"stock metadata files are missing: {missing_metadata}")
    stock_metadata = load_stock_metadata(metadata_paths)
    if args.all_symbols and (args.symbols_file is not None or args.historical_components is not None):
        raise ValueError("--all-symbols cannot be combined with a symbols or historical components file")
    if not args.all_symbols and args.symbols_file is None and args.historical_components is None:
        raise ValueError("provide --all-symbols, --symbols-file, and/or --historical-components")

    output = args.output or Path(config.data["qlib_data_path"])
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    instrument = args.instrument or config.data.get("csi1000_instrument", "csi1000")
    snapshot_dates = None
    memberships = None
    if args.all_symbols:
        history_path = None
        selected = None
        logger.info("using all valid stock CSV members from the archive")
    elif args.historical_components is not None:
        history_path = args.historical_components.resolve()
        if not history_path.is_file():
            raise FileNotFoundError(f"historical components file not found: {history_path}")
        snapshot_dates, memberships = read_historical_components(history_path, start, end)
        selected = set().union(*memberships)
        logger.info("using %s historical membership snapshots covering %s candidate symbols", len(snapshot_dates), len(selected))
    else:
        history_path = None
        selected = set()
    if not args.all_symbols and args.symbols_file is not None:
        symbols_file = args.symbols_file.resolve()
        if not symbols_file.is_file():
            raise FileNotFoundError(f"symbols file not found: {symbols_file}")
        static_symbols = read_symbols(symbols_file)
        if memberships is None:
            selected = static_symbols
        elif not selected.issubset(static_symbols):
            logger.warning("static symbols file is ignored because it omits historical members")
    calendar = load_calendar(args.calendar_source, start, end)
    if calendar is None or calendar.empty:
        logger.warning("trading calendar CSV is unavailable; deriving it from selected ZIP members")
        calendar = infer_calendar_from_archive(archive, selected, start, end)
    if calendar.empty:
        raise ValueError("no trading dates found in the selected ZIP members")

    features_root = output / "features"
    instruments_root = output / "instruments"
    calendars_root = output / "calendars"
    features_root.mkdir(parents=True, exist_ok=True)
    instruments_root.mkdir(parents=True, exist_ok=True)
    calendars_root.mkdir(parents=True, exist_ok=True)

    imported, skipped = [], {
        "not_selected": 0,
        "metadata_missing": 0,
        "invalid": 0,
        "out_of_membership": 0,
        "insufficient_history": 0,
    }
    with zipfile.ZipFile(archive) as source:
        entries = [item for item in source.infolist() if item.filename.lower().endswith(".csv")]
        for entry in entries:
            code = archive_code(entry.filename)
            if selected is not None and code not in selected:
                skipped["not_selected"] += 1
                continue
            if args.max_symbols is not None and len(imported) >= args.max_symbols:
                break
            metadata = stock_metadata.get(code)
            if metadata is None:
                skipped["metadata_missing"] += 1
                continue
            try:
                with source.open(entry) as handle:
                    frame = prepare_frame(
                        pd.read_csv(handle, encoding="utf-8-sig"), start, end, metadata
                    )
            except Exception as exc:
                logger.warning("skip %s: %s", code, exc)
                skipped["invalid"] += 1
                continue
            if frame is None:
                skipped["invalid"] += 1
                continue
            if memberships is not None:
                frame = filter_historical_membership(frame, code, snapshot_dates, memberships)
                if frame.empty:
                    skipped["out_of_membership"] += 1
                    continue
            train_rows = len(frame.loc["2010-01-01":"2022-12-31"])
            val_rows = len(frame.loc["2023-01-01":"2024-12-31"])
            if train_rows < args.min_train_rows or val_rows < args.min_val_rows:
                skipped["insufficient_history"] += 1
                continue

            symbol = qlib_symbol(code)
            feature_dir = features_root / symbol.lower()
            feature_dir.mkdir(exist_ok=True)
            aligned = frame.reindex(calendar)
            for feature in QLIB_FEATURES:
                write_feature(feature_dir / f"{feature}.day.bin", aligned[feature], calendar)
            first, last = frame.index.min().strftime("%Y-%m-%d"), frame.index.max().strftime("%Y-%m-%d")
            imported.append({
                "code": code,
                "qlib_symbol": symbol,
                "start": first,
                "end": last,
                "first_day_dropped": int(frame.attrs.get("first_day_dropped", 0)),
                "limit_up_count": int(frame.attrs.get("limit_up_count", 0)),
                "limit_down_count": int(frame.attrs.get("limit_down_count", 0)),
            })
            if len(imported) % 100 == 0:
                logger.info("converted %s symbols", len(imported))

    (calendars_root / "day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    instrument_lines = [f"{item['qlib_symbol']}\t{item['start']}\t{item['end']}" for item in imported]
    (instruments_root / f"{instrument}.txt").write_text("\n".join(instrument_lines) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "archive": str(archive),
        "adjustment": adjustment,
        "stock_metadata": [str(path) for path in metadata_paths],
        "first_listing_day_policy": "drop exact metadata listing date",
        "st_history_policy": (
            "historical 5% limits are conservatively inferred from pct change plus close at high/low; "
            "the supplied metadata has no historical name timeline"
        ),
        "tokenizer_limit_edge_policy": "limit-up high * 1.02; limit-down low * 0.98",
        "backtest_limit_policy": "limit-up blocks buy; limit-down blocks sell",
        "archive_members": len(entries),
        "selection_mode": "all_symbols" if args.all_symbols else "historical_components" if memberships is not None else "static_symbols",
        "historical_components": str(history_path) if history_path else None,
        "membership_snapshot_count": len(snapshot_dates) if snapshot_dates is not None else None,
        "instrument": instrument,
        "calendar_start": str(calendar.min().date()),
        "calendar_end": str(calendar.max().date()),
        "symbols": imported,
        "first_days_dropped": sum(item["first_day_dropped"] for item in imported),
        "limit_up_days": sum(item["limit_up_count"] for item in imported),
        "limit_down_days": sum(item["limit_down_count"] for item in imported),
        "skipped": skipped,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("completed: %s symbols -> %s", len(imported), output)
    return 0 if imported else 2


if __name__ == "__main__":
    raise SystemExit(main())
