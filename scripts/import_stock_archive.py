#!/usr/bin/env python3
"""Import A-share daily ZIP archives into an HDF5 store for training.

The source archives use Chinese column names and contain one CSV per stock.  This
script reads them directly from the ZIP archive, so the server does not need to
extract thousands of intermediate files.
"""

import argparse
import json
import sys
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tables.exceptions import NaturalNameWarning


warnings.filterwarnings("ignore", category=NaturalNameWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.logger import get_logger, setup_logger


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SOURCE_COLUMNS = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def _read_symbols(path: Path) -> set[str]:
    """Read stock codes from the first column of a text or CSV file."""
    table = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if table.empty:
        return set()
    return {
        str(value).split(".")[0].zfill(6)
        for value in table.iloc[:, 0].dropna()
        if str(value).strip()
    }


def _archive_code(name: str) -> str:
    return Path(name).name.split("_")[0].zfill(6)


def _prepare_frame(raw: pd.DataFrame, train_start: str, train_end: str) -> pd.DataFrame | None:
    raw = raw.rename(columns=SOURCE_COLUMNS)
    if not {"date", *OHLCV_COLUMNS}.issubset(raw.columns):
        return None

    frame = raw[["date", *OHLCV_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", *OHLCV_COLUMNS])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    # Kronos price tokenization requires positive OHLC prices.  This also removes
    # early invalid forward-adjusted records if the qfq archive is selected.
    frame = frame[
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["high"] >= frame["low"])
        & (frame["volume"] >= 0)
    ]
    frame = frame.set_index("date")
    if frame.empty or not ((frame.index >= train_start) & (frame.index <= train_end)).any():
        return None
    return frame


def _normalize(frame: pd.DataFrame, train_start: str, train_end: str) -> pd.DataFrame:
    train = frame.loc[train_start:train_end, OHLCV_COLUMNS]
    mean = train.mean()
    std = train.std().replace(0, 1.0).fillna(1.0)
    return (frame[OHLCV_COLUMNS] - mean) / std


def main() -> int:
    parser = argparse.ArgumentParser(description="Import A-share daily ZIP data into component HDF5")
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "A股数据" / "daily.zip",
        help="daily.zip, daily_qfq.zip, or daily_hfq.zip",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="optional CSV/TXT; first column contains stock codes (for example a CSI 1000 snapshot)",
    )
    parser.add_argument("--symbols", help="optional comma-separated stock codes")
    parser.add_argument("--max-symbols", type=int, help="limit imported symbols for a smoke test")
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--min-val-rows", type=int, default=65)
    parser.add_argument("--output", type=Path, help="default: data/processed/components.h5")
    parser.add_argument("--config", help="system YAML path")
    args = parser.parse_args()

    setup_logger(log_dir="logs/", console=True)
    logger = get_logger("scripts.import_stock_archive")
    config = load_config(args.config)

    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")

    output = args.output or Path(config.data["processed_dir"]) / "components.h5"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    selected: set[str] | None = None
    if args.symbols_file:
        selected = _read_symbols(args.symbols_file)
    if args.symbols:
        selected = selected or set()
        selected.update(code.strip().zfill(6) for code in args.symbols.split(",") if code.strip())
    if selected is None:
        catalog_path = Path(config.data.get("a_share_catalog", ROOT / "A股数据" / "股票列表.csv"))
        if not catalog_path.is_absolute():
            catalog_path = ROOT / catalog_path
        if catalog_path.exists():
            selected = _read_symbols(catalog_path)
            logger.info("restricting import to %s A-share catalog symbols", len(selected))

    train_start = config.data["finetune_train_start"]
    train_end = config.data["finetune_train_end"]
    val_start = config.data["finetune_val_start"]
    val_end = config.data["finetune_val_end"]

    imported: list[str] = []
    skipped: dict[str, int] = {"not_selected": 0, "invalid": 0, "insufficient_history": 0}
    with zipfile.ZipFile(archive) as source, pd.HDFStore(output, mode="w", complevel=5, complib="zlib") as store:
        entries = [entry for entry in source.infolist() if entry.filename.lower().endswith(".csv")]
        for entry in entries:
            code = _archive_code(entry.filename)
            if selected is not None and code not in selected:
                skipped["not_selected"] += 1
                continue
            if args.max_symbols is not None and len(imported) >= args.max_symbols:
                break
            try:
                with source.open(entry) as handle:
                    raw = pd.read_csv(handle, encoding="utf-8-sig")
                frame = _prepare_frame(raw, train_start, train_end)
            except Exception as exc:
                logger.warning("skip %s: %s", code, exc)
                skipped["invalid"] += 1
                continue

            if frame is None:
                skipped["invalid"] += 1
                continue
            train_rows = len(frame.loc[train_start:train_end])
            val_rows = len(frame.loc[val_start:val_end])
            if train_rows < args.min_train_rows or val_rows < args.min_val_rows:
                skipped["insufficient_history"] += 1
                continue

            store.put(f"/raw/{code}", _normalize(frame, train_start, train_end), format="fixed")
            imported.append(code)
            if len(imported) % 100 == 0:
                logger.info("imported %s symbols", len(imported))

        metadata = pd.DataFrame([{
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "archive": str(archive),
            "archive_members": len(entries),
            "imported_symbols": len(imported),
            "train_range": f"{train_start}~{train_end}",
            "val_range": f"{val_start}~{val_end}",
            "adjustment_note": archive.name,
        }])
        store.put("/metadata", metadata, format="fixed")

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"symbols": imported, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("completed: %s symbols -> %s", len(imported), output)
    logger.info("manifest: %s", manifest)
    return 0 if imported else 2


if __name__ == "__main__":
    raise SystemExit(main())
