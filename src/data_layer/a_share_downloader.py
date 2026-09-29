"""A-share end-of-day snapshot downloader and incremental persistence."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import akshare as ak
import pandas as pd

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger
from src.data_layer.storage import StorageLayer


logger = get_logger(__name__)

_COLUMN_MAP = {
    "代码": "code",
    "名称": "name",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "最新价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}


class AShareDailyDownloader:
    """Fetch one post-close snapshot for the complete local A-share catalog."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.storage = StorageLayer(self.config)
        self.storage.init_database()
        self.raw_dir = PROJECT_ROOT / self.config.data.get("raw_dir", "data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.raw_dir / "a_share_download_meta.json"

    def download(self, as_of: Optional[str] = None, symbols: Optional[Sequence[str]] = None) -> dict:
        """Fetch the current A-share snapshot and persist it atomically."""
        trade_date = as_of or date.today().isoformat()
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            source = "akshare.stock_zh_a_spot_em"
            try:
                raw = ak.stock_zh_a_spot_em()
            except Exception as primary_error:
                logger.warning("Eastmoney snapshot failed, trying Sina: %s", primary_error)
                raw = ak.stock_zh_a_spot()
                source = "akshare.stock_zh_a_spot"
            frame = self.prepare_snapshot(raw, trade_date)
            if symbols:
                requested = {str(symbol).zfill(6) for symbol in symbols}
                frame = frame[frame["code"].isin(requested)]
            if frame.empty:
                raise RuntimeError("AKShare returned no valid A-share rows")
            inserted = self.persist_snapshot(frame)
        except Exception as exc:
            logger.error("A-share daily update failed: %s", exc)
            return {
                "status": "failed",
                "date": trade_date,
                "rows": 0,
                "started_at": started_at,
                "error": str(exc),
            }

        result = {
            "status": "completed",
            "date": trade_date,
            "rows": inserted,
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
        }
        self._record_success(result)
        logger.info("A-share daily update completed: %s rows for %s", inserted, trade_date)
        return result

    def import_snapshot(self, path: Path, as_of: Optional[str] = None) -> dict:
        """Import a supplied daily market-wide CSV through the same validation path."""
        raw = pd.read_csv(path, dtype={"股票代码": str, "代码": str}, encoding="utf-8-sig")
        if as_of is None and "日期" in raw.columns and not raw.empty:
            as_of = str(raw["日期"].iloc[0])[:10]
        trade_date = as_of or date.today().isoformat()
        frame = self.prepare_snapshot(raw, trade_date)
        if frame.empty:
            raise ValueError(f"snapshot contains no valid A-share rows: {path}")
        inserted = self.persist_snapshot(frame)
        result = {
            "status": "completed",
            "date": trade_date,
            "rows": inserted,
            "source": str(path),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._record_success(result)
        return result

    def prepare_snapshot(self, raw: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """Normalize, validate, and restrict a snapshot to the local A-share catalog."""
        frame = raw.rename(columns=_COLUMN_MAP).copy()
        if "股票代码" in frame.columns and "code" not in frame.columns:
            frame = frame.rename(columns={"股票代码": "code"})
        if "股票名称" in frame.columns and "name" not in frame.columns:
            frame = frame.rename(columns={"股票名称": "name"})
        chinese_prices = {
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount", "换手率": "turnover",
        }
        frame = frame.rename(columns=chinese_prices)

        required = {"code", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"snapshot missing columns: {sorted(missing)}")

        frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        catalog = self._catalog()
        frame = frame[frame["code"].isin(catalog)].copy()
        frame["name"] = frame["code"].map(catalog)
        frame["date"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")

        for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if column not in frame.columns:
                frame[column] = None
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=["code", "open", "high", "low", "close", "volume"])
        frame = frame[
            (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (frame["high"] >= frame["low"])
            & (frame["volume"] >= 0)
        ]
        return frame.drop_duplicates("code", keep="last")

    def persist_snapshot(self, frame: pd.DataFrame) -> int:
        """Upsert a validated snapshot in one SQLite transaction."""
        rows = [
            (
                row.code, row.date, float(row.open), float(row.high), float(row.low),
                float(row.close), float(row.volume),
                None if pd.isna(row.amount) else float(row.amount),
                None if pd.isna(row.turnover) else float(row.turnover),
            )
            for row in frame.itertuples(index=False)
        ]
        info_rows = [(row.code, row.name, "stock") for row in frame.itertuples(index=False)]
        connection = self.storage.get_connection()
        try:
            connection.executemany(
                """INSERT INTO stock_info (code, name, type, is_active)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(code) DO UPDATE SET
                     name=excluded.name, type='stock', is_active=1, updated_at=datetime('now')""",
                info_rows,
            )
            connection.executemany(
                """INSERT INTO stock_daily
                   (code, date, open, high, low, close, volume, amount, turnover)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(code, date) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     amount=excluded.amount, turnover=excluded.turnover""",
                rows,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return len(rows)

    def last_success_date(self) -> Optional[str]:
        if not self.meta_path.exists():
            return None
        try:
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return payload.get("last_success", {}).get("date")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None

    def _record_success(self, result: dict) -> None:
        history = []
        if self.meta_path.exists():
            try:
                history = json.loads(self.meta_path.read_text(encoding="utf-8")).get("history", [])
            except (OSError, json.JSONDecodeError, AttributeError):
                history = []
        history.append(result)
        payload = {"last_success": result, "history": history[-100:]}
        temporary = self.meta_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.meta_path)

    @staticmethod
    def _catalog() -> dict[str, str]:
        path = PROJECT_ROOT / "A股数据" / "股票列表.csv"
        if not path.exists():
            raise FileNotFoundError(f"A-share catalog not found: {path}")
        catalog = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
        return {
            str(row["股票代码"]).zfill(6): str(row["股票名称"])
            for _, row in catalog.iterrows()
            if str(row.get("股票代码", "")).strip()
        }
