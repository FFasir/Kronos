"""Per-stock fresh 后复权 (hfq) daily overlay store.

Why a full-history overlay instead of appending to the archive?
  - The archived ``A股数据/daily_hfq.zip`` series is Eastmoney-calibrated and
    frozen at 2026-07-29.  Eastmoney endpoints are blocked from this cloud
    egress, while Sina (``stock_zh_a_daily``) works but anchors its hfq scale
    differently per symbol.
  - Mixing two providers inside one series would recreate the price-cliff bug
    (e.g. EM hfq 000001 ~1844 vs Sina hfq ~1790 for the same day).
  - Therefore each symbol gets one self-consistent *full* hfq series from a
    single provider (Sina) written under ``A股数据/daily_hfq_live/``; the
    storage layer prefers that overlay over the frozen archive member.  The
    model z-normalizes each input window, so the absolute scale does not
    matter — only internal consistency and a correct latest close do.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger

logger = get_logger(__name__)

# 覆盖层 CSV 与归档 zip 成员保持一致的中文列名，_load_csv_raw_data 的列映射可直接复用。
_OVERLAY_COLUMNS = [
    "日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
    "振幅", "涨跌幅", "涨跌额", "换手率",
]


def live_dir(config: Optional[Config] = None) -> Path:
    config = config or Config()
    rel = str(config.get("data", "hfq_live_dir", default="A股数据/daily_hfq_live"))
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def overlay_path(symbol: str, config: Optional[Config] = None) -> Path:
    return live_dir(config) / f"{str(symbol).zfill(6)}_daily_hfq.csv"


def _exchange_prefix(code: str) -> Optional[str]:
    """Map a bare A-share code to its Sina exchange prefix (or None for unknown)."""
    if code.startswith(("6", "9", "688")):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return None


# ─── 进程内防抖：同一 symbol 一天最多完整刷新一次；失败可冷却后重试 ───
_refresh_lock = threading.Lock()
_refreshed_on: dict[str, str] = {}          # symbol -> YYYY-MM-DD (成功日)
_last_attempt: dict[str, float] = {}        # symbol -> time.monotonic()
_ATTEMPT_COOLDOWN_SECONDS = 600.0           # 失败后 10 分钟允许重试
_symbol_locks: dict[str, threading.Lock] = {}
_symbol_locks_guard = threading.Lock()


def _symbol_lock(symbol: str) -> threading.Lock:
    with _symbol_locks_guard:
        lock = _symbol_locks.get(symbol)
        if lock is None:
            lock = threading.Lock()
            _symbol_locks[symbol] = lock
        return lock


def _attempt_allowed(code: str, force: bool) -> bool:
    """Return True when a network attempt is permitted (cooldown aware)."""
    if force:
        return True
    today = date.today().isoformat()
    with _refresh_lock:
        if _refreshed_on.get(code) == today:
            return False
        last = _last_attempt.get(code)
        if last is not None and (time.monotonic() - last) < _ATTEMPT_COOLDOWN_SECONDS:
            return False
        _last_attempt[code] = time.monotonic()
        return True


def _mark_refreshed(code: str) -> None:
    with _refresh_lock:
        _refreshed_on[code] = date.today().isoformat()


def expected_trade_date(config: Optional[Config] = None) -> Optional[str]:
    """Estimate the latest day the market should have data through.

    Uses the trading calendar when it reaches at least today; otherwise falls
    back to the last weekday.  A stale calendar must never suppress a refresh,
    so the weekday estimate is the floor.
    """
    config = config or Config()
    today = date.today()
    # 工作日估计：今天(工作日)或最近一个周五。
    estimate = today
    while estimate.weekday() >= 5:
        estimate -= timedelta(days=1)
    estimate_text = estimate.isoformat()

    calendar = PROJECT_ROOT / str(config.get(
        "data", "a_share_calendar", default="A股数据/交易日历.csv"
    ))
    calendar_max: Optional[str] = None
    if calendar.exists():
        try:
            df = pd.read_csv(calendar, dtype=str, encoding="utf-8-sig")
            if "日期" in df.columns:
                frame = df[["日期"]].copy()
                if "是否交易" in df.columns:
                    traded = df["是否交易"].astype(str).str.strip().str.upper()
                    frame = frame[traded.isin(["1", "TRUE", "是", "Y", "交易"])]
                dates = pd.to_datetime(frame["日期"], errors="coerce").dropna()
                dates = dates[dates <= pd.Timestamp(today)]
                if len(dates):
                    calendar_max = str(dates.max().date())
        except Exception as exc:
            logger.warning("read trading calendar failed: %s", exc)

    if calendar_max is None:
        return estimate_text
    # 取两者中较新者：保证旧日历不会让"已是最新"判断失效。
    return max(calendar_max, estimate_text)


def _fetch_sina_hfq(code: str, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch Sina hfq daily bars (full history unless start_date given)."""
    import akshare as ak

    prefix = _exchange_prefix(code)
    if prefix is None:
        raise ValueError(f"unsupported A-share code: {code}")
    symbol = f"{prefix}{code}"
    kwargs: dict = {"symbol": symbol, "adjust": "hfq"}
    if start_date:
        kwargs["start_date"] = start_date
        kwargs["end_date"] = date.today().strftime("%Y%m%d")
    frame = ak.stock_zh_a_daily(**kwargs)
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    if "date" not in frame.columns:
        raise ValueError(f"Sina hfq response for {code} lacks date column: {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame


def _to_overlay_rows(code: str, frame: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Convert a Sina hfq frame into the archive-style Chinese column layout.

    ``scale`` anchors the Sina hfq scale to the frozen EM archive level so the
    overlay continues seamlessly from the archive tail (both display and
    previously stored predictions stay on one scale).  A constant multiplier
    keeps every day-to-day return identical to pure Sina.

    Unit normalizations to match the EM archive convention:
      - volume: Sina gives shares -> /100 to 手 (lot).
      - turnover: Sina gives fraction -> *100 to percent.
    """
    out = pd.DataFrame(index=frame.index)
    out["日期"] = frame["date"].dt.strftime("%Y-%m-%d")
    out["股票代码"] = str(code).zfill(6)
    out["开盘"] = frame["open"] * scale
    out["收盘"] = frame["close"] * scale
    out["最高"] = frame["high"] * scale
    out["最低"] = frame["low"] * scale
    out["成交量"] = frame["volume"] / 100.0
    out["成交额"] = frame.get("amount", pd.Series(0.0, index=frame.index))
    if "turnover" in frame.columns:
        out["换手率"] = frame["turnover"] * 100.0
    for column in ["振幅", "涨跌幅", "涨跌额"]:
        out[column] = 0.0
    # 由收盘价推算日涨跌幅（供展示参考）
    closes = frame["close"].astype(float) * scale
    out["涨跌幅"] = (closes.pct_change().fillna(0.0) * 100.0).round(4)
    out["涨跌额"] = closes.diff().fillna(0.0).round(4)
    out["振幅"] = (
        ((frame["high"].astype(float) - frame["low"].astype(float))
         / frame["close"].astype(float).shift(1).replace(0, pd.NA)).fillna(0.0) * 100.0
    ).round(4).fillna(0.0)
    out = out[_OVERLAY_COLUMNS]
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).sort_values("日期")
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    return out.drop_duplicates("日期", keep="last")


def _archive_scale_anchor(code: str, frame: pd.DataFrame, config: Config) -> float:
    """Ratio that maps Sina hfq onto the EM archive scale.

    The overlay replaces the whole archived series for this symbol, so it must
    agree with the archive at the archive's last trading day.  When the symbol
    is absent from the archive (e.g. listed after the archive cutoff, BJ or new
    codes) a neutral scale of 1.0 is used.
    """
    try:
        from src.data_layer.storage import StorageLayer

        archive = StorageLayer(config).read_archive_member(code)
        if archive is None or archive.empty:
            return 1.0
        renamed = archive.rename(columns={"日期": "date", "收盘": "close"})
        renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
        renamed = renamed.dropna(subset=["date"]).sort_values("date")
        if renamed.empty:
            return 1.0
        anchor_date = renamed["date"].iloc[-1]
        anchor_close = float(renamed["close"].iloc[-1])
        sina_row = frame[frame["date"] == anchor_date]
        if sina_row.empty or not sina_row["close"].iloc[0]:
            return 1.0
        scale = anchor_close / float(sina_row["close"].iloc[0])
        if not (0.05 < scale < 20.0) or not (scale == scale):  # 防脏数据
            logger.warning("%s: archive scale anchor out of range %.4f", code, scale)
            return 1.0
        return scale
    except Exception as exc:
        logger.warning("%s: archive scale anchor failed: %s", code, exc)
        return 1.0


def refresh_symbol(
    code: str,
    config: Optional[Config] = None,
    force: bool = False,
    wait_seconds: float = 0.0,
) -> dict:
    """Refresh one symbol's full Sina-hfq overlay (atomic tmp+rename write)."""
    config = config or Config()
    code = str(code).zfill(6)

    if not _attempt_allowed(code, force):
        return {"code": code, "status": "skipped-cooldown-or-refreshed"}

    lock = _symbol_lock(code)
    with lock:
        # 再次确认：并发调用可能在前一次完成后进入，直接复用最新覆盖层。
        if not force:
            last_date = overlay_last_date(code, config)
            expected = expected_trade_date(config)
            if last_date is not None and expected is not None and last_date >= expected:
                _mark_refreshed(code)
                return {"code": code, "status": "up-to-date", "last_date": last_date}
        try:
            frame = _fetch_sina_hfq(code)
        except Exception as exc:
            return {"code": code, "status": "failed", "error": str(exc)[:300]}
        if frame is None or frame.empty:
            return {"code": code, "status": "no-data"}
        scale = _archive_scale_anchor(code, frame, config)
        rows = _to_overlay_rows(code, frame, scale=scale)
        if rows.empty:
            return {"code": code, "status": "no-rows"}
        target_dir = live_dir(config)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{code}_daily_hfq.csv"
        temporary = target.with_suffix(".csv.tmp")
        # 纯 utf-8（无 BOM），与 storage._load_csv_raw_data 的默认读取一致。
        rows.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(target)
        _mark_refreshed(code)
        if wait_seconds:
            time.sleep(wait_seconds)
        return {
            "code": code,
            "status": "refreshed",
            "rows": len(rows),
            "last_date": str(rows["日期"].iloc[-1]),
            "last_close": float(rows["收盘"].iloc[-1]),
            "scale": round(scale, 6),
        }


def refresh_symbols(
    symbols: Sequence[str],
    config: Optional[Config] = None,
    force: bool = False,
    request_interval: float = 1.0,
) -> dict:
    """Refresh many symbols with a polite inter-request pause."""
    config = config or Config()
    results = []
    for index, code in enumerate(symbols):
        results.append(refresh_symbol(code, config=config, force=force,
                                      wait_seconds=request_interval))
    ok = [item for item in results if item.get("status") == "refreshed"]
    failed = [item for item in results if item.get("status") == "failed"]
    return {
        "total": len(results),
        "refreshed": len(ok),
        "failed": failed,
        "results": results,
    }


def overlay_last_date(symbol: str, config: Optional[Config] = None) -> Optional[str]:
    """Read the newest date present in the overlay CSV without loading it all."""
    path = overlay_path(symbol, config)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["日期"], encoding="utf-8")
        if df.empty:
            return None
        dates = pd.to_datetime(df["日期"], errors="coerce").dropna()
        return str(dates.max().date()) if len(dates) else None
    except Exception:
        return None


def ensure_current(
    symbol: str,
    config: Optional[Config] = None,
    force: bool = False,
) -> dict:
    """Refresh the overlay only when it lags behind the expected trade date.

    In-process guard: only one real network attempt per symbol per day is
    allowed; failures cool down and may retry later in the day.
    """
    config = config or Config()
    code = str(symbol).zfill(6)

    last_date = overlay_last_date(code, config)
    expected = expected_trade_date(config)
    if not force and last_date is not None and expected is not None:
        if last_date >= expected:
            _mark_refreshed(code)
            return {"code": code, "status": "up-to-date", "last_date": last_date}
    return refresh_symbol(code, config=config, force=force)
