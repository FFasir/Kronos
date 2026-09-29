"""
Kronos — A股实时行情 API（新浪 hq.sinajs.cn）

端点:
  GET /api/v1/quote/list?codes=600519,000001  — 批量实时行情（最多 100 只）

设计:
  - 优先新浪实时接口（GBK，需 Referer），带进程内 TTL 缓存与请求限速；
  - 新浪失败时降级到 stock_daily 最新不复权快照（只有收盘价，无涨跌幅）；
  - 全部失败时仍返回请求的代码，行情字段为 null，前端显示 "--"。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from src.common.logger import get_logger
from src.service_layer.models import QuoteInfo, QuoteListResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/quote", tags=["实时行情"])

# 单代码行情缓存 TTL；新浪限流保护
_QUOTE_TTL_SECONDS = 10.0
_BATCH_MIN_INTERVAL_SECONDS = 3.0
_MAX_CODES_PER_REQUEST = 100

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()
_last_fetch_monotonic = 0.0


def _sina_symbol(code: str) -> Optional[str]:
    """映射 A 股代码到新浪前缀 (sh/sz/bj)。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return None


def _parse_num(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed == 0:
        return None
    return parsed


def _parse_sina_line(line: str) -> Optional[dict]:
    """
    解析新浪行情行:
      var hq_str_sh600519="贵州茅台,今开,昨收,最新,最高,最低,...,日期,时间,...";
    空载荷（"..."=""）表示停牌或无数据。
    """
    if '="' not in line:
        return None
    _, _, payload = line.partition('="')
    payload = payload.rstrip().rstrip(';').rstrip('"')
    fields = payload.split(",")
    if len(fields) < 32 or not fields[0]:
        return None
    price = _parse_num(fields[3])
    prev_close = _parse_num(fields[2])
    change = None
    change_pct = None
    if price is not None and prev_close:
        change = round(price - prev_close, 4)
        change_pct = round(change / prev_close * 100, 2)
    return {
        "name": fields[0],
        "open": _parse_num(fields[1]),
        "prev_close": prev_close,
        "price": price,
        "high": _parse_num(fields[4]),
        "low": _parse_num(fields[5]),
        "volume": _parse_num(fields[8]),
        "amount": _parse_num(fields[9]),
        "date": fields[30] if len(fields) > 30 else None,
        "time": fields[31] if len(fields) > 31 else None,
        "change": change,
        "change_pct": change_pct,
        "suspended": price is None,
        "source": "sina_realtime",
    }


def _fetch_sina(codes: list[str]) -> dict[str, dict]:
    """批量抓取新浪实时行情。"""
    import requests

    symbols = [s for code in codes if (s := _sina_symbol(code)) is not None]
    if not symbols:
        return {}
    url = "https://hq.sinajs.cn/list=" + ",".join(symbols)
    resp = requests.get(
        url,
        headers={
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        timeout=8,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"sina quote HTTP {resp.status_code}")
    text = resp.content.decode("gbk", errors="ignore")
    result: dict[str, dict] = {}
    for line in text.splitlines():
        if '="' not in line:
            continue
        key = line.split('="')[0].replace("var hq_str_", "").strip()
        if len(key) < 3:
            continue
        code = key[2:]
        parsed = _parse_sina_line(line)
        if parsed:
            result[str(code).zfill(6)] = parsed
    return result


def _snapshot_quotes(codes: list[str]) -> dict[str, dict]:
    """降级：stock_daily 最新不复权快照（仅收盘价）。"""
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        storage = StorageLayer(load_config())
        placeholders = ",".join("?" for _ in codes)
        rows = storage.execute_sql(
            f"""SELECT code, date, close FROM stock_daily
                WHERE code IN ({placeholders}) AND close IS NOT NULL AND close > 0""",
            tuple(codes),
        )
        result: dict[str, dict] = {}
        for r in rows:
            code = str(r[0]).zfill(6)
            result[code] = {
                "name": None,
                "price": float(r[2]),
                "prev_close": None,
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "amount": None,
                "date": str(r[1]),
                "time": None,
                "change": None,
                "change_pct": None,
                "suspended": False,
                "source": "daily_snapshot",
            }
        return result
    except Exception as exc:
        logger.warning("snapshot quote fallback failed: %s", exc)
        return {}


def _get_quotes_sync(codes: list[str]) -> list[dict]:
    """带缓存与限速的行情获取（在线程池中执行）。"""
    global _last_fetch_monotonic

    now = time.monotonic()
    fresh: dict[str, dict] = {}
    stale_codes: list[str] = []
    with _lock:
        for code in codes:
            item = _cache.get(code)
            if item is not None and now - item[0] < _QUOTE_TTL_SECONDS:
                fresh[code] = item[1]
            else:
                stale_codes.append(code)

    if stale_codes:
        with _lock:
            wait = _BATCH_MIN_INTERVAL_SECONDS - (now - _last_fetch_monotonic)
        if wait > 0:
            time.sleep(wait)
        with _lock:
            _last_fetch_monotonic = time.monotonic()

        fetched: dict[str, dict] = {}
        try:
            fetched = _fetch_sina(stale_codes)
        except Exception as exc:
            logger.warning("sina realtime quote fetch failed: %s", exc)

        missing = [c for c in stale_codes if c not in fetched]
        if missing:
            fetched.update(_snapshot_quotes(missing))

        with _lock:
            stored_at = time.monotonic()
            for code, quote in fetched.items():
                _cache[code] = (stored_at, quote)
        fresh.update(fetched)

    return [
        {"code": code, **(fresh.get(code) or {"suspended": True, "source": "unavailable"})}
        for code in codes
    ]


@router.get("/list", response_model=QuoteListResponse)
async def get_quote_list(
    codes: str = Query(..., description="逗号分隔的股票代码，如 600519,000001，最多 100 只"),
):
    """批量获取 A 股实时行情（价格/涨跌幅等）。"""
    code_list = []
    for part in codes.split(","):
        part = part.strip()
        if part and part.isdigit():
            code_list.append(part.zfill(6))
    code_list = code_list[:_MAX_CODES_PER_REQUEST]
    if not code_list:
        raise HTTPException(status_code=400, detail="codes 不能为空，请提供股票代码列表")

    rows = await run_in_threadpool(_get_quotes_sync, code_list)
    quotes = [
        QuoteInfo(
            code=row["code"],
            name=row.get("name"),
            price=row.get("price"),
            prev_close=row.get("prev_close"),
            change=row.get("change"),
            change_pct=row.get("change_pct"),
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            volume=row.get("volume"),
            amount=row.get("amount"),
            date=row.get("date"),
            time=row.get("time"),
            suspended=bool(row.get("suspended")),
            source=row.get("source"),
        )
        for row in rows
    ]
    as_of = max((q.time for q in quotes if q.time), default=None)
    logger.info(f"GET /quote/list → {len(quotes)} 只")
    return QuoteListResponse(quotes=quotes, as_of=as_of)
