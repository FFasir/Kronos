"""
Kronos 指数/ETF 择时策略系统 — 指数/ETF 数据 API

端点:
  GET /api/v1/index/list        — 标的列表
  GET /api/v1/index/{code}      — 标的详情 + K线
  GET /api/v1/index/{code}/kline — 指定日期范围 K线
"""

from __future__ import annotations

from typing import Optional
from functools import lru_cache
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger
from src.service_layer.models import (
    IndexInfo, IndexDetail, IndexListResponse,
    OHLCVPoint, KlineResponse, AssetTypeEnum,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/index", tags=["指数/ETF数据"])


# ═══════════════════════════════════════════════════════════════════
# Mock 数据生成
# ═══════════════════════════════════════════════════════════════════

def _make_mock_kline(code: str, days: int = 120) -> list[dict]:
    """生成模拟 K 线数据（当真实数据不可用时）。"""
    import random
    random.seed(hash(code) % 2**31)

    end_date = datetime.now()
    data = []
    base_price = _get_base_price(code)

    for i in range(days, 0, -1):
        d = end_date - timedelta(days=i)
        if d.weekday() >= 5:  # 跳过周末
            continue
        change = random.gauss(0, 0.015)
        close = base_price * (1 + change)
        open_p = close * (1 + random.gauss(0, 0.005))
        high = max(open_p, close) * (1 + abs(random.gauss(0, 0.008)))
        low = min(open_p, close) * (1 - abs(random.gauss(0, 0.008)))
        volume = random.uniform(50000000, 300000000)
        base_price = close

        data.append({
            "date": d.strftime("%Y-%m-%d"),
            "open": round(open_p, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(volume, 0),
            "amount": round(volume * close, 0),
            "turnover": round(random.uniform(0.5, 3.0), 2),
        })

    return data


def _get_base_price(code: str) -> float:
    """根据代码获取模拟基准价。"""
    prices = {
        "000852": 6234.50, "000300": 3890.20, "000905": 5678.30,
        "399006": 2150.80, "000688": 980.50,  "000016": 2645.70,
        "512100": 2.45,    "510300": 3.98,    "510500": 6.12,
        "159915": 2.15,    "588000": 0.98,    "510050": 2.85,
    }
    return prices.get(code, 1000.0)


def _get_all_symbols(config: Config) -> list[dict]:
    """获取所有标的（尝试真实数据，失败则用配置中的列表）。"""
    try:
        symbols = config.get_all_symbols()
        if symbols:
            return symbols
    except Exception:
        pass

    # Fallback: 从配置中合并 index_list + etf_list
    indices = config.get_index_list()
    etfs = config.get_etf_list()
    return indices + etfs


@lru_cache(maxsize=1)
def _stock_catalog() -> dict[str, dict]:
    path = PROJECT_ROOT / "A股数据" / "股票列表.csv"
    if not path.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(path, dtype=str).fillna("")
    catalog = {}
    for _, row in df.iterrows():
        code = str(row.get("股票代码", "")).zfill(6)
        if not code or code == "000000":
            continue
        catalog[code] = {
            "code": code,
            "name": str(row.get("股票名称", code)),
            "type": "stock",
            "industry": str(row.get("所属行业", "")),
            "is_active": str(row.get("上市状态", "上市")) == "上市",
        }
    return catalog


def _find_symbol(code: str, config: Config) -> Optional[dict]:
    configured = next((item for item in _get_all_symbols(config) if item["code"] == code), None)
    return configured or _stock_catalog().get(code)


def _is_configured_code(code: str, config: Config) -> bool:
    """True when code is one of the configured market assets (index/ETF).

    These assets are served from their own up-to-date downloader CSV files and
    must never be shadowed by the per-stock hfq overlay (code collisions like
    000852/000688 exist between stocks and configured indices).
    """
    return any(str(item.get("code")) == code for item in _get_all_symbols(config))


def _enrich_symbol_info(symbol: dict) -> IndexInfo:
    """丰富标的的基本信息（添加最新价格等）。"""
    code = symbol.get("code", "")
    base_price = _get_base_price(code)

    # 尝试从 HDF5 获取真实最新价格
    latest_close = base_price
    change_pct = 0.0
    update_date = datetime.now().strftime("%Y-%m-%d")

    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        cfg = load_config()
        storage = StorageLayer(cfg)
        raw = storage.load_raw_data(code)
        if raw is not None and len(raw) > 0:
            latest = raw.iloc[-1]
            latest_close = float(latest.get("close", base_price))
            update_date = str(raw.index[-1])[:10] if hasattr(raw.index[-1], 'strftime') else str(raw.index[-1])[:10]
            if len(raw) >= 2:
                prev_close = float(raw.iloc[-2].get("close", latest_close))
                if prev_close > 0:
                    change_pct = round((latest_close / prev_close - 1) * 100, 2)
    except Exception:
        pass

    asset_type = AssetTypeEnum(symbol.get("type", "stock"))

    return IndexInfo(
        code=code,
        name=symbol.get("name", code),
        type=asset_type,
        track_index=symbol.get("track_index"),
        latest_close=round(latest_close, 2),
        change_pct=change_pct,
        update_date=update_date,
        industry=symbol.get("industry"),
    )


# ═══════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════

@router.get("/list", response_model=IndexListResponse)
async def get_index_list(
    type: Optional[str] = Query(None, description="筛选类型: index | etf | stock"),
    keyword: Optional[str] = Query(None, min_length=1),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    获取所有标的（12个指数+ETF）列表。

    返回每个标的的基本信息：代码、名称、类型、最新收盘价、涨跌幅。
    可按类型筛选（index 仅指数，etf 仅 ETF）。
    """
    from src.common.config import load_config

    config = load_config()
    symbols = _get_all_symbols(config)
    if type == "stock" or keyword:
        symbols.extend(_stock_catalog().values())

    # 类型筛选
    if type:
        symbols = [s for s in symbols if s.get("type") == type]
    if keyword:
        key = keyword.lower()
        symbols = [
            s for s in symbols
            if key in s["code"].lower()
            or key in s.get("name", "").lower()
            or key in s.get("industry", "").lower()
        ]
    total = len(symbols)
    symbols = symbols[offset:offset + limit]

    # Market browsing must stay cheap. Prices are loaded on the detail page.
    if type == "stock":
        indices = [IndexInfo(
            code=s["code"], name=s.get("name", s["code"]), type=AssetTypeEnum.STOCK,
            industry=s.get("industry"),
        ) for s in symbols]
    else:
        indices = [_enrich_symbol_info(s) for s in symbols]

    logger.info(f"GET /index/list → {len(indices)} 个标的")
    return IndexListResponse(total=total, indices=indices)


@router.get("/{code}", response_model=IndexDetail)
async def get_index_detail(
    code: str,
    days: int = Query(1250, ge=1, le=2000, description="返回天数"),
    scale: Optional[str] = Query(
        "raw",
        pattern="^(raw|hfq)$",
        description="价格口径: raw(不复权市价,默认) | hfq(后复权,模型输入口径)",
    ),
):
    """
    获取单个标的详情 + 最近 N 天 OHLCV。

    返回基本信息 + K线序列，用于前端 K 线图渲染。
    个股默认返回不复权市价口径（scale=raw），后复权为可选切换（scale=hfq）。
    """
    config = _get_config()
    match = _find_symbol(code, config)

    if match is None:
        raise HTTPException(status_code=404, detail=f"标的 {code} 不存在")

    asset_type = AssetTypeEnum(match.get("type", "stock"))
    is_plain_stock = asset_type == AssetTypeEnum.STOCK and not _is_configured_code(code, config)

    # 尝试获取真实 K 线
    kline_raw = None
    try:
        from src.data_layer.storage import StorageLayer
        storage = StorageLayer(config)
        # 个股(非配置资产)：若本地后复权覆盖层滞后，先按日尝试一次在线刷新
        # (进程内防抖+冷却；失败不影响返回旧数据)。异步端点里放线程池跑网络。
        if is_plain_stock:
            try:
                from starlette.concurrency import run_in_threadpool
                from src.data_layer.stock_hfq_live import ensure_current
                await run_in_threadpool(ensure_current, code, config)
            except Exception:
                pass
        raw = storage.load_raw_data(code)
        if raw is not None and len(raw) > 0:
            kline_raw = raw.tail(days)
    except Exception:
        pass

    if kline_raw is None or len(kline_raw) == 0:
        raise HTTPException(status_code=404, detail=f"no local OHLCV data available for {code}")

    # ── 价格口径: 个股默认不复权市价 ──────────────────────────
    # 本地个股序列为后复权(hfq)。默认(scale=raw)按比例换算成不复权市价：
    #   raw_t = hfq_t × (最新不复权市价 / 序列最新后复权价)
    # 该线性换算精确保留每日涨跌幅，仅平移价格水平。
    price_scale = "raw"
    market_close = None
    market_date = None
    scale_factor = 1.0
    if is_plain_stock:
        try:
            snapshots = storage.execute_sql(
                """SELECT date, close FROM stock_daily
                   WHERE code = ? AND close IS NOT NULL AND close > 0
                   ORDER BY date DESC LIMIT 1""",
                (code,),
            )
            if snapshots:
                market_close = float(snapshots[0]["close"])
                market_date = str(snapshots[0]["date"])
        except Exception:
            pass

        if scale == "raw":
            hfq_last_close = float(kline_raw.iloc[-1].get("close", 0) or 0)
            if market_close and hfq_last_close > 0:
                scale_factor = market_close / hfq_last_close
            else:
                # 无不复权快照可换算 → 诚实返回后复权口径
                price_scale = "hfq"
        else:
            price_scale = "hfq"

    kline_data = []
    for idx, row in kline_raw.iterrows():
        date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)[:10]
        kline_data.append({
            "date": date_str,
            "open": float(row.get("open", 0)) * scale_factor,
            "high": float(row.get("high", 0)) * scale_factor,
            "low": float(row.get("low", 0)) * scale_factor,
            "close": float(row.get("close", 0)) * scale_factor,
            "volume": float(row.get("volume", 0)),
            "amount": float(row.get("amount", 0)) if "amount" in row else None,
            "turnover": float(row.get("turnover", 0)) if "turnover" in row else None,
        })

    # 构建 OHLCVPoint 列表
    ohlcv = [
        OHLCVPoint(
            date=d["date"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            volume=d["volume"],
            amount=d.get("amount"),
            turnover=d.get("turnover"),
        )
        for d in kline_data
    ]

    # 最新价格
    latest = ohlcv[-1] if ohlcv else None
    prev = ohlcv[-2] if len(ohlcv) >= 2 else None
    change_pct = None
    if latest and prev and prev.close > 0:
        change_pct = round((latest.close / prev.close - 1) * 100, 2)

    # index/ETF: 序列本身即真实价格 → 市价取序列最新价
    if not is_plain_stock and latest:
        market_close = latest.close
        market_date = latest.date

    logger.info(f"GET /index/{code} → {len(ohlcv)} 条K线 (price_scale={price_scale})")
    return IndexDetail(
        code=code,
        name=match.get("name", code),
        type=asset_type,
        track_index=match.get("track_index"),
        latest_close=latest.close if latest else None,
        change_pct=change_pct,
        kline=ohlcv,
        price_scale=price_scale,
        market_close=market_close,
        market_date=market_date,
    )


@router.get("/{code}/kline", response_model=KlineResponse)
async def get_index_kline(
    code: str,
    start: str = Query(..., description="起始日期 YYYY-MM-DD"),
    end: str = Query(..., description="结束日期 YYYY-MM-DD"),
):
    """
    获取指定日期范围的 K 线数据。

    用于回测中心的历史数据查询/图表展示。
    """
    config = _get_config()
    match = _find_symbol(code, config)
    name = match["name"] if match else code

    if match is None:
        raise HTTPException(status_code=404, detail=f"标的 {code} 不存在")

    # 尝试从 HDF5 加载
    kline_data = []
    try:
        from src.data_layer.storage import StorageLayer
        storage = StorageLayer(config)
        raw = storage.load_raw_data(code, start=start, end=end)
        if raw is not None and len(raw) > 0:
            for idx, row in raw.iterrows():
                date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)[:10]
                kline_data.append({
                    "date": date_str,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "amount": float(row.get("amount", 0)) if "amount" in row else None,
                    "turnover": float(row.get("turnover", 0)) if "turnover" in row else None,
                })
    except Exception:
        pass

    if not kline_data:
        raise HTTPException(status_code=404, detail=f"no local OHLCV data available for {code} in requested range")

    ohlcv = [
        OHLCVPoint(**d) for d in kline_data
    ]

    logger.info(f"GET /index/{code}/kline → {len(ohlcv)} 条 ({start} ~ {end})")
    return KlineResponse(
        code=code,
        name=name,
        start=start,
        end=end,
        count=len(ohlcv),
        data=ohlcv,
    )


# ═══════════════════════════════════════════════════════════════════
# 个股预测 K 线数据（20 日 OHLC 预测 + 置信区间）
# ═══════════════════════════════════════════════════════════════════

from src.service_layer.models import StockPredictionResponse, PredictedOHLCVPoint


@router.get("/{code}/prediction", response_model=StockPredictionResponse)
async def get_stock_prediction(
    code: str,
    days: int = Query(20, description="预测天数", ge=1, le=60),
):
    """
    获取个股未来 N 日 OHLC 预测 + 置信区间。

    用于个股详情页的预测 K 线图。
    数据源: price_prediction 表 + 模型实时推理降级。
    """
    from src.data_layer.storage import StorageLayer
    from src.common.config import load_config
    import pandas as pd
    import numpy as np

    config = load_config()
    storage = StorageLayer(config)

    # 查询标的名称
    name = _resolve_name(code, storage)

    # 尝试从 price_prediction 表读取已有预测
    try:
        rows = storage.execute_sql(
            """SELECT pred_date, pred_open, pred_high, pred_low, pred_close,
                      pred_lower, pred_upper, prediction_std
               FROM price_prediction
               WHERE code = ?
               ORDER BY pred_date ASC
               LIMIT ?""",
            (code, days),
        )
        if rows and len(rows) > 0:
            points = []
            for r in rows:
                d = dict(r)
                pred_std = float(d.get("prediction_std", 0.02) or 0.02)
                confidence = max(0.0, min(1.0, 1.0 - pred_std * 10))
                points.append(PredictedOHLCVPoint(
                    date=str(d.get("pred_date", "")),
                    open=float(d.get("pred_open", 0) or 0),
                    high=float(d.get("pred_high", 0) or 0),
                    low=float(d.get("pred_low", 0) or 0),
                    close=float(d.get("pred_close", 0) or 0),
                    volume=0.0,
                    confidence=confidence,
                    lower_bound=float(d.get("pred_lower", 0) or 0),
                    upper_bound=float(d.get("pred_upper", 0) or 0),
                ))
            logger.info(f"GET /index/{code}/prediction → {len(points)} 条 (DB)")
            return StockPredictionResponse(
                code=code,
                name=name,
                pred_days=len(points),
                predictions=points,
            )
    except Exception as e:
        logger.warning(f"price_prediction 查询失败: {e}")

    # 降级: 从最新预测记录推算简单预测序列
    try:
        raw = storage.load_raw_data(code)
        if raw is None or len(raw) == 0:
            raise HTTPException(status_code=404, detail=f"标的 {code} 无行情数据")

        last_close = float(raw["close"].iloc[-1])
        last_date = raw.index[-1]

        # 从 model_prediction 表获取该股最新预测
        pred_rows = storage.execute_sql(
            """SELECT pred_return, confidence
               FROM model_prediction
               WHERE code = ?
               ORDER BY date DESC, pred_date DESC LIMIT 1""",
            (code,),
        )

        pred_return = 0.0
        confidence = 0.5
        if pred_rows and len(pred_rows) > 0:
            pred_return = float(dict(pred_rows[0]).get("pred_return", 0) or 0)
            confidence = float(dict(pred_rows[0]).get("confidence", 0.5) or 0.5)

        # 生成简单 20 日预测: 基于预测收益率线性外推 + 均值回归
        daily_ret = pred_return / 5  # 5日预测 → 日均
        points = []
        for i in range(1, days + 1):
            # 均值回归: 预测收益随天数衰减
            decay = 0.95 ** i
            pred_close_i = last_close * (1 + daily_ret * i * decay)
            # OHLC 估算 (基于历史波动率)
            vol = 0.02 * (1 + 0.1 * i)  # 波动率随天数递增
            spread = pred_close_i * vol
            pred_date_i = pd.Timestamp(last_date) + pd.Timedelta(days=i)
            # 跳过周末
            while pred_date_i.weekday() >= 5:
                pred_date_i += pd.Timedelta(days=1)

            points.append(PredictedOHLCVPoint(
                date=str(pred_date_i.date()),
                open=pred_close_i - spread * 0.3,
                high=pred_close_i + spread,
                low=pred_close_i - spread,
                close=pred_close_i,
                volume=0,
                confidence=confidence * (0.98 ** i),
                lower_bound=pred_close_i - spread * 2,
                upper_bound=pred_close_i + spread * 2,
            ))

        logger.info(f"GET /index/{code}/prediction → {len(points)} 条 (估算)")
        return StockPredictionResponse(
            code=code,
            name=name,
            pred_days=len(points),
            predictions=points,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测K线生成失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测K线生成失败: {e}")


def _get_config():
    """安全获取配置。"""
    from src.common.config import load_config
    return load_config()


def _resolve_name(code: str, storage=None) -> str:
    """解析标的名称：先查 _find_symbol，再查 storage，最后返回 code。"""
    try:
        config = _get_config()
        match = _find_symbol(code, config)
        if match:
            return match.get("name", code)
    except Exception:
        pass
    if storage is not None:
        try:
            rows = storage.execute_sql(
                "SELECT name FROM stock_info WHERE code = ? LIMIT 1", (code,)
            )
            if rows and len(rows) > 0:
                return str(dict(rows[0]).get("name", code))
        except Exception:
            pass
    return code
