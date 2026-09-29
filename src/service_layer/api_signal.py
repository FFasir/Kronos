"""
Kronos 指数/ETF 择时策略系统 — 择时信号 API

端点:
  GET /api/v1/signal/latest      — 所有标的最新信号
  GET /api/v1/signal/{code}      — 单个标的历史信号
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

from src.common.logger import get_logger
from src.service_layer.models import (
    SignalResult, SignalListResponse, SignalHistoryResponse,
    SignalTypeEnum,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/signal", tags=["择时信号"])


# ═══════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════

_MOCK_SYMBOLS = [
    {"code": "000852", "name": "中证1000"},
    {"code": "000300", "name": "沪深300"},
    {"code": "000905", "name": "中证500"},
    {"code": "399006", "name": "创业板指"},
    {"code": "000688", "name": "科创50"},
    {"code": "000016", "name": "上证50"},
    {"code": "512100", "name": "中证1000ETF"},
    {"code": "510300", "name": "沪深300ETF"},
    {"code": "510500", "name": "中证500ETF"},
    {"code": "159915", "name": "创业板ETF"},
    {"code": "588000", "name": "科创50ETF"},
    {"code": "510050", "name": "上证50ETF"},
]


# ═══════════════════════════════════════════════════════════════════
# Mock 数据生成
# ═══════════════════════════════════════════════════════════════════

def _make_mock_signal(symbol: str, date: str, include_history: bool = False) -> SignalResult:
    """生成模拟择时信号。"""
    import random
    import hashlib

    # 可复现随机
    seed = int(hashlib.md5((symbol + date).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # 模拟: 约40%概率 BUY
    rand_val = rng.random()
    if rand_val < 0.4:
        signal_type = SignalTypeEnum.BUY
        pred_return = rng.uniform(0.005, 0.04)
        position_pct = min(rng.uniform(0.2, 0.35), 0.30)
        reason = f"预测收益率 {pred_return:.2%} > 0.5%"
    elif rand_val < 0.7:
        signal_type = SignalTypeEnum.CASH
        pred_return = rng.uniform(-0.02, 0.005)
        position_pct = 0.0
        reason = f"预测收益率 {pred_return:.2%} ≤ 0.5%"
    elif rand_val < 0.9:
        signal_type = SignalTypeEnum.HOLD
        pred_return = rng.uniform(0.002, 0.03)
        position_pct = 0.25
        reason = f"持仓中 (第{rng.randint(2, 5)}天/5天)"
    else:
        signal_type = SignalTypeEnum.SELL
        pred_return = rng.uniform(-0.03, 0.01)
        position_pct = 0.0
        reason = "持仓到期 (5天)" if rng.random() > 0.3 else "止损触发"

    info = next((s for s in _MOCK_SYMBOLS if s["code"] == symbol), None)
    name = info["name"] if info else symbol

    return SignalResult(
        symbol=symbol,
        name=name,
        date=date,
        signal_type=signal_type,
        pred_return=round(pred_return, 4),
        pred_close=round(_base_price(symbol) * (1 + pred_return), 2),
        confidence=round(rng.uniform(0.5, 0.9), 2),
        position_pct=round(position_pct, 2),
        reason=reason,
    )


def _base_price(code: str) -> float:
    """获取基准价。"""
    prices = {
        "000852": 6234.50, "000300": 3890.20, "000905": 5678.30,
        "399006": 2150.80, "000688": 980.50,  "000016": 2645.70,
        "512100": 2.45,    "510300": 3.98,    "510500": 6.12,
        "159915": 2.15,    "588000": 0.98,    "510050": 2.85,
    }
    return prices.get(code, 1000.0)


# ═══════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════

@router.get("/latest", response_model=SignalListResponse)
async def get_latest_signals():
    """
    获取所有 12 个标的最新择时信号。

    按信号优先级排序：BUY > HOLD > CASH > SELL。
    前端 ETF 择时池页面主数据源。
    """
    # Persisted signals are the production source of truth. They are cheap to
    # serve and preserve the exact model/configuration used for the decision.
    real_signals = _try_get_real_signals()
    if real_signals is not None:
        logger.info(f"GET /signal/latest → {len(real_signals)} 条 (数据库)")
        signal_date = max(signal.date for signal in real_signals)
        return SignalListResponse(total=len(real_signals), date=signal_date, signals=real_signals)

    # Live inference is only a development fallback when no audited batch has
    # been materialized yet.
    model_signals = _try_model_signals()
    if model_signals is not None:
        logger.info(f"GET /signal/latest → {len(model_signals)} 条 (模型预测)")
        signal_date = max(signal.date for signal in model_signals)
        return SignalListResponse(total=len(model_signals), date=signal_date, signals=model_signals)

    raise HTTPException(
        status_code=503,
        detail="No real trading signals are available. Generate signals from a prediction batch first.",
    )


@router.get("/{code}", response_model=SignalHistoryResponse)
async def get_signal_history(
    code: str,
    days: int = Query(30, ge=1, le=365, description="回溯天数"),
):
    """
    获取单个标的最近 N 天的历史信号流。

    返回信号流（BUY → HOLD → CASH → ...），用于前端信号标记。
    """
    symbol_info = {"name": _symbol_name(code)}

    # 尝试真实数据
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            """SELECT code, date, signal_type, pred_return, pred_close,
                      confidence, position_pct, reason
               FROM trading_signal
               WHERE code = ?
               ORDER BY date DESC LIMIT ?""",
            (code, days),
        )
        if rows and len(rows) > 0:
            signals = []
            for r in rows:
                try:
                    st = SignalTypeEnum(r[2])
                except (ValueError, KeyError):
                    st = SignalTypeEnum.CASH
                signals.append(SignalResult(
                    symbol=r[0],
                    name=symbol_info["name"],
                    date=str(r[1]),
                    signal_type=st,
                    pred_return=float(r[3]) if len(r) > 3 else 0.0,
                    pred_close=float(r[4]) if len(r) > 4 and r[4] is not None else None,
                    confidence=float(r[5]) if len(r) > 5 and r[5] is not None else None,
                    position_pct=float(r[6]) if len(r) > 6 else 0.0,
                    reason=str(r[7]) if len(r) > 7 and r[7] else None,
                ))
            logger.info(f"GET /signal/{code} → {len(signals)} 条 (真实数据)")
            return SignalHistoryResponse(
                symbol=code, name=symbol_info["name"],
                days=days, count=len(signals), signals=signals,
            )
    except Exception as e:
        logger.debug(f"真实信号历史加载失败，降级 mock: {e}")

    raise HTTPException(status_code=404, detail=f"No persisted signal history is available for {code}")


def _try_model_signals() -> Optional[list]:
    """尝试使用本地训练好的模型生成真实信号。"""
    try:
        from src.service_layer.prediction_service import get_prediction_service
        svc = get_prediction_service()
        raw = svc.get_latest_signals()
        if raw:
            return [SignalResult(**s) for s in raw]
    except Exception as e:
        logger.warning(f"模型信号生成失败，降级: {e}")
    return None


def _try_get_real_signals() -> Optional[list]:
    """Load the latest complete current-CSI1000 signal batch."""
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        from src.strategy_layer.realtime_timing import load_current_csi1000_components

        members = load_current_csi1000_components(refresh=False)
        member_codes = set(members["code"].astype(str))
        rows = storage.execute_sql(
            """SELECT t.code, t.date, t.signal_type, t.pred_return, t.pred_close,
                      t.confidence, t.position_pct, t.reason
               FROM trading_signal t
               INNER JOIN (
                   SELECT code, MAX(date) AS max_date
                   FROM trading_signal
                   GROUP BY code
               ) latest ON t.code = latest.code AND t.date = latest.max_date
               ORDER BY t.code"""
        )

        rows = [row for row in rows if str(row[0]) in member_codes]
        if rows and len(rows) > 0:
            newest_date = max(str(row[1]) for row in rows)
            rows = [row for row in rows if str(row[1]) == newest_date]
            signals = []
            names = _symbol_names(storage, [str(row[0]) for row in rows])
            for r in rows:
                try:
                    st = SignalTypeEnum(r[2])
                except (ValueError, KeyError):
                    st = SignalTypeEnum.CASH

                signals.append(SignalResult(
                    symbol=r[0],
                    name=names.get(str(r[0]), str(r[0])),
                    date=str(r[1]),
                    signal_type=st,
                    pred_return=float(r[3]) if len(r) > 3 else 0.0,
                    pred_close=float(r[4]) if len(r) > 4 and r[4] is not None else None,
                    confidence=float(r[5]) if len(r) > 5 and r[5] is not None else None,
                    position_pct=float(r[6]) if len(r) > 6 else 0.0,
                    reason=str(r[7]) if len(r) > 7 and r[7] else None,
                ))
            return signals
    except Exception as e:
        logger.debug(f"真实信号加载失败: {e}")

    return None


def _symbol_names(storage, codes: list[str]) -> dict[str, str]:
    if not codes:
        return {}
    placeholders = ",".join("?" for _ in codes)
    rows = storage.execute_sql(
        f"SELECT code, name FROM stock_info WHERE code IN ({placeholders})",
        tuple(codes),
    )
    return {str(row[0]): str(row[1]) for row in rows}


def _symbol_name(code: str) -> str:
    try:
        from src.common.config import load_config
        from src.data_layer.storage import StorageLayer

        return _symbol_names(StorageLayer(load_config()), [code]).get(code, code)
    except Exception:
        return code
