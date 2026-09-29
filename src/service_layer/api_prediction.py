"""
Kronos 指数/ETF 择时策略系统 — 预测数据 API

端点:
  GET /api/v1/prediction/latest    — 最新预测
  GET /api/v1/prediction/history   — 历史预测归档
  GET /api/v1/prediction/accuracy  — 预测准确率统计
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.common.logger import get_logger
from src.service_layer.models import (
    PredictionResult, PredictionListResponse,
    PredictionHistoryResponse, AccuracyStats,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/prediction", tags=["预测数据"])


# ═══════════════════════════════════════════════════════════════════
# Mock 数据生成
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


def _make_mock_prediction(symbol: str, pred_date: str) -> PredictionResult:
    """生成单条模拟预测。"""
    import random
    import hashlib

    seed = int(hashlib.md5((symbol + pred_date).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    base_prices = {
        "000852": 6234.50, "000300": 3890.20, "000905": 5678.30,
        "399006": 2150.80, "000688": 980.50,  "000016": 2645.70,
        "512100": 2.45,    "510300": 3.98,    "510500": 6.12,
        "159915": 2.15,    "588000": 0.98,    "510050": 2.85,
    }
    last_close = base_prices.get(symbol, 1000.0) * (1 + rng.uniform(-0.05, 0.05))
    pred_return = rng.gauss(0.003, 0.015)
    pred_close = last_close * (1 + pred_return)

    symbol_info = next((s for s in _MOCK_SYMBOLS if s["code"] == symbol), None)
    name = symbol_info["name"] if symbol_info else symbol

    return PredictionResult(
        symbol=symbol,
        name=name,
        pred_date=pred_date,
        pred_close_t5=round(pred_close, 2),
        pred_return=round(pred_return, 4),
        last_close=round(last_close, 2),
        confidence=round(rng.uniform(0.5, 0.9), 2),
        temperature=1.0,
        top_p=0.95,
        lookback_window=60,
    )


def _try_model_predictions(symbol: Optional[str] = None) -> Optional[list]:
    """尝试使用本地训练好的模型生成真实预测。"""
    try:
        from src.service_layer.prediction_service import get_prediction_service
        svc = get_prediction_service()
        preds = svc.get_latest_predictions(symbol)
        if preds:
            from src.service_layer.models import PredictionResult
            return [PredictionResult(**p) for p in preds]
    except Exception as e:
        logger.warning(f"模型预测失败，降级: {e}")
    return None


def _snapshot_market_prices(storage) -> dict[str, tuple[str, float]]:
    """stock_daily 最新不复权快照: code -> (date, close)。单次批量查询，随调用方缓存。"""
    try:
        rows = storage.execute_sql(
            "SELECT code, date, close FROM stock_daily WHERE close IS NOT NULL AND close > 0"
        )
        snap = {}
        for r in rows:
            try:
                code = str(r[0]).zfill(6)
                snap[code] = (str(r[1]), float(r[2]))
            except (ValueError, TypeError):
                continue
        return snap
    except Exception:
        return {}


def _try_get_real_predictions(symbol: Optional[str] = None) -> Optional[list]:
    """尝试从数据库获取真实预测。"""
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)

        # 注意：model_prediction 表未持久化 last_close，故由 pred_close/(1+pred_return)
        # 反推；列名为 lookback（非 lookback_window）。
        if symbol:
            rows = storage.execute_sql(
                """SELECT p.code, p.pred_date, p.pred_close, p.pred_return,
                          p.last_close, p.confidence, p.temperature, p.top_p,
                          p.lookback, p.score, s.name, p.date, p.prediction_std
                   FROM model_prediction p
                   LEFT JOIN stock_info s ON s.code = p.code
                   WHERE p.code = ?
                   ORDER BY p.date DESC, p.pred_date DESC LIMIT 1""",
                (symbol,),
            )
        else:
            rows = storage.execute_sql(
                """SELECT p.code, p.pred_date, p.pred_close, p.pred_return,
                          p.last_close, p.confidence, p.temperature, p.top_p,
                          p.lookback, p.score, s.name, p.date, p.prediction_std
                   FROM model_prediction p
                   LEFT JOIN stock_info s ON s.code = p.code
                   WHERE p.date = (SELECT MAX(date) FROM model_prediction)
                   ORDER BY p.code""",
            )

        if rows and len(rows) > 0:
            snapshots = _snapshot_market_prices(storage)
            anomaly_threshold = float(config.get(
                "strategy", "stock_selection_max_abs_return", default=0.40
            ))
            results = []
            data_date = None
            for r in rows:
                pred_close = float(r[2])
                pred_return = float(r[3])
                last_close = float(r[4]) if r[4] is not None else (
                    pred_close / (1 + pred_return) if (1 + pred_return) != 0 else pred_close
                )
                as_of = str(r[11]) if len(r) > 11 and r[11] is not None else None
                if as_of:
                    data_date = as_of if (data_date is None or as_of > data_date) else data_date

                # 市价口径换算：后复权价 × (最新不复权市价 / 后复权最新价)
                snap = snapshots.get(str(r[0]).zfill(6))
                last_close_raw = None
                pred_close_t5_raw = None
                if snap is not None and last_close and last_close > 0:
                    last_close_raw = snap[1]
                    pred_close_t5_raw = round(last_close_raw * (pred_close / last_close), 4)

                results.append(PredictionResult(
                    symbol=r[0],
                    name=str(r[10]) if len(r) > 10 and r[10] else r[0],
                    pred_date=str(r[1]),
                    pred_close_t5=pred_close,
                    pred_return=pred_return,
                    last_close=round(last_close, 4),
                    confidence=float(r[5]) if r[5] is not None else 0.5,
                    prob_up=float(r[5]) if r[5] is not None else 0.5,
                    score=float(r[9]) if r[9] is not None else None,
                    temperature=float(r[6]) if r[6] is not None else 0.6,
                    top_p=float(r[7]) if r[7] is not None else 0.9,
                    lookback_window=int(r[8]) if r[8] is not None else 250,
                    as_of_date=as_of,
                    last_close_raw=last_close_raw,
                    pred_close_t5_raw=pred_close_t5_raw,
                    price_scale="hfq",
                    is_anomalous=abs(pred_return) > anomaly_threshold,
                    prediction_std=float(r[12]) if len(r) > 12 and r[12] is not None else None,
                ))
            return results
    except Exception as e:
        logger.debug(f"真实预测加载失败，降级 mock: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════

@router.get("/latest", response_model=PredictionListResponse)
async def get_latest_prediction(
    symbol: Optional[str] = Query(None, description="标的代码，不传则返回全部"),
):
    """
    获取最新预测结果。

    - 不传 symbol → 返回所有 12 个标的最新预测
    - 指定 symbol → 仅返回该标的的最新预测
    """
    # A persisted batch represents one complete, auditable prediction universe.
    real = _try_get_real_predictions(symbol)
    if real is not None:
        if symbol:
            real = [p for p in real if p.symbol == symbol]
        data_date = max((p.as_of_date for p in real if p.as_of_date), default=None)
        logger.info(f"GET /prediction/latest → {len(real)} 条 (数据库)")
        return PredictionListResponse(total=len(real), predictions=real, data_date=data_date)

    # Development fallback for the small local fallback model only.
    model_preds = _try_model_predictions(symbol)
    if model_preds is not None:
        logger.info(f"GET /prediction/latest → {len(model_preds)} 条 (模型预测)")
        return PredictionListResponse(total=len(model_preds), predictions=model_preds)

    raise HTTPException(
        status_code=503,
        detail="No real prediction data is available. Run a prediction batch before querying this endpoint.",
    )


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    symbol: str = Query(..., description="标的代码"),
    start: Optional[str] = Query(None, description="起始生成日期(行情截止日)，如 2026-07-29"),
    end: Optional[str] = Query(None, description="结束生成日期"),
    limit: int = Query(200, ge=1, le=1000, description="最多返回条数（按生成日倒序）"),
):
    """
    历史预测归档（预测 vs 实际）。

    语义 v2：按【生成日期 date(as_of，所用行情截止日)】倒序返回该标的历次预测，
    不再用目标日(pred_date)过滤 —— 旧语义下“刚生成的批次”目标日在未来，
    会被默认的最近 30 天窗口整批漏掉，导致个股页面看似没有历史评级。

    返回每条：生成日 as_of_date / 目标日 pred_date / 预测 5 日收益 / 实际收益
    （目标日到期后由每日归档任务回填）/ 置信度。start/end 作用于生成日期，
    缺省不过滤，由 limit 控制条数。无记录时返回空列表（不抛 404）。
    """
    symbol_info = next((s for s in _MOCK_SYMBOLS if s["code"] == symbol), {"name": symbol})

    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        sql = (
            "SELECT code, date, pred_date, pred_close, pred_return, last_close, "
            "       confidence, score, actual_close, actual_return "
            "FROM model_prediction WHERE code = ?"
        )
        params: list = [symbol]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        sql += " ORDER BY date DESC, pred_date DESC LIMIT ?"
        params.append(int(limit))
        rows = storage.execute_sql(sql, tuple(params))
    except Exception as e:
        logger.debug(f"真实预测历史加载失败: {e}")
        rows = []

    records: list[PredictionResult] = []
    as_of_dates: list[str] = []
    for r in rows:
        as_of = str(r["date"]) if r["date"] else None
        if as_of:
            as_of_dates.append(as_of)
        pred_close = float(r["pred_close"])
        pred_return = float(r["pred_return"])
        last_close = float(r["last_close"]) if r["last_close"] is not None else (
            pred_close / (1 + pred_return) if (1 + pred_return) != 0 else pred_close
        )
        records.append(PredictionResult(
            symbol=str(r["code"]),
            name=symbol_info["name"],
            as_of_date=as_of,
            pred_date=str(r["pred_date"]),
            pred_close_t5=pred_close,
            pred_return=pred_return,
            last_close=round(last_close, 4),
            confidence=float(r["confidence"]) if r["confidence"] is not None else 0.5,
            prob_up=float(r["confidence"]) if r["confidence"] is not None else 0.5,
            score=float(r["score"]) if r["score"] is not None else None,
            actual_close=float(r["actual_close"]) if r["actual_close"] is not None else None,
            actual_return=float(r["actual_return"]) if r["actual_return"] is not None else None,
        ))

    resp_start = start or (min(as_of_dates) if as_of_dates else "")
    resp_end = end or (max(as_of_dates) if as_of_dates else "")
    logger.info(f"GET /prediction/history {symbol} → {len(records)} 条 (按生成日归档)")
    return PredictionHistoryResponse(
        symbol=symbol, name=symbol_info["name"],
        start=resp_start, end=resp_end, count=len(records), records=records,
    )


@router.get("/accuracy", response_model=AccuracyStats)
async def get_prediction_accuracy(
    symbol: str = Query(..., description="标的代码"),
):
    """
    获取指定标的的预测准确率统计。

    包含：方向准确率、MAE、RMSE、Spearman 相关系数。
    """
    symbol_info = next((s for s in _MOCK_SYMBOLS if s["code"] == symbol), {"name": symbol})

    # 尝试真实数据
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            """SELECT pred_return, actual_return FROM model_prediction
               WHERE code = ? AND actual_return IS NOT NULL
               ORDER BY pred_date DESC LIMIT 200""",
            (symbol,),
        )
        if rows and len(rows) >= 10:
            import numpy as np
            from scipy.stats import spearmanr

            preds = np.array([float(r[0]) for r in rows])
            actuals = np.array([float(r[1]) for r in rows])

            direction_match = np.mean(np.sign(preds) == np.sign(actuals))
            mae = np.mean(np.abs(preds - actuals))
            rmse = np.sqrt(np.mean((preds - actuals) ** 2))
            sp, _ = spearmanr(preds, actuals)

            logger.info(f"GET /prediction/accuracy → 方向准确率={direction_match:.2%}")
            return AccuracyStats(
                symbol=symbol,
                name=symbol_info["name"],
                direction_accuracy=round(float(direction_match), 4),
                mae=round(float(mae), 4),
                rmse=round(float(rmse), 4),
                spearman=round(float(sp), 4),
                total_predictions=len(rows),
            )
    except Exception as e:
        logger.debug(f"真实准确率计算失败，降级 mock: {e}")

    raise HTTPException(status_code=404, detail=f"Insufficient realized predictions to calculate accuracy for {symbol}")
