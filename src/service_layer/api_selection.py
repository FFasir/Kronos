"""Stock-pool and strategy-preset endpoints backed by real prediction records."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd

from src.common.config import load_config
from src.common.logger import get_logger
from src.service_layer.models import (
    PredictionListResponse, PredictionResult,
    StrategyPreset, StrategyPresetListResponse,
    SelectionStats, EliminatedItem,
    SelectionGenerateRequest,
)
from src.service_layer.api_prediction import _try_get_real_predictions, _try_model_predictions
from src.strategy_layer.stock_selector import StockSelector, PRESETS

logger = get_logger(__name__)

router = APIRouter(prefix="/selection", tags=["stock selection"])


# ═══════════════════════════════════════════════════════════════════
# 策略预设
# ═══════════════════════════════════════════════════════════════════

@router.get("/presets", response_model=StrategyPresetListResponse, tags=["strategy"])
async def get_strategy_presets():
    """
    获取策略风格预设列表（保守/均衡/激进）。

    前端选股池页面和策略设置页面使用。
    """
    presets = StockSelector.list_presets()
    return StrategyPresetListResponse(
        presets=[StrategyPreset(**p) for p in presets]
    )


# ═══════════════════════════════════════════════════════════════════
# 选股池
# ═══════════════════════════════════════════════════════════════════

def _available_predictions() -> list[PredictionResult]:
    """Persisted batch predictions are preferred."""
    predictions = _try_get_real_predictions()
    if predictions is None:
        predictions = _try_model_predictions()
    if not predictions:
        raise HTTPException(
            status_code=503,
            detail="No real prediction batch is available. Run the prediction job before requesting a stock pool.",
        )
    return predictions


def _with_liquidity(predictions: list[PredictionResult]) -> list[dict]:
    """Attach liquidity for the whole universe with one SQLite query."""
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(load_config())
    rows = storage.execute_sql(
        """WITH recent_dates AS (
               SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 20
           )
           SELECT d.code, d.date, d.close, d.volume, d.amount
           FROM stock_daily d
           INNER JOIN recent_dates r ON r.date = d.date
           ORDER BY d.code, d.date"""
    )
    market = {}
    latest_market_date = None
    if rows:
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["date"] = frame["date"].astype(str)
        for column in ["close", "volume", "amount"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["effective_amount"] = frame["amount"].where(
            frame["amount"] > 0, frame["close"] * frame["volume"]
        )
        latest_market_date = str(frame["date"].max())
        for code, group in frame.groupby("code", sort=False):
            latest = group.iloc[-1]
            market[code] = {
                "daily_amount": float(group["effective_amount"].mean()),
                "last_date": str(latest["date"]),
                "last_volume": float(latest["volume"] or 0.0),
            }
    enriched = []
    for prediction in predictions:
        item = prediction.model_dump()
        stats = market.get(str(prediction.symbol).zfill(6))
        if stats is None:
            item["suspended"] = True
        else:
            item["daily_amount"] = stats["daily_amount"]
            item["suspended"] = bool(
                stats["last_volume"] <= 0
                or (latest_market_date and stats["last_date"] < latest_market_date)
            )
        enriched.append(item)
    return enriched


@router.get("/latest")
async def get_latest_selection(
    preset: Optional[str] = Query(None, description="策略预设: conservative/balanced/aggressive"),
):
    """
    获取最新选股池，含筛选统计和被剔除原因。

    Query params:
      - preset: 可选，应用策略预设后筛选
    """
    config = load_config()
    selector = StockSelector(config)

    if preset:
        if not selector.apply_preset(preset):
            raise HTTPException(status_code=400, detail=f"未知预设: {preset}")

    predictions_with_liquidity = _with_liquidity(_available_predictions())
    result = selector.select(predictions_with_liquidity, include_stats=True)

    selected = result["selected"]
    stats = result["stats"]
    data_date = max(
        (str(item.get("as_of_date")) for item in selected if item.get("as_of_date")),
        default=None,
    )

    return {
        "total": len(selected),
        "predictions": [PredictionResult(**item) for item in selected],
        "stats": stats,
        "data_date": data_date,
    }


@router.post("/generate")
async def generate_selection(req: SelectionGenerateRequest):
    """
    手动生成选股池（支持自定义阈值覆盖）。

    将结果写入 stock_selection 表并返回筛选统计。
    """
    config = load_config()
    selector = StockSelector(config)

    # 应用预设
    if req.preset:
        selector.apply_preset(req.preset)

    # 覆盖参数
    if req.prob_up_threshold is not None:
        selector.prob_up_threshold = req.prob_up_threshold
    if req.score_threshold is not None:
        selector.score_threshold = req.score_threshold
    if req.min_daily_amount is not None:
        selector.min_daily_amount = req.min_daily_amount
    if req.max_per_industry is not None:
        selector.max_per_industry = req.max_per_industry
    if req.top_k is not None:
        selector.top_k = req.top_k

    predictions_with_liquidity = _with_liquidity(_available_predictions())
    result = selector.select(predictions_with_liquidity, include_stats=True)

    selected = result["selected"]
    stats = result["stats"]

    # 写入 stock_selection 表
    try:
        from src.data_layer.storage import StorageLayer
        from datetime import date

        storage = StorageLayer(config)
        today = str(date.today())
        rows = [
            (
                today,
                item.get("symbol", ""),
                item.get("prob_up", 0.0),
                item.get("score", 0.0),
                item.get("rank", 0),
                1,  # selected
            )
            for item in selected
        ]
        if rows:
            storage.execute_many(
                """INSERT OR REPLACE INTO stock_selection
                   (date, code, prob_up, score, rank, selected)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
            logger.info(f"已保存 {len(rows)} 条选股结果到数据库")
    except Exception as e:
        logger.warning(f"选股结果保存失败: {e}")

    data_date = max(
        (str(item.get("as_of_date")) for item in selected if item.get("as_of_date")),
        default=None,
    )

    return {
        "total": len(selected),
        "predictions": [PredictionResult(**item) for item in selected],
        "stats": stats,
        "data_date": data_date,
    }
