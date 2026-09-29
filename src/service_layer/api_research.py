"""Read-only API for the frozen model-selection research results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from src.service_layer.api_admin import require_admin


router = APIRouter(prefix="/admin/research", tags=["research"], dependencies=[Depends(require_admin)])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_RUNS = (
    {
        "key": "kronos_base",
        "name": "Kronos Base",
        "version": "base-selected-20260716",
        "checkpoint": "models/kronos_base",
        "tokenizer": "models/kronos_tokenizer",
        "dev": "base_csi1000_etf_dev_2025",
        "holdout": "base_csi1000_etf_holdout_2025q4_2026h1",
        "selected": True,
    },
    {
        "key": "legacy_all_a_share",
        "name": "旧全 A 微调",
        "version": "all-a-share-20260715",
        "checkpoint": "models/official_finetune/all_a_share/all_a_share_predictor/checkpoints/best_model",
        "tokenizer": "models/official_finetune/all_a_share/all_a_share_tokenizer/checkpoints/best_model",
        "dev": "all_a_share_etf_walk_forward",
        "holdout": "all_a_share_etf_final_holdout_2025q4_2026h1",
        "selected": False,
    },
    {
        "key": "new_all_a_share_hfq_v2",
        "name": "新全 A / HFQ v2",
        "version": "all-a-share-hfq-v2-20260716",
        "checkpoint": "models/official_finetune/all_a_share_hfq_v2_20260715/predictor/checkpoints/best_model",
        "tokenizer": "models/official_finetune/all_a_share_hfq_v2_20260715/tokenizer/checkpoints/best_model",
        "dev": "new_all_a_share_hfq_v2_csi1000_etf_dev_2025",
        "holdout": "new_all_a_share_hfq_v2_csi1000_etf_holdout_2025q4_2026h1",
        "selected": False,
    },
    {
        "key": "legacy_csi1000",
        "name": "旧 CSI1000 微调",
        "version": "legacy-csi1000-20260715",
        "checkpoint": "models/official_finetune/csi1000_predictor/checkpoints/best_model",
        "tokenizer": "models/official_finetune/csi1000_tokenizer/checkpoints/best_model",
        "dev": "legacy_csi1000_etf_dev_2025",
        "holdout": "legacy_csi1000_etf_holdout_2025q4_2026h1",
        "selected": False,
    },
)


def _read_summary(run_dir: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "outputs" / run_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "net_compounded_return",
        "benchmark_buy_and_hold_return",
        "total_relative_excess_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_zero_rf",
        "max_drawdown",
        "directional_accuracy",
        "prediction_spearman",
        "trade_count",
        "win_rate_before_cost",
        "in_market_ratio",
        "trading_days",
    )
    return {key: summary.get(key) for key in keys}


def _read_equity(run_dir: str) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "data" / "outputs" / run_dir / "daily_equity.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "date": row["date"],
                "equity": float(row["equity"]),
                "benchmark_equity": float(row["benchmark_equity"]),
                "in_market": row["in_market"].lower() == "true",
            }
            for row in csv.DictReader(handle)
        ]


@router.get("/summary")
def get_research_summary() -> dict[str, Any]:
    """Return the frozen timing comparison and selected-model equity curve."""
    try:
        comparisons = []
        for model in MODEL_RUNS:
            comparisons.append(
                {
                    **{key: model[key] for key in ("key", "name", "version", "checkpoint", "tokenizer", "selected")},
                    "development": _metrics(_read_summary(model["dev"])),
                    "holdout": _metrics(_read_summary(model["holdout"])),
                }
            )
        selected = next(model for model in MODEL_RUNS if model["selected"])
        equity = _read_equity(selected["holdout"])
        from src.model_layer.model_registry import list_models

        registry = list_models()
        active = next(model for model in registry["models"] if model["active"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"research artifacts unavailable: {exc}") from exc

    return {
        "as_of": "2026-07-16",
        "research_status": "candidate_passed_observed_holdout",
        "production_model": {
            "key": active["key"],
            "name": active["name"],
            "version": active["version"],
            "checkpoint": active["predictor_path"],
            "tokenizer": active["tokenizer_path"],
            "parameter_count": 102_300_000 if active["kind"] == "builtin" else None,
            "purpose": "中证1000指数信号与 512100 ETF 五日择时",
        },
        "recommended_model_key": selected["key"],
        "comparison": comparisons,
        "selected_holdout_equity": equity,
        "experiment": {
            "development_period": "2025-02 至 2025-09",
            "holdout_period": "2025-10 至 2026-06",
            "signal_asset": "中证1000指数 000852",
            "execution_asset": "中证1000 ETF 512100",
            "selection": "每个测试月仅使用前一个完整月，按成本后净收益从 48 组参数中选参",
            "inference": "10 条随机轨迹；5 日预测；预测收益大于 0.5% 入场",
            "execution": "信号日收盘后决策，下一交易日开盘买入，第五个交易日收盘卖出，持仓不重叠",
            "costs": "佣金双边 0.03%，滑点双边 0.10%，ETF 印花税 0",
        },
        "risk_notice": (
            "该区间已被本项目观察，结果是研究证据而非未来实盘保证。"
            "上线后必须用 2026-07 之后新增数据继续做真正前瞻检验。"
        ),
    }
