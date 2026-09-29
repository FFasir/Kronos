"""
Kronos 指数/ETF 择时策略系统 — 股票选择器（个股扩展）

设计意图:
  - 基于 Kronos 预测的上涨概率（prob_up）筛选个股
  - 阈值: 上涨概率 ≥ 0.65，收益分值 ≥ 6.0（均衡模式）
  - 流动性过滤: 日均成交额 ≥ 500 万
  - 行业二次筛选（避免过度集中）
  - 策略风格预设: 保守/均衡/激进三档
  - 筛选统计: 各阶段数量 + 被剔除原因
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 策略风格预设
# ═══════════════════════════════════════════════════════════════════

PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "label": "保守",
        "icon": "🛡️",
        "description": "高概率门槛、高分值、高流动性，严格控制风险",
        "prob_up_threshold": 0.75,
        "score_threshold": 7.0,
        "min_daily_amount": 10_000_000.0,
        "max_per_industry": 2,
        "top_k": 20,
    },
    "balanced": {
        "label": "均衡",
        "icon": "⚖️",
        "description": "概率与收益平衡，适度行业分散",
        "prob_up_threshold": 0.65,
        "score_threshold": 6.0,
        "min_daily_amount": 5_000_000.0,
        "max_per_industry": 3,
        "top_k": 50,
    },
    "aggressive": {
        "label": "激进",
        "icon": "🚀",
        "description": "低门槛宽松筛选，追求更多标的和更高收益潜力",
        "prob_up_threshold": 0.55,
        "score_threshold": 4.5,
        "min_daily_amount": 3_000_000.0,
        "max_per_industry": 5,
        "top_k": 100,
    },
}


@dataclass
class SelectionStats:
    """选股筛选统计"""
    initial_count: int = 0
    anomaly_count: int = 0
    after_prob_score: int = 0
    after_st_suspended: int = 0
    after_liquidity: int = 0
    after_industry: int = 0
    final_count: int = 0
    eliminated: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_count": self.initial_count,
            "anomaly_count": self.anomaly_count,
            "after_prob_score": self.after_prob_score,
            "after_st_suspended": self.after_st_suspended,
            "after_liquidity": self.after_liquidity,
            "after_industry": self.after_industry,
            "final_count": self.final_count,
            "eliminated": self.eliminated,
        }


class StockSelector:
    """
    个股选股器。

    支持:
      - 三档策略预设（保守/均衡/激进）
      - 上涨概率 + 收益分值阈值筛选
      - ST / 停牌过滤
      - 流动性过滤
      - 行业集中度过滤
      - 综合排序 Top-K
      - 筛选统计 + 剔除原因
    """

    def __init__(self, config):
        self.config = config
        self.prob_up_threshold: float = config.get("strategy", "stock_selection_prob_up", default=0.65)
        self.score_threshold: float = config.get("strategy", "stock_selection_score", default=6.0)
        self.min_daily_amount: float = config.get("strategy", "stock_selection_min_amount", default=5000000.0)
        self.max_per_industry: int = config.get("strategy", "max_per_industry", default=3)
        self.top_k: int = config.get("strategy", "stock_selection_top_k", default=50)
        self._current_preset: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════
    # 策略预设
    # ═══════════════════════════════════════════════════════════════

    def apply_preset(self, name: str) -> bool:
        """
        应用策略风格预设。

        Args:
            name: 预设名称 (conservative / balanced / aggressive)

        Returns:
            是否成功应用
        """
        preset = PRESETS.get(name)
        if preset is None:
            logger.warning(f"未知策略预设: {name}")
            return False

        self.prob_up_threshold = preset["prob_up_threshold"]
        self.score_threshold = preset["score_threshold"]
        self.min_daily_amount = preset["min_daily_amount"]
        self.max_per_industry = preset["max_per_industry"]
        self.top_k = preset["top_k"]
        self._current_preset = name

        logger.info(
            f"已应用策略预设 [{preset['icon']} {preset['label']}]: "
            f"prob_up≥{self.prob_up_threshold}, score≥{self.score_threshold}, "
            f"min_amount≥{self.min_daily_amount:,.0f}, "
            f"max_per_industry={self.max_per_industry}, top_k={self.top_k}"
        )
        return True

    @staticmethod
    def get_preset(name: str) -> Optional[Dict[str, Any]]:
        """获取指定预设的完整配置。"""
        return PRESETS.get(name)

    @staticmethod
    def list_presets() -> List[Dict[str, Any]]:
        """返回所有预设列表。"""
        return [
            {"name": key, **value}
            for key, value in PRESETS.items()
        ]

    @property
    def current_preset(self) -> Optional[str]:
        """当前应用的预设名称。"""
        return self._current_preset

    # ═══════════════════════════════════════════════════════════════
    # 核心选股
    # ═══════════════════════════════════════════════════════════════

    def select(
        self,
        predictions: List[Dict[str, Any]],
        market_data: Optional[pd.DataFrame] = None,
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        从候选池中筛选个股。

        Args:
            predictions: KronosPredictor 对个股的预测结果
            market_data: 全市场行情数据
            include_stats: 是否返回筛选统计

        Returns:
            {"selected": [...], "stats": SelectionStats} 或 {"selected": [...]}
        """
        stats = SelectionStats(initial_count=len(predictions))
        eliminated: List[Dict[str, Any]] = []

        # ── 阶段0: 数据异常过滤 ──
        # |5日预测收益| 超过合理区间（默认 ±40%）的预测直接剔除：
        # 宁可少给几条，也不让荒谬数字进入选股池。
        anomaly_threshold = float(self.config.get(
            "strategy", "stock_selection_max_abs_return", default=0.40
        ))
        after_anomaly = []
        for prediction in predictions:
            candidate = dict(prediction)
            pred_return = float(candidate.get("pred_return") or 0.0)
            if abs(pred_return) > anomaly_threshold:
                eliminated.append({
                    "symbol": candidate.get("symbol", ""),
                    "name": str(candidate.get("name") or ""),
                    "reason": (
                        f"数据异常：5日预测收益 {pred_return:+.1%} 超出 "
                        f"±{anomaly_threshold:.0%} 合理区间"
                    ),
                    "stage": "anomaly",
                })
                continue
            after_anomaly.append(candidate)

        stats.anomaly_count = len(eliminated)

        # ── 阶段1: 概率+分值筛选 ──
        after_prob_score = []
        for prediction in after_anomaly:
            candidate = dict(prediction)
            name = str(candidate.get("name") or "")

            # 概率阈值
            probability = candidate.get("prob_up")
            if probability is None:
                probability = candidate.get("confidence", 0.0)
            prob_up = float(probability or 0.0)
            expected_return = float(candidate.get("pred_return", 0.0) or 0.0)
            score = float(candidate.get("score", expected_return * 100.0) or 0.0)

            if prob_up < self.prob_up_threshold or score < self.score_threshold:
                eliminated.append({
                    "symbol": candidate.get("symbol", ""),
                    "name": name,
                    "reason": f"概率/分值不足 (prob_up={prob_up:.2f}<{self.prob_up_threshold} 或 score={score:.1f}<{self.score_threshold})",
                    "stage": "prob_score",
                })
                continue

            candidate.update({"prob_up": prob_up, "score": score})
            after_prob_score.append(candidate)

        stats.after_prob_score = len(after_prob_score)

        # ── 阶段2: ST / 停牌过滤 ──
        after_st_suspended = []
        for candidate in after_prob_score:
            name = str(candidate.get("name") or "")
            if candidate.get("is_st") or "ST" in name.upper() or candidate.get("suspended"):
                eliminated.append({
                    "symbol": candidate.get("symbol", ""),
                    "name": name,
                    "reason": f"{'ST股票' if (candidate.get('is_st') or 'ST' in name.upper()) else '停牌'}",
                    "stage": "st_suspended",
                })
                continue
            after_st_suspended.append(candidate)

        stats.after_st_suspended = len(after_st_suspended)

        # ── 阶段3: 流动性过滤 ──
        after_liquidity = []
        for candidate in after_st_suspended:
            liquidity = self._daily_amount(candidate, market_data)
            candidate["daily_amount"] = liquidity
            if liquidity is not None and liquidity < self.min_daily_amount:
                eliminated.append({
                    "symbol": candidate.get("symbol", ""),
                    "name": candidate.get("name", ""),
                    "reason": f"低流动性 (日均成交额={liquidity:,.0f}<{self.min_daily_amount:,.0f})",
                    "stage": "liquidity",
                })
                continue
            after_liquidity.append(candidate)

        stats.after_liquidity = len(after_liquidity)

        # ── 阶段4: 行业集中度过滤 ──
        after_industry = self.filter_by_industry(after_liquidity, self.max_per_industry)
        # 被行业集中度剔除的
        industry_selected_symbols = {c["symbol"] for c in after_industry}
        for candidate in after_liquidity:
            if candidate["symbol"] not in industry_selected_symbols:
                eliminated.append({
                    "symbol": candidate.get("symbol", ""),
                    "name": candidate.get("name", ""),
                    "reason": f"行业集中度超限 (行业={candidate.get('industry', 'UNKNOWN')})",
                    "stage": "industry",
                })

        stats.after_industry = len(after_industry)

        # ── 阶段5: 排序 + Top-K ──
        selected = self.rank(after_industry, self.top_k)
        # 被Top-K截断的
        top_k_symbols = {c["symbol"] for c in selected}
        for candidate in after_industry:
            if candidate["symbol"] not in top_k_symbols:
                eliminated.append({
                    "symbol": candidate.get("symbol", ""),
                    "name": candidate.get("name", ""),
                    "reason": f"Top-K 截断 (排名超出前{self.top_k})",
                    "stage": "top_k",
                })

        stats.final_count = len(selected)
        stats.eliminated = eliminated

        # 标记入选
        for candidate in selected:
            candidate["selected"] = True

        if include_stats:
            return {"selected": selected, "stats": stats.to_dict()}
        return {"selected": selected}

    # ═══════════════════════════════════════════════════════════════
    # 排序
    # ═══════════════════════════════════════════════════════════════

    def rank(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        对候选股票排序。

        Returns:
            按综合得分降序排列的股票列表
        """
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                float(item.get("pred_return", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )[:top_k]
        for index, candidate in enumerate(ranked, start=1):
            candidate["rank"] = index
        return ranked

    # ═══════════════════════════════════════════════════════════════
    # 过滤器
    # ═══════════════════════════════════════════════════════════════

    def filter_by_liquidity(
        self,
        symbols: List[str],
        daily_data: pd.DataFrame,
    ) -> List[str]:
        """
        按日均成交额过滤低流动性标的。

        Returns:
            通过流动性%流动性筛选的标的代码列表
        """
        if daily_data is None or daily_data.empty:
            return []
        result: List[str] = []
        for symbol in symbols:
            frame = daily_data
            if isinstance(daily_data.columns, pd.MultiIndex) and symbol in daily_data.columns.get_level_values(0):
                frame = daily_data[symbol]
            if "amount" in frame:
                amount = pd.to_numeric(frame["amount"], errors="coerce").tail(20).mean()
            elif {"close", "volume"}.issubset(frame.columns):
                amount = (pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(frame["volume"], errors="coerce")).tail(20).mean()
            else:
                continue
            if pd.notna(amount) and float(amount) >= self.min_daily_amount:
                result.append(symbol)
        return result

    def filter_by_industry(
        self,
        candidates: List[Dict[str, Any]],
        max_per_industry: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        行业集中度过滤。

        Returns:
            行业分散化后的候选列表
        """
        grouped: Dict[str, int] = {}
        result: List[Dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True):
            industry = str(candidate.get("industry") or "UNKNOWN")
            if industry == "UNKNOWN":
                result.append(candidate)
                continue
            if grouped.get(industry, 0) >= max_per_industry:
                continue
            grouped[industry] = grouped.get(industry, 0) + 1
            result.append(candidate)
        return result

    # ═══════════════════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _daily_amount(candidate: Dict[str, Any], market_data: Optional[pd.DataFrame]) -> Optional[float]:
        if candidate.get("daily_amount") is not None:
            return float(candidate["daily_amount"])
        if candidate.get("amount") is not None:
            return float(candidate["amount"])
        if candidate.get("last_close") is not None and candidate.get("volume") is not None:
            return float(candidate["last_close"]) * float(candidate["volume"])
        if market_data is None or market_data.empty:
            return None
        if "amount" in market_data:
            return float(pd.to_numeric(market_data["amount"], errors="coerce").tail(20).mean())
        if {"close", "volume"}.issubset(market_data.columns):
            return float((market_data["close"] * market_data["volume"]).tail(20).mean())
        return None
