"""
Kronos 指数/ETF 择时策略系统 — 择时信号生成器

基于 Kronos 预测收益率生成择时信号（论文 §3.2 方法）。

信号逻辑:
  - T-1 日收盘后，Kronos 预测 T+4 日（未来第5日）收盘价
  - 预测收益率 = pred_close_t5 / close_t_minus_1 - 1
  - 信号判定: pred_return > 0.5% → BUY（满仓做多），否则 CASH（空仓）
  - 固定持仓 5 个交易日，持仓期间信号为 HOLD
  - 仅做多 (long_only=True)

状态机:  CASH → BUY → HOLD(第2-5天) → CASH

多标的扩展:
  - 各标的独立推理 → 各自信号判定
  - 产生 BUY 信号的标的等权重分配资金
  - 单标的上限 30%（风险分散）
  - 至少保留 2 个不同标的（若仅 1 个触发 BUY，最多 30% 仓位）
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.utils import format_date

logger = get_logger(__name__)


class SignalType(str, Enum):
    """择时信号类型"""
    BUY = "BUY"        # 买入开仓（仅在 CASH 状态时生效）
    HOLD = "HOLD"      # 持仓中（买入后第2-5天）
    CASH = "CASH"      # 空仓观望
    SELL = "SELL"      # 卖出平仓（持仓到期 / 止损 / 止盈）


class PositionState(str, Enum):
    """持仓状态"""
    CASH = "CASH"      # 空仓
    LONG = "LONG"      # 持仓中


class SignalGenerator:
    """
    择时信号生成器。

    论文方法（§3.2）:
      - 使用 Kronos 预测未来第5日收盘价
      - pred_return > 0.5% → 满仓做多，持仓5天
      - 否则空仓
      - 每月滚动搜参优化 T, top_p, lookback

    ``strategy.paper_mode=true`` 时严格复现论文：只交易 ``paper_symbol``，
    每次 BUY 都是100%仓位；额外风控由 RiskManager 单独关闭。
    """

    def __init__(self, config):
        """
        Args:
            config: 系统配置 Config 实例
        """
        self.config = config

        # 信号阈值（按论文 + 配置）
        self.buy_threshold: float = config.get("strategy", "buy_threshold", default=0.005)
        self.long_only: bool = config.get("strategy", "long_only", default=True)
        self.hold_days: int = config.get("strategy", "hold_days", default=5)
        self.paper_mode: bool = config.get("strategy", "paper_mode", default=False)
        self.paper_symbol: str = config.get("strategy", "paper_symbol", default="000852")

        # 仓位管理
        self.max_single_position: float = config.get("strategy", "max_single_position", default=0.30)
        self.min_positions: int = config.get("strategy", "min_positions", default=2)

        # 风控阈值（传递给 RiskManager）
        self.stop_loss: float = config.get("strategy", "stop_loss", default=-0.08)
        self.take_profit: float = config.get("strategy", "take_profit", default=0.15)

        logger.info(
            f"SignalGenerator 初始化: buy_threshold={self.buy_threshold:.1%}, "
            f"hold_days={self.hold_days}, long_only={self.long_only}"
        )

    # ═══════════════════════════════════════════════
    # 核心信号生成
    # ═══════════════════════════════════════════════

    def generate(
        self,
        predictions: List[Dict[str, Any]],
        current_date: str,
        existing_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        基于 Kronos 预测结果生成当日择时信号。

        Args:
            predictions: KronosPredictor 输出的预测结果列表
                [{symbol, pred_close_t5, pred_return, last_close, confidence, ...}, ...]
            current_date: 当前信号生成日期 (YYYY-MM-DD)
            existing_positions: 现有持仓信息
                {symbol: {entry_date, entry_price, quantity, hold_day_count, state}}

        Returns:
            信号列表:
            [{symbol, date, signal_type, pred_return, pred_close,
              confidence, position_pct, reason}, ...]
        """
        existing_positions = existing_positions or {}
        signals = []

        for pred in predictions:
            symbol = pred.get("symbol", "")
            if self.paper_mode and symbol != self.paper_symbol:
                continue
            pred_return = pred.get("pred_return", 0.0)
            pred_close = pred.get("pred_close_t5", 0.0)
            confidence = pred.get("confidence", 0.0)

            # 跳过错误预测
            if pred.get("error"):
                signals.append(self._make_signal(
                    symbol, current_date, SignalType.CASH,
                    pred_return=0.0, pred_close=0.0, confidence=0.0,
                    position_pct=0.0, reason=f"预测错误: {pred['error']}"
                ))
                continue

            # 检查是否有现有持仓
            pos = existing_positions.get(symbol)

            if pos and pos.get("state") == PositionState.LONG:
                # ── 持仓中 ──
                hold_count = pos.get("hold_day_count", 0)
                if hold_count >= self.hold_days:
                    # 持仓到期 → 卖出
                    signals.append(self._make_signal(
                        symbol, current_date, SignalType.SELL,
                        pred_return=pred_return, pred_close=pred_close,
                        confidence=confidence, position_pct=0.0,
                        reason=f"持仓到期 ({self.hold_days}天)"
                    ))
                else:
                    # 继续持仓 → HOLD
                    signals.append(self._make_signal(
                        symbol, current_date, SignalType.HOLD,
                        pred_return=pred_return, pred_close=pred_close,
                        confidence=confidence, position_pct=pos.get("position_pct", 0.0),
                        reason=f"持仓中 (第{hold_count + 1}天/{self.hold_days}天)"
                    ))
            else:
                # ── 空仓状态 ──
                if pred_return > self.buy_threshold:
                    # 触发买入信号
                    signals.append(self._make_signal(
                        symbol, current_date, SignalType.BUY,
                        pred_return=pred_return, pred_close=pred_close,
                        confidence=confidence, position_pct=0.0,  # 仓位由 allocate 决定
                        reason=f"预测收益率 {pred_return:.2%} > {self.buy_threshold:.1%}"
                    ))
                else:
                    # 继续空仓
                    signals.append(self._make_signal(
                        symbol, current_date, SignalType.CASH,
                        pred_return=pred_return, pred_close=pred_close,
                        confidence=confidence, position_pct=0.0,
                        reason=f"预测收益率 {pred_return:.2%} ≤ {self.buy_threshold:.1%}"
                    ))

        # ── 仓位分配 ──
        signals = self._allocate_positions(signals, existing_positions)

        return signals

    def generate_from_storage(
        self,
        predictor,  # KronosPredictor
        current_date: str,
        existing_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        从 HDF5 存储加载数据 → 预测 → 生成信号（一站式）。

        Args:
            predictor: KronosPredictor 实例
            current_date: 信号生成日期
            existing_positions: 现有持仓
            symbols: 标的列表 (None = 全部)

        Returns:
            信号列表
        """
        predictions = predictor.predict_from_storage(symbols=symbols)
        return self.generate(predictions, current_date, existing_positions)

    # ═══════════════════════════════════════════════
    # 仓位分配
    # ═══════════════════════════════════════════════

    def _allocate_positions(
        self,
        signals: List[Dict[str, Any]],
        existing_positions: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        为 BUY 信号分配仓位。

        规则:
          1. 按模型置信度分配仓位
          2. 单标的上限 30%（max_single_position）
          3. 若仅 1 个标的触发 BUY → 最多 30% 仓位，其余现金
          4. 无 BUY 信号时 → 全部 CASH
        """
        if self.paper_mode:
            for signal in signals:
                if signal["signal_type"] == SignalType.BUY:
                    signal["position_pct"] = 1.0
                elif signal["signal_type"] == SignalType.CASH:
                    signal["position_pct"] = 0.0
            return signals

        # 统计 BUY 信号
        buy_symbols = [s for s in signals if s["signal_type"] == SignalType.BUY]
        n_buy = len(buy_symbols)

        if n_buy == 0:
            # 无买入信号，确保所有 CASH 信号仓位为 0
            for s in signals:
                if s["signal_type"] == SignalType.CASH:
                    s["position_pct"] = 0.0
            return signals

        weights: Dict[str, float] = {}
        if n_buy >= self.min_positions:
            confidence_sum = sum(max(float(item.get("confidence", 0.0) or 0.0), 0.01) for item in buy_symbols)
            for item in buy_symbols:
                confidence = max(float(item.get("confidence", 0.0) or 0.0), 0.01)
                weights[item["symbol"]] = min(confidence / confidence_sum, self.max_single_position)
            logger.debug(
                f"仓位分配: {n_buy} 个 BUY 信号, 按置信度加权, "
                f"单标的上限={self.max_single_position:.1%}"
            )
        else:
            for item in buy_symbols:
                weights[item["symbol"]] = self.max_single_position
            logger.debug(
                f"仓位分配: 仅 {n_buy} 个 BUY 信号, "
                f"仓位上限={self.max_single_position:.1%}"
            )

        # 分配
        for s in signals:
            if s["signal_type"] == SignalType.BUY:
                s["position_pct"] = weights[s["symbol"]]
            elif s["signal_type"] == SignalType.CASH:
                s["position_pct"] = 0.0
            # HOLD 保留已有仓位

        return signals

    # ═══════════════════════════════════════════════
    # 状态机辅助
    # ═══════════════════════════════════════════════

    def should_buy(self, pred_return: float) -> bool:
        """判断是否应该买入。"""
        return pred_return > self.buy_threshold

    def should_sell_due_to_expiry(self, hold_day_count: int) -> bool:
        """判断是否因持仓到期应卖出。"""
        return hold_day_count >= self.hold_days

    def get_next_state(
        self, current_state: PositionState, signal: SignalType
    ) -> PositionState:
        """
        状态转移。

        CASH + BUY  → LONG
        LONG + SELL → CASH
        LONG + HOLD → LONG
        CASH + CASH → CASH
        """
        if current_state == PositionState.CASH and signal == SignalType.BUY:
            return PositionState.LONG
        elif current_state == PositionState.LONG and signal == SignalType.SELL:
            return PositionState.CASH
        else:
            return current_state

    # ═══════════════════════════════════════════════
    # 批量信号
    # ═══════════════════════════════════════════════

    def generate_daily_signals(
        self,
        predictor,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        save_to_db: bool = False,
    ) -> pd.DataFrame:
        """
        逐日生成信号（用于回测）。

        在回测区间内，每天收盘后基于当天可用数据预测并生成信号。

        Args:
            predictor: KronosPredictor 实例
            start_date: 回测起始日
            end_date: 回测结束日
            symbols: 标的列表
            save_to_db: 是否保存到 SQLite

        Returns:
            DataFrame，列为:
              date, symbol, signal_type, pred_return, pred_close,
              confidence, position_pct, reason
        """
        from src.common.utils import get_trading_days

        trading_days = get_trading_days(start_date, end_date)
        logger.info(f"逐日信号生成: {start_date} ~ {end_date}, 共 {len(trading_days)} 个交易日")

        all_signals = []
        positions: Dict[str, Dict[str, Any]] = {}

        for i, day in enumerate(trading_days):
            day_str = format_date(day)

            # 每日预测 → 生成信号
            try:
                predictions = predictor.predict_from_storage(symbols=symbols)
                signals = self.generate(predictions, day_str, positions)

                # 更新持仓状态
                for sig in signals:
                    symbol = sig["symbol"]
                    if sig["signal_type"] == SignalType.BUY:
                        positions[symbol] = {
                            "entry_date": day_str,
                            "entry_price": sig.get("last_close", 0.0),
                            "quantity": 0,  # 回测时由 BacktestEngine 计算
                            "hold_day_count": 0,
                            "state": PositionState.LONG,
                            "position_pct": sig["position_pct"],
                        }
                    elif sig["signal_type"] == SignalType.HOLD:
                        if symbol in positions:
                            positions[symbol]["hold_day_count"] += 1
                    elif sig["signal_type"] == SignalType.SELL:
                        if symbol in positions:
                            del positions[symbol]

                all_signals.extend(signals)

            except Exception as e:
                logger.error(f"日期 {day_str} 信号生成失败: {e}")
                continue

            if (i + 1) % 50 == 0:
                logger.info(f"  信号生成进度: {i + 1}/{len(trading_days)}")

        df = pd.DataFrame(all_signals)
        logger.info(f"信号生成完成: {len(df)} 条信号, {df['symbol'].nunique()} 个标的")

        # 保存到数据库
        if save_to_db and len(df) > 0:
            self._save_signals_to_db(df)

        return df

    def _save_signals_to_db(self, df: pd.DataFrame):
        """将信号保存到 SQLite trading_signal 表。"""
        from src.data_layer.storage import StorageLayer

        storage = StorageLayer(self.config)
        rows = []
        for _, row in df.iterrows():
            rows.append((
                row["symbol"],
                row["date"],
                row["signal_type"],
                row.get("pred_return", 0.0),
                row.get("pred_close", 0.0),
                row.get("confidence", 0.0),
                row.get("position_pct", 0.0),
                row.get("reason", ""),
            ))

        storage.execute_many(
            """INSERT OR REPLACE INTO trading_signal
               (code, date, signal_type, pred_return, pred_close, confidence, position_pct, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.info(f"已保存 {len(rows)} 条信号到数据库")

    # ═══════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════

    def _make_signal(
        self,
        symbol: str,
        date: str,
        signal_type: SignalType,
        pred_return: float,
        pred_close: float,
        confidence: float,
        position_pct: float,
        reason: str = "",
    ) -> Dict[str, Any]:
        """构建单条信号字典。"""
        return {
            "symbol": symbol,
            "date": date,
            "signal_type": signal_type.value if isinstance(signal_type, SignalType) else signal_type,
            "pred_return": pred_return,
            "pred_close": pred_close,
            "confidence": confidence,
            "position_pct": position_pct,
            "reason": reason,
        }

    def to_dataframe(self, signals: List[Dict[str, Any]]) -> pd.DataFrame:
        """信号列表转为 DataFrame。"""
        return pd.DataFrame(signals)

    def get_buy_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取 BUY 信号。"""
        return [s for s in signals if s["signal_type"] == SignalType.BUY]

    def get_signal_summary(self, signals: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计各类型信号数量。"""
        summary = {t.value: 0 for t in SignalType}
        for s in signals:
            st = s["signal_type"]
            if st in summary:
                summary[st] += 1
        return summary

    # ═══════════════════════════════════════════════
    # 参数更新（供策略优化使用）
    # ═══════════════════════════════════════════════

    def update_params(self, **kwargs):
        """动态更新策略参数（用于 Walk-Forward 优化）。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                old = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"参数更新: {key} = {old} → {value}")
            else:
                logger.warning(f"未知参数: {key}")
