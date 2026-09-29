"""
Kronos 指数/ETF 择时策略系统 — 风控管理器

持仓期间动态风控：止损/止盈/熔断/仓位管理。

风控规则:
  1. 固定止损: 亏损 ≥ 8%（相对买入价）→ 提前平仓
  2. 移动止损: 从最高点回撤 ≥ 5% → 平仓（可选，trailing stop）
  3. 固定止盈: 盈利 ≥ 15%（相对买入价）→ 提前平仓
  4. 组合熔断: 总权益从峰值回撤 ≥ 25% → 强制平仓，暂停 10 个交易日
  5. 最大持仓数: 同时持有 ≤ 5 个标的
  6. 单标的仓位上限: 30%

所有资金计算使用 decimal.Decimal 高精度类型。
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


class RiskEventType(str, Enum):
    """风控事件类型"""
    STOP_LOSS = "STOP_LOSS"            # 固定止损触发
    TRAILING_STOP = "TRAILING_STOP"    # 移动止损触发
    TAKE_PROFIT = "TAKE_PROFIT"        # 固定止盈触发
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"  # 组合熔断
    POSITION_LIMIT = "POSITION_LIMIT"  # 仓位超限
    MAX_DRAWDOWN = "MAX_DRAWDOWN"      # 最大回撤触发


class CircuitBreakerState(str, Enum):
    """熔断状态"""
    NORMAL = "NORMAL"        # 正常交易
    TRIGGERED = "TRIGGERED"  # 已熔断，暂停交易
    COOLING = "COOLING"      # 冷却期（暂停10个交易日）


class RiskManager:
    """
    持仓期间动态风控管理器。

    使用 decimal.Decimal 进行所有资金计算（符合设计文档要求）。
    """

    def __init__(self, config):
        """
        Args:
            config: 系统配置 Config 实例
        """
        self.config = config
        self.enabled: bool = config.get("strategy", "risk_controls_enabled", default=True)

        # 止损/止盈阈值
        self.stop_loss: Decimal = Decimal(str(config.get("strategy", "stop_loss", default=-0.08)))
        self.take_profit: Decimal = Decimal(str(config.get("strategy", "take_profit", default=0.15)))

        # 移动止损
        self.trailing_stop_enabled: bool = config.get("strategy", "trailing_stop_enabled", default=False)
        self.trailing_stop_pct: Decimal = Decimal(str(config.get("strategy", "trailing_stop_pct", default=-0.05)))

        # 组合层面
        self.max_drawdown_limit: Decimal = Decimal(str(config.get("strategy", "max_drawdown_limit", default=-0.25)))
        self.circuit_breaker_cooldown: int = config.get("strategy", "circuit_breaker_cooldown", default=10)
        self.max_positions: int = config.get("strategy", "max_positions", default=5)
        self.max_single_position: Decimal = Decimal(str(config.get("strategy", "max_single_position", default=0.30)))

        # 熔断状态
        self._circuit_state = CircuitBreakerState.NORMAL
        self._circuit_trigger_date: Optional[str] = None
        self._cooldown_days_remaining: int = 0

        # 权益峰值跟踪
        self._peak_equity: Decimal = Decimal("0")
        self._peak_equity_date: Optional[str] = None

        # 风控事件记录
        self._events: List[Dict[str, Any]] = []

        logger.info(
            f"RiskManager 初始化: stop_loss={self.stop_loss}, "
            f"take_profit={self.take_profit}, max_drawdown={self.max_drawdown_limit}, "
            f"max_positions={self.max_positions}"
        )

    # ═══════════════════════════════════════════════
    # 持仓风险检查
    # ═══════════════════════════════════════════════

    def check_position(
        self,
        symbol: str,
        entry_price: Decimal,
        current_price: Decimal,
        highest_price: Decimal,
        hold_days: int,
        position_pct: Decimal,
        date: str,
    ) -> Tuple[bool, Optional[RiskEventType], str]:
        """
        检查单个持仓是否需要风控平仓。

        Args:
            symbol: 标的代码
            entry_price: 买入价
            current_price: 当前价
            highest_price: 持仓期间最高价
            hold_days: 已持仓天数
            position_pct: 仓位比例
            date: 当前日期

        Returns:
            (should_exit, event_type, reason)
            - should_exit: True 表示应强制平仓
            - event_type: 触发的风控事件类型（None 表示无需操作）
            - reason: 平仓原因描述
        """
        if not self.enabled or entry_price <= 0:
            return False, None, ""

        # 计算收益率
        pnl_pct = (current_price - entry_price) / entry_price

        # ── 1. 固定止损 ──
        if pnl_pct <= self.stop_loss:
            reason = f"固定止损触发: 亏损 {pnl_pct:.2%} ≤ {self.stop_loss}"
            self._record_event(symbol, date, RiskEventType.STOP_LOSS,
                              trigger_price=current_price, loss_pct=pnl_pct, description=reason)
            return True, RiskEventType.STOP_LOSS, reason

        # ── 2. 固定止盈 ──
        if pnl_pct >= self.take_profit:
            reason = f"固定止盈触发: 盈利 {pnl_pct:.2%} ≥ {self.take_profit}"
            self._record_event(symbol, date, RiskEventType.TAKE_PROFIT,
                              trigger_price=current_price, loss_pct=pnl_pct, description=reason)
            return True, RiskEventType.TAKE_PROFIT, reason

        # ── 3. 移动止损 ──
        if self.trailing_stop_enabled and highest_price > 0:
            drawdown_from_peak = (current_price - highest_price) / highest_price
            if drawdown_from_peak <= self.trailing_stop_pct:
                reason = f"移动止损触发: 从最高点回撤 {drawdown_from_peak:.2%} ≤ {self.trailing_stop_pct}"
                self._record_event(symbol, date, RiskEventType.TRAILING_STOP,
                                  trigger_price=current_price, loss_pct=pnl_pct, description=reason)
                return True, RiskEventType.TRAILING_STOP, reason

        return False, None, ""

    def check_portfolio(
        self,
        total_equity: Decimal,
        cash: Decimal,
        positions: Dict[str, Dict[str, Any]],
        date: str,
    ) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """
        组合层面风控检查。

        Args:
            total_equity: 总权益
            cash: 现金
            positions: 持仓字典 {symbol: {entry_price, quantity, current_price, ...}}
            date: 当前日期

        Returns:
            (circuit_breaker_triggered, reason, forced_exits)
            forced_exits: [{symbol, reason}, ...]
        """
        forced_exits = []
        if not self.enabled:
            return False, None, forced_exits

        # ── 1. 更新权益峰值 ──
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity
            self._peak_equity_date = date

        # ── 2. 熔断状态检查 ──
        if self._circuit_state == CircuitBreakerState.COOLING:
            self._cooldown_days_remaining -= 1
            if self._cooldown_days_remaining <= 0:
                logger.info(f"熔断冷却期结束，恢复正常交易 @ {date}")
                self._circuit_state = CircuitBreakerState.NORMAL
            else:
                # 仍在冷却期，强制平仓所有持仓
                for symbol in list(positions.keys()):
                    forced_exits.append({
                        "symbol": symbol,
                        "reason": f"熔断冷却期 (剩余 {self._cooldown_days_remaining} 天)",
                        "event_type": RiskEventType.CIRCUIT_BREAKER,
                    })
                return True, f"熔断冷却期", forced_exits

        # ── 3. 最大回撤熔断 ──
        if self._peak_equity > 0:
            drawdown = (total_equity - self._peak_equity) / self._peak_equity
            if drawdown <= self.max_drawdown_limit:
                reason = (
                    f"组合熔断触发: 总权益从峰值 {float(self._peak_equity):.2f} "
                    f"回撤 {drawdown:.2%} ≥ {self.max_drawdown_limit}%"
                )
                logger.warning(reason)
                self._record_event("PORTFOLIO", date, RiskEventType.CIRCUIT_BREAKER,
                                  loss_pct=drawdown, description=reason)

                # 触发熔断
                self._circuit_state = CircuitBreakerState.COOLING
                self._circuit_trigger_date = date
                self._cooldown_days_remaining = self.circuit_breaker_cooldown

                # 强制平仓所有持仓
                for symbol in list(positions.keys()):
                    forced_exits.append({
                        "symbol": symbol,
                        "reason": reason,
                        "event_type": RiskEventType.CIRCUIT_BREAKER,
                    })

                return True, reason, forced_exits

        # ── 4. 最大持仓数检查 ──
        if len(positions) > self.max_positions:
            logger.warning(f"持仓数 {len(positions)} 超过上限 {self.max_positions}")

        return False, None, forced_exits

    def check_new_position(
        self,
        symbol: str,
        position_pct: Decimal,
        current_positions: Dict[str, Dict[str, Any]],
        date: str,
    ) -> Tuple[bool, str]:
        """
        检查是否允许建立新仓位。

        Args:
            symbol: 标的代码
            position_pct: 建议仓位比例
            current_positions: 当前持仓
            date: 当前日期

        Returns:
            (allowed, reason)
        """
        if not self.enabled:
            return True, "论文复现模式：额外风控已关闭"

        # 熔断期间不允许开仓
        if self._circuit_state != CircuitBreakerState.NORMAL:
            return False, f"熔断状态: {self._circuit_state.value}"

        # 最大持仓数
        if len(current_positions) >= self.max_positions:
            return False, f"持仓数已达上限 {self.max_positions}"

        # 单标的仓位上限
        if position_pct > self.max_single_position:
            return False, f"仓位 {position_pct:.1%} 超过单标的上限 {self.max_single_position:.1%}"

        # 同标的不能重复开仓
        if symbol in current_positions:
            return False, f"标的 {symbol} 已持仓"

        return True, ""

    # ═══════════════════════════════════════════════
    # 资金管理
    # ═══════════════════════════════════════════════

    def allocate_capital(
        self,
        available_cash: Decimal,
        position_pct: Decimal,
        price: Decimal,
        min_shares: int = 100,
    ) -> Tuple[int, Decimal]:
        """
        计算实际买入数量（使用 decimal.Decimal 高精度）。

        Args:
            available_cash: 可用现金
            position_pct: 仓位比例
            price: 当前价格
            min_shares: 最小交易单位（ETF=100）

        Returns:
            (shares, allocated_amount)
        """
        if price <= 0 or available_cash <= 0:
            return 0, Decimal("0")

        # 可分配资金
        allocatable = available_cash * position_pct

        # 计算可买数量（向下取整到 min_shares 的倍数）
        raw_shares = int(allocatable / price)
        shares = (raw_shares // min_shares) * min_shares

        if shares < min_shares:
            return 0, Decimal("0")

        allocated = Decimal(str(shares)) * price
        return shares, allocated

    def kelly_criterion(
        self, win_rate: Decimal, avg_win: Decimal, avg_loss: Decimal
    ) -> Decimal:
        """
        凯利公式计算最优仓位比例。

        f = (bp - q) / b
        其中 b = avg_win / abs(avg_loss), p = win_rate, q = 1 - p

        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损（正数）

        Returns:
            最优仓位比例 [0, 1]
        """
        if avg_loss == 0:
            return Decimal("0")

        b = avg_win / abs(avg_loss)
        p = win_rate
        q = Decimal("1") - p

        f = (b * p - q) / b

        # 凯利公式通常建议使用半凯利（f/2）以降低波动
        f_half = f / Decimal("2")

        # 限制在 [0, 1]
        result = max(Decimal("0"), min(Decimal("1"), f_half))

        logger.debug(
            f"凯利公式: win_rate={win_rate:.2%}, b={b:.2f}, "
            f"full_kelly={f:.2%}, half_kelly={result:.2%}"
        )
        return result

    # ═══════════════════════════════════════════════
    # 熔断管理
    # ═══════════════════════════════════════════════

    @property
    def is_circuit_breaker_active(self) -> bool:
        """熔断是否激活（禁止交易）。"""
        return self._circuit_state != CircuitBreakerState.NORMAL

    @property
    def circuit_state(self) -> CircuitBreakerState:
        """当前熔断状态。"""
        return self._circuit_state

    def reset_circuit_breaker(self):
        """手动重置熔断（管理后台操作）。"""
        logger.warning("手动重置熔断状态")
        self._circuit_state = CircuitBreakerState.NORMAL
        self._cooldown_days_remaining = 0
        self._circuit_trigger_date = None

    # ═══════════════════════════════════════════════
    # 收益计算
    # ═══════════════════════════════════════════════

    def calculate_pnl(
        self,
        entry_price: Decimal,
        exit_price: Decimal,
        shares: int,
        commission_rate: Decimal,
        stamp_tax_rate: Decimal = Decimal("0"),
        slippage_rate: Decimal = Decimal("0"),
    ) -> Dict[str, Decimal]:
        """
        计算单笔交易的盈亏（高精度）。

        买入成本 = 成交金额 × (1 + 佣金 + 滑点)
        卖出收入 = 成交金额 × (1 - 佣金 - 印花税 - 滑点)

        Args:
            entry_price: 买入价
            exit_price: 卖出价
            shares: 股数
            commission_rate: 佣金率
            stamp_tax_rate: 印花税率（仅卖出）
            slippage_rate: 滑点率

        Returns:
            {gross_pnl, net_pnl, commission_total, stamp_tax_total, slippage_total, pnl_pct}
        """
        shares_d = Decimal(str(shares))

        # 买入成本
        buy_amount = entry_price * shares_d
        buy_commission = buy_amount * commission_rate
        buy_slippage = buy_amount * slippage_rate
        buy_total = buy_amount + buy_commission + buy_slippage

        # 卖出收入
        sell_amount = exit_price * shares_d
        sell_commission = sell_amount * commission_rate
        sell_stamp_tax = sell_amount * stamp_tax_rate
        sell_slippage = sell_amount * slippage_rate
        sell_total = sell_amount - sell_commission - sell_stamp_tax - sell_slippage

        # 净盈亏
        net_pnl = sell_total - buy_total
        gross_pnl = sell_amount - buy_amount

        # 总成本
        commission_total = buy_commission + sell_commission
        slippage_total = buy_slippage + sell_slippage

        # 收益率
        pnl_pct = net_pnl / buy_total if buy_total > 0 else Decimal("0")

        return {
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "commission_total": commission_total,
            "stamp_tax_total": sell_stamp_tax,
            "slippage_total": slippage_total,
            "buy_amount": buy_amount,
            "sell_amount": sell_amount,
            "pnl_pct": pnl_pct,
        }

    # ═══════════════════════════════════════════════
    # 事件记录
    # ═══════════════════════════════════════════════

    def _record_event(
        self,
        symbol: str,
        date: str,
        event_type: RiskEventType,
        trigger_price: Optional[Decimal] = None,
        exit_price: Optional[Decimal] = None,
        loss_amount: Optional[Decimal] = None,
        loss_pct: Optional[Decimal] = None,
        description: str = "",
    ):
        """记录风控事件。"""
        event = {
            "symbol": symbol,
            "date": date,
            "event_type": event_type.value,
            "trigger_price": float(trigger_price) if trigger_price else None,
            "exit_price": float(exit_price) if exit_price else None,
            "loss_amount": float(loss_amount) if loss_amount else None,
            "loss_pct": float(loss_pct) if loss_pct else None,
            "description": description,
            "recorded_at": datetime.now().isoformat(),
        }
        self._events.append(event)
        logger.info(f"[风控事件] {event_type.value} | {symbol} @ {date} | {description}")

    def get_events(self, since: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取风控事件列表。"""
        if since:
            return [e for e in self._events if e["date"] >= since]
        return list(self._events)

    def save_events_to_db(self, task_id: Optional[str] = None):
        """将风控事件保存到 SQLite。"""
        if not self._events:
            return

        from src.data_layer.storage import StorageLayer

        storage = StorageLayer(self.config)
        rows = []
        for e in self._events:
            rows.append((
                task_id,
                e["date"],
                e["symbol"],
                e["event_type"],
                e.get("trigger_price"),
                e.get("exit_price"),
                e.get("loss_amount"),
                e.get("loss_pct"),
                e.get("description", ""),
            ))

        storage.execute_many(
            """INSERT INTO risk_event
               (task_id, date, code, event_type, trigger_price, exit_price,
                loss_amount, loss_pct, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.info(f"已保存 {len(rows)} 条风控事件到数据库")

    def clear_events(self):
        """清空事件记录（新回测前调用）。"""
        self._events.clear()

    # ═══════════════════════════════════════════════
    # 状态导出
    # ═══════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """导出风控管理器当前状态。"""
        return {
            "circuit_state": self._circuit_state.value,
            "circuit_trigger_date": self._circuit_trigger_date,
            "cooldown_days_remaining": self._cooldown_days_remaining,
            "peak_equity": float(self._peak_equity),
            "peak_equity_date": self._peak_equity_date,
            "total_events": len(self._events),
            "config": {
                "stop_loss": float(self.stop_loss),
                "take_profit": float(self.take_profit),
                "trailing_stop_enabled": self.trailing_stop_enabled,
                "trailing_stop_pct": float(self.trailing_stop_pct),
                "max_drawdown_limit": float(self.max_drawdown_limit),
                "max_positions": self.max_positions,
                "max_single_position": float(self.max_single_position),
            },
        }

    def update_params(self, **kwargs):
        """动态更新风控参数。"""
        param_map = {
            "stop_loss": "stop_loss",
            "take_profit": "take_profit",
            "trailing_stop_pct": "trailing_stop_pct",
            "max_drawdown_limit": "max_drawdown_limit",
            "max_single_position": "max_single_position",
        }
        for key, attr in param_map.items():
            if key in kwargs:
                new_val = Decimal(str(kwargs[key]))
                old = getattr(self, attr)
                setattr(self, attr, new_val)
                logger.info(f"风控参数更新: {key} = {old} → {new_val}")

        if "max_positions" in kwargs:
            self.max_positions = int(kwargs["max_positions"])
        if "trailing_stop_enabled" in kwargs:
            self.trailing_stop_enabled = bool(kwargs["trailing_stop_enabled"])
