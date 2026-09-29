"""
Kronos 指数/ETF 择时策略系统 — 回测引擎

逐日模拟交易，含佣金/印花税/滑点成本，全程记录权益变化。

回测流程:
  每日循环:
    1. 更新持仓市值（当日收盘价 × 持仓数量）
    2. 检查止损/止盈条件 → 触发则生成 SELL 信号
    3. 检查持仓到期 → 到期则生成 SELL 信号
    4. 执行卖出订单 → 计算成本 → 更新现金
    5. 执行买入订单（如有 BUY 信号）→ 计算成本 → 更新持仓
    6. 记录当日权益（现金 + 持仓市值）
    7. 计算当日收益率

所有资金计算使用 decimal.Decimal 高精度类型（32位精度）。
"""

from __future__ import annotations

import json
import uuid
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.logger import get_logger
from src.common.utils import format_date, get_trading_days, set_seed
from src.strategy_layer.signal_generator import SignalGenerator, SignalType, PositionState
from src.strategy_layer.risk_manager import RiskManager, RiskEventType, CircuitBreakerState
from src.data_layer.a_share_rules import order_is_tradable

logger = get_logger(__name__)


class BacktestStatus(str, Enum):
    """回测任务状态"""
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BacktestEngine:
    """
    回测引擎 — 逐日模拟交易。

    职责:
      - 按交易日历逐日推进
      - 管理持仓（开仓/平仓/持仓到期）
      - 计算交易成本（佣金+印花税+滑点）
      - 调用 RiskManager 检查止损/止盈/熔断
      - 记录每日权益（现金+持仓市值）
      - 与基准（买入持有）对比

    Usage:
        engine = BacktestEngine(config)
        result = engine.run(
            predictions_df,        # 逐日预测数据
            price_data,            # {symbol: DataFrame}
            start_date="2021-07-01",
            end_date="2026-06-30",
        )
    """

    def __init__(self, config: Config):
        """
        Args:
            config: 系统配置
        """
        self.config = config

        # 回测参数
        self.initial_capital: Decimal = Decimal(
            str(config.get("backtest", "default_capital", default=1000000.0))
        )
        self.commission_rate: Decimal = Decimal(
            str(config.get("backtest", "commission_rate", default=0.0003))
        )
        self.stamp_tax_rate: Decimal = Decimal(
            str(config.get("backtest", "stamp_tax_rate", default=0.001))
        )
        self.slippage_rate: Decimal = Decimal(
            str(config.get("backtest", "slippage_rate", default=0.001))
        )
        self.benchmark_index: str = config.get("backtest", "benchmark_index", default="000852")
        self.risk_free_rate: Decimal = Decimal(
            str(config.get("backtest", "risk_free_rate", default=0.025))
        )
        self.min_shares: int = 100  # ETF 最小交易单位
        self.exit_at_close: bool = bool(
            config.get("strategy", "exit_at_close", default=True)
        )

        # 子模块
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)

        # 预测器（延迟加载，用于默认 run() 的逐日滚动预测）
        self._predictor = None
        self._predictor_loaded = False

        # 任务管理
        self._task_id: Optional[str] = None
        self._status: BacktestStatus = BacktestStatus.ACCEPTED
        self._progress: float = 0.0

        # 运行时状态
        self._cash: Decimal = self.initial_capital
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._daily_equity: List[Dict[str, Any]] = []
        self._trades: List[Dict[str, Any]] = []
        self._benchmark_equity: List[Dict[str, Any]] = []

        # 回测结果缓存
        self._result: Optional[Dict[str, Any]] = None

        logger.info(
            f"BacktestEngine 初始化: capital={self.initial_capital}, "
            f"commission={self.commission_rate}, stamp_tax={self.stamp_tax_rate}, "
            f"slippage={self.slippage_rate}"
        )

    # ═══════════════════════════════════════════════
    # 主回测循环
    # ═══════════════════════════════════════════════

    def run(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        执行回测。

        核心算法:
          在回测区间内逐日推进，每天:
            1. 基于前一日收盘数据生成信号（模拟T-1收盘后预测）
            2. 当日开盘执行订单
            3. 收盘后更新持仓市值 & 检查风控

        Args:
            symbol_dfs: {symbol: DataFrame} — 每个标的完整的 OHLCV 数据
            start_date: 回测起始日
            end_date: 回测结束日
            initial_capital: 初始资金（None = 使用配置默认值）
            progress_callback: 进度回调 (0.0 ~ 1.0)

        Returns:
            {
                task_id, status, start_date, end_date,
                initial_capital, final_equity,
                10项绩效指标,
                daily_equity: [{date, cash, position_value, total_equity, ...}],
                trades: [{date, symbol, direction, price, quantity, ...}],
                benchmark_equity: [{date, value}],
                risk_events: [...],
                config: {...},
            }
        """
        # ── 初始化 ──
        self._task_id = str(uuid.uuid4())[:8]
        self._status = BacktestStatus.RUNNING
        self._progress = 0.0

        if initial_capital is not None:
            self.initial_capital = Decimal(str(initial_capital))

        self._reset_state()

        # 获取交易日历
        trading_days = get_trading_days(start_date, end_date)
        n_days = len(trading_days)
        logger.info(f"回测开始 [{self._task_id}]: {start_date} ~ {end_date}, {n_days} 个交易日")

        if n_days == 0:
            self._status = BacktestStatus.FAILED
            raise ValueError(f"回测区间无交易日: {start_date} ~ {end_date}")

        # 预检查所有标的都有数据
        for sym in list(symbol_dfs.keys()):
            df = symbol_dfs[sym]
            if df is None or len(df) == 0:
                logger.warning(f"标的 {sym} 无数据，从回测中移除")
                del symbol_dfs[sym]

        if not symbol_dfs:
            self._status = BacktestStatus.FAILED
            raise ValueError("所有标的均无可用数据")

        # ── 构建基准权益曲线 ──
        self._build_benchmark(trading_days, symbol_dfs)

        # ── 逐日循环 ──
        start_time = time.time()

        for i, day in enumerate(trading_days):
            day_str = format_date(day)

            try:
                self._process_day(day_str, symbol_dfs, trading_days, i)
            except Exception as e:
                logger.error(f"日期 {day_str} 处理异常: {e}", exc_info=True)
                # 继续下一个交易日（记录异常但不中断回测）

            # 进度更新
            self._progress = (i + 1) / n_days
            if progress_callback:
                progress_callback(self._progress)

            if (i + 1) % 100 == 0:
                total_eq = self._daily_equity[-1]["total_equity"] if self._daily_equity else 0
                logger.info(
                    f"  回测进度: {i + 1}/{n_days} ({self._progress:.0%}) | "
                    f"权益: {total_eq:,.2f} | 持仓: {len(self._positions)}"
                )

        # ── 完成后计算绩效指标 ──
        elapsed = time.time() - start_time
        self._status = BacktestStatus.COMPLETED

        result = self._build_result(start_date, end_date, elapsed)
        self._result = result

        logger.info(
            f"回测完成 [{self._task_id}]: 最终权益={result['final_equity']:,.2f}, "
            f"累计收益={result['cumulative_return']:.2%}, "
            f"耗时={elapsed:.1f}s"
        )

        return result

    def _process_day(
        self,
        date: str,
        symbol_dfs: Dict[str, pd.DataFrame],
        trading_days: pd.DatetimeIndex,
        day_index: int,
    ):
        """
        处理单个交易日。

        步骤:
          1. 获取当日数据
          2. 检查持仓风控（止损/止盈）
          3. 持仓到期平仓
          4. 生成交易信号
          5. 执行卖出订单
          6. 执行买入订单
          7. 更新持仓市值
          8. 记录每日权益
        """
        # ── 步骤 1: 获取当日数据 ──
        day_prices = {}
        for sym, df in symbol_dfs.items():
            try:
                row = df.loc[date]
                day_prices[sym] = {
                    "open": Decimal(str(row["open"])),
                    "high": Decimal(str(row["high"])),
                    "low": Decimal(str(row["low"])),
                    "close": Decimal(str(row["close"])),
                    "volume": Decimal(str(row.get("volume", 0))),
                    "limit_up": bool(row.get("limit_up", False)),
                    "limit_down": bool(row.get("limit_down", False)),
                }
            except (KeyError, TypeError):
                # 该标的当日无数据（停牌等），跳过
                continue

        # ── 步骤 2-3: 风控检查 & 到期检查 ──
        forced_exits = []

        for sym in list(self._positions.keys()):
            pos = self._positions[sym]
            prices = day_prices.get(sym)
            if prices is None:
                continue

            # 增加持仓天数
            pos["hold_day_count"] += 1

            # 更新持仓期间最高价
            if prices["close"] > pos.get("highest_price", Decimal("0")):
                pos["highest_price"] = prices["close"]

            # 风控检查
            should_exit, event_type, reason = self.risk_manager.check_position(
                symbol=sym,
                entry_price=pos["entry_price"],
                current_price=prices["close"],
                highest_price=pos.get("highest_price", pos["entry_price"]),
                hold_days=pos["hold_day_count"],
                position_pct=pos.get("position_pct", Decimal("0")),
                date=date,
            )

            if should_exit:
                forced_exits.append({"symbol": sym, "reason": reason, "event_type": event_type})
                continue

            # 持仓到期检查
            if pos["hold_day_count"] >= self.signal_generator.hold_days:
                forced_exits.append({
                    "symbol": sym,
                    "reason": f"持仓到期 ({pos['hold_day_count']}天)",
                    "event_type": None,
                })

        # ── 步骤 4: 生成信号 ──
        # 构建伪预测（基于历史数据进行预测模拟）
        # 实际回测中，predictions 应来自 KronosPredictor 或历史预测缓存
        # 信号必须来自上一交易日收盘可见的信息，订单才可在当天开盘成交。
        if day_index == 0:
            predictions = []
        else:
            signal_as_of = format_date(trading_days[day_index - 1])
            predictions = self._simulate_predictions(day_prices, signal_as_of, symbol_dfs)

        raw_signals = self.signal_generator.generate(
            predictions=predictions,
            current_date=date,
            existing_positions=self._positions,
        )

        # ── 步骤 5: 执行卖出 ──
        sell_candidates = [s for s in raw_signals if s["signal_type"] == SignalType.SELL]
        # 加上风控强制平仓
        for fe in forced_exits:
            sym = fe["symbol"]
            if sym in self._positions:
                sell_candidates.append({
                    "symbol": sym,
                    "date": date,
                    "signal_type": SignalType.SELL,
                    "pred_return": 0.0,
                    "pred_close": 0.0,
                    "confidence": 1.0,
                    "position_pct": 0.0,
                    "reason": fe["reason"],
                })

        # 去重
        seen_sell = set()
        unique_sells = []
        for s in sell_candidates:
            if s["symbol"] not in seen_sell:
                unique_sells.append(s)
                seen_sell.add(s["symbol"])

        deferred_sells = []
        for sell_sig in unique_sells:
            if self.exit_at_close and "持仓到期" in str(sell_sig.get("reason", "")):
                deferred_sells.append(sell_sig)
            else:
                self._execute_sell(sell_sig, day_prices, date)

        # ── 组合层面风控检查 ──
        total_equity = self._calculate_total_equity(day_prices)
        cb_triggered, cb_reason, cb_exits = self.risk_manager.check_portfolio(
            total_equity=total_equity,
            cash=self._cash,
            positions=self._positions,
            date=date,
        )
        if cb_triggered:
            for cb_exit in cb_exits:
                sym = cb_exit["symbol"]
                if sym in self._positions:
                    sell_sig = {
                        "symbol": sym,
                        "date": date,
                        "signal_type": SignalType.SELL,
                        "pred_return": 0.0,
                        "pred_close": 0.0,
                        "confidence": 1.0,
                        "position_pct": 0.0,
                        "reason": cb_exit["reason"],
                    }
                    self._execute_sell(sell_sig, day_prices, date)

        # ── 步骤 6: 执行买入 ──
        buy_signals = [s for s in raw_signals if s["signal_type"] == SignalType.BUY]

        for buy_sig in buy_signals:
            sym = buy_sig["symbol"]
            prices = day_prices.get(sym)
            if prices is None:
                continue

            # 检查是否允许开仓
            allowed, reason = self.risk_manager.check_new_position(
                symbol=sym,
                position_pct=Decimal(str(buy_sig.get("position_pct", 0.3))),
                current_positions=self._positions,
                date=date,
            )
            if not allowed:
                logger.debug(f"  {sym} 不允许开仓: {reason}")
                continue

            self._execute_buy(buy_sig, day_prices, date)

        # 论文口径：固定持有期到期后，在第五个后续交易日收盘退出。
        for sell_sig in deferred_sells:
            self._execute_sell(sell_sig, day_prices, date, price_field="close")

        # ── 步骤 7-8: 更新市值 & 记录权益 ──
        total_equity = self._calculate_total_equity(day_prices)
        position_value = total_equity - self._cash

        # 计算日收益率
        prev_equity = (
            Decimal(str(self._daily_equity[-1]["total_equity"]))
            if self._daily_equity
            else self.initial_capital
        )
        daily_return = (
            (total_equity - prev_equity) / prev_equity
            if prev_equity > 0
            else Decimal("0")
        )

        # 计算回撤
        peak = self._get_peak_equity()
        drawdown = (
            (total_equity - peak) / peak if peak > 0 else Decimal("0")
        )

        # 基准权益
        benchmark_val = (
            Decimal(str(self._benchmark_equity[day_index]["value"]))
            if day_index < len(self._benchmark_equity)
            else Decimal("0")
        )

        self._daily_equity.append({
            "date": date,
            "cash": float(self._cash),
            "position_value": float(position_value),
            "total_equity": float(total_equity),
            "daily_return": float(daily_return),
            "cumulative_return": float((total_equity - self.initial_capital) / self.initial_capital),
            "drawdown": float(drawdown),
            "benchmark_value": float(benchmark_val),
            "num_positions": len(self._positions),
        })

    # ═══════════════════════════════════════════════
    # 订单执行
    # ═══════════════════════════════════════════════

    def _execute_buy(
        self,
        signal: Dict[str, Any],
        day_prices: Dict[str, Dict[str, Decimal]],
        date: str,
    ):
        """执行买入订单。"""
        sym = signal["symbol"]
        prices = day_prices.get(sym)
        if prices is None:
            return

        # 使用开盘价执行
        price = prices["open"]
        if price <= 0:
            return
        if not order_is_tradable(
            "BUY", prices.get("limit_up", False), prices.get("limit_down", False)
        ):
            logger.debug("  BUY %s blocked by limit-up on %s", sym, date)
            return

        position_pct = Decimal(str(signal.get("position_pct", 0.3)))

        # 计算买入数量
        shares, allocated = self.risk_manager.allocate_capital(
            available_cash=self._cash,
            position_pct=position_pct,
            price=price,
            min_shares=self.min_shares,
        )

        if shares == 0:
            logger.debug(f"  {sym} 资金不足: cash={self._cash}, price={price}")
            return

        # 计算成本
        amount = Decimal(str(shares)) * price
        commission = amount * self.commission_rate
        slippage = amount * self.slippage_rate
        total_cost = amount + commission + slippage

        if total_cost > self._cash:
            # 资金不够，减小买入量
            shares = int(shares * 0.95)  # 留 5% 缓冲
            shares = (shares // self.min_shares) * self.min_shares
            if shares < self.min_shares:
                return
            amount = Decimal(str(shares)) * price
            commission = amount * self.commission_rate
            slippage = amount * self.slippage_rate
            total_cost = amount + commission + slippage

            if total_cost > self._cash:
                return

        # 扣款
        self._cash -= total_cost

        # 记录持仓
        self._positions[sym] = {
            "entry_date": date,
            "entry_price": price,
            "highest_price": price,
            "shares": shares,
            "hold_day_count": 0,
            "state": PositionState.LONG,
            "position_pct": position_pct,
            "entry_cost": total_cost,
            "entry_amount": amount,
        }

        # 记录交易
        self._trades.append({
            "date": date,
            "symbol": sym,
            "direction": "BUY",
            "price": float(price),
            "quantity": shares,
            "amount": float(amount),
            "commission": float(commission),
            "stamp_tax": 0.0,
            "slippage": float(slippage),
            "net_amount": float(-total_cost),
            "reason": signal.get("reason", ""),
        })

        logger.debug(
            f"  BUY  {sym}: {shares}股 @ {price}, "
            f"金额={amount:.2f}, 成本={total_cost:.2f}, 剩余现金={self._cash:.2f}"
        )

    def _execute_sell(
        self,
        signal: Dict[str, Any],
        day_prices: Dict[str, Dict[str, Decimal]],
        date: str,
        price_field: str = "open",
    ):
        """执行卖出订单。"""
        sym = signal["symbol"]
        pos = self._positions.get(sym)
        if pos is None:
            return

        prices = day_prices.get(sym)
        if prices is None:
            # 标的无当日价格，使用最后已知价格
            price = pos["entry_price"]
        else:
            if not order_is_tradable(
                "SELL", prices.get("limit_up", False), prices.get("limit_down", False)
            ):
                logger.debug("  SELL %s blocked by limit-down on %s", sym, date)
                return
            price = prices[price_field]

        shares = pos["shares"]
        amount = Decimal(str(shares)) * price
        commission = amount * self.commission_rate
        # 沪深 ETF 不征收股票交易印花税；个股卖出仍按配置计税。
        applied_stamp_tax_rate = (
            Decimal("0") if sym.startswith(("5", "1")) else self.stamp_tax_rate
        )
        stamp_tax = amount * applied_stamp_tax_rate
        slippage = amount * self.slippage_rate
        net_proceeds = amount - commission - stamp_tax - slippage

        # 入账
        self._cash += net_proceeds

        # 计算盈亏
        pnl_result = self.risk_manager.calculate_pnl(
            entry_price=pos["entry_price"],
            exit_price=price,
            shares=shares,
            commission_rate=self.commission_rate,
            stamp_tax_rate=applied_stamp_tax_rate,
            slippage_rate=self.slippage_rate,
        )

        # 记录交易
        self._trades.append({
            "date": date,
            "symbol": sym,
            "direction": "SELL",
            "price": float(price),
            "quantity": shares,
            "amount": float(amount),
            "commission": float(commission),
            "stamp_tax": float(stamp_tax),
            "slippage": float(slippage),
            "net_amount": float(net_proceeds),
            "pnl": float(pnl_result["net_pnl"]),
            "pnl_pct": float(pnl_result["pnl_pct"]),
            "hold_days": pos.get("hold_day_count", 0),
            "reason": signal.get("reason", ""),
        })

        logger.debug(
            f"  SELL {sym}: {shares}股 @ {price}, "
            f"收入={net_proceeds:.2f}, 盈亏={pnl_result['net_pnl']:.2f} "
            f"({pnl_result['pnl_pct']:.2%}), reason={signal.get('reason', '')}"
        )

        # 移除持仓
        del self._positions[sym]

    # ═══════════════════════════════════════════════
    # 预测模拟（回测用）
    # ═══════════════════════════════════════════════

    def _get_predictor(self):
        """延迟加载 KronosPredictor（默认使用 fallback 模型）。

        加载失败（如无 checkpoint）时返回 None，调用方降级到
        仅使用历史数据的动量启发式，绝不读取未来数据。
        """
        if self._predictor_loaded:
            return self._predictor
        self._predictor_loaded = True
        try:
            from src.model_layer.model_factory import create_model
            from src.model_layer.kronos_predictor import KronosPredictor
            model, path_label = create_model(self.config)  # 返回 (model, "kronos"|"fallback")
            self._predictor = KronosPredictor(model, self.config)
            logger.info(f"回测预测器已加载（{path_label} 滚动窗口预测，无未来数据泄露）")
        except Exception as e:
            logger.warning(f"回测预测器加载失败，降级为历史动量启发式: {e}")
            self._predictor = None
        return self._predictor

    def _simulate_predictions(
        self,
        day_prices: Dict[str, Dict[str, Decimal]],
        date: str,
        symbol_dfs: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        生成回测当日的滚动预测（用于信号生成）。

        **无未来数据泄露**：每个标的仅使用截至 ``date``（含）当日的历史
        OHLCV 作为输入窗口——即 ``df.loc[:date]`` 的最后 seq_len 行，
        绝不访问 ``date`` 之后的任何行。

        优先使用 KronosPredictor（当前为 fallback 模型）做滚动窗口预测；
        模型不可用时，降级为仅基于历史数据的动量启发式。

        若需使用预计算的离线预测缓存，请改用 run_with_predictions()。
        """
        predictor = self._get_predictor()
        seq_len = self.config.get("model", "training", "seq_len", default=60)

        predictions = []
        for sym, prices in day_prices.items():
            close = float(prices["close"])
            df = symbol_dfs.get(sym)
            pred_return = 0.0
            pred_close = close
            confidence = 0.5

            if df is not None:
                # 仅取截至当日（含）的历史数据 —— 杜绝未来信息
                past_df = df.loc[:date]
                if len(past_df) >= seq_len:
                    if predictor is not None:
                        try:
                            pred = predictor.predict_single(past_df)
                            pred_close = float(pred["pred_close_t5"])
                            pred_return = float(pred["pred_return"])
                            confidence = float(pred.get("confidence", 0.6))
                        except Exception as e:
                            logger.debug(f"{sym}@{date} 模型预测失败，降级动量: {e}")
                            pred_close, pred_return = self._momentum_fallback(past_df, close)
                    else:
                        pred_close, pred_return = self._momentum_fallback(past_df, close)

            predictions.append({
                "symbol": sym,
                "pred_close_t5": pred_close,
                "pred_return": pred_return,
                "last_close": close,
                "pred_date": date,
                "confidence": confidence,
            })

        return predictions

    @staticmethod
    def _momentum_fallback(past_df: pd.DataFrame, close: float) -> Tuple[float, float]:
        """仅用历史数据的动量启发式（无未来泄露）。

        用过去第5日收盘价计算动量，做均值回归缩放。
        """
        if len(past_df) > 5:
            past_close = float(past_df["close"].iloc[-6])
            momentum = (close / past_close - 1.0) if past_close > 0 else 0.0
            pred_return = momentum * 0.3
            return close * (1.0 + pred_return), pred_return
        return close, 0.0

    # ═══════════════════════════════════════════════

    def run_with_predictions(
        self,
        predictions_df: pd.DataFrame,
        symbol_dfs: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        使用预计算的预测数据执行回测（推荐方式）。

        predictions_df 应包含列:
          date, symbol, pred_return, pred_close_t5, confidence

        每行对应某个日期对某个标的的预测。
        预测应该来自 KronosPredictor 的滚动窗口推理，
        确保训练集日期 < 预测日期（无未来信息泄露）。
        """
        # 将 predictions_df 按日期索引
        pred_by_date = {}
        for _, row in predictions_df.iterrows():
            d = format_date(row["date"])
            if d not in pred_by_date:
                pred_by_date[d] = []
            pred_by_date[d].append({
                "symbol": row["symbol"],
                "pred_close_t5": row.get("pred_close_t5", 0.0),
                "pred_return": row.get("pred_return", 0.0),
                "last_close": row.get("last_close", 0.0),
                "confidence": row.get("confidence", 0.6),
            })

        # 重写 _simulate_predictions 使用真实预测
        original_simulate = self._simulate_predictions
        self._simulate_predictions = lambda dp, d, sdf: pred_by_date.get(d, [])

        try:
            result = self.run(
                symbol_dfs=symbol_dfs,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                progress_callback=progress_callback,
            )
        finally:
            self._simulate_predictions = original_simulate

        return result

    # ═══════════════════════════════════════════════
    # 基准构建
    # ═══════════════════════════════════════════════

    def _build_benchmark(
        self,
        trading_days: pd.DatetimeIndex,
        symbol_dfs: Dict[str, pd.DataFrame],
    ):
        """
        构建基准（中证1000买入持有）权益曲线。
        """
        benchmark_df = symbol_dfs.get(self.benchmark_index)
        self._benchmark_equity = []

        if benchmark_df is None or len(benchmark_df) == 0:
            logger.warning(f"基准 {self.benchmark_index} 无数据，使用等权组合作为基准")
            # 使用所有标的等权组合作为 fallback 基准
            first_day = format_date(trading_days[0])
            base_prices = {}
            for sym, df in symbol_dfs.items():
                try:
                    base_prices[sym] = float(df.loc[first_day, "close"])
                except (KeyError, TypeError):
                    pass
            if not base_prices:
                # 没有可用基准，填充 1.0
                for day in trading_days:
                    self._benchmark_equity.append({
                        "date": format_date(day),
                        "value": float(self.initial_capital),
                    })
                return

            weights = {s: 1.0 / len(base_prices) for s in base_prices}
            for day in trading_days:
                day_str = format_date(day)
                val = 0.0
                for sym, w in weights.items():
                    try:
                        val += w * float(symbol_dfs[sym].loc[day_str, "close"]) / base_prices.get(sym, 1.0)
                    except (KeyError, TypeError):
                        val += w
                self._benchmark_equity.append({
                    "date": day_str,
                    "value": val * float(self.initial_capital),
                })
            return

        # 基准首日价格
        first_day = format_date(trading_days[0])
        try:
            base_price = float(benchmark_df.loc[first_day, "close"])
        except (KeyError, TypeError):
            # 找第一个有数据的交易日
            for d in trading_days:
                try:
                    base_price = float(benchmark_df.loc[format_date(d), "close"])
                    break
                except (KeyError, TypeError):
                    continue
            else:
                base_price = 1.0

        if base_price <= 0:
            base_price = 1.0

        for day in trading_days:
            day_str = format_date(day)
            try:
                close = float(benchmark_df.loc[day_str, "close"])
                bm_val = close / base_price * float(self.initial_capital)
            except (KeyError, TypeError):
                # 沿用前一日值
                bm_val = (
                    self._benchmark_equity[-1]["value"]
                    if self._benchmark_equity
                    else float(self.initial_capital)
                )

            self._benchmark_equity.append({"date": day_str, "value": bm_val})

    # ═══════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════

    def _reset_state(self):
        """重置回测状态。"""
        self._cash = self.initial_capital
        self._positions = {}
        self._daily_equity = []
        self._trades = []
        self._benchmark_equity = []
        self.risk_manager.clear_events()
        self.risk_manager.reset_circuit_breaker()
        self._peak_equity_val = self.initial_capital

    def _calculate_total_equity(
        self, day_prices: Dict[str, Dict[str, Decimal]]
    ) -> Decimal:
        """计算总权益 = 现金 + 持仓市值。"""
        position_value = Decimal("0")
        for sym, pos in self._positions.items():
            prices = day_prices.get(sym)
            if prices is not None:
                position_value += prices["close"] * Decimal(str(pos["shares"]))
            else:
                # 沿用买入价
                position_value += pos["entry_price"] * Decimal(str(pos["shares"]))

        return self._cash + position_value

    def _get_peak_equity(self) -> Decimal:
        """获取历史最高权益。"""
        peak = self.initial_capital
        for record in self._daily_equity:
            eq = Decimal(str(record["total_equity"]))
            if eq > peak:
                peak = eq
        return peak

    # ═══════════════════════════════════════════════
    # 结果构建
    # ═══════════════════════════════════════════════

    def _build_result(
        self, start_date: str, end_date: str, elapsed_seconds: float
    ) -> Dict[str, Any]:
        """构建回测结果，包含完整绩效指标。"""
        from src.strategy_layer.performance_metrics import PerformanceMetrics

        # 计算绩效指标
        metrics = PerformanceMetrics(self.config)
        perf = metrics.compute(
            daily_equity=self._daily_equity,
            trades=self._trades,
            initial_capital=float(self.initial_capital),
            benchmark_equity=self._benchmark_equity,
        )

        result = {
            "task_id": self._task_id,
            "status": self._status.value,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(self.initial_capital),
            "final_equity": perf.get("final_equity", 0.0),
            "elapsed_seconds": elapsed_seconds,

            # 10 项绩效指标
            "cumulative_return": perf.get("cumulative_return", 0.0),
            "annual_return": perf.get("annual_return", 0.0),
            "max_drawdown": perf.get("max_drawdown", 0.0),
            "win_rate": perf.get("win_rate", 0.0),
            "profit_loss_ratio": perf.get("profit_loss_ratio"),
            "sharpe_ratio": perf.get("sharpe_ratio"),
            "calmar_ratio": perf.get("calmar_ratio"),
            "excess_return": perf.get("excess_return", 0.0),
            "max_consecutive_wins": perf.get("max_consecutive_wins", 0),
            "max_consecutive_losses": perf.get("max_consecutive_losses", 0),
            "avg_hold_days": perf.get("avg_hold_days", 0.0),

            # 明细
            "total_trades": len(self._trades),
            "daily_equity": self._daily_equity,
            "trades": self._trades,
            "benchmark_equity": self._benchmark_equity,
            "risk_events": self.risk_manager.get_events(),

            # 月度/年度汇总
            "monthly_returns": perf.get("monthly_returns", []),
            "yearly_returns": perf.get("yearly_returns", []),

            # 回撤区间
            "max_drawdown_period": perf.get("max_drawdown_period", {}),

            # 配置
            "config": {
                "commission_rate": float(self.commission_rate),
                "stamp_tax_rate": float(self.stamp_tax_rate),
                "slippage_rate": float(self.slippage_rate),
                "benchmark_index": self.benchmark_index,
                "buy_threshold": self.signal_generator.buy_threshold,
                "hold_days": self.signal_generator.hold_days,
                "stop_loss": float(self.risk_manager.stop_loss),
                "take_profit": float(self.risk_manager.take_profit),
                "max_single_position": float(self.risk_manager.max_single_position),
                "risk_controls_enabled": self.risk_manager.enabled,
            },
        }

        return result

    # ═══════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════

    def save_to_db(self):
        """将回测结果保存到 SQLite。"""
        if self._result is None:
            raise RuntimeError("无回测结果，请先执行 run()")

        from src.data_layer.storage import StorageLayer

        storage = StorageLayer(self.config)

        # 1. backtest_summary
        r = self._result
        storage.execute_write(
            """INSERT OR REPLACE INTO backtest_summary
               (task_id, start_date, end_date, initial_capital, final_equity,
                cumulative_return, annual_return, max_drawdown, sharpe_ratio,
                calmar_ratio, win_rate, profit_loss_ratio, excess_return,
                max_consecutive_wins, max_consecutive_losses, avg_hold_days,
                total_trades, params_json, status, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                r["task_id"], r["start_date"], r["end_date"],
                r["initial_capital"], r["final_equity"],
                r["cumulative_return"], r["annual_return"], r["max_drawdown"],
                r["sharpe_ratio"], r["calmar_ratio"], r["win_rate"],
                r.get("profit_loss_ratio"), r["excess_return"],
                r["max_consecutive_wins"], r["max_consecutive_losses"],
                r["avg_hold_days"], r["total_trades"],
                json.dumps(r["config"], ensure_ascii=False),
                r["status"],
            ),
        )

        # 2. daily_equity
        if self._daily_equity:
            eq_rows = [
                (
                    r["task_id"], e["date"], e["cash"], e["position_value"],
                    e["total_equity"], e.get("daily_return", 0.0),
                    e.get("cumulative_return", 0.0), e.get("drawdown", 0.0),
                    e.get("benchmark_value", 0.0),
                )
                for e in self._daily_equity
            ]
            storage.execute_many(
                """INSERT INTO daily_equity
                   (task_id, date, cash, position_value, total_equity,
                    daily_return, cumulative_return, drawdown, benchmark_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                eq_rows,
            )

        # 3. sim_trade
        if self._trades:
            trade_rows = [
                (
                    r["task_id"], t["date"], t["symbol"], t["direction"],
                    t["price"], t["quantity"], t["amount"],
                    t["commission"], t["stamp_tax"], t["slippage"],
                    t["net_amount"], t.get("reason", ""),
                )
                for t in self._trades
            ]
            storage.execute_many(
                """INSERT INTO sim_trade
                   (task_id, trade_date, code, direction, price, quantity,
                    amount, commission, stamp_tax, slippage, net_amount, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                trade_rows,
            )

        # 4. risk_event
        self.risk_manager.save_events_to_db(task_id=r["task_id"])

        logger.info(f"回测结果已保存到数据库 [{r['task_id']}]")

    def save_to_json(self, output_path: Optional[str] = None) -> str:
        """将回测结果保存为 JSON 文件。"""
        if self._result is None:
            raise RuntimeError("无回测结果，请先执行 run()")

        if output_path is None:
            output_dir = Path(self.config.get("data", "outputs_dir", default="data/outputs"))
            output_dir = output_dir / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(
                output_dir / f"backtest_{self._task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        # 去除不可序列化的对象
        output = dict(self._result)
        output["generated_at"] = datetime.now().isoformat()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"回测结果已保存: {output_path}")
        return str(output_path)

    # ═══════════════════════════════════════════════
    # 属性/状态查询
    # ═══════════════════════════════════════════════

    @property
    def task_id(self) -> Optional[str]:
        return self._task_id

    @property
    def status(self) -> BacktestStatus:
        return self._status

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._result

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线 DataFrame。"""
        return pd.DataFrame(self._daily_equity)

    def get_trade_history(self) -> pd.DataFrame:
        """获取交易历史 DataFrame。"""
        return pd.DataFrame(self._trades)
