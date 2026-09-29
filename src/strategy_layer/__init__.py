"""
Kronos 指数/ETF 择时策略系统 — 策略层

模块:
  - SignalGenerator:    基于 Kronos 预测收益率生成 BUY/CASH/HOLD/SELL 择时信号
  - RiskManager:        动态风控（止损/止盈/熔断/仓位管理），高精度 decimal 计算
  - BacktestEngine:     逐日模拟交易（佣金/印花税/滑点），全程记录权益变化
  - PerformanceMetrics: 10项绩效指标（年化/夏普/卡玛/最大回撤/胜率/盈亏比...）
  - StockSelector:      个股选股器（Phase 6 扩展预留，当前为桩）
  - StrategyOptimizer:  策略参数优化器（Phase 6 扩展预留，当前为桩）
"""

from src.strategy_layer.signal_generator import (
    SignalGenerator,
    SignalType,
    PositionState,
)

from src.strategy_layer.risk_manager import (
    RiskManager,
    RiskEventType,
    CircuitBreakerState,
)

from src.strategy_layer.backtest_engine import (
    BacktestEngine,
    BacktestStatus,
)

from src.strategy_layer.performance_metrics import (
    PerformanceMetrics,
)

from src.strategy_layer.stock_selector import (
    StockSelector,
)

from src.strategy_layer.strategy_optimizer import (
    StrategyOptimizer,
    StrategyParams,
)

__all__ = [
    # 信号生成
    "SignalGenerator",
    "SignalType",
    "PositionState",
    # 风控
    "RiskManager",
    "RiskEventType",
    "CircuitBreakerState",
    # 回测
    "BacktestEngine",
    "BacktestStatus",
    # 绩效
    "PerformanceMetrics",
    # 选股（桩）
    "StockSelector",
    # 优化（桩）
    "StrategyOptimizer",
    "StrategyParams",
]
