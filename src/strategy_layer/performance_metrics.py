"""
Kronos 指数/ETF 择时策略系统 — 回测绩效指标计算

严格按 docs/附F-回测绩效指标计算口径说明.md 计算 10 项绩效指标。

| # | 指标 | 公式 | 说明 |
|---|------|------|------|
| 1 | 累计收益率 | (E_T - E_0) / E_0 × 100% | 含持仓浮动盈亏，已扣成本 |
| 2 | 年化收益率 | (E_T / E_0)^(244/T) - 1 | 244 交易日/年，T ≥ 63 才计算 |
| 3 | 最大回撤 | max((peak - E_t) / peak) | 含回撤区间（峰值日→谷底日→恢复日） |
| 4 | 胜率 | N_win / N × 100% | 盈利笔数/总笔数 |
| 5 | 盈亏比 | avg_win / abs(avg_loss) | 平均盈利/平均亏损，>1.5 优秀 |
| 6 | 夏普比率 | (R_ann - r_f) / σ_ann | r_f 无风险利率，σ_ann 年化波动率 |
| 7 | 卡玛比率 | R_ann / abs(MDD) | 年化收益/最大回撤 |
| 8 | 超额收益 | R_ann_strategy - R_ann_benchmark | 相对基准的超额 |
| 9 | 最大连胜/连亏 | — | 最长连续盈利/亏损笔数 |
|10 | 平均持仓天数 | Σ(hold_days) / N | 含提前止损/止盈 |

所有计算使用高精度 float64（double），最终结果保留 4 位小数。
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)

# A 股市场年交易日数
TRADING_DAYS_PER_YEAR = 244


class PerformanceMetrics:
    """
    回测绩效指标计算器。

    Usage:
        pm = PerformanceMetrics(config)
        result = pm.compute(daily_equity, trades, initial_capital, benchmark_equity)
    """

    def __init__(self, config):
        """
        Args:
            config: 系统配置
        """
        self.config = config
        self.risk_free_rate: float = float(
            config.get("backtest", "risk_free_rate", default=0.025)
        )

    # ═══════════════════════════════════════════════
    # 主计算入口
    # ═══════════════════════════════════════════════

    def compute(
        self,
        daily_equity: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        initial_capital: float,
        benchmark_equity: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        从回测输出计算全部 10 项绩效指标。

        Args:
            daily_equity: 每日权益记录
                [{date, total_equity, daily_return, cumulative_return, drawdown, ...}]
            trades: 交易记录
                [{date, symbol, direction, pnl, pnl_pct, hold_days, ...}]
            initial_capital: 初始资金
            benchmark_equity: 基准权益曲线
                [{date, value}]

        Returns:
            完整绩效指标字典
        """
        if not daily_equity or len(daily_equity) == 0:
            return self._empty_result()

        # 转为 DataFrame 方便计算
        eq_df = pd.DataFrame(daily_equity)
        trade_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # 提取关键序列
        equity = eq_df["total_equity"].values.astype(np.float64)
        daily_returns = eq_df["daily_return"].values.astype(np.float64) if "daily_return" in eq_df else None

        if daily_returns is None and len(equity) >= 2:
            daily_returns = np.diff(equity) / equity[:-1]
            daily_returns = np.insert(daily_returns, 0, 0.0)

        n_days = len(equity)

        # ── 1. 累计收益率 ──
        final_equity = float(equity[-1])
        cumulative_return = (final_equity - initial_capital) / initial_capital

        # ── 2. 年化收益率 ──
        annual_return = self._annual_return(initial_capital, final_equity, n_days)

        # ── 3. 最大回撤 ──
        max_dd, dd_period = self._max_drawdown(equity, eq_df)

        # ── 4. 胜率 ──
        win_rate, n_wins, n_losses, n_total = self._win_rate(trade_df)

        # ── 5. 盈亏比 ──
        profit_loss_ratio = self._profit_loss_ratio(trade_df)

        # ── 6. 夏普比率 ──
        sharpe = self._sharpe_ratio(daily_returns, annual_return)

        # ── 7. 卡玛比率 ──
        calmar = self._calmar_ratio(annual_return, max_dd)

        # ── 8. 超额收益（Alpha） ──
        benchmark_annual = self._benchmark_annual_return(
            benchmark_equity, n_days
        ) if benchmark_equity else 0.0
        excess_return = annual_return - benchmark_annual

        # ── 9. 最大连胜/连亏 ──
        max_consec_wins, max_consec_losses = self._max_consecutive(trade_df)

        # ── 10. 平均持仓天数 ──
        avg_hold_days = self._avg_hold_days(trade_df)

        # ── 月度/年度收益汇总 ──
        monthly_returns = self._monthly_returns(eq_df)
        yearly_returns = self._yearly_returns(eq_df, initial_capital)

        result = {
            # 基础
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "n_days": n_days,
            "n_trades": n_total,

            # 10 项指标
            "cumulative_return": round(cumulative_return, 6),
            "annual_return": round(annual_return, 6),
            "max_drawdown": round(max_dd, 6),
            "max_drawdown_period": dd_period,
            "win_rate": round(win_rate, 6),
            "profit_loss_ratio": round(profit_loss_ratio, 6) if profit_loss_ratio is not None else None,
            "sharpe_ratio": round(sharpe, 6),
            "calmar_ratio": round(calmar, 6),
            "excess_return": round(excess_return, 6),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_hold_days": round(avg_hold_days, 2),

            # 汇总
            "monthly_returns": monthly_returns,
            "yearly_returns": yearly_returns,

            # 明细统计
            "n_wins": n_wins,
            "n_losses": n_losses,
            "benchmark_annual_return": round(benchmark_annual, 6),

            # 波动率
            "annual_volatility": round(self._annual_volatility(daily_returns), 6),
        }

        logger.info(
            f"绩效指标计算完成: 累计收益={cumulative_return:.2%}, "
            f"年化={annual_return:.2%}, 最大回撤={max_dd:.2%}, "
            f"夏普={sharpe:.2f}, 胜率={win_rate:.2%}"
        )

        return result

    # ═══════════════════════════════════════════════
    # 单项指标计算
    # ═══════════════════════════════════════════════

    @staticmethod
    def _annual_return(initial: float, final: float, n_days: int) -> float:
        """
        年化收益率。

        公式: (E_T / E_0)^(244/T) - 1
        要求 T ≥ 63 个交易日才有统计意义。
        """
        if n_days < 63 or initial <= 0:
            return 0.0

        total_return = final / initial
        years = n_days / TRADING_DAYS_PER_YEAR
        annual = math.pow(total_return, 1.0 / years) - 1.0
        return annual

    @staticmethod
    def _max_drawdown(
        equity: np.ndarray, eq_df: pd.DataFrame
    ) -> Tuple[float, Dict[str, Any]]:
        """
        最大回撤 + 回撤区间。

        返回:
            (max_drawdown_value, {peak_date, trough_date, recovery_date, peak_value, trough_value})
        """
        peak = equity[0]
        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        current_peak_idx = 0

        for i in range(1, len(equity)):
            if equity[i] > peak:
                peak = equity[i]
                current_peak_idx = i
            dd = (peak - equity[i]) / peak
            if dd > max_dd:
                max_dd = dd
                peak_idx = current_peak_idx
                trough_idx = i

        # 找回撤恢复日（回撤后首次回到峰值之上的日期）
        recovery_idx = None
        if max_dd > 0 and peak_idx < len(equity) - 1:
            peak_value = equity[peak_idx]
            for i in range(trough_idx + 1, len(equity)):
                if equity[i] >= peak_value:
                    recovery_idx = i
                    break

        period = {
            "peak_date": str(eq_df["date"].iloc[peak_idx]) if "date" in eq_df else None,
            "trough_date": str(eq_df["date"].iloc[trough_idx]) if "date" in eq_df else None,
            "recovery_date": str(eq_df["date"].iloc[recovery_idx]) if recovery_idx and "date" in eq_df else None,
            "peak_value": float(equity[peak_idx]),
            "trough_value": float(equity[trough_idx]),
            "drawdown_days": trough_idx - peak_idx,
        }

        return -max_dd, period

    @staticmethod
    def _win_rate(trade_df: pd.DataFrame) -> Tuple[float, int, int, int]:
        """
        胜率 = 盈利笔数 / 总交易笔数。

        只统计 SELL 交易（完整的买卖闭环）。
        """
        sells = trade_df[trade_df["direction"] == "SELL"] if len(trade_df) > 0 else pd.DataFrame()
        n_total = len(sells)

        if n_total == 0:
            return 0.0, 0, 0, 0

        n_wins = len(sells[sells.get("pnl", 0) > 0]) if "pnl" in sells.columns else 0
        n_losses = n_total - n_wins
        win_rate = n_wins / n_total

        return win_rate, n_wins, n_losses, n_total

    @staticmethod
    def _profit_loss_ratio(trade_df: pd.DataFrame) -> Optional[float]:
        """
        盈亏比 = 平均盈利 / |平均亏损|。

        > 1.5 为优秀。
        """
        sells = trade_df[trade_df["direction"] == "SELL"] if len(trade_df) > 0 else pd.DataFrame()
        if len(sells) == 0 or "pnl" not in sells.columns:
            return None

        wins = sells[sells["pnl"] > 0]["pnl"]
        losses = sells[sells["pnl"] < 0]["pnl"]

        if len(wins) == 0 or len(losses) == 0:
            return None

        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return float("inf") if avg_win > 0 else 0.0

        return avg_win / avg_loss

    def _sharpe_ratio(self, daily_returns: np.ndarray, annual_return: float) -> float:
        """
        夏普比率 = (R_ann - r_f) / σ_ann。

        σ_ann = std(daily_returns) × sqrt(244)。
        """
        if daily_returns is None or len(daily_returns) < 2:
            return 0.0

        annual_vol = self._annual_volatility(daily_returns)
        if annual_vol == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / annual_vol

    @staticmethod
    def _calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """
        卡玛比率 = 年化收益率 / |最大回撤|。

        值越大，单位回撤获得的收益越多。
        """
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)

    @staticmethod
    def _benchmark_annual_return(
        benchmark_equity: List[Dict[str, Any]], n_days: int
    ) -> float:
        """计算基准的年化收益率。"""
        if not benchmark_equity or len(benchmark_equity) < 2:
            return 0.0

        first_val = benchmark_equity[0].get("value", 1.0)
        last_val = benchmark_equity[-1].get("value", 1.0)

        if first_val <= 0:
            return 0.0

        return PerformanceMetrics._annual_return(first_val, last_val, n_days)

    @staticmethod
    def _max_consecutive(trade_df: pd.DataFrame) -> Tuple[int, int]:
        """
        最大连胜/连亏笔数。

        遍历 SELL 交易的 pnl 序列。
        """
        sells = trade_df[trade_df["direction"] == "SELL"] if len(trade_df) > 0 else pd.DataFrame()
        if len(sells) == 0 or "pnl" not in sells.columns:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in sells["pnl"]:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    @staticmethod
    def _avg_hold_days(trade_df: pd.DataFrame) -> float:
        """平均持仓天数（含止损/止盈提前平仓）。"""
        sells = trade_df[trade_df["direction"] == "SELL"] if len(trade_df) > 0 else pd.DataFrame()
        if len(sells) == 0 or "hold_days" not in sells.columns:
            return 0.0

        return float(sells["hold_days"].mean())

    # ═══════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════

    @staticmethod
    def _annual_volatility(daily_returns: np.ndarray) -> float:
        """年化波动率 = std(daily_returns) × sqrt(244)。"""
        if daily_returns is None or len(daily_returns) < 2:
            return 0.0
        return float(np.std(daily_returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _monthly_returns(eq_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """计算月度收益率。"""
        if "date" not in eq_df.columns or "total_equity" not in eq_df.columns:
            return []

        eq_df = eq_df.copy()
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df["month"] = eq_df["date"].dt.to_period("M")

        monthly = []
        for month, group in eq_df.groupby("month"):
            first_eq = group["total_equity"].iloc[0]
            last_eq = group["total_equity"].iloc[-1]
            ret = (last_eq - first_eq) / first_eq if first_eq > 0 else 0.0
            monthly.append({
                "month": str(month),
                "return": round(float(ret), 6),
                "start_equity": float(first_eq),
                "end_equity": float(last_eq),
            })

        return monthly

    @staticmethod
    def _yearly_returns(eq_df: pd.DataFrame, initial_capital: float) -> List[Dict[str, Any]]:
        """计算年度收益率。"""
        if "date" not in eq_df.columns or "total_equity" not in eq_df.columns:
            return []

        eq_df = eq_df.copy()
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df["year"] = eq_df["date"].dt.year

        yearly = []
        prev_year_end = initial_capital

        for year, group in eq_df.groupby("year"):
            first_eq = group["total_equity"].iloc[0]
            last_eq = group["total_equity"].iloc[-1]
            ret = (last_eq - first_eq) / first_eq if first_eq > 0 else 0.0

            # 累计收益（相对初始资金）
            cum_ret = (last_eq - initial_capital) / initial_capital

            yearly.append({
                "year": int(year),
                "return": round(float(ret), 6),
                "cumulative_return": round(float(cum_ret), 6),
                "start_equity": float(first_eq),
                "end_equity": float(last_eq),
            })

            prev_year_end = float(last_eq)

        return yearly

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """返回空的绩效指标字典。"""
        return {
            "initial_capital": 0.0,
            "final_equity": 0.0,
            "n_days": 0,
            "n_trades": 0,
            "cumulative_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_period": {},
            "win_rate": 0.0,
            "profit_loss_ratio": None,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "excess_return": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_hold_days": 0.0,
            "monthly_returns": [],
            "yearly_returns": [],
            "n_wins": 0,
            "n_losses": 0,
            "benchmark_annual_return": 0.0,
            "annual_volatility": 0.0,
        }

    # ═══════════════════════════════════════════════
    # 独立计算（无需回测引擎）
    # ═══════════════════════════════════════════════

    @classmethod
    def from_equity_curve(
        cls,
        equity_curve: List[float],
        initial_capital: float,
        risk_free_rate: float = 0.025,
    ) -> Dict[str, Any]:
        """
        从纯权益曲线计算基本指标（不含交易相关指标）。

        Args:
            equity_curve: 每日总权益序列
            initial_capital: 初始资金
            risk_free_rate: 无风险利率
        """
        equity = np.array(equity_curve, dtype=np.float64)
        n_days = len(equity)

        if n_days < 2:
            return {"cumulative_return": 0.0, "n_days": n_days}

        # 日收益率
        daily_returns = np.diff(equity) / equity[:-1]
        daily_returns = np.insert(daily_returns, 0, 0.0)

        final_eq = float(equity[-1])
        cumulative_return = (final_eq - initial_capital) / initial_capital
        annual_return = cls._annual_return(initial_capital, final_eq, n_days)

        # 最大回撤
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

        max_dd = -max_dd

        annual_vol = cls._annual_volatility(daily_returns)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "initial_capital": initial_capital,
            "final_equity": final_eq,
            "n_days": n_days,
            "cumulative_return": round(cumulative_return, 6),
            "annual_return": round(annual_return, 6),
            "max_drawdown": round(max_dd, 6),
            "sharpe_ratio": round(sharpe, 6),
            "calmar_ratio": round(calmar, 6),
            "annual_volatility": round(annual_vol, 6),
        }
