"""
Kronos 指数/ETF 择时策略系统 — 策略优化器

提供策略参数优化能力:
  - 参数网格搜索 (grid_search)
  - Walk-Forward 滚动窗口优化 (walk_forward_optimize)
  - 多策略并行对比 (compare_strategies)
  - 参数敏感性分析 (sensitivity_analysis)
  - 最优参数推荐 (recommend_params)

Walk-Forward 流程:
  1. 将全量数据按时间划分为 N 个窗口
  2. 每个窗口内:
     a. 训练集: 前 window_months 个月
     b. 测试集: 后 step_months 个月 (样本外)
     c. 在训练集上网格搜索最优参数
     d. 用最优参数在测试集上回测
  3. 统计所有窗口的样本外绩效
  4. 输出最优参数组合（所有窗口中出现频率最高的参数）
"""

from __future__ import annotations

import copy
import itertools
import time
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.config import Config
from src.common.utils import get_trading_days, format_date

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# StrategyParams dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StrategyParams:
    """策略参数组合"""
    buy_threshold: float = 0.005
    hold_days: int = 5
    stop_loss: float = -0.08
    take_profit: float = 0.15
    max_single_position: float = 0.30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buy_threshold": self.buy_threshold,
            "hold_days": self.hold_days,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "max_single_position": self.max_single_position,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        return cls(
            buy_threshold=d.get("buy_threshold", 0.005),
            hold_days=d.get("hold_days", 5),
            stop_loss=d.get("stop_loss", -0.08),
            take_profit=d.get("take_profit", 0.15),
            max_single_position=d.get("max_single_position", 0.30),
        )

    def param_key(self) -> str:
        """Return a hashable string key for frequency counting."""
        return (
            f"bt={self.buy_threshold}|hd={self.hold_days}|"
            f"sl={self.stop_loss}|tp={self.take_profit}|"
            f"msp={self.max_single_position}"
        )


# ═══════════════════════════════════════════════════════════════════
# Config overlay (mirrors _RequestConfig from api_backtest.py)
# ═══════════════════════════════════════════════════════════════════

class _OverlayConfig:
    """Immutable configuration overlay that merges strategy params into a base config.

    Replicates the pattern from ``src.service_layer.api_backtest._RequestConfig``
    so that BacktestEngine / SignalGenerator / RiskManager pick up the
    optimiser-supplied parameters without mutating the global singleton.
    """

    _ALLOWED_KEYS = {
        "buy_threshold", "hold_days", "stop_loss", "take_profit",
        "max_single_position", "max_positions", "min_positions",
        "risk_controls_enabled", "trailing_stop_enabled", "trailing_stop_pct",
        "paper_mode", "paper_symbol", "exit_at_close",
    }

    def __init__(self, base: Config, strategy_params: Optional[Dict[str, Any]] = None):
        self._data = copy.deepcopy(base.raw_data)
        self._data.setdefault("strategy", {}).update({
            k: v for k, v in (strategy_params or {}).items() if k in self._ALLOWED_KEYS
        })

    def get(self, *keys, default=None) -> Any:
        value = self._data
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
        return value if value is not None else default

    @property
    def raw_data(self) -> dict:
        return self._data

    @property
    def seed(self) -> int:
        return int(self.get("reproducibility", "random_seed", default=42))


# ═══════════════════════════════════════════════════════════════════
# StrategyOptimizer
# ═══════════════════════════════════════════════════════════════════

class StrategyOptimizer:
    """
    策略参数优化器。

    Supports:
      - Exhaustive grid search over strategy parameter spaces
      - Walk-Forward rolling-window optimisation with out-of-sample validation
      - Side-by-side multi-strategy comparison
      - Parameter sensitivity analysis
      - Frequency-based best-param recommendation across windows
    """

    def __init__(self, config: Config):
        """
        Args:
            config: 系统配置 Config 实例
        """
        self.config = config
        logger.info("StrategyOptimizer 初始化")

    # ───────────────────────────────────────────────
    # Parameter grid generation
    # ───────────────────────────────────────────────

    def generate_param_grid(self, param_ranges: Dict[str, List]) -> List[StrategyParams]:
        """Generate all combinations of strategy parameters from range specifications.

        Args:
            param_ranges: Mapping of parameter name to list of candidate values, e.g.
                {
                    "buy_threshold": [0.003, 0.005, 0.008],
                    "hold_days": [3, 5, 7],
                    "stop_loss": [-0.05, -0.08],
                    "take_profit": [0.10, 0.15],
                    "max_single_position": [0.20, 0.30],
                }

        Returns:
            List of StrategyParams, one per combination (Cartesian product).
        """
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]

        combinations = []
        for combo in itertools.product(*param_values):
            d = dict(zip(param_names, combo))
            combinations.append(StrategyParams.from_dict(d))

        logger.info(f"参数网格: {len(combinations)} 种组合 ({param_ranges})")
        return combinations

    # ───────────────────────────────────────────────
    # Grid search
    # ───────────────────────────────────────────────

    def grid_search(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        predictions_df: pd.DataFrame,
        param_ranges: Dict[str, List],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        metric: str = "sharpe_ratio",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Exhaustive grid search over strategy parameters.

        For every combination in the Cartesian product of *param_ranges*:
          1. Build an overlay Config with the strategy params
          2. Instantiate BacktestEngine and run_with_predictions()
          3. Extract key metrics (sharpe, annual_return, max_drawdown, win_rate, …)
          4. Record the result

        Args:
            symbol_dfs: {symbol: DataFrame} OHLCV data per instrument
            predictions_df: Pre-computed predictions DataFrame (date, symbol, …)
            param_ranges: Parameter candidate values (see generate_param_grid)
            start_date: Backtest start (inclusive)
            end_date: Backtest end (inclusive)
            initial_capital: Starting cash
            metric: Primary ranking metric key (default "sharpe_ratio")
            progress_callback: Optional callable receiving progress fraction 0-1

        Returns:
            {
                results: [{params: dict, metrics: dict}, ...],
                best_params: dict,
                best_metrics: dict,
                ranked_results: [sorted by metric descending],
                param_ranges: dict,
                metric: str,
            }
        """
        from src.strategy_layer.backtest_engine import BacktestEngine

        param_grid = self.generate_param_grid(param_ranges)
        n_total = len(param_grid)
        results: List[Dict[str, Any]] = []

        logger.info(
            f"Grid search 开始: {n_total} 组合, 区间 {start_date}~{end_date}, "
            f"排序指标={metric}"
        )

        for idx, params in enumerate(param_grid):
            try:
                overlay = _OverlayConfig(self.config, params.to_dict())
                engine = BacktestEngine(overlay)

                bt_result = engine.run_with_predictions(
                    predictions_df=predictions_df,
                    symbol_dfs=symbol_dfs,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                )

                metrics = self._extract_metrics(bt_result)
                results.append({
                    "params": params.to_dict(),
                    "metrics": metrics,
                })

            except Exception as e:
                logger.warning(
                    f"Grid search 组合 {idx+1}/{n_total} 失败 "
                    f"(params={params.to_dict()}): {e}"
                )
                results.append({
                    "params": params.to_dict(),
                    "metrics": self._default_metrics(),
                    "error": str(e),
                })

            # Progress: grid search progress
            if progress_callback:
                progress_callback((idx + 1) / n_total)

            if (idx + 1) % max(1, n_total // 10) == 0:
                logger.info(f"  Grid search 进度: {idx + 1}/{n_total}")

        # Rank by primary metric (descending — higher sharpe / return is better;
        # for max_drawdown which is negative, higher is also better).
        ranked = sorted(
            results,
            key=lambda r: r["metrics"].get(metric, float("-inf")),
            reverse=True,
        )

        best = ranked[0] if ranked else {"params": StrategyParams().to_dict(), "metrics": self._default_metrics()}

        logger.info(
            f"Grid search 完成: best {metric}="
            f"{best['metrics'].get(metric, 'N/A')}, "
            f"params={best['params']}"
        )

        return {
            "results": results,
            "best_params": best["params"],
            "best_metrics": best["metrics"],
            "ranked_results": ranked,
            "param_ranges": param_ranges,
            "metric": metric,
        }

    # ───────────────────────────────────────────────
    # Walk-Forward optimisation
    # ───────────────────────────────────────────────

    def walk_forward_optimize(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        predictions_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        window_months: int = 12,
        step_months: int = 3,
        param_ranges: Optional[Dict[str, List]] = None,
        metric: str = "sharpe_ratio",
        initial_capital: float = 1_000_000.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Walk-Forward rolling-window strategy optimisation.

        Algorithm:
          1. Split the date range into overlapping windows.
             Each window has:
               - Train period: ``window_months`` months
               - Test period (OOS): ``step_months`` months immediately following
          2. For each window:
               a. Grid-search on the *train* period → best in-sample params
               b. Run backtest with those best params on the *test* period (OOS)
          3. Aggregate all OOS results; compute stability statistics.

        Args:
            symbol_dfs: {symbol: DataFrame}
            predictions_df: Pre-computed predictions
            start_date: Overall start
            end_date: Overall end
            window_months: Training window length in months
            step_months: OOS test window length AND rolling step in months
            param_ranges: Grid search parameter candidates (None → sensible defaults)
            metric: Primary ranking metric
            initial_capital: Starting capital per backtest
            progress_callback: Optional progress callback (0-1)

        Returns:
            {
                best_params: dict,            # frequency-based recommendation
                window_results: [{...}],      # per-window detail
                overall_sharpe: float,        # mean OOS Sharpe
                stability_report: dict,       # consistency diagnostics
            }
        """
        if param_ranges is None:
            param_ranges = {
                "buy_threshold": [0.003, 0.005, 0.008],
                "hold_days": [3, 5, 7],
                "stop_loss": [-0.05, -0.08, -0.10],
                "take_profit": [0.10, 0.15, 0.20],
            }

        windows = self._build_walk_forward_windows(
            start_date, end_date, window_months, step_months
        )

        if not windows:
            logger.warning("Walk-Forward: 无法生成有效窗口，日期范围不足")
            return {
                "best_params": StrategyParams().to_dict(),
                "window_results": [],
                "overall_sharpe": 0.0,
                "stability_report": {"status": "no_windows"},
            }

        n_windows = len(windows)
        logger.info(
            f"Walk-Forward: {n_windows} 个窗口, "
            f"train={window_months}月, test={step_months}月, "
            f"排序指标={metric}"
        )

        window_results: List[Dict[str, Any]] = []
        oos_sharpes: List[float] = []

        for w_idx, window in enumerate(windows):
            train_start = window["train_start"]
            train_end = window["train_end"]
            test_start = window["test_start"]
            test_end = window["test_end"]

            logger.info(
                f"  窗口 {w_idx+1}/{n_windows}: "
                f"train={train_start}~{train_end}, "
                f"test={test_start}~{test_end}"
            )

            # ── Step A: grid search on train period ──
            try:
                grid_result = self.grid_search(
                    symbol_dfs=symbol_dfs,
                    predictions_df=predictions_df,
                    param_ranges=param_ranges,
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=initial_capital,
                    metric=metric,
                )
                best_is_params = grid_result["best_params"]
            except Exception as e:
                logger.warning(f"  窗口 {w_idx+1} 训练期 grid search 失败: {e}")
                best_is_params = StrategyParams().to_dict()

            # ── Step B: backtest with best IS params on test period (OOS) ──
            try:
                oos_result = self._run_single_backtest(
                    symbol_dfs=symbol_dfs,
                    predictions_df=predictions_df,
                    strategy_params=best_is_params,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=initial_capital,
                )
                oos_metrics = self._extract_metrics(oos_result)
            except Exception as e:
                logger.warning(f"  窗口 {w_idx+1} OOS 回测失败: {e}")
                oos_metrics = self._default_metrics()

            oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
            oos_sharpes.append(oos_sharpe)

            window_results.append({
                "window_id": w_idx + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_is_params": best_is_params,
                "oos_metrics": oos_metrics,
                "oos_sharpe": oos_sharpe,
            })

            # Progress: each window contributes 1/n_windows
            if progress_callback:
                progress_callback((w_idx + 1) / n_windows)

        # ── Aggregate ──
        overall_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        best_params = self.recommend_params(window_results)

        stability = self._compute_stability_report(window_results, metric)

        logger.info(
            f"Walk-Forward 完成: {n_windows} 窗口, "
            f"overall_sharpe={overall_sharpe:.4f}, "
            f"best_params={best_params.to_dict()}"
        )

        return {
            "best_params": best_params.to_dict(),
            "window_results": window_results,
            "overall_sharpe": overall_sharpe,
            "stability_report": stability,
        }

    # ───────────────────────────────────────────────
    # Multi-strategy comparison
    # ───────────────────────────────────────────────

    def compare_strategies(
        self,
        strategies: List[StrategyParams],
        symbol_dfs: Dict[str, pd.DataFrame],
        predictions_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Run independent backtests for each strategy parameter set and compare.

        Args:
            strategies: List of StrategyParams to evaluate
            symbol_dfs: {symbol: DataFrame}
            predictions_df: Pre-computed predictions
            start_date: Backtest start
            end_date: Backtest end
            initial_capital: Starting capital
            progress_callback: Optional progress callback (0-1)

        Returns:
            {
                results: [{params: dict, metrics: dict}, ...],
                comparison_table: list of dicts (flat params + metrics for tabular display),
                best_by_metric: {metric_name: {params, metrics}, ...},
            }
        """
        n_strats = len(strategies)
        results: List[Dict[str, Any]] = []

        logger.info(f"多策略对比: {n_strats} 个策略, 区间 {start_date}~{end_date}")

        for idx, params in enumerate(strategies):
            try:
                bt_result = self._run_single_backtest(
                    symbol_dfs=symbol_dfs,
                    predictions_df=predictions_df,
                    strategy_params=params.to_dict(),
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                )
                metrics = self._extract_metrics(bt_result)
            except Exception as e:
                logger.warning(f"策略 {idx+1}/{n_strats} 回测失败: {e}")
                metrics = self._default_metrics()

            results.append({
                "params": params.to_dict(),
                "metrics": metrics,
            })

            if progress_callback:
                progress_callback((idx + 1) / n_strats)

        # ── Build comparison table (flat dicts for DataFrame / tabular display) ──
        comparison_table = []
        for r in results:
            row = {}
            row.update(r["params"])
            row.update(r["metrics"])
            comparison_table.append(row)

        # ── Best by each metric ──
        metric_keys = [
            "sharpe_ratio", "annual_return", "max_drawdown",
            "win_rate", "calmar_ratio", "profit_loss_ratio",
        ]
        best_by_metric: Dict[str, Dict[str, Any]] = {}
        for mk in metric_keys:
            valid = [r for r in results if mk in r["metrics"] and r["metrics"][mk] is not None]
            if not valid:
                continue
            # For max_drawdown (negative), higher is better (less negative = less drawdown)
            best = max(valid, key=lambda r: r["metrics"][mk] if r["metrics"][mk] is not None else float("-inf"))
            best_by_metric[mk] = {"params": best["params"], "metrics": best["metrics"]}

        return {
            "results": results,
            "comparison_table": comparison_table,
            "best_by_metric": best_by_metric,
        }

    # ───────────────────────────────────────────────
    # Sensitivity analysis
    # ───────────────────────────────────────────────

    def sensitivity_analysis(self, grid_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Compute per-parameter sensitivity from grid search results.

        For each parameter, calculate the average absolute change in the primary
        metric when that parameter changes (while all other parameters are held
        constant).

        Args:
            grid_results: Output of ``grid_search()`` — must contain
                ``results`` and ``metric`` keys.

        Returns:
            {
                param_name: {
                    "mean_abs_change": float,
                    "max_abs_change": float,
                    "std_change": float,
                },
                ...
            }
        """
        all_results = grid_results.get("results", [])
        primary_metric = grid_results.get("metric", "sharpe_ratio")

        if not all_results:
            return {}

        # Build a list of (params_dict, metric_value) pairs
        records = []
        for r in all_results:
            p = r.get("params", {})
            m = r.get("metrics", {}).get(primary_metric)
            if m is not None:
                records.append((p, m))

        if len(records) < 2:
            return {}

        param_names = list(records[0][0].keys())
        sensitivity: Dict[str, Dict[str, float]] = {}

        for pname in param_names:
            # Group by all other params (i.e., hold-everything-else-constant)
            other_params_values: Dict[str, List[Tuple[Any, float]]] = {}
            for p, m in records:
                key_parts = []
                for k, v in p.items():
                    if k != pname:
                        key_parts.append(f"{k}={v}")
                group_key = "|".join(sorted(key_parts))
                other_params_values.setdefault(group_key, []).append((p[pname], m))

            # Collect metric changes within each group
            changes: List[float] = []
            for group_key, values in other_params_values.items():
                if len(values) < 2:
                    continue
                # Sort by param value so we compute consecutive differences
                values.sort(key=lambda x: x[0])
                for i in range(1, len(values)):
                    delta = abs(values[i][1] - values[i - 1][1])
                    changes.append(delta)

            if changes:
                sensitivity[pname] = {
                    "mean_abs_change": float(np.mean(changes)),
                    "max_abs_change": float(np.max(changes)),
                    "std_change": float(np.std(changes)),
                }
            else:
                sensitivity[pname] = {
                    "mean_abs_change": 0.0,
                    "max_abs_change": 0.0,
                    "std_change": 0.0,
                }

        return sensitivity

    # ───────────────────────────────────────────────
    # Parameter recommendation
    # ───────────────────────────────────────────────

    def recommend_params(self, window_results: List[Dict[str, Any]]) -> StrategyParams:
        """Frequency method: most common best in-sample params across all windows.

        Each entry in *window_results* should contain ``best_is_params`` (a dict
        of strategy parameters that was optimal for that window's training period).

        The parameters that appear most frequently as the best are returned.

        Args:
            window_results: Per-window results from ``walk_forward_optimize()``

        Returns:
            StrategyParams with the most frequently winning parameter combination.
        """
        if not window_results:
            return StrategyParams()

        # Count occurrences of each param combination
        counter: Counter = Counter()
        param_by_key: Dict[str, Dict[str, Any]] = {}

        for wr in window_results:
            p = wr.get("best_is_params")
            if p is None:
                continue
            sp = StrategyParams.from_dict(p)
            key = sp.param_key()
            counter[key] += 1
            param_by_key[key] = p

        if not counter:
            return StrategyParams()

        most_common_key, freq = counter.most_common(1)[0]
        logger.info(
            f"推荐参数 (频率法): {param_by_key[most_common_key]}, "
            f"出现 {freq}/{len(window_results)} 次"
        )
        return StrategyParams.from_dict(param_by_key[most_common_key])

    # ═══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_metrics(bt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Pull the key performance metrics from a BacktestEngine result dict."""
        return {
            "sharpe_ratio": bt_result.get("sharpe_ratio"),
            "annual_return": bt_result.get("annual_return"),
            "cumulative_return": bt_result.get("cumulative_return"),
            "max_drawdown": bt_result.get("max_drawdown"),
            "win_rate": bt_result.get("win_rate"),
            "calmar_ratio": bt_result.get("calmar_ratio"),
            "profit_loss_ratio": bt_result.get("profit_loss_ratio"),
            "excess_return": bt_result.get("excess_return"),
            "total_trades": bt_result.get("total_trades", 0),
            "avg_hold_days": bt_result.get("avg_hold_days"),
            "max_consecutive_wins": bt_result.get("max_consecutive_wins", 0),
            "max_consecutive_losses": bt_result.get("max_consecutive_losses", 0),
        }

    @staticmethod
    def _default_metrics() -> Dict[str, Any]:
        """Fallback metrics when a backtest fails."""
        return {
            "sharpe_ratio": None,
            "annual_return": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "calmar_ratio": None,
            "profit_loss_ratio": None,
            "excess_return": 0.0,
            "total_trades": 0,
            "avg_hold_days": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    def _run_single_backtest(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        predictions_df: pd.DataFrame,
        strategy_params: Dict[str, Any],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Run a single backtest with a specific strategy param overlay.

        Returns the raw BacktestEngine result dict.
        """
        from src.strategy_layer.backtest_engine import BacktestEngine

        overlay = _OverlayConfig(self.config, strategy_params)
        engine = BacktestEngine(overlay)
        return engine.run_with_predictions(
            predictions_df=predictions_df,
            symbol_dfs=symbol_dfs,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )

    @staticmethod
    def _build_walk_forward_windows(
        start_date: str,
        end_date: str,
        window_months: int,
        step_months: int,
    ) -> List[Dict[str, str]]:
        """Build overlapping train/test windows for walk-forward analysis.

        Each window:
          - Train: [cursor, cursor + window_months)
          - Test:  [cursor + window_months, cursor + window_months + step_months)

        The cursor advances by ``step_months`` after each window.

        Returns:
            List of dicts with keys:
                train_start, train_end, test_start, test_end (all "YYYY-MM-DD")
        """
        from dateutil.relativedelta import relativedelta

        windows = []
        cursor = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        while True:
            train_start = cursor
            train_end = cursor + relativedelta(months=window_months, days=-1)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + relativedelta(months=step_months, days=-1)

            # Stop if the test period extends beyond the overall end date
            if test_start > end_ts:
                break

            # Clip test_end to overall end if it slightly exceeds
            clipped_test_end = min(test_end, end_ts)

            # Verify that both train and test have trading days
            train_days = get_trading_days(
                train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")
            )
            test_days = get_trading_days(
                test_start.strftime("%Y-%m-%d"), clipped_test_end.strftime("%Y-%m-%d")
            )

            if len(train_days) > 0 and len(test_days) > 0:
                windows.append({
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": clipped_test_end.strftime("%Y-%m-%d"),
                })

            cursor = cursor + relativedelta(months=step_months)

        return windows

    @staticmethod
    def _compute_stability_report(
        window_results: List[Dict[str, Any]],
        metric: str,
    ) -> Dict[str, Any]:
        """Compute stability / consistency diagnostics for walk-forward results.

        Metrics:
          - OOS metric mean, std, min, max
          - Coefficient of variation (std / |mean|)
          - Fraction of windows with positive OOS metric
          - IS-OOS degradation (if IS metrics are available)
        """
        oos_values = [
            wr["oos_metrics"].get(metric) for wr in window_results
            if wr["oos_metrics"].get(metric) is not None
        ]

        if not oos_values:
            return {
                "status": "no_valid_windows",
                "n_windows": len(window_results),
            }

        arr = np.array(oos_values, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        # Coefficient of variation
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-9 else float("inf")

        # Fraction of windows with positive OOS metric
        positive_frac = float(np.mean(arr > 0))

        # Count distinct best param sets (diversity indicator)
        param_keys = []
        for wr in window_results:
            p = wr.get("best_is_params")
            if p is not None:
                sp = StrategyParams.from_dict(p)
                param_keys.append(sp.param_key())
        n_distinct = len(set(param_keys)) if param_keys else 0

        return {
            "status": "ok",
            "n_windows": len(window_results),
            "oos_mean": mean_val,
            "oos_std": std_val,
            "oos_min": min_val,
            "oos_max": max_val,
            "coefficient_of_variation": cv,
            "positive_fraction": positive_frac,
            "distinct_param_sets": n_distinct,
        }
