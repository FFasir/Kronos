"""
Kronos 指数/ETF 择时策略系统 — 滚动网格搜参

论文核心创新: 每月在验证集上网格搜索最优推理参数 (T, top_p, lookback_window)。
优化目标: Spearman 秩相关系数（预测序列 vs 真实序列）。

搜索空间（论文图表17）:
  - T (temperature):    [0.3, 0.6, 0.8, 1.0]
  - top_p:              [0.2, 0.4, 0.6, 0.9]
  - lookback_window:    [30, 60, 90]
"""

import json
import itertools
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from src.common.logger import get_logger
from src.common.utils import set_seed

logger = get_logger(__name__)


class ParamSearch:
    """
    月度滚动网格搜参。

    对于每个月:
      1. 取前 N 个月数据作为验证集
      2. 遍历所有 (T, top_p, lookback) 组合
      3. 计算每组合的 Spearman 相关系数
      4. 选最优组合应用于当月预测
    """

    def __init__(
        self,
        predictor,     # KronosPredictor 实例
        config,
    ):
        """
        Args:
            predictor: KronosPredictor (封装了模型和推理逻辑)
            config: 系统配置
        """
        self.predictor = predictor
        self.config = config

        # 搜索空间
        ps = config._data.get("param_search", {})
        self.T_values = ps.get("T_values", [0.3, 0.6, 0.8, 1.0])
        self.top_p_values = ps.get("top_p_values", [0.2, 0.4, 0.6, 0.9])
        self.lookback_values = ps.get("lookback_values", [30, 60, 90])

        self.val_months = ps.get("val_window_months", 1)
        self.test_months = ps.get("test_window_months", 1)
        self.metric = ps.get("metric", "spearman")

        n_combos = len(self.T_values) * len(self.top_p_values) * len(self.lookback_values)
        logger.info(
            f"ParamSearch 初始化: "
            f"T×{len(self.T_values)} top_p×{len(self.top_p_values)} "
            f"lookback×{len(self.lookback_values)} = {n_combos} 组合"
        )

    def search_month(
        self,
        df: pd.DataFrame,
        val_start: str,
        val_end: str,
        symbol: str = "unknown",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        对单月进行网格搜索。

        Args:
            df: 含 OHLCV 列的 DataFrame (全量历史)
            val_start: 验证集起始日期 (YYYY-MM-DD)
            val_end: 验证集结束日期 (YYYY-MM-DD)
            symbol: 标的代码 (用于日志)
            top_k: 保留前 K 个最优参数

        Returns:
            {
                "best_params": {"T": 0.8, "top_p": 0.95, "lookback": 60},
                "best_spearman": 0.856,
                "all_results": [...],   # 所有组合结果
            }
        """
        set_seed(self.config.seed)

        df = df.sort_index().copy()
        df.index = pd.to_datetime(df.index)
        val_dates = df.index[(df.index >= val_start) & (df.index <= val_end)]
        pred_len = self.predictor.pred_len
        if len(val_dates) < 20:
            logger.warning(f"验证集数据不足 ({len(val_dates)} 天), 使用默认参数")
            return self._default_result()

        logger.info(
            f"搜参: symbol={symbol}, val_range=[{val_start}, {val_end}], "
            f"days={len(val_dates)}"
        )

        # 生成所有参数组合
        param_combos = list(itertools.product(
            self.T_values, self.top_p_values, self.lookback_values
        ))

        all_results = []
        start_time = time.time()

        for idx, (T, top_p, lookback) in enumerate(param_combos):
            # 进度日志 (每 20 组)
            if idx % 20 == 0:
                logger.debug(f"  搜参进度: {idx}/{len(param_combos)}")

            pred_prices, true_prices = [], []

            for origin_date in val_dates:
                origin_idx = df.index.get_loc(origin_date)
                target_idx = origin_idx + pred_len
                # 信号在 origin_date 收盘后生成，只比较第5个后续交易日的收盘价。
                # 目标越过验证月会窥视待测月，故不纳入验证。
                if target_idx >= len(df) or df.index[target_idx] > pd.Timestamp(val_end):
                    continue
                hist_window = df.iloc[origin_idx - lookback + 1:origin_idx + 1]
                if len(hist_window) != lookback:
                    continue

                try:
                    # 使用指定参数推理
                    result = self.predictor.predict_single(
                        hist_window,
                        lookback=lookback,
                        T=T,
                        top_p=top_p,
                    )
                    pred_price = float(result.get("pred_close_t5", 0.0))
                except Exception:
                    pred_price = float(hist_window["close"].iloc[-1])

                if pred_price > 0:
                    pred_prices.append(pred_price)
                    true_prices.append(float(df["close"].iloc[target_idx]))

            # 截取对应真实价格
            n_preds = len(pred_prices)
            if n_preds < 5:
                spearman = 0.0
            else:
                spearman = self._compute_spearman(
                    np.array(pred_prices), np.array(true_prices)
                )

            all_results.append({
                "T": T,
                "top_p": top_p,
                "lookback": lookback,
                "spearman": round(spearman, 6),
                "n_predictions": n_preds,
            })

        # 按 Spearman 降序排列
        all_results.sort(key=lambda x: x["spearman"], reverse=True)

        duration = time.time() - start_time
        best = all_results[0]

        logger.info(
            f"搜参完成: best={best} (spearman={best['spearman']:.4f}), "
            f"耗时={duration:.1f}s, 评估了{len(all_results)}组参数"
        )

        return {
            "symbol": symbol,
            "val_start": val_start,
            "val_end": val_end,
            "best_params": {
                "T": best["T"],
                "top_p": best["top_p"],
                "lookback": best["lookback"],
            },
            "best_spearman": best["spearman"],
            "top_k": all_results[:top_k],
            "all_results": all_results,
            "search_duration_seconds": round(duration, 1),
            "total_combinations": len(param_combos),
        }

    def search_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        test_month: str = None,
    ) -> Dict[str, Any]:
        """
        对所有标的执行月度搜参。

        Args:
            symbols: 标的列表 (None = 全部)
            test_month: 测试月 (如 "2025-01", None = 最近完整月)

        Returns:
            {symbol: search_result, ...}
        """
        from src.data_layer.storage import StorageLayer
        from dateutil.relativedelta import relativedelta

        storage = StorageLayer(self.config)

        if symbols is None:
            symbols = [self.config.get("strategy", "paper_symbol", default="000852")]

        # 确定测试月和验证月
        if test_month is None:
            today = pd.Timestamp.now()
            test_month = (today - relativedelta(months=1)).strftime("%Y-%m")

        test_start = f"{test_month}-01"
        test_end = pd.Timestamp(test_start) + relativedelta(months=1, days=-1)
        test_end = test_end.strftime("%Y-%m-%d")

        val_start = pd.Timestamp(test_start) - relativedelta(months=self.val_months)
        val_start = val_start.strftime("%Y-%m-%d")
        val_end = pd.Timestamp(test_start) - pd.offsets.BDay(1)
        val_end = val_end.strftime("%Y-%m-%d")

        logger.info(f"批量搜参: test_month={test_month}, val=[{val_start}, {val_end}]")

        results = {}
        for sym in symbols:
            try:
                df = storage.load_raw_data(sym)
                if df is None or len(df) == 0:
                    logger.warning(f"  {sym}: 无数据")
                    results[sym] = self._default_result()
                    continue

                results[sym] = self.search_month(
                    df, val_start, val_end, symbol=sym
                )
            except Exception as e:
                logger.error(f"  {sym} 搜参失败: {e}")
                results[sym] = self._default_result()

        return {
            "test_month": test_month,
            "val_range": {"start": val_start, "end": val_end},
            "results": results,
            "searched_at": datetime.now().isoformat(),
        }

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str = None,
    ) -> str:
        """保存搜参结果到 JSON 文件"""
        if output_dir is None:
            output_dir = self.config.get("data", "outputs_dir", default="data/outputs")
            output_dir = Path(output_dir) / "param_search"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        test_month = results.get("test_month", "unknown")
        filepath = output_dir / f"{test_month}_best_params.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"搜参结果已保存: {filepath}")
        return str(filepath)

    # ─── 辅助方法 ───

    def _compute_spearman(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """计算 Spearman 秩相关系数"""
        from scipy.stats import spearmanr

        # 对齐长度
        min_len = min(len(preds), len(targets))
        preds = preds[:min_len]
        targets = targets[:min_len]

        if min_len < 3:
            return 0.0

        try:
            corr, _ = spearmanr(preds, targets)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _default_result(self) -> Dict[str, Any]:
        """返回默认参数结果"""
        return {
            "best_params": {
                "T": self.T_values[len(self.T_values) // 2],
                "top_p": self.top_p_values[len(self.top_p_values) // 2],
                "lookback": self.lookback_values[0],
            },
            "best_spearman": 0.0,
            "top_k": [],
            "all_results": [],
        }
