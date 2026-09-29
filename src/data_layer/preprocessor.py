"""
Kronos 指数/ETF 择时策略系统 — 数据预处理模块

原始 CSV → 清洗 → 对齐 → 标准化 → HDF5 样本。

流程:
1. 批量加载 data/raw/ 中的 CSV
2. 按 date 对齐 → multi-index columns (symbol × OHLCV)
3. 缺失值处理 (前向填充 ≤3天)
4. 异常值检测与标记
5. Z-score 标准化 (按标的+按训练集拟合)
6. 月度滚动窗口切片 → HDF5
7. 生成数据质量报告
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger
from src.common.utils import format_date, split_date_range

logger = get_logger(__name__)


class Preprocessor:
    """
    数据预处理流水线。

    从 data/raw/ 读取原始 CSV，输出标准化后的 HDF5 样本文件。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.raw_dir = Path(self.config.data.get("raw_dir", "data/raw"))
        self.processed_dir = Path(self.config.data.get("processed_dir", "data/processed"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = Path(self.config.data.get("outputs_dir", "data/outputs"))
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # 输出路径
        self.h5_path = self.processed_dir / "samples.h5"
        self.scaler_params_path = self.processed_dir / "scaler_params.json"
        self.anomaly_log_path = self.processed_dir / "anomaly_log.json"
        self.quality_report_path = self.processed_dir / "quality_report.json"

        # 运行中状态
        self._scaler_params: dict = {}   # {symbol: {mean, std}}
        self._anomalies: list = []       # 异常值记录
        self._raw_dfs: Dict[str, pd.DataFrame] = {}  # 原始数据缓存

    # ─── 公开方法 ───

    def run(
        self,
        symbols: Optional[List[str]] = None,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
    ) -> str:
        """
        执行完整预处理流水线。

        Args:
            symbols: 要处理的标的代码列表，None=全部
            train_start: 训练集起始日期（用于 scaler 拟合），None=使用配置
            train_end:   训练集结束日期，None=使用 finetune_train_end

        Returns:
            HDF5 文件路径
        """
        logger.info("=" * 50)
        logger.info("开始数据预处理流水线")

        # 1. 加载原始 CSV
        logger.info("[1/6] 加载原始 CSV")
        self._load_all_csv(symbols)

        if not self._raw_dfs:
            raise RuntimeError("未加载到任何数据，请先运行 download.py")

        # 2. 信息打印
        for code, df in self._raw_dfs.items():
            logger.info(f"  {code}: {len(df)}行, {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

        # 3. 对齐到统一日期索引
        logger.info("[2/6] 按日期对齐")
        aligned = self._align_by_date()

        # 4. 缺失值处理
        logger.info("[3/6] 处理缺失值")
        aligned = self._fill_missing(aligned)

        # 5. 异常值检测
        logger.info("[4/6] 检测异常值")
        self._detect_anomalies(aligned)
        self._save_anomaly_log()

        # 6. Z-score 标准化
        logger.info("[5/6] Z-score 标准化")
        train_start = train_start or self.config.data.get("finetune_train_start", "2010-01-01")
        train_end = train_end or self.config.data.get("finetune_train_end", "2022-12-31")
        normalized, scaler_params = self._normalize(aligned, train_start, train_end)
        self._save_scaler_params(scaler_params)

        # 7. 窗口切片 → HDF5
        logger.info("[6/6] 窗口切片 → HDF5")
        self._slice_and_save(normalized)

        # 8. 质量报告
        self._generate_quality_report(aligned)

        logger.info(f"预处理完成: {self.h5_path}")
        return str(self.h5_path)

    # ─── 步骤1: 加载 CSV ───

    def _load_all_csv(self, symbols: Optional[List[str]] = None):
        """
        从 raw_dir 批量读取所有 CSV 到 self._raw_dfs。

        文件命名格式: {code}_{name}_daily.csv
        """
        all_targets = self.config.get_all_symbols()
        if symbols:
            all_targets = [t for t in all_targets if t["code"] in symbols]

        for sym in all_targets:
            code = sym["code"]
            name = sym["name"]
            filename = f"{code}_{name}_daily.csv"
            filepath = self.raw_dir / filename

            if not filepath.exists():
                logger.warning(f"  跳过 {code}: 文件不存在 ({filepath})")
                continue

            try:
                df = pd.read_csv(filepath, encoding="utf-8")
                # 标准化列名
                df.columns = [c.strip().lower() for c in df.columns]

                # 确保日期列
                if "date" not in df.columns:
                    logger.warning(f"  {code}: 缺少 'date' 列，跳过")
                    continue

                # 转换日期类型
                df["date"] = pd.to_datetime(df["date"])
                # 按日期排序 + 去重
                df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

                self._raw_dfs[code] = df
                logger.debug(f"  已加载 {code} {name}: {len(df)} 行")

            except Exception as e:
                logger.error(f"  加载 {code} 失败: {e}")

    # ─── 步骤2: 日期对齐 ───

    def _align_by_date(self) -> pd.DataFrame:
        """
        将所有标的按 date 对齐到统一索引。

        Returns:
            DataFrame，列为 multi-index: (symbol, ohlcv_field)
            如: ('000852', 'close'), ('512100', 'volume'), ...
        """
        # 为每个标的构建 (date → 多列) 子DataFrame
        frames = {}
        for code, df in self._raw_dfs.items():
            # 设置 date 为索引
            df_indexed = df.set_index("date")
            # 重命名列：添加标的代码前缀 → 用于 multi-index
            df_indexed.columns = pd.MultiIndex.from_product([[code], df_indexed.columns])
            frames[code] = df_indexed

        # 外连接：保留所有日期
        aligned = pd.concat(frames.values(), axis=1, join="outer")
        aligned = aligned.sort_index()

        logger.info(f"  对齐后: {len(aligned)} 个交易日, {len(frames)} 个标的")
        return aligned

    # ─── 步骤3: 缺失值处理 ───

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值。

        规则:
        - 单日缺失: ffill, 最多连续 fill 3 天
        - 连续缺失 > 3 天: 保留 NaN, 标记到质量报告
        - volume/amount 缺失填 0
        """
        df = df.copy()

        for code in df.columns.get_level_values(0).unique():
            for field in ["open", "high", "low", "close"]:
                col = (code, field)
                if col not in df.columns:
                    continue
                # 前向填充，限制连续 3 天
                df[col] = df[col].ffill(limit=3)

            # volume/amount: NaN → 0
            for field in ["volume", "amount"]:
                col = (code, field)
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        # 统计剩余缺失
        total_missing = df.isna().sum().sum()
        if total_missing > 0:
            logger.info(f"  剩余缺失值: {total_missing} (连续缺失>3天)")

        return df

    # ─── 步骤4: 异常值检测 ───

    def _detect_anomalies(self, df: pd.DataFrame):
        """
        检测并记录异常值。

        检测项:
        - 日收益率 > 11% 或 < -11%（涨停/跌停边界外）
        - Low > High（高低倒挂）
        - 价格连续相同但 volume > 0
        """
        self._anomalies = []

        for code in df.columns.get_level_values(0).unique():
            close = df.get((code, "close"))
            high = df.get((code, "high"))
            low = df.get((code, "low"))
            volume = df.get((code, "volume"))

            if close is None:
                continue

            # 日收益率异常
            returns = close.pct_change()
            extreme = returns[(returns > 0.11) | (returns < -0.11)]
            for date_idx in extreme.index:
                self._anomalies.append({
                    "code": code,
                    "date": format_date(date_idx),
                    "type": "extreme_return",
                    "value": float(returns[date_idx]),
                })

            # 高低倒挂
            if high is not None and low is not None:
                inverted = high < low
                for date_idx in inverted[inverted].index:
                    self._anomalies.append({
                        "code": code,
                        "date": format_date(date_idx),
                        "type": "high_low_inverted",
                        "high": float(high[date_idx]),
                        "low": float(low[date_idx]),
                    })

            # 价格连续相同且 volume > 0（疑似数据错误）
            if close is not None and volume is not None:
                same_price = (close.diff() == 0) & (volume > 0)
                for date_idx in same_price[same_price].index:
                    self._anomalies.append({
                        "code": code,
                        "date": format_date(date_idx),
                        "type": "stale_price_with_volume",
                        "close": float(close[date_idx]),
                        "volume": float(volume[date_idx]),
                    })

        logger.info(f"  检测到 {len(self._anomalies)} 条异常")

    def _save_anomaly_log(self):
        """保存异常记录 JSON。"""
        with open(self.anomaly_log_path, "w", encoding="utf-8") as f:
            json.dump(self._anomalies, f, ensure_ascii=False, indent=2)

    # ─── 步骤5: Z-score 标准化 ───

    def _normalize(
        self, df: pd.DataFrame, train_start: str, train_end: str
    ) -> Tuple[pd.DataFrame, dict]:
        """
        按标的 Z-score 标准化。

        **关键约束**: mean/std 仅在训练集上拟合，防止数据泄露。

        Args:
            df: 对齐后的 multi-index DataFrame
            train_start, train_end: 训练集日期范围（用于拟合 scaler）

        Returns:
            (normalized_df, scaler_params)
        """
        df = df.copy()
        scaler_params = {}

        # 获取训练集区间
        train_mask = (df.index >= train_start) & (df.index <= train_end)

        codes = df.columns.get_level_values(0).unique()
        for code in codes:
            scaler_params[code] = {}
            for field in ["open", "high", "low", "close", "volume", "amount"]:
                col = (code, field)
                if col not in df.columns:
                    continue

                series = df[col].astype(float)
                train_data = series[train_mask].dropna()

                if len(train_data) == 0:
                    raise ValueError(
                        f"{code}/{field}: training interval [{train_start}, {train_end}] has no valid data; "
                        "refusing to fit on future observations"
                    )

                mean_val = float(train_data.mean())
                std_val = float(train_data.std())

                # std=0 的特殊处理（如停牌期间）
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0

                scaler_params[code][field] = {"mean": mean_val, "std": std_val}

                # 应用标准化
                df[col] = (series - mean_val) / std_val

        logger.info(
            f"  标准化完成: {len(scaler_params)} 个标的, "
            f"训练集区间 [{train_start}, {train_end}]"
        )
        return df, scaler_params

    def _save_scaler_params(self, params: dict):
        """保存标准化参数到 JSON 文件（预测阶段反标准化需要）。"""
        with open(self.scaler_params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    # ─── 步骤6: 窗口切片 → HDF5 ───

    def _slice_and_save(self, df: pd.DataFrame):
        """
        按月滚动窗口切片并存入 HDF5。

        Group 结构:
          /windows/{window_id}/train  → 训练集
          /windows/{window_id}/val    → 验证集
          /windows/{window_id}/test   → 测试集
          /metadata                  → 元信息
        """
        import tables as tb  # PyTables

        # 读取滚动窗口配置（月为单位）。默认 3 个月训练窗、预测/步长各 1 个月，
        # 与 CLAUDE.md「3个月窗口，预测1个月，步长1个月」一致。
        train_months = self.config.get("param_search", "train_window_months", default=3)
        pred_months = self.config.get("param_search", "test_window_months", default=1)
        step_months = self.config.get("param_search", "step_months", default=1)

        # 获取实际数据起止日期
        start = format_date(df.index.min())
        end = format_date(df.index.max())

        windows = split_date_range(
            start, end,
            train_months=train_months,
            pred_months=pred_months,
            step_months=step_months,
        )

        logger.info(f"  生成 {len(windows)} 个滚动窗口")

        # 确保 HDF5 文件存在
        # 使用 PyTables 写入
        with pd.HDFStore(self.h5_path, mode="w", complevel=5, complib="zlib") as store:
            valid_windows = 0

            for w in windows:
                wid = w["window_id"]
                train_mask = (df.index >= w["train_start"]) & (df.index <= w["train_end"])
                val_mask = (df.index >= w["pred_start"]) & (df.index <= w["pred_end"])

                # 测试集 = 验证集之后 1 个月（若数据够）
                # 简单处理：下一个窗口的训练集起始后一天作为测试集
                # 这里简化为 val 之后 1 个月
                test_end_dt = pd.Timestamp(w["pred_end"]) + pd.DateOffset(months=1)
                test_start_dt = pd.Timestamp(w["pred_end"]) + pd.DateOffset(days=1)
                test_mask = (df.index >= test_start_dt.strftime("%Y-%m-%d")) & \
                            (df.index <= test_end_dt.strftime("%Y-%m-%d"))

                train_df = df[train_mask].copy()
                val_df = df[val_mask].copy()
                test_df = df[test_mask].copy()

                # 至少需要一定行数
                if len(train_df) < 10:
                    logger.debug(f"  跳过 {wid}: 训练集仅 {len(train_df)} 行")
                    continue

                # 写入 HDF5
                store.put(f"/windows/{wid}/train", train_df, format="table")
                if len(val_df) > 0:
                    store.put(f"/windows/{wid}/val", val_df, format="table")
                if len(test_df) > 0:
                    store.put(f"/windows/{wid}/test", test_df, format="table")

                valid_windows += 1
                logger.debug(
                    f"  窗口 {wid}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
                )

            # 写入元信息
            meta = pd.DataFrame([{
                "created_at": datetime.now().isoformat(),
                "total_windows": valid_windows,
                "date_range": f"{start} → {end}",
                "num_symbols": len(df.columns.get_level_values(0).unique()),
            }])
            store.put("/metadata", meta, format="table")

        logger.info(f"  已写入 {valid_windows} 个窗口到 {self.h5_path}")

        # 同时按标的保存原始数据（用于后续直接查询）
        self._save_raw_by_symbol(df)

    def _save_raw_by_symbol(self, df: pd.DataFrame):
        """按标的分别保存标准化后的数据到 HDF5（方便单标的查询）。"""
        with pd.HDFStore(self.h5_path, mode="r+", complevel=5, complib="zlib") as store:
            for code in df.columns.get_level_values(0).unique():
                # 提取单个标的的列
                symbol_cols = [c for c in df.columns if c[0] == code]
                symbol_df = df[symbol_cols].copy()
                # 将 multi-index 展平
                symbol_df.columns = [c[1] for c in symbol_df.columns]
                store.put(f"/raw/{code}", symbol_df, format="table")

    # ─── 步骤7: 质量报告 ───

    def _generate_quality_report(self, df: pd.DataFrame):
        """
        生成数据质量报告 JSON。

        每标的: 行数, 缺失率, 异常值数量
        整体: 总行数, 标的数, 日期范围, 数据完整性评分 (0-100)
        """
        codes = df.columns.get_level_values(0).unique()

        per_symbol = {}
        total_rows = 0
        total_missing = 0
        total_cells = 0

        for code in codes:
            symbol_cols = [c for c in df.columns if c[0] == code]
            symbol_df = df[symbol_cols]
            n_rows = len(symbol_df)
            n_missing = symbol_df.isna().sum().sum()
            n_cells = symbol_df.size
            n_anomalies = sum(1 for a in self._anomalies if a["code"] == code)

            total_rows += n_rows
            total_missing += n_missing
            total_cells += n_cells

            per_symbol[code] = {
                "rows": n_rows,
                "missing_rate": round(n_missing / n_cells * 100, 2) if n_cells > 0 else 0,
                "anomaly_count": n_anomalies,
            }

        # 完整性评分
        completeness = 100.0
        if total_cells > 0:
            completeness = max(0, 100 - (total_missing / total_cells * 100))

        report = {
            "generated_at": datetime.now().isoformat(),
            "overall": {
                "total_symbols": len(codes),
                "total_rows": len(df),
                "date_range": f"{format_date(df.index.min())} → {format_date(df.index.max())}",
                "total_missing_rate": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
                "total_anomalies": len(self._anomalies),
                "completeness_score": round(completeness, 1),
            },
            "per_symbol": per_symbol,
        }

        with open(self.quality_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"  质量报告已保存: {self.quality_report_path}")
        logger.info(f"  完整性评分: {completeness:.1f}/100")


def preprocess(config: Optional[Config] = None) -> str:
    """便捷函数：运行完整预处理流水线 → 返回 HDF5 路径"""
    return Preprocessor(config).run()
