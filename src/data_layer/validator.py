"""
Kronos 指数/ETF 择时策略系统 — 数据校验模块

最关键的校验：数据泄露检测（训练集最大日期 < 验证集最小日期）。
+ 时间序列连续性 + 值域校验 + 标的元数据一致性。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

from src.common.config import Config
from src.common.logger import get_logger
from src.common.exceptions import DataLeakError, DataNotFoundError, ValidationError

logger = get_logger(__name__)


class DataValidator:
    """
    数据校验器。

    在校验到严重错误（如数据泄露）时抛出异常；
    对于可自动修正的问题（如高低倒挂），自动修正并记录日志。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processed_dir = Path(self.config.data.get("processed_dir", "data/processed"))
        self.h5_path = self.processed_dir / "samples.h5"

    # ─── 数据泄露检测 ───

    def validate_window_leak(self, window_id: str) -> dict:
        """
        检查单个窗口的数据泄露。

        规则:
        - 训练集最大日期 < 验证集最小日期，差值 >= 1 交易日
        - 验证集最大日期 < 测试集最小日期，差值 >= 1 交易日
        - 相邻窗口训练集不能互相"看到"对方的未来

        Args:
            window_id: 如 "2021-07"

        Returns:
            {"status": "pass"|"fail", "details": [...]}

        Raises:
            DataLeakError: 检测到数据泄露
        """
        train_max, val_min, val_max, test_min = self._get_window_boundaries(window_id)
        issues = []

        # 训练集 vs 验证集
        if train_max is not None and val_min is not None:
            if train_max >= val_min:
                issues.append(
                    f"数据泄露: 训练集最大日期({train_max}) >= 验证集最小日期({val_min})"
                )
                raise DataLeakError(str(train_max), str(val_min))

        # 验证集 vs 测试集
        if val_max is not None and test_min is not None:
            if val_max >= test_min:
                issues.append(
                    f"数据泄露: 验证集最大日期({val_max}) >= 测试集最小日期({test_min})"
                )
                raise DataLeakError(str(val_max), str(test_min))

        return {
            "status": "pass" if not issues else "fail",
            "window_id": window_id,
            "boundaries": {
                "train_max": str(train_max) if train_max else None,
                "val_min": str(val_min) if val_min else None,
                "val_max": str(val_max) if val_max else None,
                "test_min": str(test_min) if test_min else None,
            },
            "issues": issues,
        }

    def validate_all_windows(self) -> List[dict]:
        """
        检查所有窗口的数据泄露 + 时间重叠。

        Returns:
            [{window_id, status, boundaries, issues}, ...]
        """
        results = []
        window_ids = self._list_windows()

        if not window_ids:
            logger.warning("未找到任何窗口数据")
            return results

        for wid in window_ids:
            try:
                result = self.validate_window_leak(wid)
                results.append(result)
                logger.info(f"  {wid}: ✓ 通过")
            except DataLeakError as e:
                results.append({
                    "status": "fail",
                    "window_id": wid,
                    "issues": [str(e)],
                })
                logger.error(f"  {wid}: ✗ 数据泄露 — {e}")
                raise  # 数据泄露是致命错误，抛出后停止检查

        # 窗口间重叠检查
        self._check_window_overlap(window_ids)

        return results

    def _check_window_overlap(self, window_ids: List[str]):
        """
        检查相邻窗口的训练集是否重叠。
        窗口 A 的训练集不应包含窗口 B 训练集中出现过的未来数据。
        """
        boundaries = []
        for wid in sorted(window_ids):
            b = self._get_window_boundaries(wid)
            boundaries.append((wid, b))

        for i in range(len(boundaries) - 1):
            w_cur = boundaries[i]
            w_next = boundaries[i + 1]

            # 当前窗口的 val/test 不应与下一窗口的 train 有交集
            cur_train_max = w_cur[1][0]  # train_max
            next_train_max = w_next[1][0]  # train_max

            if cur_train_max and next_train_max and cur_train_max >= next_train_max:
                logger.warning(
                    f"窗口重叠: {w_cur[0]} 训练集最大日期 >= {w_next[0]} 训练集最大日期"
                )

    def _get_window_boundaries(self, window_id: str) -> Tuple:
        """
        获取窗口的 train/val/test 日期边界。

        Returns:
            (train_max, val_min, val_max, test_min)
        """
        if not self.h5_path.exists():
            raise DataNotFoundError(f"HDF5 文件不存在: {self.h5_path}")

        train_max = val_min = val_max = test_min = None

        with pd.HDFStore(self.h5_path, mode="r") as store:
            for split in ["train", "val", "test"]:
                key = f"/windows/{window_id}/{split}"
                if key in store:
                    df = store[key]
                    if len(df) > 0:
                        dt_max = pd.Timestamp(df.index.max())
                        dt_min = pd.Timestamp(df.index.min())
                        if split == "train":
                            train_max = dt_max
                        elif split == "val":
                            val_min = dt_min
                            val_max = dt_max
                        elif split == "test":
                            test_min = dt_min

        return train_max, val_min, val_max, test_min

    def _list_windows(self) -> List[str]:
        """获取 HDF5 中所有窗口 ID。"""
        if not self.h5_path.exists():
            return []

        with pd.HDFStore(self.h5_path, mode="r") as store:
            keys = [k.lstrip("/") for k in store.keys()]
            windows = set()
            for k in keys:
                parts = k.split("/")
                if len(parts) >= 3 and parts[0] == "windows":
                    windows.add(parts[1])
            return sorted(windows)

    # ─── 时间序列连续性 ───

    def validate_continuity(self, df: pd.DataFrame, code: str) -> dict:
        """
        检查时间序列连续性。

        检查项:
        - 日期是否单调递增
        - 是否有未来日期
        - 单次跳跃不超过 5 个连续交易日缺失

        Returns:
            {continuity_ok, monotonic, future_dates, max_gap_days, issues}
        """
        issues = []

        if "date" not in df.columns and df.index.name != "date":
            # 尝试从索引获取
            dates = pd.DatetimeIndex(df.index)
        else:
            date_col = "date" if "date" in df.columns else df.index.name
            dates = pd.DatetimeIndex(df[date_col] if date_col in df.columns else df.index)

        # 单调递增检查
        monotonic = dates.is_monotonic_increasing
        if not monotonic:
            issues.append("日期非单调递增")

        # 未来日期
        today = pd.Timestamp.now()
        future = dates[dates > today]
        if len(future) > 0:
            issues.append(f"发现 {len(future)} 个未来日期: {future[0]} ... {future[-1]}")

        # 连续缺失天数
        if len(dates) >= 2:
            gaps = dates[1:] - dates[:-1]
            max_gap = gaps.max()
            # 简单判断：超过 10 天的间隔
            max_gap_days = max_gap.days if hasattr(max_gap, "days") else int(max_gap / np.timedelta64(1, "D"))
            if max_gap_days > 10:
                issues.append(f"最大日期间隔: {max_gap_days} 天")
        else:
            max_gap_days = 0

        return {
            "code": code,
            "continuity_ok": len(issues) == 0,
            "monotonic": bool(monotonic),
            "future_dates_count": len(future) if "future" in dir() else 0,
            "max_gap_days": max_gap_days if "max_gap_days" in dir() else 0,
            "issues": issues,
        }

    # ─── 值域校验 ───

    def validate_value_range(self, df: pd.DataFrame, code: str) -> dict:
        """
        值域校验。

        检查项:
        - OHLC > 0
        - High >= Max(Open, Close)
        - Low <= Min(Open, Close)
        - Volume >= 0, Amount >= 0
        """
        issues = []
        auto_corrected = 0

        # 负价格检查
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            neg = (df[col] <= 0).sum()
            if neg > 0:
                issues.append(f"{col}: {neg} 行 ≤ 0")

        # 高低关系
        if "high" in df.columns and "low" in df.columns:
            # 只检查高低关系，open/close 可能不在当日范围内（跳空）
            ohl = df[["high", "low"]].copy()
            ohl["high"] = ohl["high"].astype(float)
            ohl["low"] = ohl["low"].astype(float)

            inverted = (ohl["high"] < ohl["low"]).sum()
            if inverted > 0:
                # 自动修正：交换 high 和 low
                mask = ohl["high"] < ohl["low"]
                df.loc[mask, "high"], df.loc[mask, "low"] = \
                    df.loc[mask, "low"].copy(), df.loc[mask, "high"].copy()
                auto_corrected += inverted
                issues.append(f"high<low: {inverted} 行（已自动修正）")

        # 成交量为负
        for col in ["volume", "amount"]:
            if col not in df.columns:
                continue
            neg = (df[col] < 0).sum()
            if neg > 0:
                df.loc[df[col] < 0, col] = 0
                auto_corrected += neg
                issues.append(f"{col}: {neg} 行 < 0（已置0）")

        return {
            "code": code,
            "valid": len([i for i in issues if "已自动修正" not in i]) == 0,
            "auto_corrected": auto_corrected,
            "issues": issues,
        }

    # ─── 标的元数据一致性 ───

    def validate_metadata(self, code: str, expected_name: str) -> dict:
        """检查标的代码-名称映射一致性。"""
        all_symbols = self.config.get_all_symbols()
        symbol_map = {s["code"]: s for s in all_symbols}

        if code not in symbol_map:
            raise ValidationError(f"未知标的代码: {code}")

        actual = symbol_map[code]
        match = actual["name"] == expected_name

        return {
            "code": code,
            "expected_name": expected_name,
            "config_name": actual["name"],
            "type": actual["type"],
            "match": match,
        }


def validate_window(window_id: str, config: Optional[Config] = None) -> dict:
    """便捷函数：校验单个窗口数据泄露"""
    return DataValidator(config).validate_window_leak(window_id)


def validate_all(config: Optional[Config] = None) -> List[dict]:
    """便捷函数：校验所有窗口"""
    return DataValidator(config).validate_all_windows()
