"""
Kronos 指数/ETF 择时策略系统 — 工具函数

日期处理、股票代码转换、随机种子等通用工具。
"""

import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def set_seed(seed: int = 42):
    """固定所有随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device(device_str: str = "auto") -> torch.device:
    """
    自动检测可用设备。

    Args:
        device_str: "auto" | "cuda" | "cpu"

    Returns:
        torch.device
    """
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA不可用，但device被设为cuda")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def stock_code_to_akshare(code: str) -> str:
    """
    将纯数字股票代码转换为AKShare格式。

    Args:
        code: 如 "000852", "512100", "399006"

    Returns:
        AKShare格式: "sh000852", "sh512100", "sz399006"
    """
    code = str(code).zfill(6)
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    elif code.startswith(("0", "3", "4", "8")):
        return f"sz{code}"
    else:
        return f"sh{code}"


def akshare_code_to_standard(symbol: str) -> str:
    """
    将AKShare格式转回纯数字代码。

    Args:
        symbol: "sh000852" 或 "sz399006"

    Returns:
        "000852" 或 "399006"
    """
    return symbol[2:]


def is_trading_day(date: pd.Timestamp) -> bool:
    """判断是否为工作日（不精确，仅排除周末）"""
    return date.dayofweek < 5


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """
    获取日期范围内的交易日（排除周末，不含节假日）。

    Args:
        start: 起始日期 "YYYY-MM-DD"
        end: 结束日期 "YYYY-MM-DD"

    Returns:
        DatetimeIndex of trading days
    """
    dates = pd.date_range(start=start, end=end, freq="B")  # Business day
    return dates


def format_date(date) -> str:
    """将各种日期格式统一为 ISO 8601 (YYYY-MM-DD)"""
    if isinstance(date, (pd.Timestamp, datetime)):
        return date.strftime("%Y-%m-%d")
    if isinstance(date, str):
        return date[:10]
    return str(date)


def now_iso() -> str:
    """返回当前时间的 ISO 8601 字符串"""
    return datetime.now().isoformat(timespec="seconds")


def split_date_range(
    start: str, end: str, train_months: int = 3, pred_months: int = 1, step_months: int = 1
) -> list[dict]:
    """
    月度滚动窗口切分。

    Args:
        start: 整体起始日期
        end: 整体结束日期
        train_months: 训练窗口月数
        pred_months: 预测窗口月数
        step_months: 滑动步长月数

    Returns:
        [{window_id: "2021-07", train_start, train_end, pred_start, pred_end}, ...]
    """
    from dateutil.relativedelta import relativedelta

    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    windows = []

    while cur + relativedelta(months=train_months + pred_months) <= end_ts:
        train_start = cur
        train_end = cur + relativedelta(months=train_months, days=-1)
        pred_start = train_end + timedelta(days=1)
        pred_end = pred_start + relativedelta(months=pred_months, days=-1)

        # window_id: 训练窗口最后一月
        window_id = train_end.strftime("%Y-%m")

        windows.append({
            "window_id": window_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "pred_start": pred_start.strftime("%Y-%m-%d"),
            "pred_end": pred_end.strftime("%Y-%m-%d"),
        })

        cur += relativedelta(months=step_months)

    return windows
