"""
Kronos 指数/ETF 择时策略系统 — 数据下载模块

从 AKShare 下载指数/ETF 日线 OHLCV 数据。
支持三种模式：full（全量）、incremental（增量）、single（单标的调试）。

接口适配：
- 指数: ak.index_zh_a_hist()
- ETF:  ak.fund_etf_hist_em()
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import akshare as ak

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger
from src.common.utils import stock_code_to_akshare, now_iso, format_date
from src.common.exceptions import DataNotFoundError

logger = get_logger(__name__)

# ─── 列名映射 ───
# AKShare 不同接口返回的中文列名 → 统一 snake_case
_INDEX_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

_ETF_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 AKShare 中文列名映射为 snake_case 列名。"""
    # 合并两种映射
    all_maps = {**_INDEX_COLUMN_MAP, **_ETF_COLUMN_MAP}
    # 只保留实际存在的列
    rename = {k: v for k, v in all_maps.items() if k in df.columns}
    return df.rename(columns=rename)


# 标准输出列（保证至少这些列存在）
_STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


class DataDownloader:
    """
    指数/ETF 日线数据下载器。

    使用 AKShare 作为数据源，支持指数和 ETF 两种标的类型。
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: Config 实例，None 则自动加载
        """
        self.config = config or Config()
        self.raw_dir = Path(self.config.data.get("raw_dir", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # 请求间隔（秒）
        self.request_interval = 2.0
        self._last_request_time = 0.0

        # 下载元信息文件
        self._meta_path = self.raw_dir / "download_meta.json"

    # ─── 公开方法 ───

    def download(
        self,
        mode: str = "full",
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        """
        下载入口。

        Args:
            mode: "full" | "incremental" | "single"
            symbols: 标的代码列表。None→使用配置中的全部12个。single模式下必传。
            start: 起始日期 YYYY-MM-DD。None→使用配置文件值。
            end:   结束日期 YYYY-MM-DD。None→今天。

        Returns:
            {success: [...], failed: [...], meta: {...}}
        """
        # 解析标的列表
        if mode == "single":
            if not symbols or len(symbols) != 1:
                raise ValueError("single 模式需要传入恰好一个标的代码")
            target_symbols = self._resolve_targets(symbols)
        elif symbols:
            target_symbols = self._resolve_targets(symbols)
        else:
            target_symbols = self.config.get_all_symbols()

        # 解析日期范围
        if start is None:
            start = self.config.data.get("download_start", "2010-01-01")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # 增量模式：读取上次下载日期
        if mode == "incremental":
            last_date = self._get_last_download_date()
            if last_date:
                start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"增量下载：{start} → {end}")
            else:
                logger.warning("未找到上次下载记录，切换为全量下载")
                start = self.config.data.get("download_start", "2010-01-01")

        logger.info(
            f"开始下载: mode={mode}, symbols={len(target_symbols)}个, "
            f"范围={start}→{end}"
        )

        success = []
        failed = []

        for i, sym in enumerate(target_symbols):
            code = sym["code"]
            name = sym["name"]
            stype = sym["type"]

            try:
                logger.info(f"[{i+1}/{len(target_symbols)}] 下载 {code} {name} ({stype})")
                df = self._download_single(code, name, stype, start, end)
                self._validate_data(df, code)
                self._save_csv(df, code, name)
                success.append({"code": code, "name": name, "type": stype, "rows": len(df)})
                logger.info(f"  ✓ {code} {name}: {len(df)} 行")

            except Exception as e:
                logger.error(f"  ✗ {code} {name}: {e}")
                failed.append({"code": code, "name": name, "type": stype, "error": str(e)})

        # 保存下载元信息
        meta = {
            "download_time": now_iso(),
            "mode": mode,
            "date_range": {"start": start, "end": end},
            "total_symbols": len(target_symbols),
            "success_count": len(success),
            "failed_count": len(failed),
            "success": success,
            "failed": failed,
        }
        self._save_meta(meta)

        logger.info(
            f"下载完成: 成功 {len(success)}/{len(target_symbols)}, "
            f"失败 {len(failed)}/{len(target_symbols)}"
        )

        return meta

    # ─── 内部方法 ───

    def _download_single(
        self, code: str, name: str, stype: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        下载单个标的日线数据，含重试逻辑。

        Args:
            code:  纯数字代码（如 "000852"）
            name:  标的名称
            stype: "index" | "etf"
            start: YYYY-MM-DD
            end:   YYYY-MM-DD

        Returns:
            标准化列名的 DataFrame
        """
        self._rate_limit()

        if stype == "index":
            return self._download_index(code, name, start, end)
        elif stype == "etf":
            return self._download_etf(code, name, start, end)
        else:
            raise ValueError(f"未知标的类型: {stype} (code={code})")

    def _download_index(self, code: str, name: str, start: str, end: str) -> pd.DataFrame:
        """通过 AKShare 下载指数日线数据。"""
        # Index exchange rules differ from stock rules: CSI codes such as
        # 000852 are Shanghai indices even though the same bare stock code is
        # listed in Shenzhen. 399xxx indices are Shenzhen.
        akshare_symbol = f"sz{code}" if code.startswith("399") else f"sh{code}"

        try:
            # Sina and Eastmoney carry the same raw index level. Sina is the
            # primary source because some cloud egress IPs are blocked by EM.
            df = ak.stock_zh_index_daily(symbol=akshare_symbol)
            df = self._filter_date_range(df, start, end)
        except Exception as sina_error:
            logger.warning("  %s 新浪指数接口失败，切换东方财富: %s", code, sina_error)

            def _fetch():
                return ak.index_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                )

            df = self._retry_with_backoff(_fetch, code)
        df = _rename_columns(df)
        return df

    def _download_etf(self, code: str, name: str, start: str, end: str) -> pd.DataFrame:
        """通过 AKShare 下载 ETF 日线数据。"""
        akshare_symbol = stock_code_to_akshare(code)
        try:
            df = ak.fund_etf_hist_sina(symbol=akshare_symbol)
            df = self._filter_date_range(df, start, end)
        except Exception as sina_error:
            logger.warning("  %s 新浪ETF接口失败，切换东方财富: %s", code, sina_error)

            def _fetch():
                return ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    # Execution and backtests require contemporaneously
                    # tradable prices. Forward adjustment revises history.
                    adjust="",
                )

            df = self._retry_with_backoff(_fetch, code)
        df = _rename_columns(df)
        return df

    @staticmethod
    def _filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        if "date" not in df.columns:
            return df
        dates = pd.to_datetime(df["date"], errors="coerce")
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        return df.loc[dates.between(start_ts, end_ts)].copy()

    def _retry_with_backoff(self, func, code: str, max_retries: int = 3) -> pd.DataFrame:
        """
        带指数退避的重试机制。

        Args:
            func: 无参可调用对象，返回 DataFrame
            code: 标的代码（用于错误日志）
            max_retries: 最大重试次数

        Returns:
            DataFrame

        Raises:
            DataNotFoundError: 3次重试后仍失败
        """
        delays = [5, 15, 45]  # 指数退避间隔

        for attempt in range(max_retries):
            try:
                df = func()
                if df is None or len(df) == 0:
                    raise DataNotFoundError(f"{code}: AKShare 返回空数据")
                return df
            except DataNotFoundError:
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = delays[attempt]
                    logger.warning(
                        f"  {code} 第{attempt+1}次失败 ({e})，{delay}s 后重试..."
                    )
                    time.sleep(delay)
                else:
                    raise DataNotFoundError(
                        f"{code}: {max_retries}次重试后仍失败: {e}"
                    ) from e

    def _validate_data(self, df: pd.DataFrame, code: str):
        """
        数据完整性校验。

        检查项：
        - 行数 > 0
        - 不全是 NaN
        - 日期无重复
        - OHLC 值域合理

        Raises:
            DataNotFoundError: 数据验证不通过
        """
        if len(df) == 0:
            raise DataNotFoundError(f"{code}: 下载数据为空")

        # 检查必要列
        for col in ["date", "close"]:
            if col not in df.columns:
                raise DataNotFoundError(f"{code}: 缺少必要列 '{col}'")

        # 检查日期重复
        if "date" in df.columns:
            dupes = df["date"].duplicated().sum()
            if dupes > 0:
                logger.warning(f"  {code}: 发现 {dupes} 条重复日期，已去重")
                df.drop_duplicates(subset=["date"], inplace=True, keep="first")

        # 价格列不应全为 NaN
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if price_cols and df[price_cols].isna().all().all():
            raise DataNotFoundError(f"{code}: 价格数据全部为 NaN")

        # 值域检查（warning 级别，不阻断）
        if "high" in df.columns and "low" in df.columns:
            inverted = (df["high"] < df["low"]).sum()
            if inverted > 0:
                logger.warning(f"  {code}: {inverted} 行 High < Low")

    def _save_csv(self, df: pd.DataFrame, code: str, name: str):
        """保存为 CSV 文件。"""
        filename = f"{code}_{name}_daily.csv"
        filepath = self.raw_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.debug(f"  已保存: {filepath}")

    def _rate_limit(self):
        """请求间隔控制，避免 AKShare 限流。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    # ─── 元信息管理 ───

    def _save_meta(self, meta: dict):
        """保存下载元信息到 JSON 文件。"""
        # 追加模式（保留历史记录）
        history = []
        if self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    history = existing if isinstance(existing, list) else [existing]
            except (json.JSONDecodeError, IOError):
                history = []

        history.append(meta)

        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _get_last_download_date(self) -> Optional[str]:
        """从元信息文件中读取上次全量/增量下载的结束日期。"""
        if not self._meta_path.exists():
            return None

        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history and isinstance(history, list):
                last = history[-1]
                date_range = last.get("date_range", {})
                return date_range.get("end")
        except (json.JSONDecodeError, IOError, KeyError):
            pass

        return None

    def _resolve_targets(self, codes: List[str]) -> List[dict]:
        """根据代码列表查找完整的标的配置信息。"""
        all_symbols = self.config.get_all_symbols()
        symbol_map = {s["code"]: s for s in all_symbols}

        targets = []
        for code in codes:
            if code in symbol_map:
                targets.append(symbol_map[code])
            else:
                logger.warning(f"未知标的代码: {code}，已跳过")
        return targets


def download_full(config: Optional[Config] = None) -> dict:
    """便捷函数：全量下载所有标的"""
    return DataDownloader(config).download(mode="full")


def download_single(code: str, config: Optional[Config] = None) -> dict:
    """便捷函数：下载单个标的"""
    return DataDownloader(config).download(mode="single", symbols=[code])
