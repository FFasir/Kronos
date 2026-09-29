"""
Kronos 指数/ETF 择时策略系统 — 存储层

HDF5 读写抽象 + SQLite 数据库初始化与操作。

HDF5 结构:
  /raw/{symbol}/          — 按标的存储标准化后日线
  /windows/{id}/train/    — 训练集
  /windows/{id}/val/      — 验证集
  /windows/{id}/test/     — 测试集
  /metadata/              — 数据集元信息

SQLite: 15 张表，按 docs/附E-数据库设计说明书.md 建表。
"""

import os
import sqlite3
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

from src.common.config import Config, PROJECT_ROOT
from src.common.logger import get_logger

logger = get_logger(__name__)


# ─── 15 张表的建表 SQL（按附E文档） ───

_CREATE_TABLES_SQL = """

-- 1. 标的基本信息
CREATE TABLE IF NOT EXISTS stock_info (
    code        TEXT PRIMARY KEY,           -- 标的代码
    name        TEXT NOT NULL,              -- 标的名称
    type        TEXT NOT NULL,              -- index / etf / stock
    track_index TEXT,                       -- ETF跟踪的指数代码(仅ETF)
    list_date   TEXT,                       -- 上市日期
    is_active   INTEGER DEFAULT 1,          -- 是否活跃
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

-- 2. 日线行情数据
CREATE TABLE IF NOT EXISTS stock_daily (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    code    TEXT NOT NULL,
    date    TEXT NOT NULL,                  -- YYYY-MM-DD
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    amount  REAL,
    turnover REAL,                          -- 换手率(%)
    UNIQUE(code, date)
);
CREATE INDEX IF NOT EXISTS idx_stock_daily_code_date ON stock_daily(code, date);

-- 3. 模型预测结果
CREATE TABLE IF NOT EXISTS model_prediction (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,              -- 预测日期(信号生成日)
    pred_date   TEXT NOT NULL,              -- 被预测的日期
    pred_close  REAL NOT NULL,              -- 预测收盘价
    pred_return REAL,                       -- 预测收益率
    last_close  REAL,                       -- 预测时最新收盘价
    prediction_std REAL,                    -- Kronos采样预测标准差
    actual_close REAL,                      -- 到期后回填实际收盘价
    actual_return REAL,                     -- 到期后回填实际收益率
    score       REAL,                       -- 0-10 expected-return score
    confidence  REAL,                       -- 置信度
    temperature REAL,                       -- 推理参数 T
    top_p       REAL,                       -- 推理参数 top_p
    lookback    INTEGER,                    -- 回溯窗口
    model_version TEXT,                     -- 模型版本
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(code, date, pred_date)
);
CREATE INDEX IF NOT EXISTS idx_prediction_code_date ON model_prediction(code, date);

-- 4. 价格预测 (序列预测)
CREATE TABLE IF NOT EXISTS price_prediction (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL,
    created_at  TEXT NOT NULL,              -- 预测时刻
    pred_horizon INTEGER NOT NULL,          -- 预测步长 (1~20)
    pred_price  REAL,                       -- 预测价格
    pred_lower  REAL,                       -- 置信下界
    pred_upper  REAL,                       -- 置信上界
    model_version TEXT,
    as_of_date  TEXT,
    pred_date   TEXT,
    pred_open   REAL,
    pred_high   REAL,
    pred_low    REAL,
    pred_close  REAL,
    prediction_std REAL
);

-- 5. 选股结果（个股扩展预留）
CREATE TABLE IF NOT EXISTS stock_selection (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    code        TEXT NOT NULL,
    prob_up     REAL,                       -- 上涨概率
    score       REAL,                       -- 综合收益分值
    rank        INTEGER,                    -- 当日排名
    selected    INTEGER DEFAULT 0           -- 是否入选
);

-- 6. 择时交易信号
CREATE TABLE IF NOT EXISTS trading_signal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,              -- 信号生成日期
    signal_type TEXT NOT NULL,              -- BUY / CASH / HOLD / SELL
    pred_return REAL,                       -- 预测收益率
    pred_close  REAL,                       -- 预测收盘价
    confidence  REAL,                       -- 置信度
    position_pct REAL DEFAULT 0,            -- 建议仓位比例
    reason      TEXT,                       -- 信号理由
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(code, date)
);
CREATE INDEX IF NOT EXISTS idx_signal_code_date ON trading_signal(code, date);

-- 7. 回测汇总
CREATE TABLE IF NOT EXISTS backtest_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT UNIQUE NOT NULL,    -- 回测任务ID
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    final_equity    REAL,
    cumulative_return REAL,                  -- 累计收益率
    annual_return   REAL,                    -- 年化收益率
    max_drawdown    REAL,                    -- 最大回撤
    sharpe_ratio    REAL,                    -- 夏普比率
    calmar_ratio    REAL,                    -- 卡玛比率
    win_rate        REAL,                    -- 胜率
    profit_loss_ratio REAL,                  -- 盈亏比
    excess_return   REAL,                    -- 超额收益（Alpha）
    max_consecutive_wins  INTEGER,
    max_consecutive_losses INTEGER,
    avg_hold_days   REAL,                    -- 平均持仓天数
    total_trades    INTEGER,
    params_json     TEXT,                    -- 策略参数JSON
    status          TEXT DEFAULT 'running',  -- running/completed/failed
    error_message   TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    completed_at    TEXT
);

-- 8. 模拟交易明细
CREATE TABLE IF NOT EXISTS sim_trade (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    code        TEXT NOT NULL,
    direction   TEXT NOT NULL,               -- BUY / SELL
    price       REAL NOT NULL,
    quantity    INTEGER NOT NULL,
    amount      REAL NOT NULL,               -- 成交金额
    commission  REAL NOT NULL,               -- 佣金
    stamp_tax   REAL NOT NULL,               -- 印花税
    slippage    REAL NOT NULL,               -- 滑点成本
    net_amount  REAL NOT NULL,               -- 净金额
    reason      TEXT,                        -- 买卖原因
    FOREIGN KEY (task_id) REFERENCES backtest_summary(task_id)
);
CREATE INDEX IF NOT EXISTS idx_sim_trade_task ON sim_trade(task_id);

-- 9. 每日权益
CREATE TABLE IF NOT EXISTS daily_equity (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    cash            REAL NOT NULL,
    position_value  REAL NOT NULL,
    total_equity    REAL NOT NULL,
    daily_return    REAL,                    -- 日收益率
    cumulative_return REAL,                  -- 累计收益率
    drawdown       REAL,                     -- 当前回撤
    benchmark_value REAL,                    -- 基准净值
    FOREIGN KEY (task_id) REFERENCES backtest_summary(task_id)
);
CREATE INDEX IF NOT EXISTS idx_equity_task_date ON daily_equity(task_id, date);

-- 10. 风控事件
CREATE TABLE IF NOT EXISTS risk_event (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT,
    date            TEXT NOT NULL,
    code            TEXT NOT NULL,
    event_type      TEXT NOT NULL,           -- STOP_LOSS / TAKE_PROFIT / CIRCUIT_BREAKER
    trigger_price   REAL,
    exit_price      REAL,
    loss_amount     REAL,
    loss_pct       REAL,
    description     TEXT
);

-- 11. 训练日志
CREATE TABLE IF NOT EXISTS training_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id   TEXT NOT NULL,
    model_type  TEXT NOT NULL,               -- kronos_base / fallback
    start_time  TEXT NOT NULL,
    end_time    TEXT,
    status      TEXT DEFAULT 'running',      -- running/completed/failed
    epochs_completed INTEGER DEFAULT 0,
    best_train_loss   REAL,
    best_val_loss     REAL,
    spearman    REAL,
    mae         REAL,
    direction_accuracy REAL,
    checkpoint_path TEXT,
    error_message TEXT
);

-- 12. 模型版本
CREATE TABLE IF NOT EXISTS model_version (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT UNIQUE NOT NULL,
    model_type  TEXT NOT NULL,               -- kronos_base / fallback
    model_path  TEXT NOT NULL,               -- 模型文件路径
    created_at  TEXT DEFAULT (datetime('now')),
    is_active   INTEGER DEFAULT 0,
    description TEXT
);

-- 13. 预测归档
CREATE TABLE IF NOT EXISTS prediction_archive (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL,
    pred_date   TEXT NOT NULL,               -- 被预测日期
    created_at  TEXT NOT NULL,               -- 预测生成时间
    pred_close  REAL NOT NULL,
    actual_close REAL,                        -- 实际收盘价(后续回填)
    pred_return REAL,
    actual_return REAL,                       -- 实际收益率(后续回填)
    model_version TEXT
);

-- 14. 系统配置快照
CREATE TABLE IF NOT EXISTS system_config (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    config_json TEXT NOT NULL,                -- 完整配置JSON
    saved_at    TEXT DEFAULT (datetime('now')),
    description TEXT
);

-- 15. 下载日志
CREATE TABLE IF NOT EXISTS download_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL,
    start_date  TEXT,
    end_date    TEXT,
    status      TEXT NOT NULL,               -- success / failed
    rows_count  INTEGER,
    error_message TEXT,
    download_time TEXT DEFAULT (datetime('now'))
);
"""


class StorageLayer:
    """
    存储层：HDF5 读写 + SQLite 数据库操作。

    HDF5 存储标准化行情数据和窗口切片，
    SQLite 存储结构化交易/信号/回测等业务数据。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.h5_path = self._resolve_path("data", "processed_dir", default_dir="data/processed") / "samples.h5"
        self.db_path = self._resolve_path("database", "sqlite_path", default_dir="data") / "kronos.db" if \
            not str(self.config.get("database", "sqlite_path")).endswith(".db") else \
            Path(self.config.get("database", "sqlite_path"))
        # 个股"实时"后复权覆盖层：完整单一数据源(新浪hfq)序列，优先于静止的归档zip。
        # 归档 zip(东财hfq) 与新浪 hfq 复权锚不同，绝不能混拼；覆盖层整段替换才能避免悬崖。
        hfq_live_rel = self.config.get("data", "hfq_live_dir", default="A股数据/daily_hfq_live")
        hfq_live = Path(hfq_live_rel)
        if not hfq_live.is_absolute():
            hfq_live = PROJECT_ROOT / hfq_live
        self.hfq_live_dir = hfq_live
        self._h5_keys: Optional[set[str]] = None
        self._archive_handles: dict[Path, zipfile.ZipFile] = {}

    def _resolve_path(self, *keys, default_dir: str) -> Path:
        val = self.config.get(*keys)
        if val:
            p = Path(val)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return p if p.is_dir() else p.parent
        return PROJECT_ROOT / default_dir

    # ═══════════════════════════════════════════════
    # HDF5 操作
    # ═══════════════════════════════════════════════

    def save_raw_data(self, df: pd.DataFrame, symbol: str):
        """保存单个标的的原始/标准化数据到 HDF5。"""
        key = f"/raw/{symbol}"
        self._ensure_h5_dir()
        with pd.HDFStore(self.h5_path, mode="a", complevel=5, complib="zlib") as store:
            store.put(key, df, format="table")
        self._h5_keys = None

    def load_raw_data(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        按标的加载日线数据。

        Args:
            symbol: 标的代码
            start:  起始日期 YYYY-MM-DD
            end:    结束日期 YYYY-MM-DD

        Returns:
            DataFrame，列为 OHLCV，索引为 date
        """
        key = f"/raw/{symbol}"
        df = None
        if self.h5_path.exists():
            if self._h5_keys is None:
                with pd.HDFStore(self.h5_path, mode="r") as store:
                    self._h5_keys = set(store.keys())
            if key in self._h5_keys:
                with pd.HDFStore(self.h5_path, mode="r") as store:
                    df = store[key]

        # The archive importer is optional in a teaching deployment.  Read the
        # supplied per-symbol CSV directly when HDF5 has not been built yet so
        # API queries and real backtests never silently turn into mock data.
        if df is None:
            df = self._load_csv_raw_data(symbol)

        # The stock downloader uses bare six-digit codes. Some A-share codes
        # collide with configured indices (notably stock 000852 vs CSI1000).
        # Their SQLite deltas must never be merged into index/ETF price files.
        configured_market_codes = {
            str(item["code"]) for item in self.config.get_all_symbols()
        }
        delta = None if symbol in configured_market_codes else self._load_sqlite_daily(symbol)

        # 复权口径一致性防护：
        # 个股主序列（HDF5 / daily_hfq.zip 归档）是后复权价（通常数百~数千元），
        # 而 SQLite stock_daily 增量可能是下载器写入的"不复权"快照。
        # 直接拼接会在序列尾部制造价格悬崖（曾出现 1834→11.88、13.0→8.1 等，
        # 导致预测 +13998% 的假信号）。A 股有涨跌停限制（主板 ±10%、创业板/
        # 科创板 ±20%、北交所 ±30%），相邻交易日单日 |收益| 不可能超过 ~35%，
        # 因此用 delta 首日相对主序列末日的收益是否超界来判定口径是否一致。
        if delta is not None and not delta.empty and df is not None and not df.empty:
            try:
                last_close = float(df["close"].iloc[-1])
                first_delta_close = float(delta["close"].iloc[0])
                jump = first_delta_close / last_close - 1.0 if last_close > 0 else 1.0
                if abs(jump) > 0.35:
                    logger.warning(
                        f"{symbol}: SQLite delta 与主序列尾部不连续 "
                        f"(last={last_close:.2f} -> delta_first={first_delta_close:.2f}, "
                        f"jump={jump:+.1%})；视为复权口径不一致，忽略增量"
                    )
                    delta = None
            except Exception:
                pass

        if delta is not None and not delta.empty:
            if df is None:
                df = delta
            else:
                frames = [frame.dropna(axis=1, how="all") for frame in (df, delta)]
                df = pd.concat(frames)
            df = df[~df.index.duplicated(keep="last")].sort_index()
        if df is None or df.empty:
            raise FileNotFoundError(f"no local OHLCV data available for {symbol}")

        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        return df

    def _load_csv_raw_data(self, symbol: str) -> Optional[pd.DataFrame]:
        configured = next(
            (
                item for item in self.config.get_all_symbols()
                if str(item.get("code")) == symbol
            ),
            None,
        )
        candidates = []
        if configured:
            candidates.append(
                PROJECT_ROOT / "data" / "raw"
                / f"{symbol}_{configured['name']}_daily.csv"
            )
        candidates.extend((PROJECT_ROOT / "data" / "raw").glob(f"{symbol}_*_daily.csv"))
        candidates.append(PROJECT_ROOT / "A股数据" / "daily" / f"{symbol}_daily.csv")
        candidates.append(PROJECT_ROOT / "A股数据" / "daily_qfq" / f"{symbol}_daily_qfq.csv")
        candidates.append(PROJECT_ROOT / "data" / "raw" / "research" / f"{symbol}_etf_daily.csv")
        candidates.append(PROJECT_ROOT / "data" / "raw" / "research" / f"{symbol}_daily.csv")
        # 个股实时后复权覆盖层：一旦存在(单一数据源整段替换归档)，优先于 zip 归档成员。
        # 覆盖层写入与归档一致的列名，_load_csv_raw_data 的列映射可复用。
        if configured is None:
            candidates.append(self.hfq_live_dir / f"{symbol}_daily_hfq.csv")

        source = next((path for path in candidates if path.exists()), None)
        if source is not None:
            df = pd.read_csv(source)
        else:
            # A full A-share archive contains one CSV per symbol.  Reading the
            # requested member directly keeps the web service usable without
            # extracting thousands of files on the server.
            df = self._load_archive_member(symbol)
            if df is None:
                return None

        column_map = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turnover",
        }
        df = df.rename(columns=column_map)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"{source} does not contain required OHLCV columns")

        columns = [column for column in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if column in df]
        df = df[columns].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if column in df:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close", "volume"])

    def _load_sqlite_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load downloaded daily deltas without requiring a rebuilt HDF5 file."""
        if not self.db_path.exists():
            return None
        try:
            rows = self.execute_sql(
                """SELECT date, open, high, low, close, volume, amount, turnover
                   FROM stock_daily WHERE code = ? ORDER BY date""",
                (symbol,),
            )
        except sqlite3.Error:
            return None
        if not rows:
            return None
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame.dropna(subset=["date"]).set_index("date").sort_index()

    def read_archive_member(self, symbol: str) -> Optional[pd.DataFrame]:
        """Public wrapper for the archive member reader (raw Chinese columns).

        The stock refresh pipeline uses the archive tail as a *scale anchor*:
        the frozen Eastmoney hfq archive and the fresh Sina hfq series anchor
        their adjustment factors differently, so new Sina rows are rescaled to
        continue the archive level at the archive's last date.
        """
        return self._load_archive_member(symbol)

    def _load_archive_member(self, symbol: str) -> Optional[pd.DataFrame]:
        """Read one stock CSV from a supplied archive without extracting it."""
        configured = Path(self.config.get(
            "data", "a_share_archive", default="A股数据/daily.zip"
        ))
        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured
        configured_member = (
            f"{symbol}_daily_qfq.csv"
            if "qfq" in configured.name.lower()
            else f"{symbol}_daily_hfq.csv"
            if "hfq" in configured.name.lower()
            else f"{symbol}_daily.csv"
        )
        archives = (
            (configured, configured_member),
            (PROJECT_ROOT / "data" / "source" / "daily.zip", f"{symbol}_daily.csv"),
            (PROJECT_ROOT / "A股数据" / "daily.zip", f"{symbol}_daily.csv"),
        )
        for archive_path, member_name in archives:
            if not archive_path.exists():
                continue
            try:
                archive = self._archive_handles.get(archive_path)
                if archive is None:
                    archive = zipfile.ZipFile(archive_path)
                    self._archive_handles[archive_path] = archive
                with archive.open(member_name) as member:
                    return pd.read_csv(member, encoding="utf-8-sig")
            except KeyError:
                continue
            except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
                logger.warning("unable to read %s from %s: %s", member_name, archive_path, exc)
        return None

    def close(self) -> None:
        """Close cached archive readers used by bulk prediction jobs."""
        for archive in self._archive_handles.values():
            archive.close()
        self._archive_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def save_window(
        self, window_id: str, train: pd.DataFrame,
        val: Optional[pd.DataFrame] = None, test: Optional[pd.DataFrame] = None
    ):
        """保存一个窗口的 train/val/test 到 HDF5。"""
        self._ensure_h5_dir()
        with pd.HDFStore(self.h5_path, mode="a", complevel=5, complib="zlib") as store:
            store.put(f"/windows/{window_id}/train", train, format="table")
            if val is not None and len(val) > 0:
                store.put(f"/windows/{window_id}/val", val, format="table")
            if test is not None and len(test) > 0:
                store.put(f"/windows/{window_id}/test", test, format="table")

    def load_window(self, window_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载一个窗口的 train/val/test。

        Returns:
            (train_df, val_df, test_df) — 可能为空 DataFrame
        """
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {self.h5_path}")

        with pd.HDFStore(self.h5_path, mode="r") as store:
            train = store.get(f"/windows/{window_id}/train") if f"/windows/{window_id}/train" in store else pd.DataFrame()
            val = store.get(f"/windows/{window_id}/val") if f"/windows/{window_id}/val" in store else pd.DataFrame()
            test = store.get(f"/windows/{window_id}/test") if f"/windows/{window_id}/test" in store else pd.DataFrame()
        return train, val, test

    def list_windows(self) -> List[str]:
        """列出 HDF5 中所有窗口 ID，按名称排序。"""
        if not self.h5_path.exists():
            return []

        with pd.HDFStore(self.h5_path, mode="r") as store:
            keys = [k.lstrip("/") for k in store.keys() if k.startswith("/windows/")]
            windows = set()
            for k in keys:
                parts = k.split("/")
                if len(parts) >= 3:
                    windows.add(parts[1])
            return sorted(windows)

    def list_symbols(self) -> List[str]:
        """列出 HDF5 中所有标的代码。"""
        if not self.h5_path.exists():
            return []

        with pd.HDFStore(self.h5_path, mode="r") as store:
            keys = [k.lstrip("/") for k in store.keys() if k.startswith("/raw/")]
            return sorted([k.split("/")[-1] for k in keys])

    def get_metadata(self) -> dict:
        """获取数据集元信息。"""
        if not self.h5_path.exists():
            return {}

        with pd.HDFStore(self.h5_path, mode="r") as store:
            if "/metadata" in store:
                df = store["/metadata"]
                return df.to_dict(orient="records")[0] if len(df) > 0 else {}
        return {}

    def _ensure_h5_dir(self):
        """确保 HDF5 文件所在目录存在。"""
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════
    # SQLite 操作
    # ═══════════════════════════════════════════════

    def init_database(self) -> str:
        """
        初始化 SQLite 数据库 — 创建 15 张表。

        幂等操作：使用 IF NOT EXISTS，重复执行不会报错。

        Returns:
            数据库文件路径
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(_CREATE_TABLES_SQL)
            self._migrate_schema(conn)
            conn.commit()
            logger.info(f"数据库初始化完成: {self.db_path} (15张表)")
            self._seed_stock_info(conn)
        finally:
            conn.close()

        return str(self.db_path)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Add model-observability columns to databases created by older builds."""
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(model_prediction)")
        }
        additions = {
            "last_close": "REAL",
            "prediction_std": "REAL",
            "actual_close": "REAL",
            "actual_return": "REAL",
            "score": "REAL",
        }
        for name, column_type in additions.items():
            if name not in columns:
                conn.execute(
                    f'ALTER TABLE model_prediction ADD COLUMN "{name}" {column_type}'
                )

        price_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(price_prediction)")
        }
        price_additions = {
            "as_of_date": "TEXT",
            "pred_date": "TEXT",
            "pred_open": "REAL",
            "pred_high": "REAL",
            "pred_low": "REAL",
            "pred_close": "REAL",
            "prediction_std": "REAL",
        }
        for name, column_type in price_additions.items():
            if name not in price_columns:
                conn.execute(
                    f'ALTER TABLE price_prediction ADD COLUMN "{name}" {column_type}'
                )

    def _seed_stock_info(self, conn: sqlite3.Connection):
        """
        将配置中的指数/ETF 信息写入 stock_info 表。

        使用 INSERT OR IGNORE 避免重复写入。
        """
        all_symbols = self.config.get_all_symbols()
        catalog_path = PROJECT_ROOT / "A股数据" / "股票列表.csv"
        if catalog_path.exists():
            catalog = pd.read_csv(catalog_path, dtype=str, encoding="utf-8-sig").fillna("")
            all_symbols = [
                {
                    "code": str(row["股票代码"]).zfill(6),
                    "name": str(row["股票名称"]),
                    "type": "stock",
                }
                for _, row in catalog.iterrows()
                if str(row.get("股票代码", "")).strip()
            ]
        cursor = conn.cursor()

        for sym in all_symbols:
            track = sym.get("track_index")
            cursor.execute(
                """INSERT OR IGNORE INTO stock_info (code, name, type, track_index, list_date)
                   VALUES (?, ?, ?, ?, ?)""",
                (sym["code"], sym["name"], sym["type"], track, None),
            )

        conn.commit()
        logger.info(f"  已写入 {len(all_symbols)} 个标的基本信息")

    def execute_sql(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """
        执行 SELECT 查询，返回 sqlite3.Row 列表。

        返回的 Row 同时支持位置下标（``row[0]``）和列名下标
        （``row["col"]``），并可用 ``dict(row)`` 转为字典。
        这样上层既可按列名读取，也可按位置读取，避免下标口径不一致。

        Args:
            query: SQL 查询语句
            params: 查询参数

        Returns:
            [sqlite3.Row, ...]
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
        finally:
            conn.close()

    def execute_write(self, query: str, params: tuple = ()):
        """执行 INSERT/UPDATE/DELETE 语句。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(query, params)
            conn.commit()
        finally:
            conn.close()

    def execute_many(self, query: str, params_list: List[tuple]):
        """批量执行 INSERT/UPDATE 语句。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executemany(query, params_list)
            conn.commit()
        finally:
            conn.close()

    def table_exists(self, table_name: str) -> bool:
        """检查指定表是否存在。"""
        result = self.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return len(result) > 0

    def list_tables(self) -> List[str]:
        """列出数据库中所有表名。"""
        rows = self.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r["name"] for r in rows]

    def list_stock_symbols(self) -> List[str]:
        """List active A-share codes from the persisted catalog."""
        if not self.db_path.exists():
            self.init_database()
        rows = self.execute_sql(
            "SELECT code FROM stock_info WHERE type = 'stock' AND is_active = 1 ORDER BY code"
        )
        return [str(row["code"]) for row in rows]

    def get_connection(self) -> sqlite3.Connection:
        """获取原生 SQLite 连接（调用方负责关闭）。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn


def init_db(config: Optional[Config] = None) -> str:
    """便捷函数：初始化数据库"""
    return StorageLayer(config).init_database()
