"""
Kronos 指数/ETF 择时策略系统 — 回测 API

端点:
  POST /api/v1/backtest             — 提交回测任务
  GET  /api/v1/backtest/{task_id}   — 查询回测结果/进度
  GET  /api/v1/backtest/history     — 回测历史列表
"""

from __future__ import annotations

import uuid
import threading
import copy
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query

from src.common.config import Config
from src.common.logger import get_logger
from src.service_layer.models import (
    BacktestRequest, BacktestSubmitResponse,
    BacktestProgressResponse, BacktestResult,
    BacktestHistoryItem, BacktestStatusEnum,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/backtest", tags=["策略回测"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════
# 内存任务存储（生产环境应使用 Redis）
# ═══════════════════════════════════════════════════════════════════

_tasks: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# 指数择时回测使用的预测代码：000852（中证1000，信号源）与 512100（执行 ETF）。
# 数据源为 model_prediction 归档（由调度器/预测任务逐日写入），替代已废弃的
# data/outputs/base_csi1000_etf_*/test_predictions.csv 冻结文件管线。
_INDEX_SIGNAL_CODES = ("000852", "512100")


def _index_archive_signal_dates(storage, codes=None) -> list[str]:
    """model_prediction 中指定代码已存在的信号日（as-of），升序。默认指数择时对。"""
    codes = tuple(codes or _INDEX_SIGNAL_CODES)
    placeholders = ",".join("?" for _ in codes)
    rows = storage.execute_sql(
        f"""SELECT DISTINCT date FROM model_prediction
           WHERE code IN ({placeholders}) AND pred_return IS NOT NULL
           ORDER BY date""",
        codes,
    )
    return [str(r[0]) for r in rows] if rows else []


def _preflight_index_archive() -> None:
    """回测前置检查：model_prediction 中至少要有指数择时信号日。"""
    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    try:
        storage = StorageLayer(load_config())
        dates = _index_archive_signal_dates(storage)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"回测数据检查失败：{exc}") from exc
    if dates:
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "暂无回测信号数据：model_prediction 中没有任何指数择时预测"
            "（代码 000852/512100）。请先在管理端运行指数预测任务（预测 → 生成模型预测），"
            "系统会在每个交易日收盘后把信号写入归档，回测即可使用。"
        ),
    )


class _RequestConfig:
    """Immutable configuration overlay for a single backtest request."""

    def __init__(self, base, strategy_params: Optional[Dict[str, Any]] = None):
        self._data = copy.deepcopy(base.raw_data)
        allowed = {
            "buy_threshold", "hold_days", "stop_loss", "take_profit",
            "max_single_position", "max_positions", "min_positions",
            "risk_controls_enabled", "trailing_stop_enabled", "trailing_stop_pct",
            "paper_mode", "paper_symbol", "exit_at_close",
        }
        self._data.setdefault("strategy", {}).update({
            key: value for key, value in (strategy_params or {}).items() if key in allowed
        })

    def get(self, *keys, default=None):
        value = self._data
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
        return value if value is not None else default

    @property
    def raw_data(self):
        return self._data

    @property
    def seed(self):
        return int(self.get("reproducibility", "random_seed", default=42))


# ═══════════════════════════════════════════════════════════════════
# Mock 回测结果
# ═══════════════════════════════════════════════════════════════════

def _make_mock_backtest_result(task_id: str, req: BacktestRequest) -> dict:
    """生成模拟回测结果（当回测引擎不可用时）。"""
    import random
    rng = random.Random(hash(task_id) % 2**31)

    start = req.start_date
    end = req.end_date

    # 模拟日数
    try:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        n_days = max((e - s).days, 1) * 5 // 7  # 粗略估计交易日
    except ValueError:
        n_days = 250

    n_days = min(n_days, 1500)
    initial_capital = req.initial_capital

    # 生成权益曲线
    equity = [initial_capital]
    for i in range(1, n_days):
        daily_ret = rng.gauss(0.0003, 0.012)
        equity.append(equity[-1] * (1 + daily_ret))

    final_equity = equity[-1]
    cumulative_return = (final_equity - initial_capital) / initial_capital
    annual_return = ((final_equity / initial_capital) ** (244 / max(n_days, 1)) - 1) if n_days >= 63 else 0.0

    # 最大回撤
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        max_dd = min(max_dd, -dd)

    # 模拟交易
    n_trades = rng.randint(30, 80)
    win_rate = rng.uniform(0.45, 0.65)
    n_wins = int(n_trades * win_rate)
    n_losses = n_trades - n_wins

    avg_win = rng.uniform(5000, 30000)
    avg_loss = rng.uniform(-20000, -3000)
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None

    sharpe = rng.uniform(0.5, 2.5)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    # 构建每日权益序列
    daily_equity = []
    peak_val = initial_capital
    for i in range(n_days):
        date_str = f"2024-{1 + i // 22:02d}-{1 + i % 22:02d}"
        eq = equity[i]
        if eq > peak_val:
            peak_val = eq
        dd = (peak_val - eq) / peak_val if peak_val > 0 else 0.0

        daily_equity.append({
            "date": date_str,
            "cash": eq * rng.uniform(0.1, 0.4),
            "position_value": eq * rng.uniform(0.6, 0.9),
            "total_equity": eq,
            "daily_return": (equity[i] / equity[i-1] - 1) if i > 0 else 0.0,
            "cumulative_return": (eq - initial_capital) / initial_capital,
            "drawdown": -dd,
            "benchmark_value": initial_capital * (1 + rng.uniform(-0.1, 0.3)),
            "num_positions": rng.randint(0, 4),
        })

    # 模拟交易记录
    symbols = req.symbols or ["000852", "000300"]
    trades = []
    for t in range(n_trades):
        sym = symbols[t % len(symbols)]
        direction = "BUY" if t % 2 == 0 else "SELL"
        price = rng.uniform(2, 30)
        quantity = rng.randint(100, 10000)
        amount = price * quantity
        commission = amount * 0.0003
        stamp_tax = amount * 0.001 if direction == "SELL" else 0.0
        slippage = amount * 0.001
        net_amount = amount - commission - stamp_tax - slippage
        pnl = net_amount if direction == "SELL" else -net_amount

        trades.append({
            "date": f"2024-{1 + t // 10:02d}-{1 + t % 20:02d}",
            "symbol": sym,
            "direction": direction,
            "price": round(price, 2),
            "quantity": quantity,
            "amount": round(amount, 2),
            "commission": round(commission, 2),
            "stamp_tax": round(stamp_tax, 2),
            "slippage": round(slippage, 2),
            "net_amount": round(net_amount, 2),
            "pnl": round(pnl, 2) if direction == "SELL" else None,
            "pnl_pct": round(pnl / amount, 4) if direction == "SELL" and amount > 0 else None,
            "hold_days": rng.randint(1, 5) if direction == "SELL" else None,
            "reason": "持仓到期" if direction == "SELL" else "买入信号",
        })

    # 月度收益
    monthly_returns = []
    for m in range(1, 13):
        monthly_returns.append({
            "month": f"2024-{m:02d}",
            "return": round(rng.uniform(-0.05, 0.08), 4),
            "start_equity": round(initial_capital * (1 + rng.uniform(-0.1, 0.2)), 2),
            "end_equity": round(initial_capital * (1 + rng.uniform(-0.1, 0.3)), 2),
        })

    return {
        "task_id": task_id,
        "status": "completed",
        "start_date": start,
        "end_date": end,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "elapsed_seconds": rng.uniform(5, 30),
        "total_trades": n_trades,
        # 10项指标
        "cumulative_return": round(cumulative_return, 4),
        "annual_return": round(annual_return, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(profit_loss_ratio, 4) if profit_loss_ratio else None,
        "sharpe_ratio": round(sharpe, 4),
        "calmar_ratio": round(calmar, 4),
        "excess_return": round(rng.uniform(-0.05, 0.15), 4),
        "max_consecutive_wins": rng.randint(3, 10),
        "max_consecutive_losses": rng.randint(2, 7),
        "avg_hold_days": round(rng.uniform(3, 5), 1),
        # 明细
        "daily_equity": daily_equity,
        "trades": trades,
        "benchmark_equity": [
            {"date": e["date"], "value": e["benchmark_value"]}
            for e in daily_equity
        ],
        "risk_events": [],
        "monthly_returns": monthly_returns,
        "yearly_returns": [
            {"year": 2024, "return": round(rng.uniform(-0.1, 0.3), 4),
             "cumulative_return": round(rng.uniform(-0.1, 0.3), 4),
             "start_equity": initial_capital,
             "end_equity": round(final_equity, 2)},
        ],
        "max_drawdown_period": {
            "peak_date": "2024-03-15",
            "trough_date": "2024-06-20",
            "recovery_date": "2024-09-10",
            "peak_value": round(initial_capital * 1.15, 2),
            "trough_value": round(initial_capital * 0.92, 2),
            "drawdown_days": 67,
        },
        "config": {
            "commission_rate": 0.0003,
            "stamp_tax_rate": 0.001,
            "slippage_rate": 0.001,
            "benchmark_index": "000852",
            "buy_threshold": req.strategy_params.get("buy_threshold", 0.005) if req.strategy_params else 0.005,
            "hold_days": req.strategy_params.get("hold_days", 5) if req.strategy_params else 5,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# 后台回测执行
# ═══════════════════════════════════════════════════════════════════

def _run_backtest_sync(task_id: str, req: BacktestRequest):
    """同步执行回测（在独立线程中运行）。"""
    try:
        _update_task(task_id, status=BacktestStatusEnum.RUNNING, progress=0.0)

        # 尝试使用真实回测引擎
        try:
            from src.strategy_layer.backtest_engine import BacktestEngine
            from src.data_layer.storage import StorageLayer
            from src.common.config import load_config

            base_config = load_config()
            strategy_params = dict(req.strategy_params or {})
            if req.scope in ("paper_index", "custom_index"):
                paper_exec = (
                    str(req.execution_symbol or "512100")
                    if req.scope == "custom_index" else "512100"
                )
                strategy_params.update({
                    "paper_mode": True,
                    "paper_symbol": paper_exec,
                    "risk_controls_enabled": False,
                    "exit_at_close": True,
                    "max_single_position": 1.0,
                })
            else:
                strategy_params.update({"paper_mode": False, "exit_at_close": True})
            config = _RequestConfig(base_config, strategy_params)
            engine = BacktestEngine(config)

            # 加载各标的的 OHLCV 数据
            storage = StorageLayer(base_config)
            leakage_note = None
            signal_code = str(req.signal_symbol or "000852")
            execution_code = str(req.execution_symbol or "512100")
            used_online = False
            if req.scope == "paper_index":
                symbols = ["512100", "000852"]
                predictions_df = _load_frozen_index_predictions(req.start_date, req.end_date)
            elif req.scope == "custom_index":
                symbols = [execution_code, signal_code]
                # 优先读归档（快、与历史界面一致）；归档信号不足时在线重建
                try:
                    predictions_df = _load_index_archive_predictions(
                        storage, req.start_date, req.end_date,
                        codes=(signal_code,), execution_symbol=execution_code,
                    )
                    arch_dates = _index_archive_signal_dates(storage, codes=(signal_code,))
                    cutoff = pd.Timestamp(str(base_config.get(
                        "model", "training_cutoff_date", default="2026-09-01")))
                    if arch_dates and arch_dates[0] < str(cutoff.date()):
                        leakage_note = (
                            f"演示模式：本次使用归档信号（{len(predictions_df)} 个信号日），"
                            f"其中部分早于模型训练截止（{cutoff.date()}），存在前视泄漏、结果偏乐观，仅供演示。"
                        )
                    else:
                        leakage_note = None
                except ValueError:
                    predictions_df, leakage_note = _reconstruct_index_signals(
                        storage, base_config,
                        signal_symbol=signal_code,
                        execution_symbol=execution_code,
                        start_date=req.start_date, end_date=req.end_date,
                        leakage_mode=req.leakage_mode,
                        progress_callback=lambda p: _update_task(task_id, progress=p),
                    )
                    used_online = True
            else:
                symbols, predictions_df = _load_component_predictions(
                    storage, req.start_date, req.end_date
                )

            symbol_dfs = {}
            for sym in symbols:
                try:
                    df = storage.load_raw_data(sym, start=req.start_date, end=req.end_date)
                    if df is not None and len(df) > 0:
                        df.index = pd.to_datetime(df.index)
                        symbol_dfs[sym] = df
                except Exception as e:
                    logger.warning(f"加载 {sym} 数据失败: {e}")

            if not symbol_dfs:
                raise ValueError("none of the requested symbols has local OHLCV data")

            _attach_limit_flags(symbol_dfs)
            result = engine.run_with_predictions(
                predictions_df=predictions_df,
                symbol_dfs=symbol_dfs,
                start_date=req.start_date,
                end_date=req.end_date,
                initial_capital=req.initial_capital,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
            engine._task_id = task_id
            result["task_id"] = task_id
            try:
                mv_rows = storage.execute_sql(
                    "SELECT model_version FROM model_prediction "
                    "WHERE code IN ('000852','512100') ORDER BY date DESC LIMIT 1"
                )
                archive_version = str(mv_rows[0][0]) if mv_rows and mv_rows[0][0] else "model_prediction"
            except Exception:
                archive_version = "model_prediction"
            result["config"].update({
                "scope": req.scope,
                "prediction_source": "online_reconstruction" if (req.scope == "custom_index" and used_online) else "model_prediction_archive",
                "model_version": archive_version,
                "signal_asset": signal_code if req.scope == "custom_index" else ("000852" if req.scope == "paper_index" else "CSI1000 point-in-time members"),
                "execution_asset": execution_code if req.scope == "custom_index" else ("512100" if req.scope == "paper_index" else "CSI1000 constituent stocks"),
            })
            if leakage_note:
                result["config"]["leakage_note"] = leakage_note
            engine._result = result
            engine.save_to_db()
        except Exception as e:
            raise RuntimeError(f"real backtest failed: {e}") from e

        result["status"] = BacktestStatusEnum.COMPLETED.value
        _update_task(
            task_id,
            status=BacktestStatusEnum.COMPLETED,
            progress=1.0,
            result=result,
        )
        logger.info(f"回测 [{task_id}] 完成")

    except Exception as e:
        logger.error(f"回测 [{task_id}] 失败: {e}", exc_info=True)
        _update_task(
            task_id,
            status=BacktestStatusEnum.FAILED,
            progress=0.0,
            error=str(e),
        )


def _update_task(task_id: str, **kwargs):
    """线程安全地更新任务状态。"""
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].update(kwargs)


# ═══════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════

@router.get("/availability")
async def get_backtest_availability():
    """
    回测可用性诊断：各 scope 现有的信号数据（model_prediction 归档）。

    前端据此钳制日期区间、禁用不可用 scope，并在样本过小时给出明确提示，
    避免"填完表单才报错"。
    """
    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(load_config())
    index_dates = _index_archive_signal_dates(storage)

    # 逐股 scope 可用性：需要 point-in-time 成分权重文件 + ≥2 个全市场信号日
    weights_ok = (PROJECT_ROOT / "data" / "raw" / "research" / "csi1000_index_weight_history.csv").is_file()
    stock_rows = storage.execute_sql(
        "SELECT COUNT(DISTINCT date) FROM model_prediction WHERE code NOT IN ('000852','512100','000300')"
    )
    stock_dates = int(stock_rows[0][0]) if stock_rows and stock_rows[0][0] else 0

    cfg = load_config()
    pairs = _configured_asset_pairs(cfg)
    # 只保留本地有行情的信号资产
    viable_pairs = []
    for pair in pairs:
        try:
            df = storage.load_raw_data(pair["signal_symbol"])
            if df is not None and len(df) > 300:
                viable_pairs.append(pair)
        except Exception:
            continue

    return {
        "paper_index": {
            "enabled": bool(index_dates),
            "signal_dates": index_dates,
            "min_date": index_dates[0] if index_dates else None,
            "max_date": index_dates[-1] if index_dates else None,
            "count": len(index_dates),
            "note": (
                "信号日来自 model_prediction 归档（每个交易日收盘后由预测任务写入）。"
                "当前样本偏小，结果仅用于验证回测管线，不代表模型真实绩效。"
            ),
        },
        "csi1000_components": {
            "enabled": weights_ok and stock_dates >= 2,
            "count": stock_dates,
            "note": (
                "逐股回测需要 point-in-time 成分权重文件（data/raw/research/"
                "csi1000_index_weight_history.csv）与 ≥2 个全市场信号日。"
            ),
        },
        "custom_index": {
            "enabled": True,
            "pairs": viable_pairs,
            "note": (
                "自定义指数择时：在线重建每个交易日的信号（用当天为止的行情现算），"
                "样本=交易日数。注意：当前模型训练数据截止约 2026-09-01，"
                "早于该日的区间存在前视泄漏（demo 模式），结果偏乐观，仅供演示。"
            ),
        },
    }


@router.post("", response_model=BacktestSubmitResponse, status_code=202)
async def submit_backtest(req: BacktestRequest):
    """
    提交回测任务（异步执行）。

    提交后立即返回 task_id，后台执行回测计算。
    使用 GET /backtest/{task_id} 轮询进度和结果。

    回测配置:
    - 初始资金: 默认 100 万
    - 佣金: 万分之三（双边）
    - 印花税: 千分之一（仅卖出）
    - 滑点: 0.1%
    """
    _preflight_index_archive()
    task_id = str(uuid.uuid4())[:8]

    with _lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "status": BacktestStatusEnum.ACCEPTED,
            "progress": 0.0,
            "submitted_at": datetime.now().isoformat(),
            "request": req,
            "result": None,
            "error": None,
        }

    # 启动后台线程执行回测
    thread = threading.Thread(
        target=_run_backtest_sync,
        args=(task_id, req),
        daemon=True,
    )
    thread.start()

    logger.info(f"回测任务已提交 [{task_id}]: {req.symbols}, {req.start_date}~{req.end_date}")
    return BacktestSubmitResponse(
        task_id=task_id,
        status="accepted",
        submitted_at=datetime.now().isoformat(),
    )


# 注意：静态路径 /history 必须注册在动态路径 /{task_id} 之前，
# 否则 FastAPI 自上而下匹配会把 "history" 当作 task_id，导致 /history 永远 404。
@router.get("/history", response_model=list[BacktestHistoryItem])
async def get_backtest_history(
    limit: int = Query(20, ge=1, le=100, description="返回条数"),
):
    """
    获取历史回测任务列表（含状态摘要）。

    按提交时间倒序排列。
    """
    history = []

    # 先查内存中的任务
    with _lock:
        memory_items = list(_tasks.values())

    # 再尝试从数据库查询
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            """SELECT task_id, start_date, end_date, initial_capital,
                      final_equity, cumulative_return, annual_return,
                      sharpe_ratio, max_drawdown, total_trades, status, completed_at
               FROM backtest_summary
               ORDER BY completed_at DESC LIMIT ?""",
            (limit,),
        )
        if rows:
            for r in rows:
                history.append(BacktestHistoryItem(
                    task_id=str(r[0]),
                    start_date=str(r[1]),
                    end_date=str(r[2]),
                    initial_capital=float(r[3]),
                    final_equity=float(r[4]),
                    cumulative_return=float(r[5]),
                    annual_return=float(r[6]),
                    sharpe_ratio=float(r[7]),
                    max_drawdown=float(r[8]),
                    total_trades=int(r[9]) if r[9] else 0,
                    status=str(r[10]),
                    completed_at=str(r[11]) if len(r) > 11 and r[11] else None,
                ))
    except Exception as e:
        logger.debug(f"数据库回测历史查询失败: {e}")

    # 合并内存中的任务
    for task in memory_items:
        if len(history) >= limit:
            break
        result = task.get("result", {})
        if result:
            history.append(BacktestHistoryItem(
                task_id=task["task_id"],
                start_date=result.get("start_date", ""),
                end_date=result.get("end_date", ""),
                initial_capital=result.get("initial_capital", 0),
                final_equity=result.get("final_equity", 0),
                cumulative_return=result.get("cumulative_return", 0),
                annual_return=result.get("annual_return", 0),
                sharpe_ratio=result.get("sharpe_ratio", 0),
                max_drawdown=result.get("max_drawdown", 0),
                total_trades=result.get("total_trades", 0),
                status=task["status"].value,
                completed_at=None,
            ))

    logger.info(f"GET /backtest/history → {len(history)} 条")
    return history


@router.get("/{task_id}", response_model=BacktestProgressResponse | BacktestResult)
async def get_backtest_result(task_id: str):
    """
    查询回测任务状态/结果。

    返回值:
    - status=running → `{status: "running", progress: 0.45}`
    - status=completed → 完整的 BacktestResult（含10项指标+净值曲线+交易明细）
    - status=failed → `{status: "failed", error: "..."}`
    """
    with _lock:
        task = _tasks.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"回测任务 {task_id} 不存在")

    status = task["status"]

    if getattr(status, "value", status) == BacktestStatusEnum.COMPLETED.value and task.get("result"):
        # 返回完整结果
        result = task["result"]
        return result  # 完整的 BacktestResult dict

    return BacktestProgressResponse(
        task_id=task_id,
        status=status,
        progress=task.get("progress", 0.0),
        message=task.get("error"),
    )


# 延迟导入（避免循环依赖）
import pandas as pd  # noqa: E402


def _load_index_archive_predictions(storage, start_date: str, end_date: str,
                                    codes=None, execution_symbol: str = "512100") -> pd.DataFrame:
    """从 model_prediction 归档加载指数/ETF 择时预测。

    每个 as-of 日一条信号，映射为引擎需要的 date/symbol=execution_symbol/... 结构；
    稀疏信号日天然受支持。
    """
    codes = tuple(codes or _INDEX_SIGNAL_CODES)
    placeholders = ",".join("?" for _ in codes)
    rows = storage.execute_sql(
        f"""SELECT code, date, pred_date, pred_close, pred_return, last_close, confidence
            FROM model_prediction
            WHERE code IN ({placeholders}) AND date >= ? AND date <= ?
              AND pred_return IS NOT NULL
            ORDER BY date, code""",
        (*codes, start_date, end_date),
    )
    if not rows:
        available = _index_archive_signal_dates(storage, codes=codes)
        hint = (
            f"所选区间（{start_date} ~ {end_date}）内没有这些标的的择时预测记录：{','.join(codes)}。"
            f"当前归档中的信号日：{('、'.join(available)) if available else '（无）'}。"
            "请选择覆盖这些日期的区间，或先运行指数预测任务/补档生成新信号日。"
        )
        raise ValueError(hint)

    frame = pd.DataFrame([dict(row) for row in rows])
    frame["symbol"] = execution_symbol
    frame = frame.rename(columns={"pred_close": "pred_close_t5"})
    frame = frame[["date", "symbol", "pred_return", "pred_close_t5", "last_close", "confidence"]]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    return frame


def _load_frozen_index_predictions(start_date: str, end_date: str) -> pd.DataFrame:
    """兼容旧调用点：从 model_prediction 归档读取指数择时预测。"""
    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    return _load_index_archive_predictions(
        StorageLayer(load_config()), start_date, end_date
    )


def _configured_asset_pairs(config) -> list[dict]:
    """配置中的 指数→执行ETF 配对，供自定义指数择时选择。"""
    etfs = {str(item["track_index"]): item for item in config.get_etf_list() if item.get("track_index")}
    pairs = []
    for idx in config.get_index_list():
        code = str(idx["code"])
        etf = etfs.get(code)
        pairs.append({
            "signal_symbol": code,
            "signal_name": str(idx.get("name", code)),
            "execution_symbol": str(etf["code"]) if etf else "",
            "execution_name": str(etf.get("name", "")) if etf else "",
        })
    return pairs


def _reconstruct_index_signals(
    storage,
    config,
    signal_symbol: str,
    execution_symbol: str,
    start_date: str,
    end_date: str,
    leakage_mode: str = "demo",
    progress_callback=None,
) -> tuple[pd.DataFrame, str]:
    """
    在线信号重建：对区间内每个交易日，用『截至当天』的历史行情现算一次 5 日预测。

    - point-in-time：任何一天的输入都不含该日之后的数据（无未来信息）；
    - 样本量 = 区间内交易日数，不再受 model_prediction 归档稀疏限制；
    - 泄漏说明：当前微调模型训练数据截止 ~2026-09，若区间早于该日，
      demo 模式允许运行但结果会因前视而偏乐观（结果页已标注）；strict 模式
      仅保留训练截止之后的信号（当前几乎为空）。
    """
    import numpy as np

    from src.model_layer.model_factory import create_model
    from src.model_layer.kronos_predictor import KronosPredictor

    series = storage.load_raw_data(signal_symbol)
    if series is None or len(series) == 0:
        raise ValueError(f"信号资产 {signal_symbol} 没有本地行情数据，无法重建信号")
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    last_day = series.index[-1]
    end_ts = min(pd.Timestamp(end_date), last_day)
    start_ts = pd.Timestamp(start_date)

    days = [d for d in series.index if start_ts <= d <= end_ts]
    if len(days) < 5:
        raise ValueError(
            f"{signal_symbol} 在 {start_date}~{end_date} 内只有 {len(days)} 个交易日"
            f"（行情最新到 {str(last_day.date())}），无法形成有效回测；"
            "请扩大区间或换用数据更长的标的。"
        )
    if len(days) > 520:
        raise ValueError(f"区间内 {len(days)} 个交易日过多，一次在线重建上限 520 日；请缩小区间")

    # 严格模式：训练截止日（约 2026-09-01）之前视为模型已见过 → 剔除
    # 未来若提供按截止训练的新模型，可在配置中声明其训练数据末端。
    training_cutoff = pd.Timestamp(
        str(config.get("model", "training_cutoff_date", default="2026-09-01"))
    )
    if leakage_mode == "strict":
        kept = [d for d in days if d >= training_cutoff]
        dropped = len(days) - len(kept)
        days = kept
        if len(days) < 5:
            raise ValueError(
                f"strict 模式只统计 {training_cutoff.date()} 之后的信号，"
                f"区间内仅 {len(days)} 个交易日（已剔除 {dropped} 日前视样本）。"
                "当前模型训练截止晚于可用历史，请改用 demo 模式演示，或等待训练截止更早的模型。"
            )

    _update = progress_callback or (lambda p: None)
    _update(0.05)
    model, _path = create_model(config, device="auto")
    predictor = KronosPredictor(model, config)
    lookback = int(predictor.seq_len or 250)

    rows = []
    total = len(days)
    ok = 0
    for i, day in enumerate(days):
        if i % 25 == 0:
            _update(0.1 + 0.8 * i / max(total, 1))
        hist = series[series.index <= day]
        if hist is None or len(hist) < lookback:
            continue
        try:
            preds = predictor.predict_batch({signal_symbol: hist})
            if not preds or preds[0].get("error") or preds[0].get("pred_return") is None:
                continue
            p = preds[0]
            if not np.isfinite(p.get("pred_return", 0)) or not np.isfinite(p.get("pred_close_t5", 0)):
                continue
            rows.append({
                "date": day,
                "symbol": execution_symbol,
                "pred_return": float(p["pred_return"]),
                "pred_close_t5": float(p.get("pred_close_t5", 0.0)),
                "last_close": float(p.get("last_close", 0.0)),
                "confidence": float(p.get("confidence", 0.5)),
            })
            ok += 1
        except Exception as exc:
            logger.debug(f"重建信号 {day.date()} 失败: {exc}")
    _update(0.92)

    if ok < 5:
        raise ValueError(f"在线重建仅成功 {ok} 个信号日（<5），无法回测；请检查模型与行情数据")

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")

    leakage_note = (
        "演示模式：区间内信号由当前模型在线重建（point-in-time），"
        f"但该模型训练数据截止 {training_cutoff.date()}，早于该日的回测存在前视泄漏、结果偏乐观，仅用于功能演示。"
    ) if leakage_mode != "strict" else (
        f"strict 模式：仅统计 {training_cutoff.date()} 之后的在线重建信号，无训练前视。"
    )
    logger.info("在线重建 %s → %d 个信号日", signal_symbol, len(frame))
    return frame, leakage_note


def _load_component_predictions(storage, start_date: str, end_date: str):
    """Load point-in-time members and frozen per-stock forecasts for the period."""
    weights_path = PROJECT_ROOT / "data" / "raw" / "research" / "csi1000_index_weight_history.csv"
    if not weights_path.is_file():
        raise FileNotFoundError(
            "中证1000逐股回测需要 data/raw/research/csi1000_index_weight_history.csv；"
            "禁止用当前成分股快照回填历史"
        )
    weights = pd.read_csv(weights_path, dtype={"con_code": str}, parse_dates=["trade_date"])
    weights["symbol"] = weights["con_code"].str.split(".").str[0].str.zfill(6)
    snapshots = weights[
        weights["trade_date"] <= pd.Timestamp(end_date)
    ].sort_values("trade_date")
    if snapshots.empty:
        raise ValueError("历史成分权重不覆盖所选回测区间")

    rows = storage.execute_sql(
        """SELECT code AS symbol, date, pred_return, pred_close, last_close, confidence
           FROM model_prediction
           WHERE date >= ? AND date <= ? AND pred_return IS NOT NULL
           ORDER BY date, pred_return DESC""",
        (start_date, end_date),
    )
    predictions = pd.DataFrame([dict(row) for row in rows])
    if predictions.empty or predictions["date"].nunique() < 2:
        raise ValueError("所选区间缺少逐股冻结历史预测，不能用现场动量或未来行情替代")
    predictions["symbol"] = predictions["symbol"].astype(str).str.zfill(6)
    predictions["date"] = pd.to_datetime(predictions["date"])

    snapshot_dates = snapshots["trade_date"].drop_duplicates().sort_values()
    membership = {
        date: set(snapshots.loc[snapshots["trade_date"] == date, "symbol"])
        for date in snapshot_dates
    }
    def is_member(row):
        eligible = snapshot_dates[snapshot_dates <= row["date"]]
        return bool(len(eligible)) and row["symbol"] in membership[eligible.iloc[-1]]
    predictions = predictions[predictions.apply(is_member, axis=1)].copy()
    if predictions.empty:
        raise ValueError("历史预测与 point-in-time 中证1000成分股没有交集")
    predictions = predictions.rename(columns={"pred_close": "pred_close_t5"})
    return sorted(predictions["symbol"].unique()), predictions


def _attach_limit_flags(symbol_dfs: Dict[str, pd.DataFrame]) -> None:
    """Attach directional A-share price-limit flags using only prior closes."""
    for symbol, frame in symbol_dfs.items():
        if symbol.startswith(("5", "1")) or {"limit_up", "limit_down"} <= set(frame.columns):
            frame["limit_up"] = frame.get("limit_up", False)
            frame["limit_down"] = frame.get("limit_down", False)
            continue
        previous = frame["close"].shift(1)
        limit = 0.20 if symbol.startswith(("30", "68")) else 0.10
        frame["limit_up"] = frame["high"] >= previous * (1 + limit - 0.005)
        frame["limit_down"] = frame["low"] <= previous * (1 - limit + 0.005)


# ═══════════════════════════════════════════════════════════════════
# 参数优化端点
# ═══════════════════════════════════════════════════════════════════

from src.service_layer.models import (
    OptimizeRequest, OptimizeResult, OptimizeResultItem,
    CompareRequest, CompareResult, CompareStrategyResult,
    WalkForwardWindowResult,
)

_optimize_tasks: Dict[str, Dict[str, Any]] = {}
_optimize_lock = threading.Lock()


def _run_optimize_sync(task_id: str, req: OptimizeRequest):
    """同步执行参数优化任务（在独立线程中运行）。"""
    try:
        with _optimize_lock:
            _optimize_tasks[task_id]["status"] = BacktestStatusEnum.RUNNING

        from src.strategy_layer.strategy_optimizer import StrategyOptimizer
        from src.strategy_layer.backtest_engine import BacktestEngine
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        base_config = load_config()
        optimizer = StrategyOptimizer(base_config)

        # 加载数据
        storage = StorageLayer(base_config)
        if req.scope == "paper_index":
            symbols = ["512100", "000852"]
            predictions_df = _load_frozen_index_predictions(req.start_date, req.end_date)
        else:
            symbols, predictions_df = _load_component_predictions(
                storage, req.start_date, req.end_date
            )

        symbol_dfs = {}
        for sym in symbols:
            try:
                df = storage.load_raw_data(sym, start=req.start_date, end=req.end_date)
                if df is not None and len(df) > 0:
                    df.index = pd.to_datetime(df.index)
                    symbol_dfs[sym] = df
            except Exception as e:
                logger.warning(f"加载 {sym} 数据失败: {e}")

        if not symbol_dfs:
            raise ValueError("无可用数据")

        _attach_limit_flags(symbol_dfs)

        # 构建参数范围
        param_ranges = {pr.name: pr.values for pr in req.param_ranges}

        def progress_cb(progress: float):
            with _optimize_lock:
                if task_id in _optimize_tasks:
                    _optimize_tasks[task_id]["progress"] = progress

        if req.mode == "walk_forward":
            raw_result = optimizer.walk_forward_optimize(
                symbol_dfs=symbol_dfs,
                predictions_df=predictions_df,
                start_date=req.start_date,
                end_date=req.end_date,
                window_months=req.window_months,
                step_months=req.step_months,
                param_ranges=param_ranges,
                metric=req.metric,
                progress_callback=progress_cb,
            )
        else:
            raw_result = optimizer.grid_search(
                symbol_dfs=symbol_dfs,
                predictions_df=predictions_df,
                param_ranges=param_ranges,
                start_date=req.start_date,
                end_date=req.end_date,
                initial_capital=req.initial_capital,
                metric=req.metric,
                progress_callback=progress_cb,
            )

        # 构建结果
        results = [
            OptimizeResultItem(
                params=r.get("params", {}),
                cumulative_return=r.get("metrics", {}).get("cumulative_return", 0),
                annual_return=r.get("metrics", {}).get("annual_return", 0),
                max_drawdown=r.get("metrics", {}).get("max_drawdown", 0),
                sharpe_ratio=r.get("metrics", {}).get("sharpe_ratio", 0),
                win_rate=r.get("metrics", {}).get("win_rate", 0),
                total_trades=r.get("metrics", {}).get("total_trades", 0),
            )
            for r in raw_result.get("results", [])
        ]

        best_params = raw_result.get("best_params")
        best_item = results[0] if results else None

        # Walk-Forward 窗口结果
        wf_windows = []
        for w in raw_result.get("window_results", []):
            oos_metrics = w.get("oos_metrics", {})
            wf_windows.append(WalkForwardWindowResult(
                window_id=w.get("window_id", 0),
                train_start=w.get("train_start", ""),
                train_end=w.get("train_end", ""),
                test_start=w.get("test_start", ""),
                test_end=w.get("test_end", ""),
                best_params=w.get("best_is_params", w.get("best_params", {})),
                oos_sharpe=w.get("oos_sharpe", oos_metrics.get("sharpe_ratio", 0)),
                oos_return=oos_metrics.get("annual_return", 0),
                oos_drawdown=oos_metrics.get("max_drawdown", 0),
            ))

        result = OptimizeResult(
            task_id=task_id,
            status=BacktestStatusEnum.COMPLETED,
            mode=req.mode,
            metric=req.metric,
            total_combinations=len(results),
            progress=1.0,
            results=results,
            best_params=best_params,
            best_metrics=best_item,
            sensitivity=raw_result.get("sensitivity"),
            walk_forward_windows=wf_windows,
            stability_report=raw_result.get("stability_report"),
        )

        with _optimize_lock:
            _optimize_tasks[task_id].update({
                "status": BacktestStatusEnum.COMPLETED,
                "progress": 1.0,
                "result": result,
            })

        logger.info(f"参数优化 [{task_id}] 完成: {len(results)} 组合")

    except Exception as e:
        logger.error(f"参数优化 [{task_id}] 失败: {e}", exc_info=True)
        with _optimize_lock:
            _optimize_tasks[task_id].update({
                "status": BacktestStatusEnum.FAILED,
                "progress": 0.0,
                "error": str(e),
            })


@router.post("/optimize", status_code=202)
async def submit_optimize(req: OptimizeRequest):
    """
    提交参数优化任务（异步执行）。

    支持网格搜索和 Walk-Forward 两种模式。
    提交后通过 GET /backtest/optimize/{task_id} 轮询进度。
    """
    _preflight_index_archive()
    task_id = str(uuid.uuid4())[:8]

    with _optimize_lock:
        _optimize_tasks[task_id] = {
            "task_id": task_id,
            "status": BacktestStatusEnum.ACCEPTED,
            "progress": 0.0,
            "submitted_at": datetime.now().isoformat(),
            "request": req,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_optimize_sync,
        args=(task_id, req),
        daemon=True,
    )
    thread.start()

    logger.info(f"参数优化任务已提交 [{task_id}]: mode={req.mode}, {len(req.param_ranges)} 参数")
    return BacktestSubmitResponse(
        task_id=task_id,
        status="accepted",
        submitted_at=datetime.now().isoformat(),
    )


@router.get("/optimize/{task_id}")
async def get_optimize_result(task_id: str):
    """查询参数优化任务状态/结果。"""
    with _optimize_lock:
        task = _optimize_tasks.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"优化任务 {task_id} 不存在")

    status = task["status"]

    if getattr(status, "value", status) == BacktestStatusEnum.COMPLETED.value and task.get("result"):
        return task["result"]

    req = task.get("request")
    return {
        "task_id": task_id,
        "status": status,
        "progress": task.get("progress", 0.0),
        "mode": req.mode if req else "grid",
        "metric": req.metric if req else "sharpe_ratio",
        "total_combinations": 0,
        "results": [],
        "walk_forward_windows": [],
        "error": task.get("error"),
    }


# ═══════════════════════════════════════════════════════════════════
# 多策略对比端点
# ═══════════════════════════════════════════════════════════════════

_compare_tasks: Dict[str, Dict[str, Any]] = {}
_compare_lock = threading.Lock()


def _run_compare_sync(task_id: str, req: CompareRequest):
    """同步执行多策略对比任务。"""
    try:
        with _compare_lock:
            _compare_tasks[task_id]["status"] = BacktestStatusEnum.RUNNING

        from src.strategy_layer.strategy_optimizer import StrategyOptimizer
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        base_config = load_config()
        optimizer = StrategyOptimizer(base_config)

        storage = StorageLayer(base_config)
        if req.scope == "paper_index":
            symbols = ["512100", "000852"]
            predictions_df = _load_frozen_index_predictions(req.start_date, req.end_date)
        else:
            symbols, predictions_df = _load_component_predictions(
                storage, req.start_date, req.end_date
            )

        symbol_dfs = {}
        for sym in symbols:
            try:
                df = storage.load_raw_data(sym, start=req.start_date, end=req.end_date)
                if df is not None and len(df) > 0:
                    df.index = pd.to_datetime(df.index)
                    symbol_dfs[sym] = df
            except Exception as e:
                logger.warning(f"加载 {sym} 数据失败: {e}")

        if not symbol_dfs:
            raise ValueError("无可用数据")

        _attach_limit_flags(symbol_dfs)

        raw_result = optimizer.compare_strategies(
            strategies=req.strategies,
            symbol_dfs=symbol_dfs,
            predictions_df=predictions_df,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=req.initial_capital,
        )

        results = [
            CompareStrategyResult(
                strategy_id=r.get("strategy_id", i),
                params=r.get("params", {}),
                cumulative_return=r.get("metrics", {}).get("cumulative_return", 0),
                annual_return=r.get("metrics", {}).get("annual_return", 0),
                max_drawdown=r.get("metrics", {}).get("max_drawdown", 0),
                sharpe_ratio=r.get("metrics", {}).get("sharpe_ratio", 0),
                calmar_ratio=r.get("metrics", {}).get("calmar_ratio", 0),
                win_rate=r.get("metrics", {}).get("win_rate", 0),
                total_trades=r.get("metrics", {}).get("total_trades", 0),
            )
            for i, r in enumerate(raw_result.get("results", []))
        ]

        best_by_metric = raw_result.get("best_by_metric")

        result = CompareResult(
            task_id=task_id,
            status=BacktestStatusEnum.COMPLETED,
            results=results,
            best_by_metric=best_by_metric,
        )

        with _compare_lock:
            _compare_tasks[task_id].update({
                "status": BacktestStatusEnum.COMPLETED,
                "result": result,
            })

        logger.info(f"多策略对比 [{task_id}] 完成: {len(results)} 组策略")

    except Exception as e:
        logger.error(f"多策略对比 [{task_id}] 失败: {e}", exc_info=True)
        with _compare_lock:
            _compare_tasks[task_id].update({
                "status": BacktestStatusEnum.FAILED,
                "error": str(e),
            })


@router.post("/compare", status_code=202)
async def submit_compare(req: CompareRequest):
    """
    提交多策略对比任务（异步执行）。

    支持 2~5 组策略参数并行回测对比。
    """
    _preflight_index_archive()
    task_id = str(uuid.uuid4())[:8]

    with _compare_lock:
        _compare_tasks[task_id] = {
            "task_id": task_id,
            "status": BacktestStatusEnum.ACCEPTED,
            "submitted_at": datetime.now().isoformat(),
            "request": req,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_compare_sync,
        args=(task_id, req),
        daemon=True,
    )
    thread.start()

    logger.info(f"多策略对比任务已提交 [{task_id}]: {len(req.strategies)} 组策略")
    return BacktestSubmitResponse(
        task_id=task_id,
        status="accepted",
        submitted_at=datetime.now().isoformat(),
    )


@router.get("/compare/{task_id}")
async def get_compare_result(task_id: str):
    """查询多策略对比任务状态/结果。"""
    with _compare_lock:
        task = _compare_tasks.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"对比任务 {task_id} 不存在")

    status = task["status"]

    if getattr(status, "value", status) == BacktestStatusEnum.COMPLETED.value and task.get("result"):
        return task["result"]

    return {
        "task_id": task_id,
        "status": status,
        "results": [],
        "error": task.get("error"),
    }
