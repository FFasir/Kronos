"""
Kronos 指数/ETF 择时策略系统 — Pydantic 数据模型

定义所有 API 请求/响应的数据结构，用于 FastAPI 自动校验和 Swagger 文档生成。
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════
# 枚举类型
# ═══════════════════════════════════════════════════════════════════

class SignalTypeEnum(str, Enum):
    """择时信号类型"""
    BUY = "BUY"
    HOLD = "HOLD"
    CASH = "CASH"
    SELL = "SELL"


class BacktestStatusEnum(str, Enum):
    """回测任务状态"""
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AssetTypeEnum(str, Enum):
    """资产类型"""
    INDEX = "index"
    ETF = "etf"
    STOCK = "stock"


# ═══════════════════════════════════════════════════════════════════
# 指数/ETF 数据模型
# ═══════════════════════════════════════════════════════════════════

class IndexInfo(BaseModel):
    """标的基本信息"""
    code: str = Field(..., description="标的代码，如 000852", examples=["000852"])
    name: str = Field(..., description="标的名称", examples=["中证1000"])
    type: AssetTypeEnum = Field(..., description="资产类型: index | etf")
    track_index: Optional[str] = Field(None, description="ETF跟踪的指数代码")
    priority: Optional[str] = Field(None, description="优先级: P0/P1/P2")
    latest_close: Optional[float] = Field(None, description="最新收盘价")
    change_pct: Optional[float] = Field(None, description="涨跌幅 (%)")
    update_date: Optional[str] = Field(None, description="数据更新日期")
    industry: Optional[str] = Field(None, description="所属行业")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "000852",
                "name": "中证1000",
                "type": "index",
                "track_index": None,
                "priority": "P0",
                "latest_close": 6234.50,
                "change_pct": 1.25,
                "update_date": "2026-07-11"
            }
        }
    }


class IndexDetail(BaseModel):
    """单个标的详情（含最近N天OHLCV）"""
    code: str
    name: str
    type: AssetTypeEnum
    track_index: Optional[str] = None
    latest_close: Optional[float] = None
    change_pct: Optional[float] = None
    kline: Optional[List[OHLCVPoint]] = None
    # 展示口径提示：个股 K线/预测来自后复权序列("hfq")，指数/ETF 为真实价格("raw")。
    price_scale: Optional[str] = Field(None, description="价格口径: hfq(后复权) | raw(不复权)")
    # 个股真实市价（来自不复权快照），仅当有快照时返回。
    market_close: Optional[float] = Field(None, description="最新市价(不复权快照)")
    market_date: Optional[str] = Field(None, description="市价快照日期")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "000852",
                "name": "中证1000",
                "type": "index",
                "latest_close": 6234.50,
                "change_pct": 1.25,
                "kline": [
                    {"date": "2026-07-10", "open": 6180.0, "high": 6250.0,
                     "low": 6175.0, "close": 6234.5, "volume": 123456000}
                ]
            }
        }
    }


class OHLCVPoint(BaseModel):
    """单日 OHLCV 数据点"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: Optional[float] = None
    turnover: Optional[float] = None


class IndexListResponse(BaseModel):
    """标的列表响应"""
    total: int
    indices: List[IndexInfo]


class KlineResponse(BaseModel):
    """K线数据响应"""
    code: str
    name: str
    start: str
    end: str
    count: int
    data: List[OHLCVPoint]


class PredictedOHLCVPoint(BaseModel):
    """预测单日 OHLCV 数据点（含置信区间）"""
    date: str = Field(..., description="预测日期 YYYY-MM-DD")
    open: float = Field(..., description="预测开盘价")
    high: float = Field(..., description="预测最高价")
    low: float = Field(..., description="预测最低价")
    close: float = Field(..., description="预测收盘价")
    volume: float = Field(0, description="预测成交量")
    confidence: float = Field(0.5, description="预测置信度 0-1")
    lower_bound: float = Field(0, description="置信区间下界")
    upper_bound: float = Field(0, description="置信区间上界")


class StockPredictionResponse(BaseModel):
    """个股预测 K 线响应"""
    code: str
    name: str
    pred_days: int = Field(..., description="预测天数")
    predictions: List[PredictedOHLCVPoint]


# ═══════════════════════════════════════════════════════════════════
# 预测数据模型
# ═══════════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    """单条预测结果"""
    symbol: str = Field(..., description="标的代码")
    name: Optional[str] = Field(None, description="标的名称")
    pred_date: str = Field(..., description="预测生成日期")
    pred_close_t5: float = Field(..., description="预测未来第5日收盘价")
    pred_return: float = Field(..., description="预测5日收益率")
    last_close: float = Field(..., description="预测时的最新收盘价")
    confidence: float = Field(..., description="置信度 0-1", ge=0.0, le=1.0)
    prob_up: Optional[float] = Field(None, description="上涨概率 0-1", ge=0.0, le=1.0)
    score: Optional[float] = Field(None, description="预期收益分值 0-10", ge=0.0, le=10.0)
    temperature: Optional[float] = Field(None, description="推理参数 T")
    top_p: Optional[float] = Field(None, description="推理参数 top_p")
    lookback_window: Optional[int] = Field(None, description="回溯窗口天数")
    actual_close: Optional[float] = Field(None, description="实际收盘价（事后填充）")
    actual_return: Optional[float] = Field(None, description="实际收益率（事后填充）")
    error: Optional[str] = Field(None, description="预测错误信息（如有）")
    as_of_date: Optional[str] = Field(None, description="预测生成日期（所用行情截止日）")
    last_close_raw: Optional[float] = Field(None, description="最新市价（不复权）")
    pred_close_t5_raw: Optional[float] = Field(None, description="预测收盘价（市价口径，不复权）")
    price_scale: Optional[str] = Field(None, description="预测价格口径: hfq(后复权) | raw(不复权)")
    is_anomalous: Optional[bool] = Field(None, description="是否数据异常预测（|5日收益|超合理区间）")
    prediction_std: Optional[float] = Field(None, description="预测日收盘采样标准差（后复权价单位）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "000852",
                "name": "中证1000",
                "pred_date": "2026-07-10",
                "pred_close_t5": 6310.25,
                "pred_return": 0.0122,
                "last_close": 6234.50,
                "confidence": 0.72,
                "temperature": 1.0,
                "top_p": 0.95,
                "lookback_window": 60
            }
        }
    }


class PredictionListResponse(BaseModel):
    """预测列表响应"""
    total: int
    predictions: List[PredictionResult]
    data_date: Optional[str] = Field(None, description="本批预测的生成日期（最新行情截止日）")


class PredictionHistoryResponse(BaseModel):
    """历史预测响应（含实际对比）"""
    symbol: str
    name: Optional[str] = None
    start: str
    end: str
    count: int
    records: List[PredictionResult]


class AccuracyStats(BaseModel):
    """预测准确率统计"""
    symbol: str
    name: Optional[str] = None
    direction_accuracy: float = Field(..., description="方向准确率")
    mae: float = Field(..., description="平均绝对误差")
    rmse: float = Field(..., description="均方根误差")
    spearman: Optional[float] = Field(None, description="Spearman 秩相关系数")
    total_predictions: int = Field(..., description="预测总次数")
    period_start: Optional[str] = None
    period_end: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# 信号数据模型
# ═══════════════════════════════════════════════════════════════════

class SignalResult(BaseModel):
    """单条择时信号"""
    symbol: str = Field(..., description="标的代码")
    name: Optional[str] = Field(None, description="标的名称")
    date: str = Field(..., description="信号日期")
    signal_type: SignalTypeEnum = Field(..., description="信号类型")
    pred_return: float = Field(..., description="预测收益率")
    pred_close: Optional[float] = Field(None, description="预测收盘价")
    confidence: Optional[float] = Field(None, description="置信度")
    position_pct: float = Field(..., description="建议仓位比例")
    reason: Optional[str] = Field(None, description="信号原因说明")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "000852",
                "name": "中证1000",
                "date": "2026-07-11",
                "signal_type": "BUY",
                "pred_return": 0.0122,
                "pred_close": 6310.25,
                "confidence": 0.72,
                "position_pct": 0.30,
                "reason": "预测收益率 1.22% > 0.5%"
            }
        }
    }


class SignalListResponse(BaseModel):
    """信号列表响应"""
    total: int
    date: Optional[str] = None
    signals: List[SignalResult]


class SignalHistoryResponse(BaseModel):
    """单个标的历史信号"""
    symbol: str
    name: Optional[str] = None
    days: int
    count: int
    signals: List[SignalResult]


class SignalSummary(BaseModel):
    """信号汇总统计"""
    date: str
    total: int
    buy_count: int
    hold_count: int
    sell_count: int
    cash_count: int


# ═══════════════════════════════════════════════════════════════════
# 回测数据模型
# ═══════════════════════════════════════════════════════════════════

class BacktestRequest(BaseModel):
    """回测任务提交请求"""
    scope: str = Field(
        "paper_index",
        pattern="^(paper_index|csi1000_components|custom_index)$",
        description="指数择时(000852→512100) | 中证1000逐股 | 自定义指数择时(在线重建信号)",
    )
    symbols: List[str] = Field(default_factory=list, description="自定义标的列表")
    start_date: str = Field(..., description="回测起始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="回测结束日期 YYYY-MM-DD")
    # ── 自定义指数择时（scope=custom_index）──
    signal_symbol: str = Field("000852", description="信号资产（被预测的指数/ETF代码）")
    execution_symbol: str = Field("512100", description="执行资产（交易标的，默认指数对应ETF）")
    leakage_mode: str = Field(
        "demo",
        pattern="^(demo|strict)$",
        description="demo=允许前视（模型可能见过区间行情，结果偏乐观，仅供演示）；strict=仅统计训练截止之后的信号（当前几乎无样本）",
    )
    start_date: str = Field(..., description="回测起始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="回测结束日期 YYYY-MM-DD")
    initial_capital: float = Field(1000000.0, ge=10000, description="初始资金")
    strategy_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="策略参数覆盖"
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def check_date_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"日期格式必须为 YYYY-MM-DD: {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "scope": "paper_index",
                "symbols": [],
                "start_date": "2024-01-01",
                "end_date": "2026-06-30",
                "initial_capital": 1000000.0,
                "strategy_params": {"buy_threshold": 0.005, "hold_days": 5}
            }
        }
    }


class BacktestSubmitResponse(BaseModel):
    """回测任务提交响应"""
    task_id: str = Field(..., description="任务ID (UUID)")
    status: str = Field("accepted", description="任务状态")
    submitted_at: str = Field(..., description="提交时间 ISO 8601")


class BacktestProgressResponse(BaseModel):
    """回测任务进度响应"""
    task_id: str
    status: BacktestStatusEnum
    progress: float = Field(0.0, ge=0.0, le=1.0, description="进度 0-1")
    message: Optional[str] = None


class DailyEquityPoint(BaseModel):
    """每日权益记录"""
    date: str
    cash: float
    position_value: float
    total_equity: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    benchmark_value: Optional[float] = None
    num_positions: int


class TradeRecord(BaseModel):
    """交易记录"""
    date: str
    symbol: str
    direction: str  # BUY / SELL
    price: float
    quantity: int
    amount: float
    commission: float
    stamp_tax: float
    slippage: float
    net_amount: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_days: Optional[int] = None
    reason: Optional[str] = None


class MonthlyReturn(BaseModel):
    """月度收益率"""
    month: str
    return_value: float = Field(..., alias="return")
    start_equity: float
    end_equity: float


class YearlyReturn(BaseModel):
    """年度收益率"""
    year: int
    return_value: float = Field(..., alias="return")
    cumulative_return: float
    start_equity: float
    end_equity: float


class DrawdownPeriod(BaseModel):
    """回撤区间"""
    peak_date: Optional[str] = None
    trough_date: Optional[str] = None
    recovery_date: Optional[str] = None
    peak_value: float
    trough_value: float
    drawdown_days: int


class RiskEvent(BaseModel):
    """风控事件"""
    date: str
    symbol: str
    event_type: str
    reason: str
    position_pnl: Optional[float] = None


class BacktestResult(BaseModel):
    """回测完整结果"""
    task_id: str
    status: BacktestStatusEnum
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    elapsed_seconds: float
    total_trades: int

    # 10 项绩效指标
    cumulative_return: float = Field(..., description="累计收益率")
    annual_return: float = Field(..., description="年化收益率")
    max_drawdown: float = Field(..., description="最大回撤")
    win_rate: float = Field(..., description="胜率")
    profit_loss_ratio: Optional[float] = Field(None, description="盈亏比")
    sharpe_ratio: float = Field(..., description="夏普比率")
    calmar_ratio: float = Field(..., description="卡玛比率")
    excess_return: float = Field(..., description="超额收益")
    max_consecutive_wins: int = Field(..., description="最大连胜")
    max_consecutive_losses: int = Field(..., description="最大连亏")
    avg_hold_days: float = Field(..., description="平均持仓天数")

    # 明细
    daily_equity: List[DailyEquityPoint]
    trades: List[TradeRecord]
    benchmark_equity: Optional[List[Dict[str, Any]]] = None
    risk_events: List[RiskEvent]
    monthly_returns: List[MonthlyReturn]
    yearly_returns: List[YearlyReturn]
    max_drawdown_period: Optional[DrawdownPeriod] = None

    # 配置
    config: Dict[str, Any]


class BacktestHistoryItem(BaseModel):
    """回测历史记录摘要"""
    task_id: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    cumulative_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    status: str
    completed_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# 管理后台数据模型
# ═══════════════════════════════════════════════════════════════════

class AdminLoginRequest(BaseModel):
    """管理员登录请求"""
    username: str = Field("admin", min_length=1, max_length=64, description="管理员账号")
    password: str = Field(..., min_length=1, description="管理后台密码")

    model_config = {
        "json_schema_extra": {"example": {"username": "admin", "password": "your-password"}}
    }


class AdminLoginResponse(BaseModel):
    """管理员登录响应"""
    success: bool
    token: Optional[str] = None
    message: Optional[str] = None


class SystemMonitor(BaseModel):
    """系统监控状态"""
    model_status: str = Field(..., description="模型状态: running/error/idle")
    data_status: str = Field(..., description="数据状态: ready/stale/error")
    cpu_usage: Optional[float] = Field(None, description="CPU 使用率 %")
    memory_usage: Optional[float] = Field(None, description="内存使用率 %")
    gpu_usage: Optional[float] = Field(None, description="GPU 使用率 %")
    gpu_memory_used: Optional[float] = Field(None, description="GPU 显存使用 MB")
    gpu_memory_total: Optional[float] = Field(None, description="GPU 显存总量 MB")
    model_version: Optional[str] = Field(None, description="当前模型版本")
    last_training_time: Optional[str] = Field(None, description="最后训练时间")
    last_prediction_time: Optional[str] = Field(None, description="最后预测时间")
    last_data_update: Optional[str] = Field(None, description="最后数据更新时间")
    uptime: Optional[str] = Field(None, description="服务运行时间")
    pool_count: Optional[int] = Field(None, description="选股池标的数")
    active_positions: Optional[int] = Field(None, description="当前持仓数")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_status": "running",
                "data_status": "ready",
                "cpu_usage": 35.2,
                "memory_usage": 62.1,
                "gpu_usage": 78.5,
                "gpu_memory_used": 6144,
                "gpu_memory_total": 24576,
                "model_version": "kronos-base-v2",
                "last_training_time": "2026-07-01T02:30:00",
                "last_prediction_time": "2026-07-11T16:00:00",
                "last_data_update": "2026-07-11T15:35:00",
                "uptime": "5d 12h 30m",
                "pool_count": 12,
                "active_positions": 2
            }
        }
    }


class TrainingLogEntry(BaseModel):
    """训练日志条目"""
    id: Optional[int] = None
    window_id: str = Field(..., description="训练窗口ID")
    start_time: str = Field(..., description="训练开始时间")
    end_time: Optional[str] = Field(None, description="训练结束时间")
    status: str = Field(..., description="训练状态: running/completed/failed")
    epochs_completed: Optional[int] = Field(None, description="完成epochs")
    best_val_loss: Optional[float] = Field(None, description="最佳验证损失")
    best_spearman: Optional[float] = Field(None, description="最佳Spearman")
    model_path: Optional[str] = Field(None, description="模型保存路径")
    error_message: Optional[str] = Field(None, description="错误信息")


class DataSourceInfo(BaseModel):
    """数据源状态信息"""
    symbol: str
    name: str
    type: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_rows: Optional[int] = None
    status: str = Field(..., description="数据状态: ready/stale/missing/error")
    missing_dates: Optional[int] = Field(None, description="缺失日期数")


class DashboardSummary(BaseModel):
    """仪表盘综合摘要"""
    model_status: Optional[str] = None
    data_status: Optional[str] = None
    signal_summary: Optional[SignalSummary] = None
    signal_total: Optional[int] = None
    pool_count: Optional[int] = None
    last_update: Optional[str] = None
    gpu_usage: Optional[float] = None
    uptime: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# 策略预设 & 选股统计
# ═══════════════════════════════════════════════════════════════════

class StrategyPreset(BaseModel):
    """策略风格预设"""
    name: str = Field(..., description="预设标识: conservative/balanced/aggressive")
    label: str = Field(..., description="中文标签")
    icon: str = Field("", description="图标emoji")
    description: str = Field("", description="预设说明")
    prob_up_threshold: float = Field(..., description="上涨概率阈值")
    score_threshold: float = Field(..., description="收益分值阈值")
    min_daily_amount: float = Field(..., description="最低日均成交额")
    max_per_industry: int = Field(..., description="单一行业最大持仓数")
    top_k: int = Field(..., description="选股Top-K")


class StrategyPresetListResponse(BaseModel):
    """策略预设列表响应"""
    presets: List[StrategyPreset]


class EliminatedItem(BaseModel):
    """被剔除的股票记录"""
    symbol: str = Field(..., description="股票代码")
    name: str = Field("", description="股票名称")
    reason: str = Field(..., description="剔除原因")
    stage: str = Field(..., description="筛选阶段: prob_score/st_suspended/liquidity/industry/top_k")


class SelectionStats(BaseModel):
    """选股筛选统计"""
    initial_count: int = Field(0, description="初始预测总数")
    anomaly_count: int = Field(0, description="数据异常剔除数（|5日收益|超阈值）")
    after_prob_score: int = Field(0, description="概率+分值筛选后")
    after_st_suspended: int = Field(0, description="ST/停牌过滤后")
    after_liquidity: int = Field(0, description="流动性过滤后")
    after_industry: int = Field(0, description="行业集中度过滤后")
    final_count: int = Field(0, description="最终入选数")
    eliminated: List[EliminatedItem] = Field(default_factory=list, description="被剔除记录")


class SelectionGenerateRequest(BaseModel):
    """手动生成选股池请求"""
    preset: Optional[str] = Field(None, description="预设名称")
    prob_up_threshold: Optional[float] = Field(None, description="上涨概率阈值覆盖")
    score_threshold: Optional[float] = Field(None, description="收益分值阈值覆盖")
    min_daily_amount: Optional[float] = Field(None, description="最低日均成交额覆盖")
    max_per_industry: Optional[int] = Field(None, description="行业集中度上限覆盖")
    top_k: Optional[int] = Field(None, description="Top-K覆盖")


# ═══════════════════════════════════════════════════════════════════
# 策略优化 & 多策略对比
# ═══════════════════════════════════════════════════════════════════

class ParamRange(BaseModel):
    """单个参数的搜索范围"""
    name: str = Field(..., description="参数名")
    values: List[float] = Field(..., description="参数取值列表")


class OptimizeRequest(BaseModel):
    """参数优化任务请求"""
    param_ranges: List[ParamRange] = Field(..., description="参数搜索范围")
    start_date: str = Field(..., description="优化起始日")
    end_date: str = Field(..., description="优化结束日")
    initial_capital: float = Field(1000000.0, ge=10000, description="初始资金")
    mode: str = Field("grid", pattern="^(grid|walk_forward)$", description="优化模式")
    metric: str = Field("sharpe_ratio", description="优化指标")
    scope: str = Field("paper_index", pattern="^(paper_index|csi1000_components)$")
    window_months: int = Field(12, ge=3, description="Walk-Forward训练窗口月数")
    step_months: int = Field(3, ge=1, description="Walk-Forward步长月数")


class OptimizeResultItem(BaseModel):
    """单个参数组合的优化结果"""
    params: Dict[str, Any] = Field(..., description="参数组合")
    cumulative_return: float = Field(0, description="累计收益率")
    annual_return: float = Field(0, description="年化收益率")
    max_drawdown: float = Field(3, description="最大回撤")
    sharpe_ratio: float = Field(0, description="夏普比率")
    win_rate: float = Field(0, description="胜率")
    total_trades: int = Field(0, description="总交易数")


class WalkForwardWindowResult(BaseModel):
    """Walk-Forward 单窗口结果"""
    window_id: int = Field(..., description="窗口序号")
    train_start: str = Field(..., description="训练期起点")
    train_end: str = Field(..., description="训练期终点")
    test_start: str = Field(..., description="测试期起点")
    test_end: str = Field(..., description="测试期终点")
    best_params: Dict[str, Any] = Field(..., description="训练期最优参数")
    oos_sharpe: float = Field(0, description="OOS夏普比率")
    oos_return: float = Field(0, description="OOS年化收益率")
    oos_drawdown: float = Field(0, description="OOS最大回撤")


class OptimizeResult(BaseModel):
    """参数优化完整结果"""
    task_id: str
    status: BacktestStatusEnum
    mode: str = Field("grid", description="优化模式")
    metric: str = Field("sharpe_ratio", description="优化指标")
    total_combinations: int = Field(0, description="总参数组合数")
    progress: float = Field(0, ge=0, le=1, description="进度")
    results: List[OptimizeResultItem] = Field(default_factory=list, description="所有组合结果")
    best_params: Optional[Dict[str, Any]] = Field(None, description="最优参数组合")
    best_metrics: Optional[OptimizeResultItem] = Field(None, description="最优指标")
    sensitivity: Optional[Dict[str, Any]] = Field(None, description="参数敏感度分析")
    walk_forward_windows: List[WalkForwardWindowResult] = Field(default_factory=list, description="Walk-Forward窗口结果")
    stability_report: Optional[Dict[str, Any]] = Field(None, description="稳定性报告")
    error: Optional[str] = Field(None, description="错误信息")


class CompareRequest(BaseModel):
    """多策略对比请求"""
    strategies: List[Dict[str, Any]] = Field(..., min_length=2, max_length=5, description="策略参数组合列表(2-5组)")
    start_date: str = Field(..., description="对比起始日")
    end_date: str = Field(..., description="对比结束日")
    initial_capital: float = Field(1000000.0, ge=10000, description="初始资金")
    scope: str = Field("paper_index", pattern="^(paper_index|csi1000_components)$")


class CompareStrategyResult(BaseModel):
    """单个策略的对比结果"""
    strategy_id: int = Field(..., description="策略序号")
    params: Dict[str, Any] = Field(..., description="策略参数")
    cumulative_return: float = Field(0)
    annual_return: float = Field(0)
    max_drawdown: float = Field(0)
    sharpe_ratio: float = Field(0)
    calmar_ratio: float = Field(0)
    win_rate: float = Field(0)
    total_trades: int = Field(0)


class CompareResult(BaseModel):
    """多策略对比完整结果"""
    task_id: str
    status: BacktestStatusEnum
    results: List[CompareStrategyResult] = Field(default_factory=list, description="各策略结果")
    best_by_metric: Optional[Dict[str, int]] = Field(None, description="各指标最优策略ID映射")
    error: Optional[str] = Field(None, description="错误信息")


# ═══════════════════════════════════════════════════════════════════
# 通用响应模型
# ═══════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """API 错误响应"""
    error: bool = True
    error_code: str = Field(..., description="错误码，如 E200")
    message: str = Field(..., description="错误描述")
    detail: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": True,
                "error_code": "E200",
                "message": "请求参数无效",
                "detail": "start_date 格式错误"
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════
# 实时行情（新浪）模型
# ═══════════════════════════════════════════════════════════════════

class QuoteInfo(BaseModel):
    """单只股票实时行情"""
    code: str = Field(..., description="股票代码")
    name: Optional[str] = Field(None, description="股票名称")
    price: Optional[float] = Field(None, description="最新价（不复权）")
    prev_close: Optional[float] = Field(None, description="昨收价")
    change: Optional[float] = Field(None, description="涨跌额")
    change_pct: Optional[float] = Field(None, description="涨跌幅 (%)")
    open: Optional[float] = Field(None, description="今开")
    high: Optional[float] = Field(None, description="最高")
    low: Optional[float] = Field(None, description="最低")
    volume: Optional[float] = Field(None, description="成交量（股）")
    amount: Optional[float] = Field(None, description="成交额（元）")
    date: Optional[str] = Field(None, description="行情日期 YYYY-MM-DD")
    time: Optional[str] = Field(None, description="行情时间 HH:MM:SS")
    suspended: bool = Field(False, description="是否停牌/无行情")
    source: Optional[str] = Field(None, description="数据来源: sina_realtime | daily_snapshot")


class QuoteListResponse(BaseModel):
    """实时行情列表响应"""
    quotes: List[QuoteInfo]
    as_of: Optional[str] = Field(None, description="行情时间")


# ═══════════════════════════════════════════════════════════════════
# 策略观点模型
# ═══════════════════════════════════════════════════════════════════

class StrategyOpinionTarget(BaseModel):
    """单个标的的策略观点"""
    symbol: str = Field(..., description="标的代码")
    name: Optional[str] = Field(None, description="标的名称")
    signal_type: str = Field(..., description="当前信号: BUY/HOLD/SELL/CASH")
    pred_return: Optional[float] = Field(None, description="模型预测5日收益率")
    position_pct: float = Field(0.0, description="建议仓位比例 0-1")
    reason: Optional[str] = Field(None, description="决策原因")
    date: Optional[str] = Field(None, description="信号生成日期")


class StrategyOpinion(BaseModel):
    """策略观点：为什么当前是这个仓位，什么条件下会变"""
    strategy_name: str
    rule: str
    buy_threshold: float
    hold_days: int
    execution_symbol: Optional[str] = None
    risk_controls_enabled: bool = False
    risk_rules: List[str] = Field(default_factory=list)
    latest_data_date: Optional[str] = Field(None, description="最新预测数据日期（判断信号是否过期）")
    targets: List[StrategyOpinionTarget] = Field(default_factory=list)
    generated_at: str = Field(..., description="响应生成时间")
