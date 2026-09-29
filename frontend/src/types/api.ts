/**
 * API 响应类型定义 — 对应 FastAPI / src/service_layer/models.py Pydantic 模型
 *
 * 命名约定：
 * - 字段名与后端 Pydantic 模型保持一致（snake_case）
 * - 可选字段标记 `?` 对应 Optional[...]
 * - 枚举值使用字符串联合类型对应 Python str Enum
 */

// ═══════════════════════════════════════════════════════════════════
// Generic
// ═══════════════════════════════════════════════════════════════════

/** 标准 API 响应包装 — 对应后端模型（无统一包装层，直接返回 data） */
export interface ApiError {
  error: true
  error_code: string
  message: string
  detail?: string
}

/** 分页信息 */
export interface PageInfo {
  page: number
  page_size: number
  total: number
}

// ═══════════════════════════════════════════════════════════════════
// Index / ETF (api_index.py)
// ═══════════════════════════════════════════════════════════════════

/** 资产类型 */
export type AssetType = 'index' | 'etf' | 'stock'

/** 标的基本信息 — IndexInfo */
export interface IndexInfo {
  code: string
  name: string
  type: AssetType
  track_index?: string
  priority?: string
  latest_close?: number
  change_pct?: number
  update_date?: string
  industry?: string
}

/** 标的列表响应 — IndexListResponse */
export interface IndexListResponse {
  total: number
  indices: IndexInfo[]
}

/** 单日 OHLCV 数据点 — OHLCVPoint */
export interface KlineBar {
  date: string // YYYY-MM-DD
  open: number
  high: number
  low: number
  close: number
  volume: number
  amount?: number
  turnover?: number
}

/** 单个标的详情 + K线 — IndexDetail */
export interface IndexDetail {
  code: string
  name: string
  type: AssetType
  track_index?: string
  latest_close?: number
  change_pct?: number
  kline?: KlineBar[]
  /** 价格口径: hfq(后复权) | raw(不复权真实价) */
  price_scale?: 'hfq' | 'raw' | string
  /** 个股真实市价(不复权快照) */
  market_close?: number
  /** 市价快照日期 */
  market_date?: string
}

/** K线数据响应 — KlineResponse */
export interface KlineResponse {
  code: string
  name: string
  start: string
  end: string
  count: number
  data: KlineBar[]
}

/** 预测单日 OHLCV 数据点（含置信区间）— PredictedOHLCVPoint */
export interface PredictedOHLCVPoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  confidence: number
  lower_bound: number
  upper_bound: number
}

/** 个股预测 K 线响应 — StockPredictionResponse */
export interface StockPredictionResponse {
  code: string
  name: string
  pred_days: number
  predictions: PredictedOHLCVPoint[]
}

// ═══════════════════════════════════════════════════════════════════
// Prediction (api_prediction.py)
// ═══════════════════════════════════════════════════════════════════

/** 单条预测结果 — PredictionResult */
export interface PredictionResult {
  symbol: string
  name?: string
  pred_date: string
  pred_close_t5: number
  pred_return: number
  last_close: number
  confidence: number // 0-1
  prob_up?: number // 0-1
  score?: number // 0-10
  temperature?: number
  top_p?: number
  lookback_window?: number
  actual_close?: number // 事后填充
  actual_return?: number // 事后填充
  error?: string
  /** 预测生成日期（所用行情截止日） */
  as_of_date?: string
  /** 最新市价（不复权） */
  last_close_raw?: number
  /** 预测收盘价（市价口径，不复权） */
  pred_close_t5_raw?: number
  /** 预测价格口径 */
  price_scale?: 'hfq' | 'raw' | string
  /** 是否数据异常预测（|5日收益|超合理区间） */
  is_anomalous?: boolean
  /** 预测日收盘采样标准差（后复权价单位；除以最新价即收益标准差） */
  prediction_std?: number
}

/** 预测列表响应 — PredictionListResponse */
export interface PredictionListResponse {
  total: number
  predictions: PredictionResult[]
  /** 本批预测的生成日期（最新行情截止日） */
  data_date?: string
}

/** 历史预测响应 — PredictionHistoryResponse */
export interface PredictionHistoryResponse {
  symbol: string
  name?: string
  start: string
  end: string
  count: number
  records: PredictionResult[]
}

/** 预测准确率统计 — AccuracyStats */
export interface AccuracyStats {
  symbol: string
  name?: string
  direction_accuracy: number
  mae: number
  rmse: number
  spearman?: number
  total_predictions: number
  period_start?: string
  period_end?: string
}

// ═══════════════════════════════════════════════════════════════════
// Signal (api_signal.py)
// ═══════════════════════════════════════════════════════════════════

/** 信号类型 */
export type SignalType = 'BUY' | 'HOLD' | 'CASH' | 'SELL'

/** 单条择时信号 — SignalResult */
export interface SignalResult {
  symbol: string
  name?: string
  date: string
  signal_type: SignalType
  pred_return: number
  pred_close?: number
  confidence?: number
  position_pct: number
  reason?: string
}

/** 信号列表响应 — SignalListResponse */
export interface SignalListResponse {
  total: number
  date?: string
  signals: SignalResult[]
}

/** 单个标的历史信号 — SignalHistoryResponse */
export interface SignalHistoryResponse {
  symbol: string
  name?: string
  days: number
  count: number
  signals: SignalResult[]
}

/** 信号汇总统计 — SignalSummary */
export interface SignalSummary {
  date: string
  total: number
  buy_count: number
  hold_count: number
  sell_count: number
  cash_count: number
}

/** 个股详情（信号 + K线 + 评分）— 前端组合类型 */
export interface StockDetail {
  stock_code: string
  stock_name: string
  industry?: string
  market_cap?: number
  kline: KlineBar[]
  predicted_kline?: KlineBar[]
  confidence_interval?: [number, number][]
  signals: SignalResult[]
  score: {
    trend: number
    momentum: number
    volume: number
    composite: number
  }
}

// ═══════════════════════════════════════════════════════════════════
// Backtest (api_backtest.py)
// ═══════════════════════════════════════════════════════════════════

/** 回测任务状态 */
export type BacktestStatus = 'accepted' | 'running' | 'completed' | 'failed' | 'timeout'

/** 回测任务提交请求 — BacktestRequest */
export interface BacktestRequest {
  scope?: 'paper_index' | 'csi1000_components' | 'custom_index'
  symbols: string[]
  start_date: string
  end_date: string
  initial_capital?: number
  strategy_params?: Record<string, unknown>
  /** scope=custom_index：信号资产（被预测的指数） */
  signal_symbol?: string
  /** scope=custom_index：执行资产（交易标的） */
  execution_symbol?: string
  /** scope=custom_index：demo=允许前视（演示）；strict=仅统计训练截止后信号 */
  leakage_mode?: 'demo' | 'strict'
}

/** 回测任务提交响应 — BacktestSubmitResponse */
export interface BacktestSubmitResponse {
  task_id: string
  status: string
  submitted_at: string
}

/** 回测可用性诊断 — GET /api/v1/backtest/availability */
export interface BacktestScopeAvailability {
  enabled: boolean
  signal_dates?: string[]
  min_date?: string | null
  max_date?: string | null
  count: number
  note: string
}

export interface BacktestAvailability {
  paper_index: BacktestScopeAvailability
  csi1000_components: BacktestScopeAvailability
}

/** 回测任务进度 — BacktestProgressResponse */
export interface BacktestProgress {
  task_id: string
  status: BacktestStatus
  progress: number // 0-1
  message?: string
}

/** 每日权益记录 — DailyEquityPoint */
export interface DailyEquityPoint {
  date: string
  cash: number
  position_value: number
  total_equity: number
  daily_return: number
  cumulative_return: number
  drawdown: number
  benchmark_value?: number
  num_positions: number
}

/** 交易记录 — TradeRecord */
export interface TradeRecord {
  date: string
  symbol: string
  direction: string // BUY / SELL
  price: number
  quantity: number
  amount: number
  commission: number
  stamp_tax: number
  slippage: number
  net_amount: number
  pnl?: number
  pnl_pct?: number
  hold_days?: number
  reason?: string
}

/** 月度收益率 — MonthlyReturn */
export interface MonthlyReturn {
  month: string
  return: number // 后端用别名
  start_equity: number
  end_equity: number
}

/** 年度收益率 — YearlyReturn */
export interface YearlyReturn {
  year: number
  return: number
  cumulative_return: number
  start_equity: number
  end_equity: number
}

/** 回撤区间 — DrawdownPeriod */
export interface DrawdownPeriod {
  peak_date?: string
  trough_date?: string
  recovery_date?: string
  peak_value: number
  trough_value: number
  drawdown_days: number
}

/** 风控事件 — RiskEvent */
export interface RiskEvent {
  date: string
  symbol: string
  event_type: string
  reason: string
  position_pnl?: number
}

/** 回测完整结果 — BacktestResult */
export interface BacktestResult {
  task_id: string
  status: BacktestStatus
  /** 运行进度 0-1（轮询时后端实时返回） */
  progress?: number
  /** 进度消息 */
  message?: string
  start_date: string
  end_date: string
  initial_capital: number
  final_equity: number
  elapsed_seconds: number
  total_trades: number

  // 绩效指标
  cumulative_return: number
  annual_return: number
  max_drawdown: number
  win_rate: number
  profit_loss_ratio?: number
  sharpe_ratio: number
  calmar_ratio: number
  excess_return: number
  max_consecutive_wins: number
  max_consecutive_losses: number
  avg_hold_days: number

  // 明细
  daily_equity: DailyEquityPoint[]
  trades: TradeRecord[]
  benchmark_equity?: { date: string; value: number }[]
  risk_events: RiskEvent[]
  monthly_returns: MonthlyReturn[]
  yearly_returns: YearlyReturn[]
  max_drawdown_period?: DrawdownPeriod

  // 配置
  config: Record<string, unknown>
}

/** 回测历史记录摘要 — BacktestHistoryItem */
export interface BacktestHistoryItem {
  task_id: string
  start_date: string
  end_date: string
  initial_capital: number
  final_equity: number
  cumulative_return: number
  annual_return: number
  sharpe_ratio: number
  max_drawdown: number
  total_trades: number
  status: string
  completed_at?: string
}

// ═══════════════════════════════════════════════════════════════════
// Admin (api_admin.py)
// ═══════════════════════════════════════════════════════════════════

/** 管理员登录请求 — AdminLoginRequest */
export interface AdminLoginRequest {
  username: string
  password: string
}

/** 管理员登录响应 — AdminLoginResponse */
export interface AdminLoginResponse {
  success: boolean
  token?: string
  message?: string
}

/** 系统监控状态 — SystemMonitor */
export interface SystemMonitor {
  model_status: string // running | error | idle
  data_status: string // ready | stale | error
  cpu_usage?: number
  memory_usage?: number
  gpu_usage?: number
  gpu_memory_used?: number
  gpu_memory_total?: number
  model_version?: string
  last_training_time?: string
  last_prediction_time?: string
  last_data_update?: string
  uptime?: string
  pool_count?: number
  active_positions?: number
}

/** 免登录系统状态摘要 — GET /api/v1/system/status */
export interface SystemStatus {
  model_status: string // running | error | idle
  data_status: string // ready | stale | error
  model_version?: string
  last_data_update?: string
  last_prediction_time?: string
  /** 预测覆盖统计 */
  prediction_coverage?: PredictionCoverage
}

/** 预测覆盖统计 */
export interface PredictionCoverage {
  universe: number
  predicted: number
  prediction_date?: string
  coverage_pct?: number
}

// ═══════════════════════════════════════════════════════════════════
// 实时行情（api_quote.py）
// ═══════════════════════════════════════════════════════════════════

/** 单只股票实时行情 — QuoteInfo */
export interface QuoteInfo {
  code: string
  name?: string
  price?: number
  prev_close?: number
  change?: number
  change_pct?: number
  open?: number
  high?: number
  low?: number
  volume?: number
  amount?: number
  date?: string
  time?: string
  suspended?: boolean
  source?: string
}

/** 实时行情列表响应 — QuoteListResponse */
export interface QuoteListResponse {
  quotes: QuoteInfo[]
  as_of?: string
}

// ═══════════════════════════════════════════════════════════════════
// 策略观点（api_strategy.py）
// ═══════════════════════════════════════════════════════════════════

/** 单个标的的策略观点 — StrategyOpinionTarget */
export interface StrategyOpinionTarget {
  symbol: string
  name?: string
  signal_type: string // BUY/HOLD/SELL/CASH
  pred_return?: number
  position_pct: number
  reason?: string
  date?: string
}

/** 策略观点 — StrategyOpinion */
export interface StrategyOpinion {
  strategy_name: string
  rule: string
  buy_threshold: number
  hold_days: number
  execution_symbol?: string
  risk_controls_enabled: boolean
  risk_rules: string[]
  latest_data_date?: string
  targets: StrategyOpinionTarget[]
  generated_at: string
}

/** 训练日志条目 — TrainingLogEntry */
export interface TrainingLogEntry {
  id?: number
  window_id: string
  start_time: string
  end_time?: string
  status: string // running | completed | failed
  epochs_completed?: number
  best_val_loss?: number
  best_spearman?: number
  model_path?: string
  error_message?: string
}

/** 数据源状态 — DataSourceInfo */
export interface DataSourceInfo {
  symbol: string
  name: string
  type: string
  start_date?: string
  end_date?: string
  total_rows?: number
  status: string // ready | stale | missing | error
  missing_dates?: number
}

export interface BackgroundJob {
  job_id: string
  name: string
  status: 'accepted' | 'running' | 'completed' | 'failed'
  progress: number
  submitted_at: string
  started_at?: string
  completed_at?: string
  result?: Record<string, unknown>
  error?: string
}

/** 仪表盘综合摘要 — DashboardSummary */
export interface DashboardSummary {
  model_status?: string
  data_status?: string
  signal_summary?: SignalSummary
  signal_total?: number
  pool_count?: number
  last_update?: string
  gpu_usage?: number
  uptime?: string
}

// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════

/** 策略参数 */
export interface StrategyConfig {
  buy_threshold: number
  long_only: boolean
  hold_days: number
  stop_loss: number
  take_profit: number
  max_single_position: number
  min_positions: number
}

/** 回测参数 */
export interface BacktestParams {
  default_capital: number
  commission_rate: number
  stamp_tax_rate: number
  slippage_rate: number
  benchmark_index: string
  risk_free_rate: number
}

/** 模型参数 */
export interface ModelParams {
  kronos_version: string
  device: string
  training: {
    seq_len: number
    pred_len: number
    epochs: number
    batch_size: number
    learning_rate: number
  }
}

/** 超参搜索配置 */
export interface ParamSearchConfig {
  T_values: number[]
  top_p_values: number[]
  lookback_values: number[]
}

/** 系统级配置 — GET /api/v1/admin/config 响应 */
export interface SystemConfigResponse {
  strategy: StrategyConfig
  backtest: BacktestParams
  model: ModelParams
  param_search: ParamSearchConfig
  server: {
    host: string
    port: number
  }
}

// ═══════════════════════════════════════════════════════════════════
// Strategy Presets & Selection Stats
// ═══════════════════════════════════════════════════════════════════

/** 策略风格预设 */
export interface StrategyPreset {
  name: string // conservative / balanced / aggressive
  label: string
  icon: string
  description: string
  prob_up_threshold: number
  score_threshold: number
  min_daily_amount: number
  max_per_industry: number
  top_k: number
}

/** 策略预设列表响应 */
export interface StrategyPresetListResponse {
  presets: StrategyPreset[]
}

/** 被剔除的股票 */
export interface EliminatedItem {
  symbol: string
  name: string
  reason: string
  stage: 'anomaly' | 'prob_score' | 'st_suspended' | 'liquidity' | 'industry' | 'top_k'
}

/** 选股筛选统计 */
export interface SelectionStats {
  initial_count: number
  anomaly_count: number
  after_prob_score: number
  after_st_suspended: number
  after_liquidity: number
  after_industry: number
  final_count: number
  eliminated: EliminatedItem[]
}

// ═══════════════════════════════════════════════════════════════════
// Strategy Optimization & Multi-Strategy Compare
// ═══════════════════════════════════════════════════════════════════

/** 参数搜索范围 */
export interface ParamRange {
  name: string
  values: number[]
}

/** 优化请求 */
export interface OptimizeRequest {
  param_ranges: ParamRange[]
  start_date: string
  end_date: string
  initial_capital?: number
  mode?: 'grid' | 'walk_forward'
  metric?: string
  scope?: 'paper_index' | 'csi1000_components'
  window_months?: number
  step_months?: number
}

/** 单个参数组合优化结果 */
export interface OptimizeResultItem {
  params: Record<string, unknown>
  cumulative_return: number
  annual_return: number
  max_drawdown: number
  sharpe_ratio: number
  win_rate: number
  total_trades: number
}

/** Walk-Forward 窗口结果 */
export interface WalkForwardWindowResult {
  window_id: number
  train_start: string
  train_end: string
  test_start: string
  test_end: string
  best_params: Record<string, unknown>
  oos_sharpe: number
  oos_return: number
  oos_drawdown: number
}

/** 参数优化完整结果 */
export interface OptimizeResult {
  task_id: string
  status: BacktestStatus
  mode: string
  metric: string
  total_combinations: number
  progress: number
  results: OptimizeResultItem[]
  best_params?: Record<string, unknown>
  best_metrics?: OptimizeResultItem
  sensitivity?: Record<string, unknown>
  walk_forward_windows: WalkForwardWindowResult[]
  stability_report?: Record<string, unknown>
  error?: string
}

/** 多策略对比请求 */
export interface CompareRequest {
  strategies: Record<string, unknown>[]
  start_date: string
  end_date: string
  initial_capital?: number
  scope?: 'paper_index' | 'csi1000_components'
}

/** 单策略对比结果 */
export interface CompareStrategyResult {
  strategy_id: number
  params: Record<string, unknown>
  cumulative_return: number
  annual_return: number
  max_drawdown: number
  sharpe_ratio: number
  calmar_ratio: number
  win_rate: number
  total_trades: number
}

/** 多策略对比完整结果 */
export interface CompareResult {
  task_id: string
  status: BacktestStatus
  results: CompareStrategyResult[]
  best_by_metric?: Record<string, number>
  error?: string
}
