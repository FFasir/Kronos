export interface ResearchMetrics {
  net_compounded_return: number | null
  benchmark_buy_and_hold_return: number | null
  total_relative_excess_return: number | null
  annualized_return: number | null
  annualized_volatility: number | null
  sharpe_zero_rf: number | null
  max_drawdown: number | null
  directional_accuracy: number | null
  prediction_spearman: number | null
  trade_count: number | null
  win_rate_before_cost: number | null
  in_market_ratio: number | null
  trading_days: number | null
}

export interface ModelComparison {
  key: string
  name: string
  version: string
  checkpoint: string
  tokenizer: string
  selected: boolean
  development: ResearchMetrics
  holdout: ResearchMetrics
}

export interface EquityPoint {
  date: string
  equity: number
  benchmark_equity: number
  in_market: boolean
}

export interface ResearchSummary {
  as_of: string
  research_status: string
  production_model: {
    key: string
    name: string
    version: string
    checkpoint: string
    tokenizer: string
    parameter_count: number | null
    purpose: string
  }
  recommended_model_key: string
  comparison: ModelComparison[]
  selected_holdout_equity: EquityPoint[]
  experiment: Record<string, string>
  risk_notice: string
}

export interface RegisteredModel {
  key: string
  name: string
  version: string
  predictor_path: string
  tokenizer_path: string
  kind: 'builtin' | 'custom'
  research_status: 'selected' | 'not_selected' | 'unverified'
  ready: boolean
  validation_error: string | null
  active: boolean
}

export interface ModelRegistryResponse {
  active_key: string
  models: RegisteredModel[]
  restart_required: boolean
}
