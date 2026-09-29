/**
 * 回测 UI 状态类型定义
 *
 * 回测领域类型（BacktestResult, TradeRecord 等）统一在 types/api.ts 中定义。
 * 本文件仅包含回测 UI 特有的状态类型。
 */

import type { BacktestRequest, BacktestProgress, BacktestResult } from './api'

// Re-export for convenience
export type { BacktestRequest, BacktestProgress, BacktestResult }

/** 回测 UI 状态机阶段 */
export type BacktestPhase = 'idle' | 'submitted' | 'running' | 'completed' | 'failed'

/** 回测表单配置（前端表单用，含验证规则） */
export interface BacktestFormConfig {
  scope: 'paper_index' | 'csi1000_components'
  symbols: string[]
  start_date: string
  end_date: string
  initial_capital: number
  strategy_params: {
    buy_threshold: number
    hold_days: number
    stop_loss: number
    take_profit: number
    max_single_position: number
  }
}

/** 回测表单默认值 */
export const BACKTEST_FORM_DEFAULTS: BacktestFormConfig = {
  scope: 'paper_index',
  symbols: [],
  start_date: '2025-02-01',
  end_date: '2026-06-30',
  initial_capital: 1_000_000,
  strategy_params: {
    buy_threshold: 0.005,
    hold_days: 5,
    stop_loss: -0.08,
    take_profit: 0.15,
    max_single_position: 0.3,
  },
}
