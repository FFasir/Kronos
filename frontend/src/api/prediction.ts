/**
 * 预测数据 API — api_prediction.py + api_selection.py
 *
 * 端点:
 *   GET /api/v1/prediction/latest    — 最新预测结果
 *   GET /api/v1/prediction/history   — 历史预测归档
 *   GET /api/v1/prediction/accuracy  — 预测准确率统计
 *   GET /api/v1/selection/latest     — 最新选股池
 *   GET /api/v1/selection/presets    — �+ 策略预设列表
 *   POST /api/v1/selection/generate  — 手动生成选股池
 */

import { apiClient } from './client'
import type {
  PredictionListResponse,
  PredictionHistoryResponse,
  AccuracyStats,
  StrategyPresetListResponse,
  SelectionStats,
  StockPredictionResponse,
} from '@/types/api'

// ═══════════════════════════════════════════════════════════════════
// Endpoints
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取最新预测结果。
 * GET /api/v1/prediction/latest?symbol=000852
 *
 * @param symbol - 不传则返回全部标的最新预测
 */
export async function getLatestPrediction(
  symbol?: string,
): Promise<PredictionListResponse | null> {
  let ep = '/prediction/latest'
  if (symbol) ep += `?symbol=${encodeURIComponent(symbol)}`
  return apiClient.get<PredictionListResponse>(ep, {
    cache: true,
    cacheTtl: 30_000,
  })
}

/** Get the server-filtered stock pool rather than raw prediction records. */
export async function getLatestSelection(): Promise<PredictionListResponse | null> {
  return apiClient.get<PredictionListResponse>('/selection/latest', {
    cache: true,
    cacheTtl: 30_000,
  })
}

/**
 * 获取历史预测归档（预测 vs 实际对比）。
 * GET /api/v1/prediction/history?symbol=...&start=...&end=...
 */
export async function getPredictionHistory(
  symbol: string,
  start?: string,
  end?: string,
): Promise<PredictionHistoryResponse | null> {
  const params = new URLSearchParams({ symbol })
  if (start) params.set('start', start)
  if (end) params.set('end', end)
  return apiClient.get<PredictionHistoryResponse>(
    `/prediction/history?${params.toString()}`,
  )
}

/**
 * 获取预测准确率统计。
 * GET /api/v1/prediction/accuracy?symbol=...
 */
export async function getPredictionAccuracy(
  symbol: string,
): Promise<AccuracyStats | null> {
  return apiClient.get<AccuracyStats>(
    `/prediction/accuracy?symbol=${encodeURIComponent(symbol)}`,
    { cache: true, cacheTtl: 60_000 },
  )
}

// ═══════════════════════════════════════════════════════════════════
// Strategy Presets
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取策略风格预设列表（保守/均衡/激进）。
 * GET /api/v1/selection/presets
 */
export async function getStrategyPresets(): Promise<StrategyPresetListResponse | null> {
  return apiClient.get<StrategyPresetListResponse>('/selection/presets', {
    cache: true,
    cacheTtl: 300_000,
  })
}

/**
 * 手动生成选股池（支持预设和自定义阈值）。
 * POST /api/v1/selection/generate
 */
export async function generateSelection(params: {
  preset?: string
  prob_up_threshold?: number
  score_threshold?: number
  min_daily_amount?: number
  max_per_industry?: number
  top_k?: number
}): Promise<PredictionListResponse & { stats: SelectionStats } | null> {
  return apiClient.post<PredictionListResponse & { stats: SelectionStats }>(
    '/selection/generate',
    params,
  )
}

// ═══════════════════════════════════════════════════════════════════
// Stock Prediction K-line
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取个股未来 N 日 OHLC 预测 + 置信区间。
 * GET /api/v1/index/{code}/prediction?days=20
 */
export async function getStockPrediction(
  code: string,
  days: number = 20,
): Promise<StockPredictionResponse | null> {
  return apiClient.get<StockPredictionResponse>(
    `/index/${encodeURIComponent(code)}/prediction?days=${days}`,
    { cache: true, cacheTtl: 60_000 },
  )
}
