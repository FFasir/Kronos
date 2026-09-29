/**
 * 交易信号 API — api_signal.py
 *
 * 端点:
 *   GET /api/v1/signal/latest   — 所有标的最新择时信号
 *   GET /api/v1/signal/{code}   — 单个标的历史信号流
 */

import { apiClient } from './client'
import type { SignalListResponse, SignalHistoryResponse } from '@/types/api'

// ═══════════════════════════════════════════════════════════════════
// Endpoints
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取所有标的最新择时信号。
 * GET /api/v1/signal/latest
 */
export async function getLatestSignals(): Promise<SignalListResponse | null> {
  return apiClient.get<SignalListResponse>('/signal/latest', {
    cache: true,
    cacheTtl: 30_000,
  })
}

/**
 * 获取单个标的历史信号流。
 * GET /api/v1/signal/{code}?days=30
 */
export async function getSignalHistory(
  code: string,
  days = 30,
): Promise<SignalHistoryResponse | null> {
  return apiClient.get<SignalHistoryResponse>(
    `/signal/${encodeURIComponent(code)}?days=${days}`,
  )
}
