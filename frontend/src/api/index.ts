/**
 * 指数/ETF 数据 API — api_index.py
 *
 * 端点:
 *   GET /api/v1/index/list      — 标的列表
 *   GET /api/v1/index/{code}     — 标的详情 + K线
 *   GET /api/v1/index/{code}/kline — 指定日期范围 K线
 */

import { apiClient } from './client'
import type { IndexListResponse, IndexDetail, KlineResponse } from '@/types/api'

// ═══════════════════════════════════════════════════════════════════
// Endpoints
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取所有标的列表（12 个指数+ETF）。
 * GET /api/v1/index/list
 */
export async function getIndexList(
  type?: 'index' | 'etf' | 'stock',
  keyword?: string,
  limit = 100,
  offset = 0,
): Promise<IndexListResponse | null> {
  const params = new URLSearchParams()
  if (type) params.set('type', type)
  if (keyword) params.set('keyword', keyword)
  params.set('limit', String(limit))
  params.set('offset', String(offset))
  const qs = `?${params.toString()}`
  return apiClient.get<IndexListResponse>(`/index/list${qs}`, {
    cache: true,
    cacheTtl: 60_000,
  })
}

/**
 * 获取单个标的详情 + 最近 N 天 OHLCV。
 * GET /api/v1/index/{code}?days=120&scale=raw
 *
 * @param scale - 价格口径：raw(不复权市价，默认) | hfq(后复权，模型输入口径)
 */
export async function getIndexDetail(
  code: string,
  days = 1250,
  scale: 'raw' | 'hfq' = 'raw',
): Promise<IndexDetail | null> {
  return apiClient.get<IndexDetail>(
    `/index/${encodeURIComponent(code)}?days=${days}&scale=${scale}`,
    { cache: true, cacheTtl: 30_000 },
  )
}

/**
 * 获取指定日期范围的 K 线数据。
 * GET /api/v1/index/{code}/kline?start=...&end=...
 */
export async function getIndexKline(
  code: string,
  start: string,
  end: string,
): Promise<KlineResponse | null> {
  const params = new URLSearchParams({ start, end })
  return apiClient.get<KlineResponse>(
    `/index/${encodeURIComponent(code)}/kline?${params.toString()}`,
    { cache: true, cacheTtl: 60_000 },
  )
}
