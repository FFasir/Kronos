/**
 * 实时行情 API — api_quote.py
 *
 * 端点:
 *   GET /api/v1/quote/list?codes=600519,000001 — 批量实时行情（最多 100 只）
 */

import { apiClient } from './client'
import type { QuoteListResponse } from '@/types/api'

/**
 * 批量获取 A 股实时行情（价格/涨跌幅等）。
 * @param codes - 股票代码数组
 */
export async function getQuotes(codes: string[]): Promise<QuoteListResponse | null> {
  if (!codes || codes.length === 0) return null
  return apiClient.get<QuoteListResponse>(
    `/quote/list?codes=${encodeURIComponent(codes.join(','))}`,
    { cache: true, cacheTtl: 8_000 },
  )
}
