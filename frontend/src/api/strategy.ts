/**
 * 策略观点 API — api_strategy.py
 *
 * 端点:
 *   GET /api/v1/strategy/opinion — 解释"为什么当前是这个仓位、什么条件下会变成买入"
 */

import { apiClient } from './client'
import type { StrategyOpinion } from '@/types/api'

/**
 * 获取当前策略观点（信号 + 预测收益率 + 决策原因 + 触发条件）。
 */
export async function getStrategyOpinion(): Promise<StrategyOpinion | null> {
  return apiClient.get<StrategyOpinion>('/strategy/opinion', {
    cache: true,
    cacheTtl: 30_000,
  })
}
