/** 股票相关类型定义 */

/** 股票基本信息 */
export interface Stock {
  code: string
  name: string
  industry: string
  market_cap: number | null
}

/** 盘口概览 */
export interface MarketOverview {
  up_count: number
  down_count: number
  flat_count: number
  total_count: number
  avg_change: number
  update_time: string
}
