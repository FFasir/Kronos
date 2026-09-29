/**
 * MonthlyReturnsChart — 月度收益分布图组件（柱状图热力色）
 *
 * 替换旧版 KronosCharts.buildMonthlyReturnsOption()。
 *
 * 用法:
 *   <MonthlyReturnsChart months={months} returns={returns} />
 */

import { useEffect } from 'react'
import { useChart, buildMonthlyReturnsOption } from '@/hooks/useChart'

export interface MonthlyReturnsChartProps {
  months: string[]
  returns: number[]
  height?: number
}

export default function MonthlyReturnsChart({
  months,
  returns: returnValues,
  height = 280,
}: MonthlyReturnsChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  useEffect(() => {
    const option = buildMonthlyReturnsOption(months, returnValues)
    setOption(option)
  }, [months, returnValues, setOption, instanceKey])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
