/**
 * EquityCurveChart — 收益曲线图组件（策略 vs 基准）
 *
 * 替换旧版 KronosCharts.buildEquityCurveOption()。
 *
 * 用法:
 *   <EquityCurveChart dates={dates} strategyValues={[...]} benchmarkValues={[...]} />
 */

import { useEffect } from 'react'
import { useChart, buildEquityCurveOption } from '@/hooks/useChart'

export interface EquityCurveChartProps {
  dates: string[]
  strategyValues: number[]
  benchmarkValues?: number[]
  benchmarkLabel?: string
  height?: number
}

export default function EquityCurveChart({
  dates,
  strategyValues,
  benchmarkValues,
  benchmarkLabel,
  height = 350,
}: EquityCurveChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  useEffect(() => {
    const option = buildEquityCurveOption(dates, strategyValues, benchmarkValues, benchmarkLabel)
    setOption(option)
  }, [dates, strategyValues, benchmarkValues, benchmarkLabel, setOption, instanceKey])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
