/**
 * DrawdownChart — 回撤区域图组件
 *
 * 替换旧版 KronosCharts.buildDrawdownOption()。
 *
 * 用法:
 *   <DrawdownChart dates={dates} values={[...]} />
 */

import { useEffect } from 'react'
import { useChart, buildDrawdownOption } from '@/hooks/useChart'

export interface DrawdownChartProps {
  dates: string[]
  values: number[]
  height?: number
}

export default function DrawdownChart({ dates, values, height = 250 }: DrawdownChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  useEffect(() => {
    const option = buildDrawdownOption(dates, values)
    setOption(option)
  }, [dates, values, setOption, instanceKey])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
