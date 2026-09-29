/**
 * KlineChart — K线图组件（含 MA5/MA10/MA20）
 *
 * 替换旧版 KronosCharts.buildKlineOption()。
 * 接收 KlineBar[] 数据，内部调用 useChart() hook  + buildKlineOption()。
 *
 * 用法:
 *   <KlineChart data={klineData} height={400} />
 */

import { useEffect } from 'react'
import { useChart, buildKlineOption, buildVolumeOption } from '@/hooks/useChart'
import type { KlineBar } from '@/types/api'
import type { KlineDatum } from '@/hooks/useChart'

export interface KlineChartProps {
  data: KlineBar[]
  height?: number
  showVolume?: boolean
}

/** KlineBar → KlineDatum */
function toKlineDatum(bars: KlineBar[]): KlineDatum[] {
  return bars.map((d) => ({
    date: d.date,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close,
    volume: d.volume,
  }))
}

export default function KlineChart({ data, height = 400, showVolume = false }: KlineChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  useEffect(() => {
    const bars = toKlineDatum(data)
    const option = buildKlineOption(bars, { showVolume })
    setOption(option)
  }, [data, showVolume, setOption, instanceKey])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}

/**
 * VolumeChart — 成交量柱状图组件
 */
export interface VolumeChartProps {
  data: KlineBar[]
  height?: number
}

export function VolumeChart({ data, height = 120 }: VolumeChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  useEffect(() => {
    const bars = toKlineDatum(data)
    const option = buildVolumeOption(bars)
    setOption(option)
  }, [data, setOption, instanceKey])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}

export { toKlineDatum }
