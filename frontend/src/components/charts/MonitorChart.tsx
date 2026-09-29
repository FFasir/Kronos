/**
 * MonitorChart — 管理后台监控图表（GPU/Loss）
 *
 * 替换旧版 KronosCharts.buildMonitorLossOption() + buildMonitorGpuOption()。
 * 根据 type 渲染 Loss 趋势或 GPU 利用率。
 *
 * 用法:
 *   <MonitorChart type="loss" months={months} data={lossData} />
 *   <MonitorChart type="gpu" hours={hours} gpuData={[...]} memData={[...]} />
 */

import { useEffect } from 'react'
import { useChart, buildMonitorLossOption, buildMonitorGpuOption } from '@/hooks/useChart'

export interface MonitorLossChartProps {
  type: 'loss'
  months: string[]
  data: number[]
  height?: number
}

export interface MonitorGpuChartProps {
  type: 'gpu'
  hours: string[]
  gpuData: number[]
  memData: number[]
  height?: number
}

export type MonitorChartProps = MonitorLossChartProps | MonitorGpuChartProps

export default function MonitorChart(props: MonitorChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()
  const height = props.height || (props.type === 'loss' ? 300 : 250)

  useEffect(() => {
    if (props.type === 'loss') {
      setOption(buildMonitorLossOption(props.months, props.data))
    } else {
      setOption(buildMonitorGpuOption(props.hours, props.gpuData, props.memData))
    }
  }, [props, setOption, instanceKey])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
