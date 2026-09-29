/**
 * ParamSensitivityChart — 参数敏感度分析图
 *
 * 展示每个策略参数对绩效指标的影响趋势：
 *   - 折线图：X轴=参数值，Y轴=指标值（Sharpe/Return/Drawdown）
 *   - 支持多指标切换
 *
 * 数据来源: OptimizeResult.sensitivity
 */

import { useEffect, useMemo } from 'react'
import { useChart } from '@/hooks/useChart'
import type { EChartsOption } from 'echarts'

export interface SensitivityData {
  [paramName: string]: {
    values: number[]
    sharpe_ratio: number[]
    annual_return: number[]
    max_drawdown: number[]
  }
}

export interface ParamSensitivityChartProps {
  data: SensitivityData
  metric?: 'sharpe_ratio' | 'annual_return' | 'max_drawdown'
  height?: number
}

const METRIC_LABELS: Record<string, string> = {
  sharpe_ratio: '夏普比率',
  annual_return: '年化收益率',
  max_drawdown: '最大回撤',
}

export default function ParamSensitivityChart({
  data,
  metric = 'sharpe_ratio',
  height = 300,
}: ParamSensitivityChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  const option = useMemo<EChartsOption>(() => {
    const entries = Object.entries(data)
    if (entries.length === 0) {
      return { title: { text: '无敏感度数据', left: 'center', top: 'middle' } }
    }

    const series: EChartsOption['series'] = []
    const legendData: string[] = []
    const xAxisDataByParam: Record<string, number[]> = {}

    for (const [paramName, paramData] of entries) {
      if (!paramData?.values?.length) continue
      xAxisDataByParam[paramName] = paramData.values
      legendData.push(paramName)

      series.push({
        name: paramName,
        type: 'line',
        data: paramData.values.map((v, i) => [v, paramData[metric]?.[i] ?? 0]),
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { width: 2 },
      })
    }

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const xVal = params[0].data?.[0] ?? params[0].axisValue
          let html = `<strong>${xVal}</strong><br/>`
          for (const p of params) {
            html += `${p.marker} ${p.seriesName}: <strong>${(p.data?.[1] ?? 0).toFixed(4)}</strong><br/>`
          }
          return html
        },
      },
      legend: {
        data: legendData,
        bottom: 0,
        type: 'scroll',
      },
      grid: {
        left: 60,
        right: 20,
        top: 20,
        bottom: 40,
      },
      xAxis: {
        type: 'value',
        name: '参数值',
        nameLocation: 'middle',
        nameGap: 25,
        axisLabel: { fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        name: METRIC_LABELS[metric] ?? metric,
        nameLocation: 'end',
        axisLabel: { fontSize: 11 },
      },
      series,
    }
  }, [data, metric, instanceKey])

  useEffect(() => {
    setOption(option)
  }, [option, setOption])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
