/**
 * MultiEquityCurveChart — 多策略净值曲线叠加图
 *
 * 对比模式下，将 2~5 组策略的权益曲线叠加在同一张图上，
 * 不同颜色区分，支持 hover 联动。
 *
 * 用法:
 *   <MultiEquityCurveChart
 *     dates={dates}
 *     curves={[
 *       { name: '策略A', values: [...] },
 *       { name: '策略B', values: [...] },
 *     ]}
 *   />
 */

import { useEffect, useMemo } from 'react'
import { useChart } from '@/hooks/useChart'
import type { EChartsOption } from 'echarts'

export interface EquityCurve {
  name: string
  values: number[]
}

export interface MultiEquityCurveChartProps {
  dates: string[]
  curves: EquityCurve[]
  height?: number
}

/** 最多5条曲线的颜色序列 */
const CURVE_COLORS = ['#1890ff', '#52c41a', '#faad14', '#722ed1', '#ff4d4f']

export default function MultiEquityCurveChart({
  dates,
  curves,
  height = 350,
}: MultiEquityCurveChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  const option = useMemo<EChartsOption>(() => {
    if (dates.length === 0 || curves.length === 0) {
      return { title: { text: '无净值数据', left: 'center', top: 'middle' } }
    }

    const series: EChartsOption['series'] = curves.map((curve, i) => ({
      name: curve.name,
      type: 'line' as const,
      data: curve.values,
      smooth: false,
      symbol: 'none',
      lineStyle: {
        width: 2,
        color: CURVE_COLORS[i % CURVE_COLORS.length],
      },
      itemStyle: {
        color: CURVE_COLORS[i % CURVE_COLORS.length],
      },
    }))

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
      },
      legend: {
        data: curves.map((c) => c.name),
        bottom: 0,
        type: 'scroll',
      },
      grid: {
        left: 70,
        right: 20,
        top: 20,
        bottom: 40,
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          fontSize: 11,
          formatter: (v: string) => v?.slice(5) ?? v, // MM-DD
        },
      },
      yAxis: {
        type: 'value',
        name: '权益 (¥)',
        nameLocation: 'end',
        axisLabel: {
          fontSize: 11,
          formatter: (v: number) => {
            if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
            if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(0)}K`
            return v.toFixed(0)
          },
        },
        scale: true,
      },
      series,
    }
  }, [dates, curves, instanceKey])

  useEffect(() => {
    setOption(option)
  }, [option, setOption])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
