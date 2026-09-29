/**
 * PredictedKlineChart — 预测 K 线组件（含置信区间带状图）
 *
 * 支持两种数据格式:
 *   1. 简单模式: PredictedKlineDatum[] (仅收盘价 + CI)
 *   2. OHLC 模式: PredictedOHLCVPoint[] (完整 OHLC + CI)
 *
 * 可与历史 K 线叠加，dividerIndex 指定历史/预测分界点。
 *
 * 用法:
 *   <PredictedKlineChart data={predictedData} dividerIndex={60} />
 *   <PredictedKlineChart ohlcData={ohlcPredictions} />
 */

import { useEffect, useMemo } from 'react'
import { useChart, buildPredictedKlineOption } from '@/hooks/useChart'
import type { PredictedKlineDatum } from '@/hooks/useChart'
import type { PredictedOHLCVPoint } from '@/types/api'
import type { EChartsOption } from 'echarts'

export interface PredictedKlineChartProps {
  /** 简单模式数据 (close + CI) */
  data?: PredictedKlineDatum[]
  /** OHLC 模式数据 (完整 OHLCV + CI) */
  ohlcData?: PredictedOHLCVPoint[]
  /** 历史/预测分界线索引 */
  dividerIndex?: number
  height?: number
}

export default function PredictedKlineChart({
  data,
  ohlcData,
  dividerIndex,
  height = 300,
}: PredictedKlineChartProps) {
  const { chartRef, setOption, instanceKey } = useChart()

  const option = useMemo<EChartsOption>(() => {
    // OHLC 模式: 蜡烛图 + 置信区间
    if (ohlcData && ohlcData.length > 0) {
      const dates = ohlcData.map((d) => d.date)
      // ECharts candlestick: [open, close, low, high]
      const candleData = ohlcData.map((d) => [d.open, d.close, d.low, d.high])
      const ciLower = ohlcData.map((d) => d.lower_bound || null)
      const ciUpper = ohlcData.map((d) => d.upper_bound || null)

      // 分界线
      const markLine = dividerIndex != null
        ? {
            data: [{ xAxis: dividerIndex }],
            lineStyle: { color: '#f59e0b', width: 2, type: 'dashed' as const },
            label: {
              show: true,
              formatter: '历史 | 预测',
              position: 'insideEndTop' as const,
              color: '#f59e0b',
              fontSize: 11,
            },
            symbol: 'none',
          }
        : undefined

      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: (params: any) => {
            if (!Array.isArray(params)) return ''
            const idx = params[0]?.dataIndex ?? 0
            const d = ohlcData[idx]
            if (!d) return ''
            let html = `<strong>${d.date}</strong><br/>`
            html += `开: ${d.open.toFixed(2)}  收: ${d.close.toFixed(2)}<br/>`
            html += `高: ${d.high.toFixed(2)}  低: ${d.low.toFixed(2)}<br/>`
            if (d.confidence) html += `置信度: ${(d.confidence * 100).toFixed(0)}%<br/>`
            if (d.lower_bound && d.upper_bound) {
              html += `区间: [${d.lower_bound.toFixed(2)}, ${d.upper_bound.toFixed(2)}]`
            }
            return html
          },
        },
        grid: { left: 60, right: 20, top: 20, bottom: 30 },
        xAxis: {
          type: 'category',
          data: dates,
          axisLabel: { fontSize: 11, formatter: (v: string) => v?.slice(5) ?? v },
        },
        yAxis: {
          type: 'value',
          scale: true,
          axisLabel: { fontSize: 11 },
        },
        series: [
          {
            name: '预测K线',
            type: 'candlestick',
            data: candleData,
            itemStyle: {
              color: '#ef4444',       // up (red CN style)
              color0: '#22c55e',      // down (green)
              borderColor: '#ef4444',
              borderColor0: '#22c55e',
            },
            markLine,
          },
          {
            name: '置信上界',
            type: 'line',
            data: ciUpper,
            smooth: true,
            lineStyle: { width: 0 },
            areaStyle: { color: 'rgba(79,110,246,0.12)' },
            symbol: 'none',
            stack: 'ci-band',
            z: 1,
          },
          {
            name: '置信下界',
            type: 'line',
            data: ciLower,
            smooth: true,
            lineStyle: { width: 0 },
            areaStyle: { color: 'rgba(79,110,246,0.08)' },
            symbol: 'none',
            stack: 'ci-band',
            z: 1,
          },
        ],
      }
    }

    // 简单模式 (close + CI)
    if (data && data.length > 0) {
      return buildPredictedKlineOption(data, { dividerIndex })
    }

    return { title: { text: '暂无预测数据', left: 'center', top: 'middle' } }
  }, [data, ohlcData, dividerIndex, instanceKey])

  useEffect(() => {
    setOption(option)
  }, [option, setOption])

  return <div ref={chartRef} className="chart-container" style={{ height }} />
}
