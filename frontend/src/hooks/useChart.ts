/**
 * useChart — ECharts 实例生命周期管理 Hook
 *
 * 职责:
 *   1. ECharts 实例 init / dispose（React 安全）
 *   2. ResizeObserver 自动缩放
 *   3. 暗色模式自适应
 *   4. 配置构建器纯函数（从 chart-utils.js 移植）
 *
 * 用法:
 *   const { chartRef, setOption } = useChart()
 *   // <div ref={chartRef} className="chart-container" />
 *   // setOption({ ... })  // 直接设置 ECharts option
 */

import { useRef, useEffect, useCallback, useState } from 'react'
import * as echarts from 'echarts'
import type { EChartsOption, EChartsType } from 'echarts'
import { useAppStore } from '@/stores/useAppStore'

// ═══════════════════════════════════════════════════════════════════
// Hook
// ═══════════════════════════════════════════════════════════════════

export function useChart() {
  const chartRef = useRef<HTMLDivElement>(null)
  const instanceRef = useRef<EChartsType | null>(null)
  const observerRef = useRef<ResizeObserver | null>(null)
  const lastOptionRef = useRef<EChartsOption | null>(null)
  const theme = useAppStore((s) => s.theme)

  // instanceKey 递增 → 驱动消费组件重新 setOption（主题切换时必需）
  const [instanceKey, setInstanceKey] = useState(0)

  // Init / re-init（主题变化时销毁旧实例、创建新实例）
  useEffect(() => {
    const dom = chartRef.current
    if (!dom) return

    // 销毁旧实例
    if (instanceRef.current) {
      instanceRef.current.dispose()
      instanceRef.current = null
    }

    const instance = echarts.init(dom, theme === 'dark' ? 'dark' : undefined)
    instanceRef.current = instance
    setInstanceKey((k) => k + 1)

    // 主题切换后立即恢复上一次的 option（避免空白闪烁）
    if (lastOptionRef.current) {
      try {
        instance.setOption(lastOptionRef.current, true)
      } catch {
        /* ignore */
      }
    }

    // ResizeObserver
    try {
      const observer = new ResizeObserver(() => {
        try {
          instance.resize()
        } catch {
          /* ignore */
        }
      })
      observer.observe(dom)
      observerRef.current = observer
    } catch {
      /* ResizeObserver not supported */
    }

    // Cleanup
    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect()
        observerRef.current = null
      }
      instance.dispose()
      instanceRef.current = null
    }
  }, [theme])

  /** 安全设置 option（同时缓存最后 option 供主题切换恢复使用） */
  const setOption = useCallback((option: EChartsOption, notMerge = true) => {
    const inst = instanceRef.current
    if (!inst || inst.isDisposed()) return
    lastOptionRef.current = option
    try {
      inst.setOption(option, notMerge)
    } catch {
      /* chart disposed during render */
    }
  }, [])

  /** 手动 resize */
  const resize = useCallback(() => {
    try {
      instanceRef.current?.resize()
    } catch {
      /* ignore */
    }
  }, [])

  /** 获取 ECharts 实例（高级用法） */
  const getInstance = useCallback(() => instanceRef.current, [])

  return { chartRef, setOption, resize, getInstance, instanceKey }
}

// ═══════════════════════════════════════════════════════════════════
// Chart Option Builders (移植自 chart-utils.js)
// ═══════════════════════════════════════════════════════════════════

/** OHLC 数据点 */
export interface OhlcData {
  open: number
  close: number
  high: number
  low: number
  volume?: number
}

/** K线数据点（含日期） */
export interface KlineDatum extends OhlcData {
  date: string
}

// ── K线图 (含 MA5/MA10/MA20) ────────────────────────────────

export function buildKlineOption(
  klineData: KlineDatum[],
  _opts?: { showVolume?: boolean },
): EChartsOption {
  if (!klineData || klineData.length === 0) return {}

  const dates: string[] = []
  const ohlc: number[][] = []
  const volumes: number[] = []
  const ma5: [number, number][] = []
  const ma10: [number, number][] = []
  const ma20: [number, number][] = []

  for (let i = 0; i < klineData.length; i++) {
    const d = klineData[i]
    dates.push(d.date)
    const o = +d.open,
      h = +d.high,
      l = +d.low,
      c = +d.close
    ohlc.push([o, c, l, h])
    volumes.push(+((d as KlineDatum).volume || 0))

    if (i >= 4) {
      let sum5 = 0
      for (let j = i - 4; j <= i; j++) sum5 += +klineData[j].close
      ma5.push([i, +((sum5 / 5).toFixed(2))])
    }
    if (i >= 9) {
      let sum10 = 0
      for (let j = i - 9; j <= i; j++) sum10 += +klineData[j].close
      ma10.push([i, +((sum10 / 10).toFixed(2))])
    }
    if (i >= 19) {
      let sum20 = 0
      for (let j = i - 19; j <= i; j++) sum20 += +klineData[j].close
      ma20.push([i, +((sum20 / 20).toFixed(2))])
    }
  }

  const series: EChartsOption['series'] = [
    { name: 'K线', type: 'candlestick', data: ohlc },
    {
      name: 'MA5', type: 'line', data: ma5, smooth: true,
      lineStyle: { width: 1, color: '#e74c5e' }, symbol: 'none',
    },
    {
      name: 'MA10', type: 'line', data: ma10, smooth: true,
      lineStyle: { width: 1, color: '#f59e0b' }, symbol: 'none',
    },
    {
      name: 'MA20', type: 'line', data: ma20, smooth: true,
      lineStyle: { width: 1, color: '#7c5cf0' }, symbol: 'none',
    },
  ]

  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: { left: '8%', right: '2%', top: 20, bottom: 40 },
    xAxis: {
      type: 'category', data: dates,
      axisLabel: { formatter: (v: string) => v.slice(2, 7) },
    },
    yAxis: { type: 'value', scale: true },
    dataZoom: [
      { type: 'inside' },
      { type: 'slider', bottom: 10, height: 20 },
    ],
    series,
  }
}

// ── 成交量图 ─────────────────────────────────────────────

export function buildVolumeOption(klineData: KlineDatum[]): EChartsOption {
  if (!klineData || klineData.length === 0) return {}

  const dates = klineData.map((d) => d.date)
  const volumes = klineData.map((d, i) => ({
    value: +(d.volume || 0),
    itemStyle: {
      color: +d.close >= +klineData[Math.max(0, i - 1)].close ? '#ef5350' : '#26a69a',
    },
  }))

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '8%', right: '2%', top: 5, bottom: 10 },
    xAxis: {
      type: 'category', data: dates, show: false,
    },
    yAxis: { type: 'value', axisLabel: { formatter: (v: number) => (v / 1e8).toFixed(1) + '亿' } },
    series: [{
      name: '成交量', type: 'bar', data: volumes,
    }],
  }
}

// ── 收益曲线（策略 vs 基准）─────────────────────────────────

export function buildEquityCurveOption(
  dates: string[],
  strategyValues: number[],
  benchmarkValues?: number[],
  benchmarkLabel = '沪深300基准',
): EChartsOption {
  const series: EChartsOption['series'] = [
    {
      name: '策略净值', type: 'line', data: strategyValues, smooth: true,
      lineStyle: { color: '#4f6ef6', width: 2 },
      areaStyle: { color: 'rgba(79,110,246,0.15)' }, symbol: 'none',
    },
  ]
  if (benchmarkValues && benchmarkValues.length > 0) {
    series.push({
      name: benchmarkLabel, type: 'line', data: benchmarkValues, smooth: true,
      lineStyle: { color: '#8c919f', width: 1.5, type: 'dashed' }, symbol: 'none',
    })
  }
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (ps: unknown) => {
        const params = ps as { axisValue: string; color: string; seriesName: string; value: number }[]
        const header = params[0]?.axisValue ?? ''
        return header + '<br/>' + params.map((p) =>
          `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${p.color};margin-right:4px;"></span>${p.seriesName}: ${p.value}`,
        ).join('<br/>')
      },
    },
    legend: { data: series.map((s) => (s as { name: string }).name), bottom: 0 },
    grid: { left: '8%', right: '4%', top: 10, bottom: 52 },
    xAxis: { type: 'category', data: dates, axisLabel: { formatter: (v: string) => v.slice(0, 7) } },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}' } },
    series,
  }
}

// ── 回撤曲线 ─────────────────────────────────────────────

export function buildDrawdownOption(dates: string[], drawdownValues: number[]): EChartsOption {
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (ps: unknown) => {
        const p = (ps as { axisValue: string; value: number }[])[0]
        return `${p.axisValue}<br/>最大回撤: ${p.value.toFixed(2)}%`
      },
    },
    grid: { left: '8%', right: '4%', top: 10, bottom: 30 },
    xAxis: { type: 'category', data: dates, axisLabel: { formatter: (v: string) => v.slice(0, 7) } },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%' }, inverse: true },
    series: [{
      name: '回撤', type: 'line', data: drawdownValues, smooth: true,
      lineStyle: { color: '#e74c5e', width: 2 },
      areaStyle: { color: 'rgba(231,76,94,0.2)' }, symbol: 'none',
    }],
  }
}

// ── 月度收益分布（柱状图）───────────────────────────────────

export function buildMonthlyReturnsOption(months: string[], returns: number[]): EChartsOption {
  const colors = returns.map((v) =>
    v >= 3 ? '#22b07d' : v >= 0 ? '#7ecba4' : v >= -3 ? '#f5a0a0' : '#e74c5e',
  )
  return {
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (ps: unknown) => {
        const p = (ps as { axisValue: string; value: number }[])[0]
        return `${p.axisValue}<br/>收益率: ${p.value}%`
      },
    },
    grid: { left: '6%', right: '2%', top: 10, bottom: 10 },
    xAxis: {
      type: 'category', data: months,
      axisLabel: { show: true, rotate: 45, fontSize: 9, formatter: (v: string) => v.slice(2) },
    },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
    dataZoom: [{ type: 'slider', bottom: 0, height: 18, start: 0, end: 100 }],
    series: [{
      name: '月度收益', type: 'bar',
      data: returns.map((v, i) => ({
        value: v,
        itemStyle: { color: colors[i], borderRadius: [2, 2, 0, 0] },
      })),
    }],
  }
}

// ── 多策略对比 ───────────────────────────────────────────

export interface StrategyCurve {
  name: string
  values: number[]
  color: string
  dashed?: boolean
}

export function buildMultiStrategyOption(dates: string[], curves: StrategyCurve[]): EChartsOption {
  const series = curves.map((s) => ({
    name: s.name, type: 'line' as const, data: s.values, smooth: true,
    lineStyle: { color: s.color, width: s.dashed ? 1.5 : 2, type: s.dashed ? ('dashed' as const) : ('solid' as const) },
    symbol: 'none',
  }))
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: series.map((s) => s.name), bottom: 0 },
    grid: { left: '8%', right: '4%', top: 10, bottom: 30 },
    xAxis: { type: 'category', data: dates, axisLabel: { formatter: (v: string) => v.slice(0, 7) } },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}' } },
    series,
  }
}

// ── 预测 K 线（含置信区间）──────────────────────────────────

export interface PredictedKlineDatum {
  date: string
  close: number
  ci_lower?: number
  ci_upper?: number
}

export function buildPredictedKlineOption(
  data: PredictedKlineDatum[],
  opts?: { dividerIndex?: number },
): EChartsOption {
  if (!data || data.length === 0) return {}

  const dates = data.map((d) => d.date)
  const closeValues = data.map((d) => d.close)
  const ciLower = data.map((d) => d.ci_lower ?? null)
  const ciUpper = data.map((d) => d.ci_upper ?? null)

  // 黄色虚线分隔线：历史数据 vs 预测数据
  const markLine = opts?.dividerIndex != null
    ? {
        data: [{ xAxis: opts.dividerIndex }],
        lineStyle: { color: '#f59e0b', width: 2, type: 'dashed' as const },
        label: { show: true, formatter: '历史 | 预测', position: 'insideEndTop' as const, color: '#f59e0b', fontSize: 11 },
        symbol: 'none',
      }
    : undefined

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '8%', right: '4%', top: 20, bottom: 10 },
    xAxis: { type: 'category', data: dates, axisLabel: { formatter: (v: string) => v.slice(5) } },
    yAxis: { type: 'value', scale: true },
    series: [
      {
        name: '置信区间', type: 'line', data: ciUpper, smooth: true,
        lineStyle: { color: 'rgba(79,110,246,0.3)', width: 0 },
        areaStyle: { color: 'rgba(79,110,246,0.1)' },
        symbol: 'none', stack: 'confidence-band',
      },
      {
        name: '置信下限', type: 'line', data: ciLower, smooth: true,
        lineStyle: { color: 'rgba(79,110,246,0.3)', width: 0 },
        areaStyle: { color: 'rgba(79,110,246,0.05)' },
        symbol: 'none', stack: 'confidence-band',
      },
      {
        name: '预测价', type: 'line', data: closeValues, smooth: true,
        lineStyle: { color: '#4f6ef6', width: 2.5, type: 'dashed' },
        symbol: 'circle', symbolSize: 6,
        markLine,
      },
    ],
  }
}

// ── 监控图表（Loss 趋势）────────────────────────────────────

export function buildMonitorLossOption(months: string[], lossData: number[]): EChartsOption {
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (ps: unknown) => {
        const p = (ps as { axisValue: string; value: number }[])[0]
        return `${p.axisValue}<br/>Loss: <b>${p.value.toFixed(3)}</b>`
      },
    },
    grid: { left: '10%', right: '5%', top: 15, bottom: 25 },
    xAxis: { type: 'category', data: months, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', name: 'Loss', axisLabel: { formatter: '{value}' } },
    series: [{
      name: 'Training Loss', type: 'line', data: lossData, smooth: true,
      lineStyle: { color: '#4f6ef6', width: 2.5 },
      areaStyle: { color: 'rgba(79,110,246,0.12)' },
      symbol: 'circle', symbolSize: 6,
      markLine: {
        data: [{ yAxis: 0.035, name: '达标线', label: { formatter: '达标 0.035' } }],
        lineStyle: { color: '#22b07d', type: 'dashed' },
      },
    }],
  }
}

// ── 监控图表（GPU 利用率）───────────────────────────────────

export function buildMonitorGpuOption(
  hours: string[],
  gpuData: number[],
  memData: number[],
): EChartsOption {
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['GPU利用率', '显存占用'], bottom: 0 },
    grid: { left: '10%', right: '5%', top: 15, bottom: 30 },
    xAxis: { type: 'category', data: hours },
    yAxis: { type: 'value', name: '%', max: 100 },
    series: [
      {
        name: 'GPU利用率', type: 'line', data: gpuData, smooth: true,
        lineStyle: { color: '#f59e0b', width: 2 },
        areaStyle: { color: 'rgba(245,158,11,0.1)' }, symbol: 'circle', symbolSize: 4,
      },
      {
        name: '显存占用', type: 'line', data: memData, smooth: true,
        lineStyle: { color: '#7c5cf0', width: 2 },
        areaStyle: { color: 'rgba(124,92,240,0.08)' }, symbol: 'circle', symbolSize: 4,
      },
    ],
  }
}
