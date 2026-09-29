/**
 * 回测中心页面
 * 对应旧版 user_portal.html #page-backtest
 *
 * 功能：参数表单 → 提交任务 → 轮询进度 → 绩效报告（指标卡片 + 图表 + 详情Tabs）
 * 数据来源：POST /api/v1/backtest          (提交)
 *           GET  /api/v1/backtest/{task_id} (轮询结果)
 *           GET  /api/v1/backtest/history   (历史记录)
 */

import { useEffect, useState, useCallback, useMemo } from 'react'
import dayjs from 'dayjs'
import {
  Card,
  Typography,
  Row,
  Col,
  Form,
  Segmented,
  InputNumber,
  DatePicker,
  Button,
  Progress,
  Tabs,
  Table,
  Descriptions,
  Tag,
  Space,
  Divider,
  Tooltip,
  Select,
  Alert,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  ExperimentOutlined,
  FallOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  HistoryOutlined,
  InfoCircleOutlined,
  BarChartOutlined,
  LineChartOutlined,
  ThunderboltOutlined,
  SwapOutlined,
  PlusOutlined,
  DeleteOutlined,
} from '@ant-design/icons'
import {
  submitBacktest, getBacktestResult, getBacktestHistory, pollBacktest,
  submitOptimize, pollOptimize,
  submitCompare, getCompareResult, getBacktestAvailability,
} from '@/api/backtest'
import type {
  BacktestRequest,
  BacktestAvailability,
  BacktestHistoryItem,
  TradeRecord,
  DailyEquityPoint,
  RiskEvent,
  YearlyReturn,
  MonthlyReturn,
  OptimizeRequest,
  OptimizeResult,
  OptimizeResultItem,
  ParamRange,
  CompareRequest,
  CompareResult,
  CompareStrategyResult,
  WalkForwardWindowResult,
} from '@/types/api'
import { useBacktestStore } from '@/stores/useBacktestStore'
import { BACKTEST_FORM_DEFAULTS } from '@/types/backtest'
import EquityCurveChart from '@/components/charts/EquityCurveChart'
import DrawdownChart from '@/components/charts/DrawdownChart'
import MonthlyReturnsChart from '@/components/charts/MonthlyReturnsChart'
import ParamSensitivityChart from '@/components/charts/ParamSensitivityChart'
import StatCard from '@/components/common/StatCard'
import PctText from '@/components/common/PctText'
import ErrorState from '@/components/common/ErrorState'
import EmptyState from '@/components/common/EmptyState'
import Toast from '@/components/common/Toast'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

/** 状态标签配置 */
const STATUS_TAG: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  accepted: { color: 'processing', icon: <ClockCircleOutlined />, label: '已提交' },
  running: { color: 'processing', icon: <ClockCircleOutlined />, label: '运行中' },
  completed: { color: 'success', icon: <CheckCircleOutlined />, label: '已完成' },
  failed: { color: 'error', icon: <CloseCircleOutlined />, label: '失败' },
  timeout: { color: 'warning', icon: <CloseCircleOutlined />, label: '超时' },
}

/** 交易原因术语词典（表头 ⓘ 与 行内翻译共用） */
const REASON_GLOSSARY =
  '持仓到期：买入后固定持有 5 个交易日，第 5 日收盘按规则无条件卖出（赚亏都到点走）。' +
  '止损：亏损达到 -8% 提前平仓；止盈：盈利达到 +15% 提前止盈。' +
  '信号买入：当日模型预测收益 > 阈值，触发买入。'

/** 把引擎返回的英文/缩写原因翻译成大白话 */
function friendlyReason(reason?: string): string {
  if (!reason) return '--'
  if (reason.includes('持仓到期') || /到期|expire/i.test(reason)) return '持有满 N 日，到期卖出'
  if (reason.includes('止损') || /stop_loss/i.test(reason)) return '触发止损（-8%）平仓'
  if (reason.includes('止盈') || /take_profit/i.test(reason)) return '触发止盈（+15%）平仓'
  if (reason.includes('买入信号') || /BUY|信号/i.test(reason) && !reason.includes('卖出')) return '信号触发买入'
  if (reason.includes('跌停') || /limit_down/i.test(reason)) return '跌停禁止卖出'
  if (reason.includes('涨停') || /limit_up/i.test(reason)) return '涨停禁止买入'
  return reason
}

/** 回测页面模式 */
type PageMode = 'backtest' | 'optimize' | 'compare'

/** 参数范围配置行 */
interface ParamRangeRow {
  name: string
  min: number
  max: number
  step: number
}

/** 对比策略行 */
interface CompareStrategyRow {
  key: string
  buy_threshold: number
  hold_days: number
  stop_loss: number
  take_profit: number
}

/** 优化排序指标选项 */
const METRIC_OPTIONS = [
  { label: '夏普比率', value: 'sharpe_ratio' },
  { label: '年化收益', value: 'annual_return' },
  { label: '最大回撤', value: 'max_drawdown' },
  { label: '胜率', value: 'win_rate' },
]

/** 默认参数范围 */
const DEFAULT_PARAM_RANGES: ParamRangeRow[] = [
  { name: 'buy_threshold', min: 0.003, max: 0.01, step: 0.001 },
  { name: 'hold_days', min: 3, max: 7, step: 1 },
  { name: 'stop_loss', min: -0.12, max: -0.05, step: 0.01 },
  { name: 'take_profit', min: 0.08, max: 0.20, step: 0.02 },
]

/** 默认对比策略 */
const DEFAULT_COMPARE_STRATEGIES: CompareStrategyRow[] = [
  { key: 's1', buy_threshold: 0.005, hold_days: 5, stop_loss: -0.08, take_profit: 0.15 },
  { key: 's2', buy_threshold: 0.008, hold_days: 3, stop_loss: -0.06, take_profit: 0.12 },
]

export default function BacktestPage() {
  const { phase, progress, result, lastConfig, setPhase, setProgress, setResult, setLastConfig, reset } =
    useBacktestStore()

  const [form] = Form.useForm()
  const [submitting, setSubmitting] = useState(false)
  const [polling, setPolling] = useState(false)
  const [history, setHistory] = useState<BacktestHistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // ── 模式切换 ────────────────────────────────────────
  const [pageMode, setPageMode] = useState<PageMode>('backtest')

  // ── 回测可用性（信号样本来自 model_prediction 归档）──
  const [availability, setAvailability] = useState<BacktestAvailability | null>(null)

  useEffect(() => {
    getBacktestAvailability().then((data) => {
      if (data) setAvailability(data)
    })
  }, [])

  // 默认区间自动贴合可用信号窗口（否则默认 2026-06-30 结束 → 无信号 → 必然失败）
  useEffect(() => {
    if (!availability?.paper_index?.min_date || lastConfig) return
    const cur = form.getFieldValue('dateRange')
    if (cur?.[0]) return // 用户已手动选过
    const maxDate = availability.paper_index.max_date ?? '2026-09-01'
    form.setFieldsValue({
      dateRange: [
        dayjs(availability.paper_index.min_date).subtract(5, 'day'),
        dayjs(maxDate).add(6, 'day'),
      ],
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availability])

  const paperAvail = availability?.paper_index
  const paperEnabled = paperAvail?.enabled !== false
  const compEnabled = availability?.csi1000_components?.enabled === true
  const customAvail = availability?.custom_index
  const assetPairs = customAvail?.pairs ?? []

  // ── 优化模式状态 ────────────────────────────────────
  const [optForm] = Form.useForm()
  const [paramRanges, setParamRanges] = useState<ParamRangeRow[]>(DEFAULT_PARAM_RANGES)
  const [optimizeSubmitting, setOptimizeSubmitting] = useState(false)
  const [optimizePolling, setOptimizePolling] = useState(false)
  const [optimizeProgress, setOptimizeProgress] = useState(0)
  const [optimizeResult, setOptimizeResult] = useState<OptimizeResult | null>(null)
  const [optimizeMetric, setOptimizeMetric] = useState<string>('sharpe_ratio')
  const [optimizeSortMetric, setOptimizeSortMetric] = useState<string>('sharpe_ratio')

  // ── 对比模式状态 ────────────────────────────────────
  const [cmpForm] = Form.useForm()
  const [compareStrategies, setCompareStrategies] = useState<CompareStrategyRow[]>(DEFAULT_COMPARE_STRATEGIES)
  const [compareSubmitting, setCompareSubmitting] = useState(false)
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)
  const [comparePolling, setComparePolling] = useState(false)

  // ── 加载历史 ────────────────────────────────────────
  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    try {
      const data = await getBacktestHistory()
      if (data) setHistory(data)
    } catch {
      // silent
    } finally {
      setHistoryLoading(false)
    }
  }, [])

  useEffect(() => { loadHistory() }, [loadHistory])

  // ── 初始化表单 ──────────────────────────────────────
  useEffect(() => {
    form.setFieldsValue({
      scope: lastConfig?.scope ?? BACKTEST_FORM_DEFAULTS.scope,
      symbols: (lastConfig?.symbols ?? BACKTEST_FORM_DEFAULTS.symbols).join(', '),
      dateRange: [
        dayjs(lastConfig?.start_date ?? BACKTEST_FORM_DEFAULTS.start_date),
        dayjs(lastConfig?.end_date ?? BACKTEST_FORM_DEFAULTS.end_date),
      ],
      initial_capital: lastConfig?.initial_capital ?? BACKTEST_FORM_DEFAULTS.initial_capital,
    })
  }, [form, lastConfig])

  // ── 提交回测 ────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields()
      setSubmitting(true)

      const config: BacktestRequest = {
        scope: values.scope ?? BACKTEST_FORM_DEFAULTS.scope,
        symbols: [],
        start_date: values.dateRange?.[0]?.format('YYYY-MM-DD') ?? BACKTEST_FORM_DEFAULTS.start_date,
        end_date: values.dateRange?.[1]?.format('YYYY-MM-DD') ?? BACKTEST_FORM_DEFAULTS.end_date,
        initial_capital: values.initial_capital ?? BACKTEST_FORM_DEFAULTS.initial_capital,
        strategy_params: values.strategy_params,
        signal_symbol: values.scope === 'custom_index' ? (values.signal_symbol ?? '000852') : undefined,
        execution_symbol: values.scope === 'custom_index' ? (values.execution_symbol ?? '512100') : undefined,
        leakage_mode: values.scope === 'custom_index' ? (values.leakage_mode ?? 'demo') : undefined,
      }

      // 提交前拦截：paper_index 区间必须覆盖至少一个归档信号日
      if (values.scope !== 'custom_index' && values.scope !== 'csi1000_components' && paperAvail?.min_date) {
        if (config.end_date! < paperAvail.min_date || config.start_date > (paperAvail.max_date ?? '')) {
          Toast.warning(
            `所选区间不包含任何信号日。可用信号日：${paperAvail.min_date} ~ ${paperAvail.max_date}，请调整区间后再试。`,
          )
          setSubmitting(false)
          return
        }
      }

      setLastConfig(config)

      const resp = await submitBacktest(config)
      if (!resp) {
        Toast.error('回测提交失败，请检查后端服务')
        setSubmitting(false)
        return
      }

      setPhase('running')
      setProgress({ task_id: resp.task_id, status: 'accepted', progress: 0, message: undefined })
      Toast.success(`回测任务已提交 (${resp.task_id})`)

      // 开始轮询
      setPolling(true)
      const finalResult = await pollBacktest(
        resp.task_id,
        (prog) => setProgress({ task_id: resp.task_id, status: prog.status ?? 'running', progress: prog.progress ?? 0, message: prog.message }),
      )

      setPolling(false)

      if (finalResult?.status === 'completed') {
        // 拉取完整结果
        const full = await getBacktestResult(resp.task_id)
        if (full) {
          setResult(full)
          setPhase('completed')
          Toast.success('回测完成！')
          loadHistory()
        } else {
          setPhase('failed')
          Toast.error('无法获取回测结果')
        }
      } else if (finalResult?.status === 'failed') {
        setPhase('failed')
        Toast.error('回测任务执行失败')
      } else {
        setPhase('failed')
        Toast.warning('回测任务超时，请稍后查询')
      }
    } catch {
      // 表单验证失败
    } finally {
      setSubmitting(false)
    }
  }, [form, setLastConfig, setPhase, setProgress, setResult, loadHistory])

  // ── 重置 ────────────────────────────────────────────
  const handleReset = useCallback(() => {
    reset()
    form.resetFields()
    setActiveTab('overview')
  }, [reset, form])

  // ── 查看历史回测 ────────────────────────────────────
  const handleViewHistory = useCallback(async (taskId: string) => {
    setPhase('running')
    setProgress({ task_id: taskId, status: 'running', progress: 0.5 })
    const data = await getBacktestResult(taskId)
    if (data) {
      setResult(data)
      setPhase(data.status === 'completed' ? 'completed' : 'failed')
    } else {
      setPhase('failed')
      Toast.error('无法加载历史回测结果')
    }
  }, [setPhase, setProgress, setResult])

  // ════════════════════════════════════════════════════════
  // 参数优化
  // ════════════════════════════════════════════════════════

  const handleOptimizeSubmit = useCallback(async () => {
    try {
      const values = await optForm.validateFields()
      setOptimizeSubmitting(true)
      setOptimizeResult(null)
      setOptimizeProgress(0)

      // 构建参数范围
      const paramRangesPayload: ParamRange[] = paramRanges
        .filter((r) => r.step > 0 && r.max >= r.min)
        .map((r) => {
          const values: number[] = []
          for (let v = r.min; v <= r.max + r.step * 0.01; v += r.step) {
            values.push(Math.round(v * 10000) / 10000)
          }
          return { name: r.name, values }
        })

      if (paramRangesPayload.length === 0) {
        Toast.warning('请至少配置一个参数范围')
        setOptimizeSubmitting(false)
        return
      }

      const req: OptimizeRequest = {
        param_ranges: paramRangesPayload,
        start_date: values.dateRange?.[0]?.format('YYYY-MM-DD') ?? '2023-01-01',
        end_date: values.dateRange?.[1]?.format('YYYY-MM-DD') ?? '2024-12-31',
        initial_capital: values.initial_capital ?? 1_000_000,
        mode: values.mode ?? 'grid',
        metric: optimizeMetric,
        scope: values.scope ?? 'paper_index',
        window_months: values.window_months ?? 12,
        step_months: values.step_months ?? 3,
      }

      const resp = await submitOptimize(req)
      if (!resp) {
        Toast.error('优化任务提交失败')
        setOptimizeSubmitting(false)
        return
      }

      Toast.success(`优化任务已提交 (${resp.task_id})`)
      setOptimizePolling(true)

      const finalResult = await pollOptimize(
        resp.task_id,
        (prog) => setOptimizeProgress(prog.progress ?? 0),
      )

      setOptimizePolling(false)

      if (finalResult?.status === 'completed') {
        setOptimizeResult(finalResult)
        Toast.success('参数优化完成！')
      } else if (finalResult?.status === 'failed') {
        Toast.error('参数优化执行失败')
      } else {
        Toast.warning('参数优化超时')
      }
    } catch {
      // form validation
    } finally {
      setOptimizeSubmitting(false)
    }
  }, [optForm, paramRanges, optimizeMetric])

  // ════════════════════════════════════════════════════════
  // 多策略对比
  // ════════════════════════════════════════════════════════

  const handleCompareSubmit = useCallback(async () => {
    try {
      const values = await cmpForm.validateFields()
      setCompareSubmitting(true)
      setCompareResult(null)

      if (compareStrategies.length < 2) {
        Toast.warning('请至少配置 2 组策略参数')
        setCompareSubmitting(false)
        return
      }

      const req: CompareRequest = {
        strategies: compareStrategies.map((s, i) => ({
          strategy_id: i + 1,
          buy_threshold: s.buy_threshold,
          hold_days: s.hold_days,
          stop_loss: s.stop_loss,
          take_profit: s.take_profit,
        })),
        start_date: values.dateRange?.[0]?.format('YYYY-MM-DD') ?? '2023-01-01',
        end_date: values.dateRange?.[1]?.format('YYYY-MM-DD') ?? '2024-12-31',
        initial_capital: values.initial_capital ?? 1_000_000,
        scope: values.scope ?? 'paper_index',
      }

      const resp = await submitCompare(req)
      if (!resp) {
        Toast.error('对比任务提交失败')
        setCompareSubmitting(false)
        return
      }

      Toast.success(`对比任务已提交 (${resp.task_id})`)
      setComparePolling(true)

      // 简单轮询
      const start = Date.now()
      const timeout = 600_000
      while (Date.now() - start < timeout) {
        await new Promise((r) => setTimeout(r, 3000))
        const data = await getCompareResult(resp.task_id)
        if (!data) break
        if (data.status === 'completed') {
          setCompareResult(data)
          setComparePolling(false)
          setCompareSubmitting(false)
          Toast.success('多策略对比完成！')
          return
        }
        if (data.status === 'failed') {
          setComparePolling(false)
          setCompareSubmitting(false)
          Toast.error('对比任务执行失败')
          return
        }
      }
      setComparePolling(false)
      setCompareSubmitting(false)
      Toast.warning('对比任务超时')
    } catch {
      // form validation
    } finally {
      setCompareSubmitting(false)
    }
  }, [cmpForm, compareStrategies])

  // ── 交易记录列 ──────────────────────────────────────
  const tradeCols: ColumnsType<TradeRecord> = [
    { title: '日期', dataIndex: 'date', key: 'date', width: 100 },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: 60,
      render: (v: string) => (
        <Tag color={v === 'BUY' ? 'red' : 'green'}>{v === 'BUY' ? '买入' : '卖出'}</Tag>
      ),
    },
    { title: '代码', dataIndex: 'symbol', key: 'symbol', width: 80, render: (v: string) => <Text code>{v}</Text> },
    { title: '价格', dataIndex: 'price', key: 'price', width: 80, align: 'right', render: (v: number) => v?.toFixed(2) },
    { title: '数量', dataIndex: 'quantity', key: 'quantity', width: 70, align: 'right' },
    { title: '金额', dataIndex: 'amount', key: 'amount', width: 100, align: 'right', render: (v: number) => v?.toFixed(0) },
    { title: '佣金', dataIndex: 'commission', key: 'commission', width: 70, align: 'right', render: (v: number) => v?.toFixed(2) },
    { title: '印花税', dataIndex: 'stamp_tax', key: 'stamp_tax', width: 70, align: 'right', render: (v: number) => v?.toFixed(2) },
    {
      title: '盈亏',
      dataIndex: 'pnl',
      key: 'pnl',
      width: 80,
      align: 'right',
      render: (v: number | undefined) =>
        v != null ? <PctText value={v} raw precision={0} suffix="" bold /> : '--',
    },
    {
      title: '盈亏%',
      dataIndex: 'pnl_pct',
      key: 'pnl_pct',
      width: 80,
      align: 'right',
      render: (v: number | undefined) => (v != null ? <PctText value={v} /> : '--'),
    },
    { title: '持有天数', dataIndex: 'hold_days', key: 'hold_days', width: 70, align: 'right', render: (v: number | undefined) => v ?? '--' },
    {
      title: (
        <Tooltip title={REASON_GLOSSARY}>
          <span>原因<span style={{ color: '#999' }}> ⓘ</span></span>
        </Tooltip>
      ),
      dataIndex: 'reason',
      key: 'reason',
      width: 160,
      ellipsis: true,
      render: (v: string | undefined) =>
        v ? (
          <Tooltip title={v}>
            <span>{friendlyReason(v)}</span>
          </Tooltip>
        ) : '--',
    },
  ]

  // ── 权益明细列 ──────────────────────────────────────
  const equityCols: ColumnsType<DailyEquityPoint> = [
    { title: '日期', dataIndex: 'date', key: 'date', width: 100 },
    { title: '现金', dataIndex: 'cash', key: 'cash', width: 100, align: 'right', render: (v: number) => v.toFixed(0) },
    { title: '持仓市值', dataIndex: 'position_value', key: 'position_value', width: 100, align: 'right', render: (v: number) => v.toFixed(0) },
    { title: '总权益', dataIndex: 'total_equity', key: 'total_equity', width: 100, align: 'right', render: (v: number) => v.toFixed(0) },
    {
      title: '日收益',
      dataIndex: 'daily_return',
      key: 'daily_return',
      width: 80,
      align: 'right',
      render: (v: number) => <PctText value={v} />,
    },
    {
      title: '累计收益',
      dataIndex: 'cumulative_return',
      key: 'cumulative_return',
      width: 80,
      align: 'right',
      render: (v: number) => <PctText value={v} />,
    },
    {
      title: '回撤',
      dataIndex: 'drawdown',
      key: 'drawdown',
      width: 80,
      align: 'right',
      render: (v: number) => <span style={{ color: 'var(--danger)' }}>{(v * 100).toFixed(2)}%</span>,
    },
    { title: '持仓数', dataIndex: 'num_positions', key: 'num_positions', width: 70, align: 'right' },
  ]

  // ── 风控事件列 ──────────────────────────────────────
  const riskCols: ColumnsType<RiskEvent> = [
    { title: '日期', dataIndex: 'date', key: 'date', width: 100 },
    { title: '代码', dataIndex: 'symbol', key: 'symbol', width: 80, render: (v: string) => <Text code>{v}</Text> },
    {
      title: '事件',
      dataIndex: 'event_type',
      key: 'event_type',
      width: 100,
      render: (v: string) => {
        const isStop = v.includes('stop_loss') || v.includes('止损')
        const isTP = v.includes('take_profit') || v.includes('止盈')
        return <Tag color={isStop ? 'red' : isTP ? 'green' : 'orange'}>{v}</Tag>
      },
    },
    {
      title: (
        <Tooltip title={REASON_GLOSSARY}>
          <span>原因<span style={{ color: '#999' }}> ⓘ</span></span>
        </Tooltip>
      ),
      dataIndex: 'reason',
      key: 'reason',
      width: 200,
      ellipsis: true,
      render: (v: string | undefined) =>
        v ? (
          <Tooltip title={v}>
            <span>{friendlyReason(v)}</span>
          </Tooltip>
        ) : '--',
    },
    {
      title: '盈亏',
      dataIndex: 'position_pnl',
      key: 'position_pnl',
      width: 80,
      align: 'right',
      render: (v: number | undefined) =>
        v != null ? <PctText value={v} raw precision={0} suffix="" /> : '--',
    },
  ]

  // ── 年度收益列 ──────────────────────────────────────
  const yearlyCols: ColumnsType<YearlyReturn> = [
    { title: '年度', dataIndex: 'year', key: 'year', width: 80 },
    {
      title: '收益率',
      dataIndex: 'return',
      key: 'return',
      width: 100,
      align: 'right',
      render: (v: number) => <PctText value={v} bold />,
    },
    {
      title: '累计',
      dataIndex: 'cumulative_return',
      key: 'cumulative_return',
      width: 100,
      align: 'right',
      render: (v: number) => <PctText value={v} />,
    },
    { title: '期初权益', dataIndex: 'start_equity', key: 'start_equity', width: 100, align: 'right', render: (v: number) => v.toFixed(0) },
    { title: '期末权益', dataIndex: 'end_equity', key: 'end_equity', width: 100, align: 'right', render: (v: number) => v.toFixed(0) },
  ]

  // ── 图表数据 ────────────────────────────────────────
  const chartDates = useMemo(
    () => result?.daily_equity?.map((e) => e.date) ?? [],
    [result],
  )
  const strategyValues = useMemo(
    () => result?.daily_equity?.map((e) => e.total_equity) ?? [],
    [result],
  )
  const benchmarkValues = useMemo(
    () =>
      result?.benchmark_equity
        ? result.benchmark_equity.map((b) => b.value)
        : undefined,
    [result],
  )
  const drawdownValues = useMemo(
    () => result?.daily_equity?.map((e) => e.drawdown * 100) ?? [],
    [result],
  )
  const monthlyMonths = useMemo(
    () => result?.monthly_returns?.map((m) => m.month) ?? [],
    [result],
  )
  const monthlyReturns = useMemo(
    () => result?.monthly_returns?.map((m: MonthlyReturn) => {
      // Backend returns "return" in the MonthlyReturn type
      const r = m as MonthlyReturn & { return?: number }
      return (r.return ?? 0) * 100
    }) ?? [],
    [result],
  )

  // ════════════════════════════════════════════════════════
  // Render: Form (idle / submitted / running / failed without result)
  // ════════════════════════════════════════════════════════
  const showForm = phase === 'idle' || phase === 'failed' && !result

  // ── 优化结果列 ──────────────────────────────────────
  const optResultCols: ColumnsType<OptimizeResultItem> = [
    {
      title: '#',
      key: 'rank',
      width: 40,
      render: (_: unknown, __: OptimizeResultItem, i: number) => i + 1,
    },
    {
      title: '买入阈值',
      key: 'buy_threshold',
      width: 80,
      align: 'right',
      render: (_: unknown, r: OptimizeResultItem) => ((r.params?.buy_threshold as number) ?? 0).toFixed(4),
    },
    {
      title: '持有天数',
      key: 'hold_days',
      width: 70,
      align: 'right',
      render: (_: unknown, r: OptimizeResultItem) => r.params?.hold_days ?? '--',
    },
    {
      title: '年化收益',
      dataIndex: 'annual_return',
      key: 'annual_return',
      width: 90,
      align: 'right',
      sorter: (a, b) => a.annual_return - b.annual_return,
      render: (v: number) => <PctText value={v} />,
    },
    {
      title: '最大回撤',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      width: 90,
      align: 'right',
      sorter: (a, b) => a.max_drawdown - b.max_drawdown,
      render: (v: number) => <span style={{ color: 'var(--danger)' }}>{(v * 100).toFixed(2)}%</span>,
    },
    {
      title: '夏普比率',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      width: 80,
      align: 'right',
      sorter: (a, b) => a.sharpe_ratio - b.sharpe_ratio,
      defaultSortOrder: 'descend',
      render: (v: number) => v?.toFixed(3),
    },
    {
      title: '胜率',
      dataIndex: 'win_rate',
      key: 'win_rate',
      width: 70,
      align: 'right',
      render: (v: number) => `${(v * 100).toFixed(1)}%`,
    },
    {
      title: '交易数',
      dataIndex: 'total_trades',
      key: 'total_trades',
      width: 60,
      align: 'right',
    },
  ]

  // ── Walk-Forward 窗口列 ─────────────────────────────
  const wfCols: ColumnsType<WalkForwardWindowResult> = [
    { title: '窗口', dataIndex: 'window_id', key: 'window_id', width: 50 },
    { title: '训练期', key: 'train', width: 160, render: (_: unknown, r: WalkForwardWindowResult) => `${r.train_start} ~ ${r.train_end}` },
    { title: '测试期', key: 'test', width: 160, render: (_: unknown, r: WalkForwardWindowResult) => `${r.test_start} ~ ${r.test_end}` },
    { title: 'OOS夏普', dataIndex: 'oos_sharpe', key: 'oos_sharpe', width: 80, align: 'right', render: (v: number) => v?.toFixed(3) },
    { title: 'OOS收益', dataIndex: 'oos_return', key: 'oos_return', width: 80, align: 'right', render: (v: number) => <PctText value={v} /> },
    { title: 'OOS回撤', dataIndex: 'oos_drawdown', key: 'oos_drawdown', width: 80, align: 'right', render: (v: number) => <span style={{ color: 'var(--danger)' }}>{(v * 100).toFixed(2)}%</span> },
  ]

  // ── 对比结果列 ──────────────────────────────────────
  const cmpResultCols: ColumnsType<CompareStrategyResult> = [
    { title: '策略', dataIndex: 'strategy_id', key: 'strategy_id', width: 60, render: (v: number) => `策略 ${v}` },
    {
      title: '买入阈值',
      key: 'buy_threshold',
      width: 80,
      align: 'right',
      render: (_: unknown, r: CompareStrategyResult) => ((r.params?.buy_threshold as number) ?? 0).toFixed(4),
    },
    {
      title: '持有天数',
      key: 'hold_days',
      width: 70,
      align: 'right',
      render: (_: unknown, r: CompareStrategyResult) => r.params?.hold_days ?? '--',
    },
    { title: '年化收益', dataIndex: 'annual_return', key: 'annual_return', width: 90, align: 'right', render: (v: number) => <PctText value={v} bold /> },
    { title: '夏普比率', dataIndex: 'sharpe_ratio', key: 'sharpe_ratio', width: 80, align: 'right', render: (v: number) => v?.toFixed(3) },
    { title: '卡玛比率', dataIndex: 'calmar_ratio', key: 'calmar_ratio', width: 80, align: 'right', render: (v: number) => v?.toFixed(3) },
    { title: '最大回撤', dataIndex: 'max_drawdown', key: 'max_drawdown', width: 90, align: 'right', render: (v: number) => <span style={{ color: 'var(--danger)' }}>{(v * 100).toFixed(2)}%</span> },
    { title: '胜率', dataIndex: 'win_rate', key: 'win_rate', width: 70, align: 'right', render: (v: number) => `${(v * 100).toFixed(1)}%` },
    { title: '交易数', dataIndex: 'total_trades', key: 'total_trades', width: 60, align: 'right' },
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
        <Title level={4} style={{ margin: 0 }}>
          <ExperimentOutlined /> 回测中心
        </Title>
        <Space>
          <Segmented
            value={pageMode}
            onChange={(v) => setPageMode(v as PageMode)}
            options={[
              { label: '回测', value: 'backtest', icon: <PlayCircleOutlined /> },
              { label: '参数优化', value: 'optimize', icon: <ThunderboltOutlined /> },
              { label: '多策略对比', value: 'compare', icon: <SwapOutlined /> },
            ]}
          />
          {phase !== 'idle' && pageMode === 'backtest' && (
            <Button icon={<ReloadOutlined />} onClick={handleReset}>
              新建回测
            </Button>
          )}
        </Space>
      </div>

      {/* ── 回测可用性提示（仅回测模式；信号来自 model_prediction 归档）── */}
      {pageMode === 'backtest' && paperAvail && !paperEnabled && (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          message="暂无可用信号数据，回测无法开始"
          description={paperAvail.note}
        />
      )}
      {pageMode === 'backtest' && paperAvail && paperEnabled && (
        <Alert
          type={paperAvail.count < 30 ? 'warning' : 'info'}
          showIcon
          style={{ marginBottom: 16 }}
          message={`本次将按历史信号回放：可用的信号日共 ${paperAvail.count} 个（${paperAvail.min_date} ~ ${paperAvail.max_date}）`}
          description={
            <span>
              策略在每个信号日收盘后生成一次预测（见下方"策略范围"说明），信号日越多结果越可靠。
              {paperAvail.count < 30 && ' 当前样本偏小：结果仅用于验证回测管线，不代表模型真实绩效。'}
            </span>
          }
        />
      )}

      {/* ── 回测表单 ───────────────────────────────── */}
      {pageMode === 'backtest' && showForm && (
        <Card title="回测参数" style={{ marginBottom: 16 }}>
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              scope: BACKTEST_FORM_DEFAULTS.scope,
              symbols: BACKTEST_FORM_DEFAULTS.symbols.join(', '),
              dateRange: [
                dayjs(BACKTEST_FORM_DEFAULTS.start_date),
                dayjs(BACKTEST_FORM_DEFAULTS.end_date),
              ],
              initial_capital: BACKTEST_FORM_DEFAULTS.initial_capital,
              strategy_params: BACKTEST_FORM_DEFAULTS.strategy_params,
            }}
          >
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="scope"
                  label="策略范围"
                  rules={[{ required: true }]}
                >
                  <Segmented
                    block
                    options={[
                      { label: '指数择时(中证1000→512100)', value: 'paper_index' },
                      { label: '中证1000逐股择时', value: 'csi1000_components', disabled: !compEnabled },
                      { label: '自定义指数择时', value: 'custom_index' },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="dateRange"
                  label="回测区间"
                  rules={[{ required: true, message: '请选择日期范围' }]}
                >
                  <RangePicker style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item noStyle shouldUpdate={(prev, next) => prev.scope !== next.scope}>
              {({ getFieldValue }) => (
                <div style={{ marginBottom: 16, color: 'var(--text-muted)', fontSize: 13 }}>
                  {getFieldValue('scope') === 'csi1000_components'
                    ? '对中证1000成分股逐股做 5 日预测信号（需历史成分权重文件，当前未启用；涨停禁买、跌停禁卖）。'
                    : getFieldValue('scope') === 'custom_index'
                      ? '自定义指数择时：选一个信号指数与执行 ETF 后，对区间内每个交易日用「当天为止」的行情在线重建 5 日预测 → 次日满仓/空仓交易。样本=区间交易日数。'
                      : '每个信号日：对中证1000(000852) 收盘后预测 → 下一交易日开盘满仓/空仓交易 512100(中证1000ETF)，固定持有 5 个交易日收盘退出（论文 §3.2 口径）。'}
                </div>
              )}
            </Form.Item>

            {/* 自定义指数择时：资产与泄漏模式选择 */}
            <Form.Item noStyle shouldUpdate={(prev, next) => prev.scope !== next.scope}>
              {({ getFieldValue }) =>
                getFieldValue('scope') === 'custom_index' ? (
                  <>
                    <Row gutter={16}>
                      <Col xs={24} sm={8}>
                        <Form.Item name="signal_symbol" label="信号指数" initialValue="000852">
                          <Select
                            options={assetPairs.map((p) => ({
                              value: p.signal_symbol,
                              label: `${p.signal_name}(${p.signal_symbol})`,
                            }))}
                            placeholder="选择被预测的指数"
                          />
                        </Form.Item>
                      </Col>
                      <Col xs={24} sm={8}>
                        <Form.Item name="execution_symbol" label="执行资产（交易）" initialValue="512100">
                          <Select
                            options={assetPairs.map((p) => ({
                              value: p.execution_symbol,
                              label: `${p.execution_name}(${p.execution_symbol})`,
                            }))}
                            placeholder="选择交易的 ETF"
                          />
                        </Form.Item>
                      </Col>
                      <Col xs={24} sm={8}>
                        <Form.Item name="leakage_mode" label="泄漏模式" initialValue="demo">
                          <Select
                            options={[
                              { value: 'demo', label: '演示模式（允许前视，结果偏乐观）' },
                              { value: 'strict', label: '严格模式（仅训练截止后，样本≈0）' },
                            ]}
                          />
                        </Form.Item>
                      </Col>
                    </Row>
                    {customAvail?.note && (
                      <Alert
                        type="warning"
                        showIcon
                        style={{ marginBottom: 16 }}
                        message="前视泄漏提示"
                        description={customAvail.note}
                      />
                    )}
                  </>
                ) : null
              }
            </Form.Item>

            <Row gutter={16}>
              <Col xs={24} sm={8}>
                <Form.Item
                  name="initial_capital"
                  label={
                    <Space>
                      初始资金
                      <Tooltip title="回测初始资金（元）">
                        <InfoCircleOutlined style={{ color: 'var(--text-muted)' }} />
                      </Tooltip>
                    </Space>
                  }
                >
                  <InputNumber
                    min={10000}
                    max={100_000_000}
                    step={100000}
                    style={{ width: '100%' }}
                    formatter={(v) => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} sm={8}>
                <Form.Item
                  name={['strategy_params', 'buy_threshold']}
                  label="五日预测收益阈值"
                  initialValue={0.005}
                >
                  <InputNumber
                    min={-0.05}
                    max={0.2}
                    step={0.001}
                    style={{ width: '100%' }}
                    formatter={(v) => `${((v ?? 0) * 100).toFixed(1)}%`}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} sm={8}>
                <Form.Item
                  name={['strategy_params', 'hold_days']}
                  label="固定持有交易日"
                  initialValue={5}
                >
                  <InputNumber
                    min={1}
                    max={20}
                    step={1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </Col>
            </Row>

            <div style={{ textAlign: 'right' }}>
              <Space>
                <Button onClick={() => form.resetFields()}>重置</Button>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleSubmit}
                  loading={submitting}
                  disabled={!paperEnabled}
                  size="large"
                >
                  开始回测
                </Button>
              </Space>
            </div>
          </Form>
        </Card>
      )}

      {/* ── 进度条 ───────────────────────────────── */}
      {pageMode === 'backtest' && (phase === 'submitted' || phase === 'running' || polling) && (
        <Card style={{ marginBottom: 16 }}>
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <Title level={5}>
              <ClockCircleOutlined spin={polling} />{' '}
              {polling ? '回测运行中…' : phase === 'submitted' ? '已提交' : '运行中'}
            </Title>
            <Progress
              percent={Math.round((progress?.progress ?? 0) * 100)}
              status={polling ? 'active' : phase === 'submitted' ? 'normal' : 'active'}
              style={{ maxWidth: 500, margin: '16px auto' }}
            />
            <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
              任务 ID: <Text code>{progress?.task_id}</Text>
              {progress?.message && <span> — {progress.message}</span>}
            </div>
          </div>
        </Card>
      )}

      {/* ── 回测结果 ───────────────────────────────── */}
      {pageMode === 'backtest' && phase === 'completed' && result && (
        <>
          {/* 摘要信息 */}
          <Card style={{ marginBottom: 16 }}>
            <Row gutter={[4, 4]} style={{ marginBottom: 12 }}>
              <Col>
                <Space>
                  {STATUS_TAG[result.status]?.icon}
                  <Text strong>任务 ID:</Text>
                  <Text code>{result.task_id}</Text>
                  <Tag color={STATUS_TAG[result.status]?.color}>
                    {STATUS_TAG[result.status]?.label}
                  </Tag>
                  <Text type="secondary">
                    {result.start_date} ~ {result.end_date} | 耗时 {result.elapsed_seconds?.toFixed(1)}s
                  </Text>
                </Space>
              </Col>
            </Row>

            {/* ── 本次回测说明（讲人话）────────────────── */}
            {(result as { config?: Record<string, unknown> }).config?.scope && (
              <Descriptions
                size="small"
                column={3}
                style={{ marginBottom: 16, fontSize: 13 }}
                items={(() => {
                  const cfg = (result as { config?: Record<string, unknown> }).config as Record<string, unknown>
                  const scopeLabel =
                    cfg.scope === 'paper_index' ? '指数择时：预测 中证1000(000852) → 交易 512100'
                    : cfg.scope === 'custom_index' ? `自定义指数择时：预测 ${cfg.signal_asset ?? ''} → 交易 ${cfg.execution_asset ?? ''}`
                    : '中证1000 逐股择时'
                  const sourceLabel = String(cfg.prediction_source ?? '')
                    .includes('online')
                    ? '在线重建（每个交易日当天现算）'
                    : '历史信号归档（model_prediction）'
                  return [
                    { key: 'scope', label: '策略', children: scopeLabel },
                    { key: 'range', label: '回测区间', children: `${result.start_date} ~ ${result.end_date}` },
                    { key: 'signal-source', label: '信号来源', children: sourceLabel },
                    { key: 'hold', label: '规则', children: `预测收益 > ${((cfg.buy_threshold as number ?? 0.005) * 100).toFixed(1)}% 满仓买入，持有 ${cfg.hold_days ?? 5} 个交易日到期卖出` },
                    { key: 'model', label: '模型版本', children: String(cfg.model_version ?? '--') },
                    ...(cfg.leakage_note
                      ? [{ key: 'leak', label: '⚠ 前视提示', span: 3 as const, children: String(cfg.leakage_note) }]
                      : []),
                  ]
                })()}
              />
            )}

            {/* ── 绩效指标卡片 ─────────────────────── */}
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={4}>
                <StatCard
                  title="累计收益率"
                  value={(result.cumulative_return ?? 0) * 100}
                  suffix="%"
                  precision={2}
                />
              </Col>
              <Col xs={12} sm={4}>
                <StatCard
                  title="年化收益率"
                  value={(result.annual_return ?? 0) * 100}
                  suffix="%"
                  precision={2}
                />
              </Col>
              <Col xs={12} sm={4}>
                <StatCard
                  title="最大回撤"
                  value={(result.max_drawdown ?? 0) * 100}
                  suffix="%"
                  precision={2}
                />
              </Col>
              <Col xs={12} sm={4}>
                <StatCard
                  title="夏普比率"
                  value={result.sharpe_ratio ?? 0}
                  precision={2}
                />
              </Col>
              <Col xs={12} sm={4}>
                <StatCard
                  title="胜率"
                  value={(result.win_rate ?? 0) * 100}
                  suffix="%"
                  precision={1}
                />
              </Col>
              <Col xs={12} sm={4}>
                <StatCard
                  title="总交易"
                  value={result.total_trades ?? 0}
                  suffix=" 笔"
                  precision={0}
                />
              </Col>
            </Row>

            <Divider style={{ margin: '12px 0' }} />

            {/* 更多指标 */}
            <Row gutter={16}>
              {[
                { label: '卡玛比率', value: (result.calmar_ratio ?? 0).toFixed(2) },
                { label: '超额收益', value: ((result.excess_return ?? 0) * 100).toFixed(2) + '%' },
                { label: '盈亏比', value: (result.profit_loss_ratio ?? 0).toFixed(2) },
                { label: '最大连胜', value: result.max_consecutive_wins ?? '--' },
                { label: '最大连亏', value: result.max_consecutive_losses ?? '--' },
                { label: '平均持仓', value: (result.avg_hold_days ?? 0).toFixed(1) + ' 天' },
                { label: '期末权益', value: (result.final_equity ?? 0).toLocaleString() },
              ].map((m) => (
                <Col xs={12} sm={6} lg={3} key={m.label}>
                  <div style={{ marginBottom: 4 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>{m.label}</Text>
                  </div>
                  <Text strong style={{ fontSize: 15 }}>{m.value}</Text>
                </Col>
              ))}
            </Row>
          </Card>

          {/* ── 图表 ───────────────────────────────── */}
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col xs={24} lg={14}>
              <Card
                title={<Space><LineChartOutlined />收益曲线</Space>}
                size="small"
              >
                {chartDates.length > 0 ? (
                  <EquityCurveChart
                    dates={chartDates}
                    strategyValues={strategyValues}
                    benchmarkValues={benchmarkValues}
                    height={350}
                  />
                ) : (
                  <EmptyState message="无权益数据" />
                )}
              </Card>
            </Col>
            <Col xs={24} lg={10}>
              <Card
                title={<Space><FallOutlined />最大回撤</Space>}
                size="small"
              >
                {drawdownValues.length > 0 ? (
                  <DrawdownChart dates={chartDates} values={drawdownValues} height={350} />
                ) : (
                  <EmptyState message="无回撤数据" />
                )}
              </Card>
            </Col>
          </Row>

          {/* ── 月度收益 ──────────────────────────── */}
          {monthlyReturns.length > 0 && (
            <Card
              title={<Space><BarChartOutlined />月度收益分布</Space>}
              size="small"
              style={{ marginBottom: 16 }}
            >
              <MonthlyReturnsChart months={monthlyMonths} returns={monthlyReturns} height={280} />
            </Card>
          )}

          {/* ── 详情 Tabs ─────────────────────────── */}
          <Card>
            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              items={[
                {
                  key: 'trades',
                  label: `交易记录 (${result.trades?.length ?? 0})`,
                  children: (
                    <Table<TradeRecord>
                      columns={tradeCols}
                      dataSource={result.trades ?? []}
                      rowKey={(_, i) => String(i)}
                      size="small"
                      pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 笔` }}
                      scroll={{ x: 1100 }}
                      locale={{ emptyText: '暂无交易记录' }}
                    />
                  ),
                },
                {
                  key: 'equity',
                  label: '权益明细',
                  children: (
                    <Table<DailyEquityPoint>
                      columns={equityCols}
                      dataSource={result.daily_equity ?? []}
                      rowKey="date"
                      size="small"
                      pagination={{ pageSize: 30, showSizeChanger: true, showTotal: (t) => `共 ${t} 天` }}
                      scroll={{ x: 700 }}
                      locale={{ emptyText: '暂无权益数据' }}
                    />
                  ),
                },
                {
                  key: 'drawdown',
                  label: '回撤分析',
                  children: result.max_drawdown_period ? (
                    <Descriptions bordered size="small" column={2}>
                      <Descriptions.Item label="峰值日期">{result.max_drawdown_period.peak_date || '--'}</Descriptions.Item>
                      <Descriptions.Item label="谷底日期">{result.max_drawdown_period.trough_date || '--'}</Descriptions.Item>
                      <Descriptions.Item label="峰值净值">{result.max_drawdown_period.peak_value?.toFixed(2)}</Descriptions.Item>
                      <Descriptions.Item label="谷底净值">{result.max_drawdown_period.trough_value?.toFixed(2)}</Descriptions.Item>
                      <Descriptions.Item label="恢复日期">{result.max_drawdown_period.recovery_date || '--'}</Descriptions.Item>
                      <Descriptions.Item label="回撤天数">{result.max_drawdown_period.drawdown_days} 天</Descriptions.Item>
                    </Descriptions>
                  ) : (
                    <EmptyState message="无回撤分析数据" />
                  ),
                },
                {
                  key: 'risk',
                  label: `风控事件 (${result.risk_events?.length ?? 0})`,
                  children: (
                    <Table<RiskEvent>
                      columns={riskCols}
                      dataSource={result.risk_events ?? []}
                      rowKey={(_, i) => String(i)}
                      size="small"
                      pagination={{ pageSize: 20, showTotal: (t) => `共 ${t} 次` }}
                      scroll={{ x: 600 }}
                      locale={{ emptyText: '无风控事件' }}
                    />
                  ),
                },
                {
                  key: 'yearly',
                  label: '年度收益',
                  children: (
                    <Table<YearlyReturn>
                      columns={yearlyCols}
                      dataSource={result.yearly_returns ?? []}
                      rowKey="year"
                      size="small"
                      pagination={false}
                      locale={{ emptyText: '无年度收益数据' }}
                    />
                  ),
                },
              ]}
            />
          </Card>
        </>
      )}

      {/* ── 失败 ───────────────────────────────────── */}
      {pageMode === 'backtest' && phase === 'failed' && !result && (
        <Card>
          <ErrorState
            message="回测任务未能完成"
            detail={progress?.message || '请检查后端日志了解详情'}
            onRetry={() => {
              setPhase('idle')
            }}
          />
        </Card>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* 参数优化模式                                              */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {pageMode === 'optimize' && (
        <>
          {/* 优化表单 */}
          {!optimizeResult && !optimizePolling && (
            <Card title="参数优化配置" style={{ marginBottom: 16 }}>
              <Form form={optForm} layout="vertical" initialValues={{
                scope: 'paper_index',
                dateRange: [dayjs('2023-01-01'), dayjs('2024-12-31')],
                initial_capital: 1_000_000,
                mode: 'grid',
                window_months: 12,
                step_months: 3,
              }}>
                <Row gutter={16}>
                  <Col xs={24} sm={12}>
                    <Form.Item name="scope" label="策略范围" rules={[{ required: true }]}>
                      <Segmented block options={[
                        { label: '指数择时(中证1000→512100)', value: 'paper_index' },
                        { label: '中证1000逐股择时', value: 'csi1000_components', disabled: !compEnabled },
                      ]} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={12}>
                    <Form.Item name="dateRange" label="回测区间" rules={[{ required: true }]}>
                      <RangePicker style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col xs={24} sm={8}>
                    <Form.Item name="initial_capital" label="初始资金">
                      <InputNumber min={10000} max={100_000_000} step={100000} style={{ width: '100%' }}
                        formatter={(v) => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={8}>
                    <Form.Item name="mode" label="优化模式">
                      <Segmented block options={[
                        { label: '网格搜索', value: 'grid' },
                        { label: 'Walk-Forward', value: 'walk_forward' },
                      ]} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={8}>
                    <Form.Item label="排序指标">
                      <Select value={optimizeMetric} onChange={setOptimizeMetric} options={METRIC_OPTIONS} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>

                {/* Walk-Forward 参数 */}
                <Form.Item noStyle shouldUpdate={(prev, next) => prev.mode !== next.mode}>
                  {({ getFieldValue }) =>
                    getFieldValue('mode') === 'walk_forward' ? (
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col xs={12} sm={6}>
                          <Form.Item name="window_months" label="训练窗口(月)">
                            <InputNumber min={3} max={36} step={1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} sm={6}>
                          <Form.Item name="step_months" label="步进(月)">
                            <InputNumber min={1} max={6} step={1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                      </Row>
                    ) : null
                  }
                </Form.Item>

                {/* 参数范围配置 */}
                <Divider titlePlacement="start" style={{ fontSize: 13 }}>参数搜索范围</Divider>
                {paramRanges.map((pr, idx) => (
                  <Row gutter={8} key={idx} style={{ marginBottom: 8, alignItems: 'center' }}>
                    <Col span={5}>
                      <Text strong style={{ fontSize: 13 }}>{pr.name}</Text>
                    </Col>
                    <Col span={4}>
                      <InputNumber size="small" value={pr.min} step={pr.name === 'hold_days' ? 1 : 0.001}
                        onChange={(v) => {
                          const next = [...paramRanges]; next[idx] = { ...next[idx], min: v ?? pr.min }
                          setParamRanges(next)
                        }} style={{ width: '100%' }} />
                    </Col>
                    <Col span={4}>
                      <InputNumber size="small" value={pr.max} step={pr.name === 'hold_days' ? 1 : 0.001}
                        onChange={(v) => {
                          const next = [...paramRanges]; next[idx] = { ...next[idx], max: v ?? pr.max }
                          setParamRanges(next)
                        }} style={{ width: '100%' }} />
                    </Col>
                    <Col span={4}>
                      <InputNumber size="small" value={pr.step} step={pr.name === 'hold_days' ? 1 : 0.001}
                        onChange={(v) => {
                          const next = [...paramRanges]; next[idx] = { ...next[idx], step: v ?? pr.step }
                          setParamRanges(next)
                        }} style={{ width: '100%' }} />
                    </Col>
                    <Col span={3}>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {(() => {
                          let count = 0
                          for (let v = pr.min; v <= pr.max + pr.step * 0.01; v += pr.step) count++
                          return `${count} 值`
                        })()}
                      </Text>
                    </Col>
                  </Row>
                ))}
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
                  总组合数: <Text strong>{
                    paramRanges.reduce((acc, pr) => {
                      let count = 0
                      for (let v = pr.min; v <= pr.max + pr.step * 0.01; v += pr.step) count++
                      return acc * Math.max(count, 1)
                    }, 1)
                  }</Text>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <Space>
                    <Button onClick={() => setParamRanges(DEFAULT_PARAM_RANGES)}>重置范围</Button>
                    <Button type="primary" icon={<ThunderboltOutlined />}
                      onClick={handleOptimizeSubmit} loading={optimizeSubmitting} size="large"
                      disabled={!paperEnabled}>
                      开始优化
                    </Button>
                  </Space>
                </div>
              </Form>
            </Card>
          )}

          {/* 优化进度 */}
          {optimizePolling && (
            <Card style={{ marginBottom: 16 }}>
              <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <Title level={5}><ClockCircleOutlined spin /> 参数优化运行中…</Title>
                <Progress percent={Math.round(optimizeProgress * 100)} status="active"
                  style={{ maxWidth: 500, margin: '16px auto' }} />
              </div>
            </Card>
          )}

          {/* 优化结果 */}
          {optimizeResult && (
            <>
              <Alert type="success" showIcon style={{ marginBottom: 12 }}
                message={`优化完成：${optimizeResult.total_combinations} 组参数，模式=${optimizeResult.mode}，指标=${optimizeResult.metric}`} />

              {/* 最优参数 */}
              {optimizeResult.best_params && (
                <Card title="🏆 最优参数组合" size="small" style={{ marginBottom: 16 }}>
                  <Descriptions bordered size="small" column={3}>
                    {Object.entries(optimizeResult.best_params).map(([k, v]) => (
                      <Descriptions.Item key={k} label={k}>
                        {typeof v === 'number' ? v.toFixed(4) : String(v)}
                      </Descriptions.Item>
                    ))}
                  </Descriptions>
                  {optimizeResult.best_metrics && (
                    <div style={{ marginTop: 12 }}>
                      <Space size="large">
                        <Text>年化收益: <PctText value={optimizeResult.best_metrics.annual_return} bold /></Text>
                        <Text>夏普: <Text strong>{optimizeResult.best_metrics.sharpe_ratio?.toFixed(3)}</Text></Text>
                        <Text>回撤: <span style={{ color: 'var(--danger)' }}>{(optimizeResult.best_metrics.max_drawdown * 100).toFixed(2)}%</span></Text>
                      </Space>
                    </div>
                  )}
                </Card>
              )}

              <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                {/* 敏感度分析图 */}
                <Col xs={24} lg={12}>
                  <Card title={<Space><LineChartOutlined />参数敏感度</Space>} size="small">
                    {optimizeResult.sensitivity ? (
                      <ParamSensitivityChart data={optimizeResult.sensitivity as any} height={300} />
                    ) : (
                      <EmptyState message="无敏感度数据" />
                    )}
                  </Card>
                </Col>

                {/* Walk-Forward 窗口结果 */}
                <Col xs={24} lg={12}>
                  <Card title={<Space><BarChartOutlined />Walk-Forward 窗口</Space>} size="small">
                    {optimizeResult.walk_forward_windows?.length > 0 ? (
                      <Table<WalkForwardWindowResult>
                        columns={wfCols}
                        dataSource={optimizeResult.walk_forward_windows}
                        rowKey="window_id"
                        size="small"
                        pagination={false}
                        scroll={{ x: 600 }}
                      />
                    ) : (
                      <EmptyState message="非 Walk-Forward 模式" />
                    )}
                  </Card>
                </Col>
              </Row>

              {/* 稳定性报告 */}
              {optimizeResult.stability_report && (
                <Card title="稳定性报告" size="small" style={{ marginBottom: 16 }}>
                  <Descriptions bordered size="small" column={3}>
                    {Object.entries(optimizeResult.stability_report).map(([k, v]) => (
                      <Descriptions.Item key={k} label={k}>
                        {typeof v === 'number' ? v.toFixed(4) : String(v)}
                      </Descriptions.Item>
                    ))}
                  </Descriptions>
                </Card>
              )}

              {/* 全部组合排序表 */}
              <Card title="参数组合排行" size="small" style={{ marginBottom: 16 }}>
                <Space style={{ marginBottom: 8 }}>
                  <Text>排序指标:</Text>
                  <Select value={optimizeSortMetric} onChange={setOptimizeSortMetric} options={METRIC_OPTIONS} size="small" style={{ width: 120 }} />
                </Space>
                <Table<OptimizeResultItem>
                  columns={optResultCols}
                  dataSource={optimizeResult.results}
                  rowKey={(_, i) => String(i)}
                  size="small"
                  pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 组` }}
                  scroll={{ x: 600 }}
                />
              </Card>

              <div style={{ textAlign: 'center' }}>
                <Button onClick={() => { setOptimizeResult(null); setOptimizeProgress(0) }}>
                  重新优化
                </Button>
              </div>
            </>
          )}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* 多策略对比模式                                          */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {pageMode === 'compare' && (
        <>
          {/* 对比表单 */}
          {!compareResult && !comparePolling && (
            <Card title="多策略对比配置" style={{ marginBottom: 16 }}>
              <Form form={cmpForm} layout="vertical" initialValues={{
                scope: 'paper_index',
                dateRange: [dayjs('2023-01-01'), dayjs('2024-12-31')],
                initial_capital: 1_000_000,
              }}>
                <Row gutter={16}>
                  <Col xs={24} sm={8}>
                    <Form.Item name="scope" label="策略范围" rules={[{ required: true }]}>
                      <Segmented block options={[
                        { label: '指数择时(中证1000→512100)', value: 'paper_index' },
                        { label: '中证1000逐股择时', value: 'csi1000_components', disabled: !compEnabled },
                      ]} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={8}>
                    <Form.Item name="dateRange" label="回测区间" rules={[{ required: true }]}>
                      <RangePicker style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={8}>
                    <Form.Item name="initial_capital" label="初始资金">
                      <InputNumber min={10000} max={100_000_000} step={100000} style={{ width: '100%' }}
                        formatter={(v) => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')} />
                    </Form.Item>
                  </Col>
                </Row>

                {/* 策略参数配置 */}
                <Divider titlePlacement="start" style={{ fontSize: 13 }}>
                  策略参数 ({compareStrategies.length}/5)
                </Divider>
                {compareStrategies.map((s, idx) => (
                  <Card key={s.key} size="small" style={{ marginBottom: 8 }}
                    title={<Text strong>策略 {idx + 1}</Text>}
                    extra={compareStrategies.length > 2 && (
                      <Button size="small" danger icon={<DeleteOutlined />}
                        onClick={() => setCompareStrategies(prev => prev.filter((_, i) => i !== idx))}>
                        删除
                      </Button>
                    )}
                  >
                    <Row gutter={16}>
                      <Col xs={12} sm={6}>
                        <Text type="secondary" style={{ fontSize: 12 }}>买入阈值</Text>
                        <InputNumber size="small" value={s.buy_threshold} min={0} max={0.1} step={0.001}
                          onChange={(v) => {
                            const next = [...compareStrategies]; next[idx] = { ...next[idx], buy_threshold: v ?? 0.005 }
                            setCompareStrategies(next)
                          }} style={{ width: '100%' }} />
                      </Col>
                      <Col xs={12} sm={6}>
                        <Text type="secondary" style={{ fontSize: 12 }}>持有天数</Text>
                        <InputNumber size="small" value={s.hold_days} min={1} max={20} step={1}
                          onChange={(v) => {
                            const next = [...compareStrategies]; next[idx] = { ...next[idx], hold_days: v ?? 5 }
                            setCompareStrategies(next)
                          }} style={{ width: '100%' }} />
                      </Col>
                      <Col xs={12} sm={6}>
                        <Text type="secondary" style={{ fontSize: 12 }}>止损</Text>
                        <InputNumber size="small" value={s.stop_loss} min={-0.20} max={-0.02} step={0.01}
                          onChange={(v) => {
                            const next = [...compareStrategies]; next[idx] = { ...next[idx], stop_loss: v ?? -0.08 }
                            setCompareStrategies(next)
                          }} style={{ width: '100%' }} />
                      </Col>
                      <Col xs={12} sm={6}>
                        <Text type="secondary" style={{ fontSize: 12 }}>止盈</Text>
                        <InputNumber size="small" value={s.take_profit} min={0.05} max={0.30} step={0.01}
                          onChange={(v) => {
                            const next = [...compareStrategies]; next[idx] = { ...next[idx], take_profit: v ?? 0.15 }
                            setCompareStrategies(next)
                          }} style={{ width: '100%' }} />
                      </Col>
                    </Row>
                  </Card>
                ))}
                {compareStrategies.length < 5 && (
                  <Button icon={<PlusOutlined />} style={{ marginBottom: 16 }}
                    onClick={() => setCompareStrategies(prev => [...prev, {
                      key: `s${prev.length + 1}`,
                      buy_threshold: 0.005, hold_days: 5, stop_loss: -0.08, take_profit: 0.15,
                    }])}>
                    添加策略
                  </Button>
                )}

                <div style={{ textAlign: 'right' }}>
                  <Space>
                    <Button onClick={() => setCompareStrategies(DEFAULT_COMPARE_STRATEGIES)}>重置</Button>
                    <Button type="primary" icon={<SwapOutlined />}
                      onClick={handleCompareSubmit} loading={compareSubmitting} size="large"
                      disabled={!paperEnabled}>
                      开始对比
                    </Button>
                  </Space>
                </div>
              </Form>
            </Card>
          )}

          {/* 对比进度 */}
          {comparePolling && (
            <Card style={{ marginBottom: 16 }}>
              <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <Title level={5}><ClockCircleOutlined spin /> 多策略对比运行中…</Title>
                <Progress percent={100} status="active" style={{ maxWidth: 500, margin: '16px auto' }} />
              </div>
            </Card>
          )}

          {/* 对比结果 */}
          {compareResult && (
            <>
              <Alert type="success" showIcon style={{ marginBottom: 12 }}
                message={`对比完成：${compareResult.results.length} 组策略`} />

              {/* 绩效对比表 */}
              <Card title="绩效对比" size="small" style={{ marginBottom: 16 }}>
                <Table<CompareStrategyResult>
                  columns={cmpResultCols}
                  dataSource={compareResult.results}
                  rowKey="strategy_id"
                  size="small"
                  pagination={false}
                  scroll={{ x: 700 }}
                />
                {compareResult.best_by_metric && (
                  <div style={{ marginTop: 12 }}>
                    <Text type="secondary">各指标最优:</Text>{' '}
                    {Object.entries(compareResult.best_by_metric).map(([metric, id]) => (
                      <Tag key={metric} color="blue">{metric}: 策略 {id}</Tag>
                    ))}
                  </div>
                )}
              </Card>

              <div style={{ textAlign: 'center' }}>
                <Button onClick={() => setCompareResult(null)}>
                  重新对比
                </Button>
              </div>
            </>
          )}
        </>
      )}

      {/* ── 历史记录 ──────────────────────────────── */}
      {pageMode === 'backtest' && phase === 'idle' && (
        <Card
          title={<Space><HistoryOutlined />历史回测</Space>}
          extra={
            <Button icon={<ReloadOutlined />} onClick={loadHistory} loading={historyLoading} size="small">
              刷新
            </Button>
          }
        >
          {history.length === 0 ? (
            <EmptyState message="暂无历史回测记录，提交一个新回测吧" />
          ) : (
            <Table<BacktestHistoryItem>
              dataSource={history}
              rowKey="task_id"
              size="small"
              loading={historyLoading}
              pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }}
              onRow={(record) => ({
                onClick: () => handleViewHistory(record.task_id),
                style: { cursor: 'pointer' },
              })}
              columns={[
                {
                  title: '任务 ID',
                  dataIndex: 'task_id',
                  key: 'task_id',
                  width: 90,
                  render: (v: string) => <Text code>{v?.slice(0, 8)}</Text>,
                },
                {
                  title: '区间',
                  key: 'range',
                  width: 180,
                  render: (_: unknown, r: BacktestHistoryItem) => `${r.start_date} ~ ${r.end_date}`,
                },
                {
                  title: '累计收益',
                  dataIndex: 'cumulative_return',
                  key: 'cumulative_return',
                  width: 90,
                  align: 'right',
                  render: (v: number) => <PctText value={v} bold />,
                },
                {
                  title: '夏普',
                  dataIndex: 'sharpe_ratio',
                  key: 'sharpe_ratio',
                  width: 70,
                  align: 'right',
                  render: (v: number) => v?.toFixed(2),
                },
                {
                  title: '最大回撤',
                  dataIndex: 'max_drawdown',
                  key: 'max_drawdown',
                  width: 80,
                  align: 'right',
                  render: (v: number) => (
                    <span style={{ color: 'var(--danger)' }}>{(v * 100).toFixed(1)}%</span>
                  ),
                },
                {
                  title: '交易数',
                  dataIndex: 'total_trades',
                  key: 'total_trades',
                  width: 70,
                  align: 'right',
                },
                {
                  title: '状态',
                  dataIndex: 'status',
                  key: 'status',
                  width: 80,
                  render: (v: string) => {
                    const tag = STATUS_TAG[v]
                    return <Tag color={tag?.color}>{tag?.label ?? v}</Tag>
                  },
                },
                {
                  title: '完成时间',
                  dataIndex: 'completed_at',
                  key: 'completed_at',
                  width: 100,
                  render: (v: string | undefined) => v || '--',
                },
              ]}
            />
          )}
        </Card>
      )}
    </div>
  )
}
