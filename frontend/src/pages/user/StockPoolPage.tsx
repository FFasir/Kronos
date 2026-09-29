/**
 * 选股池页面
 * 对应旧版 user_portal.html #page-stock-pool
 *
 * 功能：Kronos 模型预测选股池表格，支持策略预设、高级参数、筛选统计、CSV 导出
 * 数据来源：GET /api/v1/prediction/latest
 *           GET /api/v1/selection/presets
 *           POST /api/v1/selection/generate
 */

import { useEffect, useState, useCallback, useMemo } from 'react'
import {
  Card,
  Typography,
  Space,
  Button,
  Select,
  Input,
  Tooltip,
  Row,
  Col,
  Slider,
  InputNumber,
  Tag,
  Collapse,
  Statistic,
  Alert,
} from 'antd'
import {
  ReloadOutlined,
  DownloadOutlined,
  SearchOutlined,
  FilterOutlined,
  SafetyCertificateOutlined,
  FundOutlined,
  RocketOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  InfoCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { getLatestSelection, getStrategyPresets, generateSelection } from '@/api/prediction'
import { getLatestSignals } from '@/api/signal'
import { apiProbe } from '@/api/client'
import type { SignalResult, StrategyPreset, SelectionStats } from '@/types/api'
import { useStockStore } from '@/stores/useStockStore'
import StockTable, { toStockTableRows } from '@/components/stock/StockTable'
import LoadingSkeleton from '@/components/common/LoadingSkeleton'
import ErrorState from '@/components/common/ErrorState'
import EmptyState from '@/components/common/EmptyState'
import Toast from '@/components/common/Toast'
import { useNavigate } from 'react-router-dom'

const { Title, Text } = Typography

/** 预设筛选方案 */
type Preset = 'all' | 'csi1000' | 'csi500' | 'hs300' | 'top20' | 'top50'

const PRESET_OPTIONS: { value: Preset; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'csi1000', label: '中证1000' },
  { value: 'top20', label: 'Top 20' },
  { value: 'top50', label: 'Top 50' },
  { value: 'csi500', label: '中证500' },
  { value: 'hs300', label: '沪深300' },
]

/** CSI 500 component stocks - partial sample for filtering */
const CSI500_SAMPLE = new Set(['000852', '000905'])

/** HS 300 component stocks - partial sample */
const HS300_SAMPLE = new Set(['000300', '000016'])

/** 策略风格预设图标 */
const PRESET_ICONS: Record<string, React.ReactNode> = {
  conservative: <SafetyCertificateOutlined style={{ fontSize: 24, color: '#1890ff' }} />,
  balanced: <FundOutlined style={{ fontSize: 24, color: '#52c41a' }} />,
  aggressive: <RocketOutlined style={{ fontSize: 24, color: '#faad14' }} />,
}

/**
 * 判断选股失败原因：后端未启动 vs 后端正常但暂无预测数据。
 * 预测数据依赖 Kronos 模型，模型未下载/未运行时选股池会诚实地为空。
 */
async function resolveSelectionError(): Promise<string> {
  const backendUp = await apiProbe()
  return backendUp
    ? '暂无选股数据：预测模型尚未就绪，需先下载 Kronos 模型并运行预测任务'
    : '选股数据加载失败，请确认后端服务已启动'
}

export default function StockPoolPage() {
  const navigate = useNavigate()
  const { pool, setPool, poolLoading, setPoolLoading } = useStockStore()

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [preset, setPreset] = useState<Preset>('all')
  const [search, setSearch] = useState('')
  const [signals, setSignals] = useState<SignalResult[]>([])

  // ── 策略预设 ────────────────────────────────────────
  const [strategyPresets, setStrategyPresets] = useState<StrategyPreset[]>([])
  const [activeStrategyPreset, setActiveStrategyPreset] = useState<string | null>(null)
  const [selectionStats, setSelectionStats] = useState<SelectionStats | null>(null)
  const [dataDate, setDataDate] = useState<string | undefined>(undefined)

  // ── 高级参数 ────────────────────────────────────────
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [probUpThreshold, setProbUpThreshold] = useState(0.65)
  const [scoreThreshold, setScoreThreshold] = useState(6.0)
  const [minDailyAmount, setMinDailyAmount] = useState(5_000_000)
  const [maxPerIndustry, setMaxPerIndustry] = useState(3)
  const [topK, setTopK] = useState(50)

  // ── 加载预设 ────────────────────────────────────────
  useEffect(() => {
    getStrategyPresets().then((data) => {
      if (data?.presets) setStrategyPresets(data.presets)
    })
  }, [])

  // ── 加载数据 ────────────────────────────────────────
  const loadData = useCallback(async (strategyPreset?: string) => {
    setLoading(true)
    setError(null)
    setPoolLoading(true)
    try {
      // 如果有策略预设，通过 selection/generate 获取带统计的结果
      if (strategyPreset) {
        const data = await generateSelection({ preset: strategyPreset })
        if (!data) {
          setError(await resolveSelectionError())
          setPool([])
          return
        }
        setPool(data.predictions ?? [])
        if ((data as any).stats) setSelectionStats((data as any).stats as SelectionStats)
        if ((data as any).data_date) setDataDate((data as any).data_date as string)
      } else {
        const [data, signalData] = await Promise.all([
          getLatestSelection(),
          getLatestSignals(),
        ])
        if (!data) {
          setError(await resolveSelectionError())
          setPool([])
          return
        }
        setPool(data.predictions ?? [])
        setSignals(signalData?.signals ?? [])
        if ((data as any).stats) setSelectionStats((data as any).stats as SelectionStats)
        if ((data as any).data_date) setDataDate((data as any).data_date as string)
      }
    } catch {
      setError(await resolveSelectionError())
    } finally {
      setLoading(false)
      setPoolLoading(false)
    }
  }, [setPool, setPoolLoading])

  useEffect(() => { loadData() }, [loadData])

  // ── 切换策略预设 ────────────────────────────────────
  const handleStrategyPresetChange = useCallback((presetName: string) => {
    setActiveStrategyPreset(presetName)

    // 用预设值更新高级参数
    const found = strategyPresets.find((p) => p.name === presetName)
    if (found) {
      setProbUpThreshold(found.prob_up_threshold)
      setScoreThreshold(found.score_threshold)
      setMinDailyAmount(found.min_daily_amount)
      setMaxPerIndustry(found.max_per_industry)
      setTopK(found.top_k)
    }

    loadData(presetName)
  }, [strategyPresets, loadData])

  // ── 用自定义参数重新生成 ────────────────────────────
  const handleRegenerate = useCallback(async () => {
    setLoading(true)
    setPoolLoading(true)
    try {
      const data = await generateSelection({
        preset: activeStrategyPreset ?? undefined,
        prob_up_threshold: probUpThreshold,
        score_threshold: scoreThreshold,
        min_daily_amount: minDailyAmount,
        max_per_industry: maxPerIndustry,
        top_k: topK,
      })
      if (!data) {
        Toast.error('生成选股池失败')
        return
      }
      setPool(data.predictions ?? [])
      if ((data as any).stats) setSelectionStats((data as any).stats as SelectionStats)
      Toast.success(`选股池已更新：${data.predictions?.length ?? 0} 只`)
    } catch {
      Toast.error('生成选股池失败')
    } finally {
      setLoading(false)
      setPoolLoading(false)
    }
  }, [activeStrategyPreset, probUpThreshold, scoreThreshold, minDailyAmount, maxPerIndustry, topK, setPool, setPoolLoading])

  // ── 筛选逻辑 ────────────────────────────────────────
  const filtered = useMemo(() => {
    let result = [...pool]

    // 预设筛选
    switch (preset) {
      case 'csi1000': {
        const predictionMap = new Map(result.map((item) => [item.symbol, item]))
        result = signals.map((signal) => predictionMap.get(signal.symbol) ?? ({
          symbol: signal.symbol,
          name: signal.name,
          pred_date: signal.date,
          pred_close_t5: signal.pred_close ?? 0,
          pred_return: signal.pred_return,
          last_close: 0,
          confidence: signal.confidence ?? 0,
          error: signal.reason,
        }))
        break
      }
      case 'csi500':
        result = result.filter((p) => CSI500_SAMPLE.has(p.symbol))
        break
      case 'hs300':
        result = result.filter((p) => HS300_SAMPLE.has(p.symbol))
        break
      case 'top20':
        result = result.sort((a, b) => b.confidence - a.confidence).slice(0, 20)
        break
      case 'top50':
        result = result.sort((a, b) => b.confidence - a.confidence).slice(0, 50)
        break
    }

    // 搜索筛选
    if (search.trim()) {
      const kw = search.trim().toLowerCase()
      result = result.filter(
        (p) => p.symbol.toLowerCase().includes(kw) || p.name?.toLowerCase().includes(kw),
      )
    }

    return result
  }, [pool, preset, search, signals])

  // ── CSV 导出 ────────────────────────────────────────
  const handleExport = useCallback(() => {
    if (filtered.length === 0) {
      Toast.warning('没有可导出的数据')
      return
    }
    try {
      const headers = ['代码', '名称', '最新价(市价)', '预测收益率', '采样看涨度', '预测生成日期']
      const rows = filtered.map((p) => [
        p.symbol,
        p.name || '',
        (p.last_close_raw ?? p.last_close).toFixed(2),
        (p.pred_return * 100).toFixed(2) + '%',
        (p.confidence * 100).toFixed(0) + '%',
        p.as_of_date ?? '',
      ])
      const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
      const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `选股池_${new Date().toISOString().slice(0, 10)}.csv`
      a.click()
      URL.revokeObjectURL(url)
      Toast.success(`已导出 ${filtered.length} 条记录`)
    } catch {
      Toast.error('导出失败')
    }
  }, [filtered])

  // ── 加载中 ──────────────────────────────────────────
  if (loading && pool.length === 0) {
    return <LoadingSkeleton type="table" rows={12} />
  }

  // ── 错误 ────────────────────────────────────────────
  if (error && pool.length === 0) {
    return <ErrorState message={error} onRetry={() => loadData()} />
  }

  return (
    <div>
      {/* ── 实验性提示条（实测准确率说明，保持诚实定位）────── */}
      <Alert
        type="warning"
        showIcon
        icon={<WarningOutlined />}
        style={{ marginBottom: 16 }}
        message="实验性预测 · 仅供研究观察，不构成投资建议"
        description={
          <span>
            该模型最新一期出样本检验方向准确率约 <b>34%</b>（低于随机水平 50%），
            且按预测排序选股无超额收益。数据异常预测（5 日 |收益| &gt; 40%，超出涨跌停合理区间）已自动剔除。
            {dataDate && <>本批预测生成于 <b>{dataDate}</b>。</>}
          </span>
        }
      />
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
        <Title level={4} style={{ margin: 0 }}>选股池</Title>
        <Space wrap>
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索代码或名称..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            allowClear
            style={{ width: 200 }}
          />
          <Select
            value={preset}
            onChange={setPreset}
            style={{ width: 120 }}
            options={PRESET_OPTIONS}
          />
          <Tooltip title="导出 CSV">
            <Button icon={<DownloadOutlined />} onClick={handleExport}>
              导出
            </Button>
          </Tooltip>
          <Button icon={<ReloadOutlined />} onClick={() => loadData(activeStrategyPreset ?? undefined)} loading={poolLoading}>
            刷新
          </Button>
        </Space>
      </div>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* 策略风格预设卡片                                        */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {strategyPresets.length > 0 && (
        <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
          {strategyPresets.map((sp) => (
            <Col xs={24} sm={8} key={sp.name}>
              <Card
                hoverable
                size="small"
                onClick={() => handleStrategyPresetChange(sp.name)}
                style={{
                  border: activeStrategyPreset === sp.name
                    ? '2px solid var(--primary, #1890ff)'
                    : undefined,
                  background: activeStrategyPreset === sp.name
                    ? 'var(--primary-bg, rgba(24,144,255,0.06))'
                    : undefined,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  {PRESET_ICONS[sp.name] ?? <SafetyCertificateOutlined style={{ fontSize: 24 }} />}
                  <div style={{ flex: 1 }}>
                    <Text strong style={{ fontSize: 15 }}>{sp.label}</Text>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                      {sp.description}
                    </div>
                  </div>
                  {activeStrategyPreset === sp.name && (
                    <Tag color="blue">当前</Tag>
                  )}
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* 筛选统计卡片                                            */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {selectionStats && (
        <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="初筛候选" value={selectionStats.initial_count} />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="数据异常剔除" value={selectionStats.anomaly_count ?? 0}
                valueStyle={{ color: 'var(--danger, #ff4d4f)' }} />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="概率/评分通过" value={selectionStats.after_prob_score}
                valueStyle={{ color: 'var(--success, #52c41a)' }} />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="ST/停牌剔除" value={selectionStats.after_prob_score - selectionStats.after_st_suspended}
                valueStyle={{ color: 'var(--warning, #faad14)' }} />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="低流动性剔除" value={selectionStats.after_st_suspended - selectionStats.after_liquidity}
                valueStyle={{ color: 'var(--warning, #faad14)' }} />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="行业集中度剔除" value={selectionStats.after_liquidity - selectionStats.after_industry}
                valueStyle={{ color: 'var(--warning, #faad14)' }} />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic title="最终入选" value={selectionStats.final_count}
                valueStyle={{ color: 'var(--primary, #1890ff)', fontWeight: 700 }} />
            </Card>
          </Col>
        </Row>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* 高级参数面板（折叠式）                                  */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <Collapse
        style={{ marginBottom: 16 }}
        activeKey={advancedOpen ? ['adv'] : []}
        onChange={(keys) => setAdvancedOpen(keys.includes('adv'))}
        items={[{
          key: 'adv',
          label: <Space><SettingOutlined /> 高级筛选参数</Space>,
          children: (
            <Row gutter={[24, 16]}>
              <Col xs={24} sm={8}>
                <div style={{ marginBottom: 4 }}>
                  <Space>
                    <Text type="secondary" style={{ fontSize: 13 }}>上涨概率阈值</Text>
                    <Tooltip title="预测上涨概率 ≥ 此值才入选。推荐: 保守0.75 / 均衡0.65 / 激进0.55">
                      <InfoCircleOutlined style={{ color: 'var(--text-muted)', fontSize: 12 }} />
                    </Tooltip>
                  </Space>
                </div>
                <Slider min={0.3} max={0.95} step={0.05} value={probUpThreshold}
                  marks={{ 0.3: '0.3', 0.55: '0.55', 0.65: '0.65', 0.75: '0.75', 0.95: '0.95' }}
                  onChange={setProbUpThreshold} />
                <InputNumber size="small" value={probUpThreshold} min={0} max={1} step={0.05}
                  onChange={(v) => setProbUpThreshold(v ?? 0.65)} style={{ width: 80 }} />
              </Col>
              <Col xs={24} sm={8}>
                <div style={{ marginBottom: 4 }}>
                  <Space>
                    <Text type="secondary" style={{ fontSize: 13 }}>综合评分阈值</Text>
                    <Tooltip title="Kronos综合评分 ≥ 此值才入选。推荐: 保守7.0 / 均衡6.0 / 激进4.5">
                      <InfoCircleOutlined style={{ color: 'var(--text-muted)', fontSize: 12 }} />
                    </Tooltip>
                  </Space>
                </div>
                <Slider min={0} max={10} step={0.5} value={scoreThreshold}
                  marks={{ 0: '0', 4.5: '4.5', 6: '6', 7: '7', 10: '10' }}
                  onChange={setScoreThreshold} />
                <InputNumber size="small" value={scoreThreshold} min={0} max={10} step={0.5}
                  onChange={(v) => setScoreThreshold(v ?? 6.0)} style={{ width: 80 }} />
              </Col>
              <Col xs={24} sm={8}>
                <div style={{ marginBottom: 4 }}>
                  <Text type="secondary" style={{ fontSize: 13 }}>最小日均成交额 (万元)</Text>
                </div>
                <InputNumber value={minDailyAmount / 10000} min={100} max={50000} step={100}
                  onChange={(v) => setMinDailyAmount((v ?? 500) * 10000)} style={{ width: 120 }} />
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 13 }}>行业最多入选</Text>
                  <InputNumber value={maxPerIndustry} min={1} max={10} step={1}
                    onChange={(v) => setMaxPerIndustry(v ?? 3)} style={{ width: 60, marginLeft: 8 }} />
                </div>
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 13 }}>最大入选数</Text>
                  <InputNumber value={topK} min={10} max={200} step={10}
                    onChange={(v) => setTopK(v ?? 50)} style={{ width: 60, marginLeft: 8 }} />
                </div>
              </Col>
              <Col xs={24} style={{ textAlign: 'right' }}>
                <Space>
                  <Button icon={<PlayCircleOutlined />} type="primary" onClick={handleRegenerate} loading={poolLoading}>
                    用当前参数重新选股
                  </Button>
                  <Button onClick={() => navigate('/user/backtest')}>
                    运行回测验证
                  </Button>
                </Space>
              </Col>
            </Row>
          ),
        }]}
      />

      {/* 概要 */}
      <div style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-muted)' }}>
        <Space>
          <FilterOutlined />
          <span>
            共 <Text strong>{pool.length}</Text> 条预测，当前显示{' '}
            <Text strong>{filtered.length}</Text> 条
          </span>
          {preset !== 'all' && (
            <Text type="secondary">
              （方案：{PRESET_OPTIONS.find((o) => o.value === preset)?.label}）
            </Text>
          )}
          {activeStrategyPreset && (
            <Tag color="blue">
              策略预设：{strategyPresets.find((p) => p.name === activeStrategyPreset)?.label ?? activeStrategyPreset}
            </Tag>
          )}
        </Space>
      </div>

      <Card>
        {filtered.length === 0 ? (
          <EmptyState message={pool.length === 0 ? '暂无选股数据，请等待模型预测完成' : '当前筛选条件下无匹配结果'} />
        ) : (
          <StockTable
            dataSource={toStockTableRows(filtered, signals)}
            dataDate={dataDate}
            loading={poolLoading}
            onRowClick={(record) => navigate(`/user/stock/${record.code}`)}
          />
        )}
      </Card>
    </div>
  )
}
