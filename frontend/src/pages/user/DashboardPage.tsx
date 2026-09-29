/**
 * 仪表盘页面
 * 对应旧版 user_portal.html #page-dashboard
 *
 * 功能：市场概览卡片 + 最新信号表格
 * 数据来源：GET /api/v1/index/list（指数行情）
 *           GET /api/v1/signal/latest（最新信号）
 *           GET /api/v1/prediction/latest（选股池预测）
 */

import { useEffect, useState, useCallback } from 'react'
import { Row, Col, Card, Typography, Spin, Space, Button, Tag, Divider } from 'antd'
import {
  RiseOutlined,
  FallOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  StockOutlined,
  SelectOutlined,
  AlertOutlined,
  CloudServerOutlined,
  ClockCircleOutlined,
  FundOutlined,
} from '@ant-design/icons'
import { getIndexList } from '@/api/index'
import { getLatestSignals } from '@/api/signal'
import { getLatestPrediction, getLatestSelection } from '@/api/prediction'
import { getSystemStatus } from '@/api/admin'
import { getStrategyOpinion } from '@/api/strategy'
import type { IndexInfo, SignalResult, SystemStatus, StrategyOpinion } from '@/types/api'
import { useStockStore } from '@/stores/useStockStore'
import StatCard from '@/components/common/StatCard'
import StockTable, { toStockTableRows } from '@/components/stock/StockTable'
import LoadingSkeleton from '@/components/common/LoadingSkeleton'
import ErrorState from '@/components/common/ErrorState'
import EmptyState from '@/components/common/EmptyState'
import { useNavigate } from 'react-router-dom'

const { Title, Text } = Typography

/** 数据日期角标：显示数据日期，超过一周（自然日）标为滞后 */
function dataDateCaption(updateDate?: string): React.ReactNode {
  if (!updateDate) return null
  const parsed = new Date(updateDate)
  if (Number.isNaN(parsed.getTime())) {
    return <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>数据日期 {updateDate}</span>
  }
  const diffDays = Math.floor((Date.now() - parsed.getTime()) / 86_400_000)
  const stale = diffDays > 5
  return (
    <span style={{ fontSize: 12, color: stale ? 'var(--warning)' : 'var(--text-muted)' }}>
      数据日期 {updateDate}
      {stale && ` · 滞后 ${diffDays} 天`}
    </span>
  )
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { pool, setPool, poolLoading, setPoolLoading } = useStockStore()

  const [indices, setIndices] = useState<IndexInfo[]>([])
  const [signals, setSignals] = useState<SignalResult[]>([])
  const [poolCount, setPoolCount] = useState<number>(0)
  const [monitor, setMonitor] = useState<SystemStatus | null>(null)
  const [opinion, setOpinion] = useState<StrategyOpinion | null>(null)
  const [dataDate, setDataDate] = useState<string | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // ── 加载数据 ────────────────────────────────────────
  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    setPoolLoading(true)

    try {
      const [indexData, signalData, predData, selectionData, monitorData, opinionData] = await Promise.all([
        getIndexList(),
        getLatestSignals(),
        getLatestPrediction(),
        getLatestSelection(),
        getSystemStatus(),
        getStrategyOpinion(),
      ])

      // apiClient 失败时返回 null（不抛异常），需显式判断
      if (!indexData && !signalData && !predData) {
        setError('数据加载失败，请确认后端服务已启动')
        return
      }

      setIndices(indexData?.indices ?? [])
      setSignals(signalData?.signals ?? [])
      setPool(predData?.predictions ?? [])
      setPoolCount(selectionData?.total ?? predData?.predictions?.length ?? 0)
      setMonitor(monitorData)
      setOpinion(opinionData)
      if (predData?.data_date) setDataDate(predData.data_date)
    } catch {
      setError('数据加载失败')
    } finally {
      setLoading(false)
      setPoolLoading(false)
    }
  }, [setPool, setPoolLoading])

  useEffect(() => { loadData() }, [loadData])

  // ── 加载中 ──────────────────────────────────────────
  if (loading) {
    return (
      <div>
        <Title level={4} style={{ marginBottom: 16 }}>市场概览</Title>
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          {[1, 2, 3, 4].map((i) => (
            <Col xs={12} sm={6} key={i}>
              <Card><Spin /></Card>
            </Col>
          ))}
        </Row>
        <Title level={4} style={{ marginBottom: 16 }}>最新信号</Title>
        <LoadingSkeleton type="table" rows={6} />
      </div>
    )
  }

  // ── 错误 ────────────────────────────────────────────
  if (error && indices.length === 0) {
    return <ErrorState message={error} onRetry={loadData} />
  }

  // ── 找出主要指数（与 config/system.yaml 配置的标的一致）──
  const hs300 = indices.find((i) => i.code === '000300')  // 沪深300
  const zz1000 = indices.find((i) => i.code === '000852') // 中证1000
  const cyb = indices.find((i) => i.code === '399006')    // 创业板指

  const indexCards = [
    {
      title: '沪深300',
      value: hs300?.latest_close ?? 0,
      change: hs300?.change_pct ?? 0,
      updateDate: hs300?.update_date,
      icon: <StockOutlined />,
    },
    {
      title: '中证1000',
      value: zz1000?.latest_close ?? 0,
      change: zz1000?.change_pct ?? 0,
      updateDate: zz1000?.update_date,
      icon: <RiseOutlined />,
    },
    {
      title: '创业板指',
      value: cyb?.latest_close ?? 0,
      change: cyb?.change_pct ?? 0,
      updateDate: cyb?.update_date,
      icon: <FallOutlined />,
    },
    {
      title: '最新信号',
      value: signals.length,
      suffix: ' 条',
      updateDate: undefined,
      icon: <ThunderboltOutlined />,
    },
  ]
  const predictionMap = new Map(pool.map((item) => [item.symbol, item]))
  const componentPool = signals.map((signal) => predictionMap.get(signal.symbol) ?? ({
    symbol: signal.symbol,
    name: signal.name,
    pred_date: signal.date,
    pred_close_t5: signal.pred_close ?? 0,
    pred_return: signal.pred_return,
    last_close: 0,
    confidence: signal.confidence ?? 0,
    error: signal.reason,
  }))
  const signalCounts = signals.reduce<Record<string, number>>((result, item) => {
    result[item.signal_type] = (result[item.signal_type] ?? 0) + 1
    return result
  }, {})

  return (
    <div>
      {/* ── 状态卡片 ───────────────────────────────── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <StatCard
            title="股票池数量"
            value={poolCount}
            suffix=" 只"
            precision={0}
          />
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ background: 'var(--surface-1)', borderRadius: 'var(--radius)' }}>
            <Typography.Text type="secondary" style={{ fontSize: 13, marginBottom: 8, display: 'block' }}>
              今日信号分布
            </Typography.Text>
            <Space size={4}>
              <Tag color="red">买 {signalCounts['BUY'] ?? 0}</Tag>
              <Tag color="orange">持 {signalCounts['HOLD'] ?? 0}</Tag>
              <Tag color="green">卖 {signalCounts['SELL'] ?? 0}</Tag>
            </Space>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ background: 'var(--surface-1)', borderRadius: 'var(--radius)' }}>
            <Typography.Text type="secondary" style={{ fontSize: 13, marginBottom: 8, display: 'block' }}>
              模型状态
            </Typography.Text>
            <Space>
              <CloudServerOutlined style={{ color: monitor?.model_status === 'running' ? 'var(--success)' : 'var(--warning)' }} />
              <Typography.Text strong style={{ fontSize: 16 }}>
                {monitor?.model_status === 'running' ? '运行中' : monitor?.model_status === 'error' ? '异常' : monitor?.model_status ?? '--'}
              </Typography.Text>
            </Space>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ background: 'var(--surface-1)', borderRadius: 'var(--radius)' }}>
            <Typography.Text type="secondary" style={{ fontSize: 13, marginBottom: 8, display: 'block' }}>
              数据更新时间
            </Typography.Text>
            <Space>
              <ClockCircleOutlined style={{ color: 'var(--text-muted)' }} />
              <Typography.Text style={{ fontSize: 14 }}>
                {monitor?.last_data_update ?? monitor?.last_prediction_time ?? '--'}
              </Typography.Text>
            </Space>
            {monitor?.prediction_coverage && (
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>
                预测覆盖 {monitor.prediction_coverage.predicted}/{monitor.prediction_coverage.universe} 只
                {monitor.prediction_coverage.coverage_pct != null && ` (${monitor.prediction_coverage.coverage_pct}%)`}
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* ── 策略观点：为什么是当前仓位、什么条件会变 ────── */}
      {opinion && opinion.targets.length > 0 && (
        <Card
          style={{ marginBottom: 24 }}
          title={<Space><FundOutlined />策略观点 — {opinion.strategy_name}</Space>}
          extra={
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              生成于 {opinion.generated_at?.replace('T', ' ').slice(0, 16) ?? '--'}
            </span>
          }
        >
          {opinion.targets.map((t) => {
            const stale = Boolean(t.date && opinion.latest_data_date && t.date < opinion.latest_data_date)
            const tagColor =
              t.signal_type === 'BUY' ? 'red'
              : t.signal_type === 'SELL' ? 'green'
              : t.signal_type === 'HOLD' ? 'orange'
              : 'default'
            const tagLabel =
              t.signal_type === 'BUY' ? '买入'
              : t.signal_type === 'SELL' ? '卖出'
              : t.signal_type === 'HOLD' ? '持有'
              : '空仓'
            return (
              <div key={t.symbol} style={{ marginBottom: 12 }}>
                <Space wrap>
                  <Text strong>{t.name ?? t.symbol}</Text>
                  <Text code>{t.symbol}</Text>
                  <Tag color={tagColor}>{tagLabel}</Tag>
                  {t.pred_return != null && (
                    <span>模型预测未来5日收益 <Text strong>{`${(t.pred_return * 100).toFixed(2)}%`}</Text></span>
                  )}
                  <span>建议仓位 {`${(t.position_pct * 100).toFixed(0)}%`}</span>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>信号日期 {t.date ?? '--'}</span>
                  {stale && <Tag color="red">已过期</Tag>}
                </Space>
                {t.reason && (
                  <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
                    {t.reason}
                  </div>
                )}
              </div>
            )
          })}
          <Divider style={{ margin: '8px 0 12px' }} />
          <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.8 }}>
            <Text strong style={{ color: 'var(--text, rgba(0,0,0,0.85))' }}>决策规则：</Text>
            {opinion.rule}
          </div>
          <div style={{ fontSize: 13, marginTop: 6 }}>
            <Text strong>触发买入条件：</Text>
            模型预测未来 5 日收益率 &gt; {`${(opinion.buy_threshold * 100).toFixed(1)}%`}；
            <Text type="secondary" style={{ marginLeft: 8 }}>
              风控：{opinion.risk_rules.join('；')}
            </Text>
          </div>
        </Card>
      )}

      {/* ── 快捷操作 ───────────────────────────────── */}
      <Card size="small" style={{ marginBottom: 24 }}>
        <Space size={12}>
          <Typography.Text type="secondary" style={{ fontSize: 13 }}>快捷操作</Typography.Text>
          <Button
            type="primary"
            icon={<SelectOutlined />}
            onClick={() => navigate('/user/pool')}
          >
            运行选股
          </Button>
          <Button
            icon={<FundOutlined />}
            onClick={() => navigate('/user/backtest')}
          >
            运行回测
          </Button>
          <Button
            icon={<AlertOutlined />}
            onClick={() => navigate('/user/market')}
          >
            查看信号
          </Button>
        </Space>
      </Card>

      {/* ── 市场概览 ───────────────────────────────── */}
      <Title level={4} style={{ marginBottom: 16 }}>市场概览</Title>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {indexCards.map((item) => (
          <Col xs={12} sm={6} key={item.title}>
            <StatCard
              title={item.title}
              value={item.value}
              change={item.change}
              suffix={item.suffix}
              precision={item.suffix ? 0 : 2}
              extra={dataDateCaption(item.updateDate)}
            />
          </Col>
        ))}
      </Row>

      {/* ── 最新信号 ───────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          {signals[0]?.name ? `${signals[0].name} 实时择时` : '指数择时信号'}
        </Title>
        <Space>
          <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            更新于 {signals[0]?.date || '--'}
          </span>
        </Space>
      </div>

      <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
        {[
          { title: '买入', key: 'BUY' },
          { title: '持有', key: 'HOLD' },
          { title: '卖出', key: 'SELL' },
          { title: '空仓', key: 'CASH' },
        ].map((item) => (
          <Col xs={12} sm={6} key={item.key}>
            <StatCard title={item.title} value={signalCounts[item.key] ?? 0} suffix=" 只" precision={0} />
          </Col>
        ))}
      </Row>

      <Card>
        {componentPool.length === 0 ? (
          <EmptyState message="暂无指数择时信号，请在后台生成交易信号" icon={<BarChartOutlined />} />
        ) : (
          <StockTable
            dataSource={toStockTableRows(componentPool, signals)}
            dataDate={dataDate}
            loading={poolLoading}
            onRowClick={(record) => navigate(`/user/stock/${record.code}`)}
          />
        )}
      </Card>
    </div>
  )
}
