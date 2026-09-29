/**
 * QuickPreview — 右侧滑出预览面板
 *
 * 替换旧版右侧滑出面板，封装 Ant Design Drawer。
 * 内嵌关键指标卡片 + 迷你 90 天趋势图 + 预测方向摘要。
 *
 * 用法:
 *   <QuickPreview visible={show} onClose={() => setShow(false)} stockCode="000852" stockName="中证1000" />
 */

import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Drawer,
  Card,
  Row,
  Col,
  Statistic,
  Tag,
  Button,
  Space,
  Typography,
  Spin,
  Divider,
} from 'antd'
import {
  RightOutlined,
  RiseOutlined,
  FallOutlined,
  MinusOutlined,
} from '@ant-design/icons'
import { getIndexDetail } from '@/api/index'
import { getLatestPrediction } from '@/api/prediction'
import type { KlineBar, PredictionResult } from '@/types/api'
import { useChart, buildPredictedKlineOption } from '@/hooks/useChart'
import type { PredictedKlineDatum } from '@/hooks/useChart'
import PctText from '@/components/common/PctText'

const { Text, Title } = Typography

export interface QuickPreviewProps {
  visible: boolean
  onClose: () => void
  stockCode: string
  stockName?: string
}

export default function QuickPreview({ visible, onClose, stockCode, stockName }: QuickPreviewProps) {
  const navigate = useNavigate()

  // ── 数据状态 ──────────────────────────────────────
  const [kline, setKline] = useState<KlineBar[]>([])
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [marketClose, setMarketClose] = useState<number | null>(null)
  const [marketDate, setMarketDate] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // ── 迷你趋势图 ────────────────────────────────────
  const { chartRef, setOption, instanceKey } = useChart()

  // ── 加载数据 ──────────────────────────────────────
  const loadData = useCallback(async (code: string) => {
    setLoading(true)
    try {
      const [detail, predResp] = await Promise.all([
        getIndexDetail(code, 90, 'raw'),
        getLatestPrediction(code),
      ])
      setKline(detail?.kline ?? [])
      setPrediction(predResp?.predictions?.[0] ?? null)
      setMarketClose(detail?.market_close ?? null)
      setMarketDate(detail?.market_date ?? null)
    } catch {
      setKline([])
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (visible && stockCode) {
      loadData(stockCode)
    }
  }, [visible, stockCode, loadData])

  // ── 迷你趋势图 option ─────────────────────────────
  useEffect(() => {
    if (kline.length === 0) return
    const trendData: PredictedKlineDatum[] = kline.map((b) => ({
      date: b.date,
      close: b.close,
    }))
    setOption(buildPredictedKlineOption(trendData))
  }, [kline, setOption, instanceKey])

  // ── 预测方向摘要 ──────────────────────────────────
  const predDirection = prediction
    ? prediction.pred_return > 0.005
      ? '看涨'
      : prediction.pred_return < -0.005
        ? '看跌'
        : '震荡'
    : '--'

  const directionColor = prediction
    ? prediction.pred_return > 0.005
      ? 'red'
      : prediction.pred_return < -0.005
        ? 'green'
        : 'orange'
    : undefined

  const lastClose = kline.length > 0 ? kline[kline.length - 1].close : null
  const lastDate = kline.length > 0 ? kline[kline.length - 1].date : null
  const compositeScore = prediction
    ? Math.round(prediction.confidence * 70 + 15)
    : null

  return (
    <Drawer
      title={
        <span>
          {stockCode}{' '}
          <Text type="secondary" style={{ fontSize: 14 }}>
            {stockName || ''}
          </Text>
        </span>
      }
      placement="right"
      width={420}
      open={visible}
      onClose={onClose}
      styles={{
        body: { padding: '16px 20px' },
      }}
      extra={
        <Button
          type="link"
          icon={<RightOutlined />}
          onClick={() => navigate(`/user/stock/${stockCode}`)}
          style={{ padding: 0, fontSize: 13 }}
        >
          查看完整详情
        </Button>
      }
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}>
          <Spin size="large" tip="加载中…" />
        </div>
      ) : (
        <>
          {/* ── 关键指标卡片 ──────────────────────── */}
          <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
            <Col span={12}>
              <Card size="small" style={{ background: 'var(--surface-1)' }}>
                <Statistic
                  title="最新价(市价)"
                  value={lastClose ?? '--'}
                  precision={2}
                  valueStyle={{ fontSize: 20, fontWeight: 600 }}
                />
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  不复权市价口径
                  {lastDate ? ` · 截至 ${lastDate}` : ''}
                  {marketDate && marketDate !== lastDate ? ` · 快照 ${marketDate}` : ''}
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" style={{ background: 'var(--surface-1)' }}>
                <Statistic
                  title="预测收益"
                  value={prediction ? (prediction.pred_return * 100) : '--'}
                  precision={2}
                  suffix={prediction ? '%' : ''}
                  valueStyle={{
                    fontSize: 20,
                    fontWeight: 600,
                    color: prediction
                      ? prediction.pred_return > 0 ? 'var(--danger)' : prediction.pred_return < 0 ? 'var(--success)' : 'var(--text)'
                      : undefined,
                  }}
                  prefix={
                    prediction
                      ? prediction.pred_return > 0.005
                        ? <RiseOutlined />
                        : prediction.pred_return < -0.005
                          ? <FallOutlined />
                          : <MinusOutlined />
                      : null
                  }
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" style={{ background: 'var(--surface-1)' }}>
                <Statistic
                  title="置信度"
                  value={prediction ? (prediction.confidence * 100) : '--'}
                  precision={0}
                  suffix={prediction ? '%' : ''}
                  valueStyle={{ fontSize: 20, fontWeight: 600 }}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" style={{ background: 'var(--surface-1)' }}>
                <Statistic
                  title="综合评分"
                  value={compositeScore ?? '--'}
                  valueStyle={{
                    fontSize: 20,
                    fontWeight: 600,
                    color: compositeScore != null
                      ? compositeScore >= 60 ? 'var(--success)' : compositeScore >= 40 ? 'var(--warning)' : 'var(--danger)'
                      : undefined,
                  }}
                />
              </Card>
            </Col>
          </Row>

          {/* ── 迷你 90 天趋势图 ──────────────────── */}
          <Title level={5} style={{ marginTop: 8, marginBottom: 8 }}>90 日趋势</Title>
          {kline.length > 0 ? (
            <div ref={chartRef} className="chart-container" style={{ height: 180, marginBottom: 16 }} />
          ) : (
            <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
              暂无趋势数据
            </div>
          )}

          <Divider style={{ margin: '8px 0 12px' }} />

          {/* ── 预测方向摘要 ──────────────────────── */}
          <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>预测方向</Title>
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Tag color={directionColor} style={{ fontSize: 14, padding: '2px 12px' }}>
                {predDirection}
              </Tag>
              {prediction && (
                <Text type="secondary" style={{ fontSize: 13 }}>
                  预测日期 {prediction.pred_date}
                </Text>
              )}
            </div>
            {prediction && (
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                预测收盘价(市价)：
                <Text strong>
                  ¥{(
                    prediction.pred_close_t5_raw ??
                    ((lastClose ?? 0) * (1 + (prediction.pred_return ?? 0)))
                  ).toFixed(2)}
                </Text>
                ，收益率：<PctText value={prediction.pred_return} bold />
              </div>
            )}
          </Space>

          <Divider style={{ margin: '16px 0 12px' }} />

          {/* ── 查看完整详情按钮 ──────────────────── */}
          <Button
            type="primary"
            block
            icon={<RightOutlined />}
            onClick={() => navigate(`/user/stock/${stockCode}`)}
          >
            查看完整详情
          </Button>
        </>
      )}
    </Drawer>
  )
}
