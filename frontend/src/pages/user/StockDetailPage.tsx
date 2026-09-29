/**
 * 个股看板页面
 * 对应旧版 user_portal.html #page-stock-detail
 *
 * 功能：股票搜索 → K线 + 成交量 → 预测K线（含置信区间）→ 交易信号 → 综合评分
 * 数据来源：GET /api/v1/index/{code}?days=120   (K线)
 *           GET /api/v1/prediction/latest?symbol=... (预测)
 *           GET /api/v1/signal/{code}?days=30        (信号)
 */

import { useEffect, useState, useCallback, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Card,
  Typography,
  Row,
  Col,
  Space,
  Tag,
  Tooltip,
  Descriptions,
  Divider,
  Alert,
  Button,
} from 'antd'
import {
  LineChartOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  TrophyOutlined,
  WarningOutlined,
  SyncOutlined,
  StarOutlined,
  StarFilled,
  HistoryOutlined,
} from '@ant-design/icons'
import { getIndexDetail } from '@/api/index'
import { getLatestPrediction, getPredictionHistory } from '@/api/prediction'
import { getSignalHistory } from '@/api/signal'
import { getQuotes } from '@/api/quote'
import type {
  KlineBar,
  PredictionResult,
  QuoteInfo,
  SignalResult,
} from '@/types/api'
import { useStockStore } from '@/stores/useStockStore'
import { useWatchlistStore } from '@/stores/useWatchlistStore'
import SearchBox from '@/components/common/SearchBox'
import PctText, { pctColor } from '@/components/common/PctText'
import KlineChart, { VolumeChart } from '@/components/charts/KlineChart'
import PredictedKlineChart from '@/components/charts/PredictedKlineChart'
import LoadingSkeleton from '@/components/common/LoadingSkeleton'
import ErrorState from '@/components/common/ErrorState'
import EmptyState from '@/components/common/EmptyState'
import type { Stock } from '@/types/stock'

const { Title, Text } = Typography

/** 与 config/system.yaml strategy.buy_threshold 一致（预测收益 > 0.5% → 买入） */
const BUY_THRESHOLD = 0.005

/** 实时报价轮询间隔 (ms) */
const QUOTE_POLL_MS = 15_000

/** 预测自动刷新间隔 (ms)：预测批次每日收盘后由服务端生成，5 分钟轻量检查即可发现新批次 */
const PREDICTION_POLL_MS = 5 * 60_000

/** 评分维度颜色 */
function scoreColor(v: number): string {
  return v >= 60 ? 'var(--success)' : v >= 40 ? 'var(--warning)' : 'var(--danger)'
}

export default function StockDetailPage() {
  const { code: routeCode } = useParams<{ code?: string }>()
  const navigate = useNavigate()
  const { currentStock, setCurrentStock, setStockLoading, searchLoading } =
    useStockStore()

  const [klineBars, setKlineBars] = useState<KlineBar[]>([])
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [predictionHistory, setPredictionHistory] = useState<PredictionResult[]>([])
  const [signals, setSignals] = useState<SignalResult[]>([])
  const [stockType, setStockType] = useState<'stock' | 'etf' | 'index' | 'other'>('other')
  const [marketClose, setMarketClose] = useState<number | null>(null)
  const [marketDate, setMarketDate] = useState<string | null>(null)
  const [realQuote, setRealQuote] = useState<QuoteInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── 自选股 ────────────────────────────────────────
  const watchCodes = useWatchlistStore((s) => s.codes)
  const toggleWatch = useWatchlistStore((s) => s.toggle)
  const isWatched = Boolean(routeCode && watchCodes.includes(routeCode))

  // ── 实时价自动刷新（新浪报价，15 秒轮询）───────────
  useEffect(() => {
    if (!routeCode) return
    let alive = true
    const fetchQuote = async () => {
      const resp = await getQuotes([routeCode])
      if (!alive) return
      const q = resp?.quotes?.[0] ?? null
      setRealQuote(q && q.source === 'sina_realtime' && q.price != null && !q.suspended ? q : null)
    }
    fetchQuote()
    const timer = window.setInterval(fetchQuote, QUOTE_POLL_MS)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [routeCode])

  // ── 预测自动刷新（5 分钟轻量轮询）──────────────────
  // 预测批次由服务端每日收盘后生成：轮询发现新批次(as_of 变化)即自动更新界面，
  // 并自动摘掉“已过期”标记；预测内容未变时不触发重渲染。
  useEffect(() => {
    if (!routeCode) return
    let alive = true
    const refreshPrediction = async () => {
      try {
        const [pred, history] = await Promise.all([
          getLatestPrediction(routeCode),
          getPredictionHistory(routeCode),
        ])
        if (!alive) return
        const latestPred = pred?.predictions?.[0] ?? null
        setPredictionHistory(history?.records ?? [])
        setPrediction((prev) => {
          if (
            latestPred && prev &&
            prev.as_of_date === latestPred.as_of_date &&
            prev.pred_date === latestPred.pred_date &&
            prev.pred_return === latestPred.pred_return
          ) {
            return prev // 批次未变：保持原引用，不触发重渲染
          }
          return latestPred ?? prev
        })
      } catch {
        // 网络抖动忽略：下次轮询自动重试，不影响当前显示
      }
    }
    refreshPrediction()
    const timer = window.setInterval(refreshPrediction, PREDICTION_POLL_MS)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [routeCode])

  // ── 评分与最新预测同步：预测批次变化后按同一公式重算评分卡 ──
  const derivedScore = useMemo(() => {
    if (!prediction) return null
    return {
      trend: prediction.confidence ? prediction.confidence * 80 : 50,
      momentum: prediction.pred_return ? Math.min(100, Math.max(0, 50 + prediction.pred_return * 200)) : 50,
      volume: 50, // 量能暂为占位值，与 loadStock 一致
      composite: prediction.confidence ? prediction.confidence * 70 + 15 : 50,
    }
  }, [prediction])

  useEffect(() => {
    if (!routeCode || !derivedScore || !currentStock || currentStock.stock_code !== routeCode) return
    const s = currentStock.score
    if (
      s &&
      s.trend === derivedScore.trend &&
      s.momentum === derivedScore.momentum &&
      s.volume === derivedScore.volume &&
      s.composite === derivedScore.composite
    ) {
      return // 评分未变，避免无谓写入
    }
    setCurrentStock({ ...currentStock, score: derivedScore })
  }, [derivedScore, routeCode, currentStock, setCurrentStock])

  // ── 加载个股数据 ────────────────────────────────────
  // 价格一律不复权市价口径（用户默认；后复权仅存在于模型内部，不向用户展示）
  const loadStock = useCallback(async (code: string) => {
    setLoading(true)
    setError(null)
    setStockLoading(true)

    try {
      const [detail, pred, sigHistory, history] = await Promise.all([
        getIndexDetail(code, 1250, 'raw'),
        getLatestPrediction(code),
        getSignalHistory(code, 30),
        getPredictionHistory(code),
      ])

      if (detail) {
        setKlineBars(detail.kline ?? [])
        setStockType(detail.type)
        setMarketClose(detail.market_close ?? null)
        setMarketDate(detail.market_date ?? null)
      } else {
        setKlineBars([])
      }

      const latestPred = pred?.predictions?.[0] ?? null
      setPrediction(latestPred)
      setPredictionHistory(history?.records ?? [])

      if (sigHistory) {
        setSignals(sigHistory.signals ?? [])
      } else {
      setSignals([])
      }

      // 构建 StockDetail 用于 QuickPreview（用本次请求的 latestPred，不能用组件状态 prediction — 那是上一只股票的旧值）
      setCurrentStock({
        stock_code: code,
        stock_name: detail?.name || code,
        kline: detail?.kline ?? [],
        signals: sigHistory?.signals ?? [],
        score: {
          trend: latestPred?.confidence ? latestPred.confidence * 80 : 50,
          momentum: latestPred?.pred_return ? Math.min(100, Math.max(0, 50 + latestPred.pred_return * 200)) : 50,
          volume: 50, // volume score computed from data
          composite: latestPred?.confidence ? latestPred.confidence * 70 + 15 : 50,
        },
      })
    } catch {
      setError('个股数据加载失败')
    } finally {
      setLoading(false)
      setStockLoading(false)
    }
  }, [setCurrentStock, setStockLoading])

  // ── 路由参数变化时自动加载 ─────────────────────────
  useEffect(() => {
    if (routeCode) {
      loadStock(routeCode)
    }
  }, [routeCode, loadStock])

  // ── 搜索回调 ────────────────────────────────────────
  const handleSelect = useCallback(
    (stock: Stock) => {
      navigate(`/user/stock/${stock.code}`)
    },
    [navigate],
  )

  // 将 KlineBar[] 转换为预测数据格式
  const toPredictedData = (bars: KlineBar[]) =>
    bars.map((b) => ({
      date: b.date,
      close: b.close,
    }))

  /** YYYY-MM-DD 纯日期运算（UTC，避免时区偏移） */
  function addDays(dateStr: string, days: number): string {
    const d = new Date(`${dateStr}T00:00:00Z`)
    d.setUTCDate(d.getUTCDate() + days)
    return d.toISOString().slice(0, 10)
  }

  /**
   * 构造"今日 → 模型 5 日目标"序列：
   * 模型只输出第 5 个交易日一个目标（pred_date），中间 4 个点仅为线性示意插值，
   * 区间宽度取采样标准差 prediction_std（≈±1σ，随天数线性放大）。
   * 不再伪造 20 天轨迹。
   */
  const forecastSeries = useMemo(() => {
    if (!prediction || klineBars.length === 0) return null
    const lastBar = klineBars[klineBars.length - 1]
    const lastClose = lastBar?.close ?? 0
    if (!lastClose || Number.isNaN(lastClose)) return null
    // 目标价（市价口径）
    const target =
      prediction.pred_close_t5_raw ??
      (marketClose != null ? marketClose * (1 + (prediction.pred_return ?? 0)) : lastClose * (1 + (prediction.pred_return ?? 0)))
    // 收益标准差（std 为后复权价单位，除以模型侧最新价即无量纲，天然不受口径影响）
    const stdFrac =
      prediction.prediction_std && prediction.last_close > 0
        ? Math.max(0.002, Math.abs(prediction.prediction_std / prediction.last_close))
        : 0.03
    const startDate = lastBar.date
    const endDate = prediction.pred_date || addDays(startDate, 5)
    const spanDays = Math.max(5, Math.round((Date.parse(endDate) - Date.parse(startDate)) / 86_400_000))
    const points: Array<{ date: string; close: number; ci_lower: number; ci_upper: number }> = []
    for (let k = 1; k <= 5; k++) {
      const frac = k / 5
      const mid = lastClose + (target - lastClose) * frac
      const halfWidth = Math.abs(mid * stdFrac * frac)
      points.push({
        date: addDays(startDate, Math.round(spanDays * frac)),
        close: +mid.toFixed(2),
        ci_lower: +(mid - halfWidth).toFixed(2),
        ci_upper: +(mid + halfWidth).toFixed(2),
      })
    }
    return {
      targetClose: +target.toFixed(2),
      history: toPredictedData(klineBars.slice(-30)),
      points,
      lastDate: startDate,
      endDate,
      predReturn: prediction.pred_return,
      stdFrac,
    }
  }, [prediction, klineBars, marketClose])

  // ── 无路由代码：显示搜索提示 ────────────────────────
  if (!routeCode) {
    return (
      <div>
        <Title level={4} style={{ marginBottom: 16 }}>个股看板</Title>
        <Card>
          <div style={{ maxWidth: 400, margin: '0 auto', padding: '40px 0', textAlign: 'center' }}>
            <LineChartOutlined style={{ fontSize: 36, color: 'var(--text-muted)', marginBottom: 16 }} />
            <p style={{ marginBottom: 16, color: 'var(--text-muted)' }}>
              请输入股票代码查看 K 线、预测和交易信号
            </p>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <SearchBox
                onSelect={handleSelect}
                loading={searchLoading}
                placeholder="搜索股票代码，如 000852、600519..."
              />
            </div>
          </div>
        </Card>
      </div>
    )
  }

  // ── 加载中（已有数据时保留旧图，避免口径切换整页闪烁）──
  if (loading && klineBars.length === 0) {
    return (
      <div>
        <div style={{ marginBottom: 16 }}>
          <Title level={4} style={{ margin: 0 }}>
            {routeCode} <Text type="secondary">加载中…</Text>
          </Title>
        </div>
        <LoadingSkeleton type="chart" rows={3} />
      </div>
    )
  }

  // ── 错误 ────────────────────────────────────────────
  if (error && klineBars.length === 0) {
    return (
      <div>
        <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
          <Title level={4} style={{ margin: 0 }}>{routeCode}</Title>
          <SearchBox onSelect={handleSelect} placeholder="搜索其他股票..." />
        </div>
        <ErrorState message={error} onRetry={() => loadStock(routeCode)} />
      </div>
    )
  }

  // ── 无数据 ──────────────────────────────────────────
  if (klineBars.length === 0) {
    return (
      <div>
        <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
          <Title level={4} style={{ margin: 0 }}>{routeCode}</Title>
          <SearchBox onSelect={handleSelect} placeholder="搜索其他股票..." />
        </div>
        <Card>
          <EmptyState message={`未找到 ${routeCode} 的 K 线数据`} />
        </Card>
      </div>
    )
  }

  // ── 最新信号 ────────────────────────────────────────
  const latestSignal = signals[0]
  const signalColor =
    latestSignal?.signal_type === 'BUY'
      ? 'red'
      : latestSignal?.signal_type === 'SELL'
        ? 'green'
        : 'orange'

  /**
   * 无实盘信号时，按论文规则由模型预测即时推导参考信号：
   *   预测 5 日收益 > 0.5% → 参考买入（持 5 个交易日）；否则 → 参考空仓。
   * 覆盖任意股票，保证"点开就有信号"。
   * 注意：这里是普通计算而非 hook——组件在加载中会提前 return，hook 数量必须稳定。
   */
  const derivedSignal = (() => {
    if (latestSignal || !prediction || prediction.is_anomalous) return null
    const ret = prediction.pred_return ?? 0
    const buy = ret > BUY_THRESHOLD
    return {
      type: (buy ? 'BUY' : 'CASH') as 'BUY' | 'CASH',
      label: buy ? '参考买入' : '参考空仓',
      ret,
    }
  })()

  // ── 预测过期判断：生成日期落后于最新行情日期 ──────────
  const predictionStale = Boolean(
    prediction?.as_of_date && marketDate && prediction.as_of_date < marketDate,
  )

  // ── 历史档案统计（预测 vs 实际；仅展示逐批对错，不冒充胜率）──
  const archiveVerified = predictionHistory.filter((p) => p.actual_return != null)
  const archiveHitCount = archiveVerified.filter(
    (p) => p.actual_return != null && Math.sign(p.pred_return) === Math.sign(p.actual_return),
  ).length

  return (
    <div>
      {/* ── 顶栏：代码 + 搜索 ──────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
        <Title level={4} style={{ margin: 0 }}>
          {routeCode}{' '}
          <Text type="secondary" style={{ fontSize: 15 }}>
            {currentStock?.stock_name || ''}
          </Text>
          {prediction && (
            <>
              {prediction.is_anomalous ? (
                <Tag color="red" style={{ marginLeft: 8 }}>数据异常 · 仅供参考</Tag>
              ) : (
                <Tag style={{ marginLeft: 8 }}>实验性预测</Tag>
              )}
              {predictionStale && (
                <Tag color="red" style={{ marginLeft: 4 }}>已过期</Tag>
              )}
            </>
          )}
        </Title>
        <Space wrap>
          {routeCode && (
            <Tooltip title={isWatched ? '移出自选' : '加入自选（存本机浏览器）'}>
              <Button
                type={isWatched ? 'primary' : 'default'}
                icon={isWatched ? <StarFilled /> : <StarOutlined />}
                onClick={() => toggleWatch(routeCode)}
              >
                {isWatched ? '已自选' : '自选'}
              </Button>
            </Tooltip>
          )}
          {realQuote && (
            <Space size={8} style={{ background: 'var(--surface-1, rgba(0,0,0,0.03))', borderRadius: 6, padding: '3px 10px' }}>
              <SyncOutlined spin style={{ fontSize: 12, color: 'var(--text-muted)' }} />
              <Text type="secondary" style={{ fontSize: 12 }}>实时</Text>
              <Text strong style={{ fontSize: 16, color: pctColor(realQuote.change_pct ?? 0) }}>
                ¥{realQuote.price!.toFixed(2)}
              </Text>
              <PctText value={realQuote.change_pct} raw bold />
              {realQuote.time && (
                <Text type="secondary" style={{ fontSize: 11 }}>{realQuote.time}</Text>
              )}
              <Tooltip title="来自新浪实时行情，每 15 秒自动刷新（仅交易时段有变化）">
                <span style={{ fontSize: 11, color: '#999', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </Space>
          )}
          <SearchBox onSelect={handleSelect} placeholder="搜索其他股票..." />
        </Space>
      </div>

      {/* ── K线图 ──────────────────────────────────── */}
      <Card
        title={<Space><BarChartOutlined />K线图</Space>}
        style={{ marginBottom: 16 }}
      >
        <KlineChart data={klineBars} height={400} showVolume={false} />
        <VolumeChart data={klineBars} height={120} />
        {stockType === 'stock' && (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.7 }}>
            注：K线为<b>不复权市价</b>（与交易所行情一致）。
            最近行情截至 <Text code>{klineBars[klineBars.length - 1]?.date ?? '--'}</Text>
            {marketClose != null && (
              <>；最新市价：<Text strong style={{ color: 'var(--success)' }}>¥{marketClose.toFixed(2)}</Text>
                {marketDate ? `（${marketDate}）` : ''}
              </>
            )}
            。预测收益率与K线同口径，可直接比较。
          </div>
        )}
      </Card>

      {/* ── 预测K线走势（诚实版：模型 5 日目标点，不伪造 20 天）── */}
      <Card
        title={<Space><LineChartOutlined />预测K线走势</Space>}
        style={{ marginBottom: 16 }}
      >
        {prediction?.is_anomalous ? (
          <Alert
            type="warning"
            showIcon
            icon={<WarningOutlined />}
            message="该预测结果数据异常，已隐藏预测走势图"
            description={
              <span>
                模型给出的 5 日预测收益为 <PctText value={prediction.pred_return} bold />，
                超出个股 5 日涨跌停合理区间（约 ±40%），属明显失真的预测，
                <b>仅供参考，不作为任何交易依据</b>。
              </span>
            }
          />
        ) : prediction && forecastSeries ? (
          <div>
            <PredictedKlineChart
              data={[
                ...forecastSeries.history,
                ...forecastSeries.points,
              ]}
              dividerIndex={forecastSeries.history.length}
              height={320}
            />
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.7 }}>
              <Tag color="blue" style={{ marginRight: 6 }}>模型 5 日目标</Tag>
              虚线为目标日（{forecastSeries.endDate}）的预测收盘 ¥{forecastSeries.targetClose.toFixed(2)}，
              5 日收益 <PctText value={forecastSeries.predReturn} bold />
              （约 ±{Math.round(forecastSeries.stdFrac * 100)}% 采样标准差区间）。
              模型按论文口径仅预测未来 <b>5 个交易日</b>，中间点为线性示意连线，
              <b>非模型逐日预测</b>；更长期限需另行训练。
            </div>
          </div>
        ) : (
          <EmptyState message="暂无预测数据，无法生成预测K线" />
        )}
      </Card>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        {/* ── 预测K线 ────────────────────────────── */}
        <Col xs={24} lg={12}>
          <Card title={<Space><LineChartOutlined />未来预测</Space>} size="small">
            {prediction ? (
              <div>
                {prediction.is_anomalous ? (
                  <Alert
                    type="warning"
                    showIcon
                    message="数据异常预测，仅供参考"
                    style={{ marginBottom: 12 }}
                  />
                ) : (
                  <PredictedKlineChart
                    data={[...toPredictedData(klineBars.slice(-30)), {
                      date: prediction.pred_date,
                      close: prediction.pred_close_t5_raw ?? (marketClose != null ? marketClose * (1 + prediction.pred_return) : prediction.pred_close_t5),
                    }]}
                    height={280}
                  />
                )}
                <Divider style={{ margin: '12px 0' }} />
                <Descriptions column={2} size="small">
                  <Descriptions.Item label="预测目标日">{prediction.pred_date}</Descriptions.Item>
                  <Descriptions.Item label="预测生成日期">
                    <Space size={6}>
                      <span>{prediction.as_of_date ?? '--'}</span>
                      {predictionStale && <Tag color="red">已过期</Tag>}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="预测收盘(市价)">
                    <Text strong>
                      {(prediction.pred_close_t5_raw ?? (marketClose != null ? marketClose * (1 + prediction.pred_return) : null))?.toFixed(2) ?? '--'}
                    </Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="预测收益率">
                    <PctText value={prediction.pred_return} bold />
                  </Descriptions.Item>
                  <Descriptions.Item label="最新市价">
                    <Text strong style={{ color: 'var(--success)' }}>
                      {marketClose != null ? `¥${marketClose.toFixed(2)}` : '--'}
                    </Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="采样看涨度(参考)">
                    <Tooltip title="模型采样统计值，并非历史胜率；实测该模型方向准确率低于随机，仅供参考">
                      <Tag color={prediction.confidence >= 0.8 ? 'green' : prediction.confidence >= 0.5 ? 'orange' : 'red'}>
                        {(prediction.confidence * 100).toFixed(0)}%
                      </Tag>
                    </Tooltip>
                  </Descriptions.Item>
                  <Descriptions.Item label="状态" span={2}>
                    {prediction.is_anomalous ? (
                      <Tag color="red">数据异常 · 仅供参考</Tag>
                    ) : predictionStale ? (
                      <Tag color="red">已过期</Tag>
                    ) : (
                      <Tag color="blue">实验性预测</Tag>
                    )}
                  </Descriptions.Item>
                </Descriptions>
              </div>
            ) : (
              <EmptyState message="暂无预测数据" />
            )}
          </Card>
        </Col>

        {/* ── 信号 + 评分 ────────────────────────── */}
        <Col xs={24} lg={12}>
          <Card title={<Space><ThunderboltOutlined />交易信号 & 评分</Space>} size="small">
            <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>
              <TrophyOutlined /> 综合评分
              <Tooltip title="综合评分 = 由模型预测换算的研究性参考分（0–100）：趋势≈采样看涨度×80；动量≈50+预测5日收益×200；综合≈采样看涨度×70+15。与基本面、技术面无关；模型为实验性，评分仅供横向参考，不构成投资建议。">
                <span style={{ marginLeft: 4, color: 'var(--text-muted)', fontSize: 12, cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </Title>
            {currentStock?.score ? (
              <Descriptions column={2} size="small" bordered>
                <Descriptions.Item label="趋势">
                  <span style={{ color: scoreColor(currentStock.score.trend), fontWeight: 600 }}>
                    {currentStock.score.trend.toFixed(0)}
                  </span>
                </Descriptions.Item>
                <Descriptions.Item label="动量">
                  <span style={{ color: scoreColor(currentStock.score.momentum), fontWeight: 600 }}>
                    {currentStock.score.momentum.toFixed(0)}
                  </span>
                </Descriptions.Item>
                <Descriptions.Item label="量能">
                  <span style={{ color: scoreColor(currentStock.score.volume), fontWeight: 600 }}>
                    {currentStock.score.volume.toFixed(0)}
                  </span>
                </Descriptions.Item>
                <Descriptions.Item label="综合">
                  <span style={{ color: scoreColor(currentStock.score.composite), fontWeight: 600 }}>
                    {currentStock.score.composite.toFixed(0)}
                  </span>
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <EmptyState message="暂无评分数据" />
            )}
            {currentStock?.score && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, lineHeight: 1.6 }}>
                说明：评分由模型预测（采样看涨度、预测收益）按固定公式换算，属研究性参考；
                “量能”暂为固定占位值，未接入成交量计算；不构成投资建议。
              </div>
            )}

            <Divider style={{ margin: '16px 0 12px' }} />

            <Title level={5} style={{ marginBottom: 8 }}>
              <ThunderboltOutlined /> 交易信号
              <Tooltip title="实盘信号仅对模型选股池、中证1000成分股与指数择时标的生成；其余股票在打开时会根据其模型预测按论文规则即时推导参考信号（阈值：预测5日收益 > 0.5% → 买入）。">
                <span style={{ marginLeft: 4, color: 'var(--text-muted)', fontSize: 12, cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </Title>
            {latestSignal ? (
              <Descriptions column={2} size="small" bordered style={{ marginBottom: 16 }}>
                <Descriptions.Item label="最新信号">
                  <Tag color={signalColor}>{latestSignal.signal_type}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="置信度">
                  {latestSignal.confidence != null ? `${(latestSignal.confidence * 100).toFixed(0)}%` : '--'}
                </Descriptions.Item>
                <Descriptions.Item label="建议仓位">
                  {latestSignal.position_pct != null ? `${(latestSignal.position_pct * 100).toFixed(0)}%` : '--'}
                </Descriptions.Item>
                <Descriptions.Item label="信号日期">
                  {latestSignal.date}
                </Descriptions.Item>
                {latestSignal.reason && (
                  <Descriptions.Item label="原因" span={2}>
                    {latestSignal.reason}
                  </Descriptions.Item>
                )}
              </Descriptions>
            ) : derivedSignal ? (
              <div style={{ padding: '10px 12px', background: 'var(--surface-1, rgba(0,0,0,0.02))', borderRadius: 6, lineHeight: 1.7 }}>
                <Space wrap>
                  <Tag color={derivedSignal.type === 'BUY' ? 'red' : 'default'} style={{ fontSize: 13, padding: '1px 10px' }}>
                    {derivedSignal.label}
                  </Tag>
                  <span style={{ fontSize: 13 }}>
                    模型预测未来 5 日收益 <PctText value={derivedSignal.ret} bold />
                    {derivedSignal.type === 'BUY'
                      ? ` > 阈值 ${(BUY_THRESHOLD * 100).toFixed(1)}%，建议持有 5 个交易日`
                      : ` ≤ 阈值 ${(BUY_THRESHOLD * 100).toFixed(1)}%，保持观望`}
                  </span>
                </Space>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>
                  该股不在实盘信号池，此信号为按论文规则由模型预测<b>即时推导的参考信号</b>，
                  未经实盘持仓跟踪。模型为实验性（方向准确率低于随机），<b>不构成投资建议</b>。
                </div>
              </div>
            ) : (
              <div style={{ padding: '8px 12px', background: 'var(--surface-1, rgba(0,0,0,0.02))', borderRadius: 6, fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                <Tag>暂无信号</Tag>
                <div style={{ marginTop: 6 }}>
                  该股暂无模型预测数据，无法推导参考信号（可能为新股/长期停牌，或预测任务未覆盖）。
                </div>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* ── 历史档案：预测 vs 实际 + 实盘信号记录 ────── */}
      <Card
        title={<Space><HistoryOutlined />预测 vs 实际 档案</Space>}
        size="small"
        style={{ marginBottom: 16 }}
      >
        {/* ① 逐批预测对照（评级） */}
        <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>
          逐批预测对照（评级）
          <Tooltip title="每一行 = 一次预测批次：生成日(行情截止日) → 目标日(未来第5个交易日)。实际收益在目标日收盘后按与预测同口径(后复权)回填，可逐批对比。旧版按“目标日”过滤，刚生成批次的未来目标日会被整批漏掉，现改为按生成日归档。">
            <span style={{ marginLeft: 4, color: 'var(--text-muted)', fontSize: 12, cursor: 'help' }}>ⓘ</span>
          </Tooltip>
        </Title>

        {predictionHistory.length > 0 ? (
          <div>
            <Space wrap size={[8, 4]} style={{ marginBottom: 8 }}>
              <Tag>归档 {predictionHistory.length} 次</Tag>
              {archiveVerified.length > 0 ? (
                <Tag color={archiveVerified.length >= 5 && archiveHitCount / archiveVerified.length >= 0.5 ? 'success' : 'default'}>
                  已到期验证 {archiveVerified.length} 次 · 方向命中 {archiveHitCount} 次
                </Tag>
              ) : (
                <Tag>尚无到期批次（目标日未到，由每日归档任务回填实际收益）</Tag>
              )}
            </Space>
            <div style={{ maxHeight: 240, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>生成日</th>
                    <th style={{ textAlign: 'left', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>目标日</th>
                    <th style={{ textAlign: 'right', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>预测5日收益</th>
                    <th style={{ textAlign: 'right', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>实际收益</th>
                    <th style={{ textAlign: 'center', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>到期验证</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionHistory.map((item) => {
                    const asOf = item.as_of_date ?? '--'
                    const verified = item.actual_return != null
                    const hit = verified && Math.sign(item.pred_return) === Math.sign(item.actual_return!)
                    return (
                      <tr key={`${asOf}-${item.pred_date}`}>
                        <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>{asOf}</td>
                        <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>{item.pred_date}</td>
                        <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>
                          <PctText value={item.pred_return} />
                        </td>
                        <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>
                          {verified ? <PctText value={item.actual_return} /> : <Text type="secondary">--</Text>}
                        </td>
                        <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'center' }}>
                          {verified ? (
                            hit ? <Tag color="success">✓ 命中</Tag> : <Tag color="error">✗ 未中</Tag>
                          ) : (
                            <Tag>待验证</Tag>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.7 }}>
              “命中” = 预测涨跌方向与实际一致。归档覆盖历次全市场快照批次
              （含 2026-07-29、09-03）与每日新增批次，实际收益在目标日到期后由每日归档任务自动回填。
              实验性说明：实测该模型全市场方向准确率约 <b>34%</b>（低于随机 50%），
              个股归档样本极少、无统计意义，此表仅如实展示逐次对照，<b>不构成投资建议</b>。
            </div>
          </div>
        ) : (
          <div style={{ padding: '20px 16px', background: 'var(--surface-1, rgba(0,0,0,0.02))', borderRadius: 6, lineHeight: 1.8 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>该股暂无已归档的预测批次</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              历史评级来自“全市场快照”批次（已收录 2026-07-29、09-03 两次）与每日 16:30 自动新增批次。
              本股尚未被任何批次收录，常见于新股 / 长期停牌 / 暂停上市；
              下次快照收录后此处会自动出现每批 预测 vs 实际 对照。
            </div>
          </div>
        )}

        <Divider style={{ margin: '16px 0 12px' }} />

        {/* ② 实盘信号记录 */}
        <Title level={5} style={{ marginBottom: 8 }}>
          <ThunderboltOutlined /> 实盘信号记录
          <Tooltip title="实盘信号由历史持仓跟踪生成，仅覆盖模型选股池、中证1000成分与指数择时标的；其它股票没有实盘信号属正常现象，请使用本页上方「交易信号」的即时参考信号。">
            <span style={{ marginLeft: 4, color: 'var(--text-muted)', fontSize: 12, cursor: 'help' }}>ⓘ</span>
          </Tooltip>
        </Title>
        {signals.length > 0 ? (
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>日期</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>信号</th>
                  <th style={{ textAlign: 'right', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>置信度</th>
                  <th style={{ textAlign: 'right', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>仓位</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>原因</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s, i) => {
                  const color =
                    s.signal_type === 'BUY' ? 'var(--danger)'
                    : s.signal_type === 'SELL' ? 'var(--success)'
                    : 'var(--warning)'
                  return (
                    <tr key={i}>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>{s.date}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>
                        <Tag color={color === 'var(--danger)' ? 'red' : color === 'var(--success)' ? 'green' : 'orange'}>
                          {s.signal_type}
                        </Tag>
                      </td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>
                        {s.confidence != null ? `${(s.confidence * 100).toFixed(0)}%` : '--'}
                      </td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>
                        {`${(s.position_pct * 100).toFixed(0)}%`}
                      </td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {s.reason || '--'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
            该股无实盘信号记录 —— 实盘信号仅对<b>模型选股池 / 中证1000成分 / 指数择时标的</b>生成；
            本页上方「交易信号」为按当前模型预测即时推导的<b>参考信号</b>，两者请勿混淆。
          </div>
        )}
      </Card>
    </div>
  )
}
