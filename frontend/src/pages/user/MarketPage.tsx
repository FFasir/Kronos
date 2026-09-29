import { useCallback, useEffect, useState } from 'react'
import { Button, Input, Space, Switch, Table, Typography, Tooltip, Tag } from 'antd'
import { EyeOutlined, ReloadOutlined, SyncOutlined } from '@ant-design/icons'
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table'
import { useNavigate } from 'react-router-dom'

import { getIndexList } from '@/api/index'
import { getQuotes } from '@/api/quote'
import type { IndexInfo, QuoteInfo } from '@/types/api'
import { useWatchlistStore } from '@/stores/useWatchlistStore'
import ErrorState from '@/components/common/ErrorState'
import PctText, { pctColor } from '@/components/common/PctText'
import { StarFilled, StarOutlined } from '@ant-design/icons'


const { Title, Text } = Typography
const PAGE_SIZE = 50
/** 报价自动刷新间隔（毫秒） */
const AUTO_REFRESH_MS = 20_000

export default function MarketPage() {
  const navigate = useNavigate()
  const [rows, setRows] = useState<IndexInfo[]>([])
  const [quotes, setQuotes] = useState<Record<string, QuoteInfo>>({})
  const [quotesLoading, setQuotesLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [keyword, setKeyword] = useState('')
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── 自选（行内快捷加入）──────────────────────────
  const watchCodes = useWatchlistStore((s) => s.codes)
  const toggleWatch = useWatchlistStore((s) => s.toggle)

  /** 单独刷新当前页报价（不重拉列表，不打断浏览位置） */
  const loadQuotesOnly = useCallback(async (codes: string[]) => {
    if (codes.length === 0) return
    try {
      const quoteResp = await getQuotes(codes)
      const map: Record<string, QuoteInfo> = {}
      for (const q of quoteResp?.quotes ?? []) map[q.code] = q
      setQuotes(map)
    } catch {
      // 忽略单次失败，下一轮重试
    }
  }, [])

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getIndexList('stock', query || undefined, PAGE_SIZE, (page - 1) * PAGE_SIZE)
      const pageRows = result?.indices ?? []
      setRows(pageRows)
      setTotal(result?.total ?? 0)
      setQuotes({})
      setQuotesLoading(pageRows.length > 0)
      try {
        await loadQuotesOnly(pageRows.map((r) => r.code))
      } finally {
        setQuotesLoading(false)
      }
    } catch {
      setError('A股目录加载失败')
    } finally {
      setLoading(false)
    }
  }, [page, query, loadQuotesOnly])

  useEffect(() => { load() }, [load])

  // ── 自动刷新报价（默认开启，20 秒一次）─────────────
  useEffect(() => {
    if (!autoRefresh || rows.length === 0) return
    const timer = window.setInterval(() => {
      loadQuotesOnly(rows.map((r) => r.code))
    }, AUTO_REFRESH_MS)
    return () => window.clearInterval(timer)
  }, [autoRefresh, rows, loadQuotesOnly])

  const columns: ColumnsType<IndexInfo> = [
    { title: '代码', dataIndex: 'code', width: 110, render: (value: string) => <Text code>{value}</Text> },
    { title: '名称', dataIndex: 'name', width: 170 },
    { title: '行业', dataIndex: 'industry', width: 140, render: (value?: string) => value || '--' },
    {
      title: '最新价',
      key: 'price',
      width: 100,
      align: 'right',
      render: (_: unknown, row: IndexInfo) => {
        const q = quotes[row.code]
        if (!q || q.price == null) return <Text type="secondary">--</Text>
        return (
          <Text strong style={{ color: pctColor(q.change_pct ?? 0) }}>
            {q.price.toFixed(2)}
          </Text>
        )
      },
    },
    {
      title: '涨跌幅',
      key: 'change_pct',
      width: 100,
      align: 'right',
      render: (_: unknown, row: IndexInfo) => {
        const q = quotes[row.code]
        if (!q) return <Text type="secondary">--</Text>
        if (q.suspended) return <Tag color="orange">停牌</Tag>
        if (q.change_pct == null) return <Text type="secondary">--</Text>
        return <PctText value={q.change_pct} raw bold />
      },
    },
    {
      title: '行情时间',
      key: 'time',
      width: 110,
      align: 'center',
      render: (_: unknown, row: IndexInfo) => {
        const q = quotes[row.code]
        if (!q?.time) return <Text type="secondary">--</Text>
        return <Text type="secondary" style={{ fontSize: 12 }}>{q.time}</Text>
      },
    },
    {
      title: '自选',
      key: 'watch',
      width: 60,
      align: 'center',
      render: (_: unknown, row: IndexInfo) => {
        const watched = watchCodes.includes(row.code)
        return (
          <Button
            type="text"
            size="small"
            icon={watched ? <StarFilled style={{ color: '#fadb14' }} /> : <StarOutlined />}
            title={watched ? '已自选，点击移出' : '加入自选（存本机浏览器）'}
            onClick={(e) => { e.stopPropagation(); toggleWatch(row.code) }}
          />
        )
      },
    },
    {
      title: '查看',
      key: 'action',
      width: 70,
      align: 'center',
      render: (_: unknown, row: IndexInfo) => (
        <Button
          type="text"
          icon={<EyeOutlined />}
          title="查看个股"
          aria-label={`查看 ${row.name}`}
          onClick={() => navigate(`/user/stock/${row.code}`)}
        />
      ),
    },
  ]

  const changePage = (pagination: TablePaginationConfig) => setPage(pagination.current ?? 1)

  if (error && rows.length === 0) return <ErrorState message={error} onRetry={load} />

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 16 }} wrap>
        <div>
          <Title level={4} style={{ margin: 0 }}>A股行情</Title>
          <Text type="secondary">
            共 {total.toLocaleString()} 只股票 · 价格与涨跌幅来自新浪实时接口（约 15 秒延迟）
            <Tooltip title="实时行情由服务器抓取新浪数据并缓存；停牌或接口失败时显示 '--'">
              <span style={{ marginLeft: 4, color: '#999' }}>ⓘ</span>
            </Tooltip>
          </Text>
        </div>
        <Space wrap>
          <Space size={6}>
            <SyncOutlined spin={autoRefresh} style={{ color: autoRefresh ? 'var(--primary, #1890ff)' : 'var(--text-muted)' }} />
            <span style={{ fontSize: 13 }}>自动刷新</span>
            <Switch
              size="small"
              checked={autoRefresh}
              onChange={setAutoRefresh}
              title="每 20 秒自动刷新价格与涨跌幅（不刷新列表）"
            />
          </Space>
          <Input.Search
            value={keyword}
            placeholder="代码、名称或行业"
            allowClear
            style={{ width: 260 }}
            onChange={(event) => setKeyword(event.target.value)}
            onSearch={(value) => { setPage(1); setQuery(value.trim()) }}
          />
          <Button icon={<ReloadOutlined />} title="刷新" aria-label="刷新" onClick={load} loading={loading} />
        </Space>
      </Space>
      <Table<IndexInfo>
        rowKey="code"
        columns={columns}
        dataSource={rows}
        loading={loading || quotesLoading}
        onChange={changePage}
        pagination={{ current: page, pageSize: PAGE_SIZE, total, showSizeChanger: false }}
        scroll={{ x: 800 }}
        size="middle"
      />
    </div>
  )
}
