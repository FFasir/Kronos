/**
 * 我的自选 — 自选股实时行情页
 *
 * 数据来源：
 *   useWatchlistStore（localStorage 本地持久化）
 *   GET /api/v1/quote/list          (实时行情)
 *
 * 功能：自选列表 → 实时价格/涨跌幅（20 秒自动刷新）→ 点击进入个股看板
 */

import { useCallback, useEffect, useState } from 'react'
import { Button, Card, Empty, Space, Table, Tag, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { DeleteOutlined, EyeOutlined, StarOutlined, SyncOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { getQuotes } from '@/api/quote'
import type { QuoteInfo } from '@/types/api'
import { useWatchlistStore } from '@/stores/useWatchlistStore'
import PctText, { pctColor } from '@/components/common/PctText'

const { Title, Text } = Typography
const REFRESH_MS = 20_000

interface Row {
  code: string
  quote?: QuoteInfo
}

export default function WatchlistPage() {
  const navigate = useNavigate()
  const codes = useWatchlistStore((s) => s.codes)
  const remove = useWatchlistStore((s) => s.remove)
  const clear = useWatchlistStore((s) => s.clear)
  const [quotes, setQuotes] = useState<Record<string, QuoteInfo>>({})
  const [loading, setLoading] = useState(false)

  const loadQuotes = useCallback(async () => {
    if (codes.length === 0) return
    try {
      const resp = await getQuotes(codes)
      const map: Record<string, QuoteInfo> = {}
      for (const q of resp?.quotes ?? []) map[q.code] = q
      setQuotes(map)
    } catch {
      // 单次失败忽略，下一轮重试
    }
  }, [codes])

  useEffect(() => {
    setLoading(true)
    loadQuotes().finally(() => setLoading(false))
  }, [loadQuotes])

  // 自动刷新报价
  useEffect(() => {
    if (codes.length === 0) return
    const timer = window.setInterval(loadQuotes, REFRESH_MS)
    return () => window.clearInterval(timer)
  }, [codes, loadQuotes])

  const columns: ColumnsType<Row> = [
    {
      title: '代码',
      dataIndex: 'code',
      width: 110,
      render: (v: string) => <Text code>{v}</Text>,
    },
    {
      title: '名称',
      key: 'name',
      width: 170,
      render: (_: unknown, r: Row) => r.quote?.name ?? r.code,
    },
    {
      title: '最新价',
      key: 'price',
      width: 110,
      align: 'right',
      render: (_: unknown, r: Row) => {
        const q = r.quote
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
      render: (_: unknown, r: Row) => {
        const q = r.quote
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
      render: (_: unknown, r: Row) => {
        if (!r.quote?.time) return <Text type="secondary">--</Text>
        return <Text type="secondary" style={{ fontSize: 12 }}>{r.quote.time}</Text>
      },
    },
    {
      title: '查看',
      key: 'action',
      width: 120,
      align: 'center',
      render: (_: unknown, r: Row) => (
        <Space size={4}>
          <Button
            type="text"
            icon={<EyeOutlined />}
            title="查看个股"
            onClick={(e) => { e.stopPropagation(); navigate(`/user/stock/${r.code}`) }}
          />
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            title="移出自选"
            onClick={(e) => { e.stopPropagation(); remove(r.code) }}
          />
        </Space>
      ),
    },
  ]

  const rows: Row[] = codes.map((code) => ({ code, quote: quotes[code] }))

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 16 }} wrap>
        <div>
          <Title level={4} style={{ margin: 0 }}>
            <StarOutlined style={{ color: '#fadb14' }} /> 我的自选
          </Title>
          <Text type="secondary">
            共 {codes.length} 只 · 价格来自新浪实时行情（每 20 秒自动刷新，存在本机浏览器）
            <Space size={6} style={{ marginLeft: 8 }}>
              <SyncOutlined spin style={{ fontSize: 12, color: 'var(--text-muted)' }} />
            </Space>
          </Text>
        </div>
        {codes.length > 0 && (
          <Button danger type="text" onClick={clear}>清空自选</Button>
        )}
      </Space>

      {codes.length === 0 ? (
        <Card>
          <Empty
            description="还没有自选股票。到个股看板打开任意股票，点右上角 ☆ 加入自选。"
            style={{ padding: '48px 0' }}
          >
            <Button type="primary" onClick={() => navigate('/user/market')}>
              去 A股行情逛逛
            </Button>
          </Empty>
        </Card>
      ) : (
        <Table<Row>
          rowKey="code"
          columns={columns}
          dataSource={rows}
          loading={loading}
          size="middle"
          pagination={false}
          scroll={{ x: 640 }}
          onRow={(r) => ({
            onClick: () => navigate(`/user/stock/${r.code}`),
            style: { cursor: 'pointer' },
          })}
          locale={{ emptyText: '暂无数据' }}
        />
      )}
    </div>
  )
}
