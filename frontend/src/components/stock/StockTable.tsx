/**
 * StockTable — 可排序/筛选股票表格组件
 *
 * 替换旧版手写表格 HTML，封装 Ant Design Table。
 * 支持排序、搜索筛选、自定义列、行点击回调。
 *
 * 信任改造要点:
 *  - 价格默认展示不复权市价（last_close_raw），与个股 K 线同口径；
 *  - 每行标注预测生成日期，落后于整批最新日期（dataDate）时标红"已过期"；
 *  - 数据异常预测（|5日收益|>40%）以红色标记警示。
 *
 * 用法:
 *   <StockTable
 *     dataSource={pool}
 *     dataDate="2026-09-03"
 *     loading={false}
 *     onRowClick={(record) => navigate(`/user/stock/${record.code}`)}
 *   />
 */

import { Table, Tag, Tooltip, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import type { PredictionResult, SignalResult } from '@/types/api'
import PctText from '@/components/common/PctText'

const { Text } = Typography

/** 表格行数据（选股池） */
export interface StockTableRow {
  key: string
  code: string
  name: string
  close: number
  pred_return: number
  confidence: number
  signal_type: string
  position_pct: number
  error?: string
  /** 预测生成日期 */
  as_of_date?: string
  /** 数据异常预测 */
  is_anomalous?: boolean
}

export interface StockTableProps {
  /** 数据 */
  dataSource: StockTableRow[]
  /** 整批预测的生成日期（最新数据日期），用于判断单行是否过期 */
  dataDate?: string
  /** 加载中 */
  loading?: boolean
  /** 行点击回调 */
  onRowClick?: (record: StockTableRow) => void
}

/**
 * 从 PredictionResult 数组转换为 StockTableRow。
 * @param signals - 最新信号列表，按 symbol 合并信号类型与建议仓位
 */
export function toStockTableRows(
  predictions: PredictionResult[],
  signals?: SignalResult[],
): StockTableRow[] {
  const sigMap = new Map((signals ?? []).map((s) => [s.symbol, s]))
  return predictions.map((p, i) => {
    const sig = sigMap.get(p.symbol)
    return {
      key: p.symbol || String(i),
      code: p.symbol,
      name: p.name || p.symbol,
      close: p.last_close_raw ?? p.last_close,
      pred_return: p.pred_return,
      confidence: p.confidence,
      signal_type: sig?.signal_type ?? '--',
      position_pct: sig?.position_pct ?? 0,
      error: p.error,
      as_of_date: p.as_of_date,
      is_anomalous: p.is_anomalous,
    }
  })
}

const SIGNAL_COLORS: Record<string, string> = {
  BUY: 'red',
  HOLD: 'orange',
  SELL: 'green',
  CASH: 'default',
}

const SIGNAL_LABELS: Record<string, string> = {
  BUY: '买入',
  HOLD: '持有',
  SELL: '卖出',
  CASH: '空仓',
}

/** 判断单行预测是否已过期（生成日期落后于整批最新日期） */
function isStaleRow(asOfDate?: string, dataDate?: string): boolean {
  if (!asOfDate || !dataDate) return false
  return asOfDate < dataDate
}

export default function StockTable({
  dataSource,
  dataDate,
  loading = false,
  onRowClick,
}: StockTableProps) {
  const columns: ColumnsType<StockTableRow> = [
    {
      title: '代码',
      dataIndex: 'code',
      key: 'code',
      width: 100,
      sorter: (a, b) => a.code.localeCompare(b.code),
      render: (code: string) => <Text code>{code}</Text>,
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      width: 120,
    },
    {
      title: (
        <Tooltip title="不复权市价，与个股 K 线、行情页同口径">
          <span>最新价(市价)<span style={{ color: '#999' }}> ⓘ</span></span>
        </Tooltip>
      ),
      dataIndex: 'close',
      key: 'close',
      width: 120,
      align: 'right',
      sorter: (a, b) => a.close - b.close,
      render: (v: number, record) => record.error ? '--' : (v?.toFixed(2) ?? '--'),
    },
    {
      title: '预测收益率',
      dataIndex: 'pred_return',
      key: 'pred_return',
      width: 130,
      align: 'right',
      sorter: (a, b) => a.pred_return - b.pred_return,
      render: (v: number, record) => {
        if (record.error) return '--'
        return (
          <span style={record.is_anomalous ? { opacity: 0.55 } : undefined}>
            <PctText value={v} bold />
            {record.is_anomalous && (
              <Tag color="red" style={{ marginLeft: 6, fontSize: 11 }}>数据异常</Tag>
            )}
          </span>
        )
      },
    },
    {
      title: (
        <Tooltip title="模型采样统计值，并非历史胜率。实测该模型方向准确率低于随机，仅供研究参考。">
          <span>采样看涨度<span style={{ color: '#999' }}> ⓘ</span></span>
        </Tooltip>
      ),
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      align: 'right',
      sorter: (a, b) => a.confidence - b.confidence,
      render: (v: number, record) => {
        if (record.error) return <Tag>缺失</Tag>
        const color = v >= 0.8 ? 'var(--success)' : v >= 0.5 ? 'var(--warning)' : 'var(--danger)'
        return <Tag color={color}>{(v * 100).toFixed(0)}%</Tag>
      },
    },
    {
      title: '生成日期',
      dataIndex: 'as_of_date',
      key: 'as_of_date',
      width: 120,
      align: 'center',
      sorter: (a, b) => (a.as_of_date ?? '').localeCompare(b.as_of_date ?? ''),
      render: (v: string | undefined, record) => {
        if (!v) return '--'
        if (isStaleRow(v, dataDate)) {
          return <Tag color="red">已过期 · {v}</Tag>
        }
        return <span>{v}</span>
      },
    },
    {
      title: '信号',
      dataIndex: 'signal_type',
      key: 'signal_type',
      width: 80,
      filters: [
        { text: '买入', value: 'BUY' },
        { text: '持有', value: 'HOLD' },
        { text: '卖出', value: 'SELL' },
        { text: '空仓', value: 'CASH' },
      ],
      onFilter: (value, record) => record.signal_type === value,
      render: (v: string) => (
        <Tag color={SIGNAL_COLORS[v] || 'default'}>{SIGNAL_LABELS[v] || v}</Tag>
      ),
    },
    {
      title: '建议仓位',
      dataIndex: 'position_pct',
      key: 'position_pct',
      width: 100,
      align: 'right',
      render: (v: number) => `${((v ?? 0) * 100).toFixed(0)}%`,
    },
  ]

  return (
    <Table<StockTableRow>
      columns={columns}
      dataSource={dataSource}
      loading={loading}
      size="middle"
      pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
      onRow={(record) => ({
        onClick: () => onRowClick?.(record),
        style: { cursor: onRowClick ? 'pointer' : 'default' },
      })}
      scroll={{ x: 900 }}
      locale={{ emptyText: '暂无数据' }}
    />
  )
}
