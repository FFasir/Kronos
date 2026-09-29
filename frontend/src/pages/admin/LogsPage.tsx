/**
 * 运行日志页面
 * 对应旧版 admin_portal.html #page-logs
 *
 * 功能：模型训练日志列表，支持状态筛选和分页
 * 数据来源：GET /api/v1/admin/training/log?limit=50
 */

import { useEffect, useState, useCallback } from 'react'
import { Table, Tag, Typography, Card, Select, Space, Button } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { ReloadOutlined, FileTextOutlined } from '@ant-design/icons'
import { getTrainingLogs } from '@/api/admin'
import type { TrainingLogEntry } from '@/types/api'
import LoadingSkeleton from '@/components/common/LoadingSkeleton'
import EmptyState from '@/components/common/EmptyState'
import ErrorState from '@/components/common/ErrorState'

const { Title, Text } = Typography

/** 训练状态颜色映射 */
const STATUS_MAP: Record<string, { color: string; label: string }> = {
  completed: { color: 'success', label: '已完成' },
  running: { color: 'processing', label: '运行中' },
  failed: { color: 'error', label: '失败' },
}

export default function LogsPage() {
  const [logs, setLogs] = useState<TrainingLogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filterStatus, setFilterStatus] = useState<string | undefined>(undefined)

  // ── 加载日志 ────────────────────────────────────────
  const fetchLogs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getTrainingLogs(50)
      if (data) {
        setLogs(data)
      } else {
        setLogs([])
      }
    } catch {
      setError('无法加载训练日志')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchLogs() }, [fetchLogs])

  // ── 过滤 ────────────────────────────────────────────
  const filtered = filterStatus
    ? logs.filter((l) => l.status === filterStatus)
    : logs

  // ── 表格列定义 ──────────────────────────────────────
  const columns: ColumnsType<TrainingLogEntry> = [
    {
      title: '训练窗口',
      dataIndex: 'window_id',
      key: 'window_id',
      width: 130,
      sorter: (a, b) => a.window_id.localeCompare(b.window_id),
      render: (v: string) => <Text code>{v}</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 90,
      filters: [
        { text: '已完成', value: 'completed' },
        { text: '运行中', value: 'running' },
        { text: '失败', value: 'failed' },
      ],
      onFilter: (value, record) => record.status === value,
      render: (v: string) => {
        const m = STATUS_MAP[v]
        return <Tag color={m?.color}>{m?.label ?? v}</Tag>
      },
    },
    {
      title: '开始时间',
      dataIndex: 'start_time',
      key: 'start_time',
      width: 160,
      sorter: (a, b) => a.start_time.localeCompare(b.start_time),
    },
    {
      title: '结束时间',
      dataIndex: 'end_time',
      key: 'end_time',
      width: 160,
      render: (v: string | undefined) => v || '--',
    },
    {
      title: 'Epochs',
      dataIndex: 'epochs_completed',
      key: 'epochs_completed',
      width: 80,
      align: 'right',
      render: (v: number | undefined) => v ?? '--',
    },
    {
      title: '最佳 Val Loss',
      dataIndex: 'best_val_loss',
      key: 'best_val_loss',
      width: 120,
      align: 'right',
      render: (v: number | undefined) => (v != null ? v.toFixed(4) : '--'),
    },
    {
      title: 'Spearman',
      dataIndex: 'best_spearman',
      key: 'best_spearman',
      width: 100,
      align: 'right',
      render: (v: number | undefined) => (v != null ? v.toFixed(4) : '--'),
    },
    {
      title: '模型路径',
      dataIndex: 'model_path',
      key: 'model_path',
      ellipsis: true,
      render: (v: string | undefined) => v || '--',
    },
    {
      title: '错误信息',
      dataIndex: 'error_message',
      key: 'error_message',
      width: 180,
      ellipsis: true,
      render: (v: string | undefined) => v || '--',
    },
  ]

  // ── 加载中 ──────────────────────────────────────────
  if (loading && logs.length === 0) {
    return <LoadingSkeleton type="table" rows={8} />
  }

  // ── 加载失败 ────────────────────────────────────────
  if (error && logs.length === 0) {
    return <ErrorState message={error} onRetry={fetchLogs} />
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>运行日志</Title>
        <Space>
          <Select
            allowClear
            placeholder="状态筛选"
            style={{ width: 120 }}
            value={filterStatus}
            onChange={setFilterStatus}
            options={[
              { label: '已完成', value: 'completed' },
              { label: '运行中', value: 'running' },
              { label: '失败', value: 'failed' },
            ]}
          />
          <Button icon={<ReloadOutlined />} onClick={fetchLogs} loading={loading}>
            刷新
          </Button>
        </Space>
      </div>

      <Card>
        {filtered.length === 0 ? (
          <EmptyState message="暂无训练日志" icon={<FileTextOutlined />} />
        ) : (
          <Table<TrainingLogEntry>
            columns={columns}
            dataSource={filtered}
            rowKey={(r) => r.id?.toString() ?? r.window_id}
            loading={loading}
            size="middle"
            pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
            scroll={{ x: 1100 }}
            locale={{ emptyText: '暂无数据' }}
          />
        )}
      </Card>
    </div>
  )
}
