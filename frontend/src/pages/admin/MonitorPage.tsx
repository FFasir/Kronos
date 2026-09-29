/**
 * 系统监控页面
 * 对应旧版 admin_portal.html #page-monitor
 *
 * 功能：系统状态卡片 + GPU/Loss 图表 + 数据源状态
 * 数据来源：GET /api/v1/admin/monitor
 *           GET /api/v1/admin/data/sources
 *           GET /api/v1/admin/training/log?limit=12
 */

import { useEffect, useState, useCallback } from 'react'
import { Row, Col, Card, Typography, Tag, Table, Space, Button, Alert, Popconfirm } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  ClockCircleOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  DatabaseOutlined,
  PlayCircleOutlined,
  FilterOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
} from '@ant-design/icons'
import { getSystemMonitor, getDataSources, getTrainingLogs, runAdminJob, getAdminJobStatus } from '@/api/admin'
import type { AdminJobName } from '@/api/admin'
import type { SystemMonitor, DataSourceInfo, TrainingLogEntry, BackgroundJob } from '@/types/api'
import StatCard from '@/components/common/StatCard'
import MonitorChart from '@/components/charts/MonitorChart'
import LoadingSkeleton from '@/components/common/LoadingSkeleton'
import ErrorState from '@/components/common/ErrorState'
import EmptyState from '@/components/common/EmptyState'

const { Title } = Typography

/** 系统状态颜色映射 */
const STATUS_COLOR: Record<string, string> = {
  running: '#22b07d',
  ready: '#22b07d',
  idle: '#f59e0b',
  error: '#e74c5e',
  stale: '#f59e0b',
  missing: '#e74c5e',
}

const STATUS_ICON: Record<string, React.ReactNode> = {
  running: <CheckCircleOutlined />,
  ready: <CheckCircleOutlined />,
  idle: <ClockCircleOutlined />,
  error: <CloseCircleOutlined />,
  stale: <WarningOutlined />,
  missing: <CloseCircleOutlined />,
}

/** 手工任务按钮元数据。
 *  - confirm: 高成本任务（触发全市场推理 / GPU 训练 / 大量写库）需要二次确认。
 *  - 提醒文案点明当前生产配置为沪深300指数择时（paper_symbol=000300, execution=510300），
 *    而 predict.py 不带 --symbols 时默认遍历全 A 股，个股预测与当前指数模型不匹配。
 */
const JOB_BUTTONS: {
  name: AdminJobName
  label: string
  icon: React.ReactNode
  type?: 'primary'
  danger?: boolean
  confirm?: string
  confirmDesc?: string
}[] = [
  {
    name: 'data_update',
    label: '更新日线',
    icon: <DatabaseOutlined />,
  },
  {
    name: 'timing_data_update',
    label: '更新择时行情',
    icon: <DatabaseOutlined />,
  },
  {
    name: 'stock_hfq_update',
    label: '刷新个股后复权行情',
    icon: <SyncOutlined />,
    confirm: '从新浪增量刷新近期个股的后复权(hfq)日线？',
    confirmDesc: '刷新被预测/选中的个股到最新交易日，K线与个股预测随后可用最新数据。',
  },
  {
    name: 'preprocess',
    label: '构建预处理库',
    icon: <PlayCircleOutlined />,
    confirm: '构建预处理库（全 A 个股 HDF5）？',
    confirmDesc: '耗时较长并占用磁盘；若当前只需要指数/ETF 择时可跳过。',
  },
  {
    name: 'timing_signal',
    label: '生成交易信号',
    icon: <ThunderboltOutlined />,
    type: 'primary',
    confirm: '生成交易信号（按配置的 paper_symbol=000300）？',
    confirmDesc: '会先对 000300 做模型预测，再生成指数信号并落库。',
  },
  {
    name: 'prediction',
    label: '生成模型预测',
    icon: <ThunderboltOutlined />,
    danger: true,
    confirm: '生成模型预测（范围 = 配置 prediction_symbols，当前 000300/510300）？',
    confirmDesc: '当前激活的是沪深300指数对齐模型（timing-idx300），对个股价格尺度不匹配；若确需全 A 预测请用命令行 --all-a-share 显式执行。',
  },
  {
    name: 'rolling_training',
    label: '滚动训练',
    icon: <ExperimentOutlined />,
    confirm: '启动滚动训练（GPU 长任务）？',
    confirmDesc: '耗时可达数小时；训练过程中请勿重复触发其他任务。',
  },
  {
    name: 'selection_signal',
    label: '生成选股与信号',
    icon: <FilterOutlined />,
    danger: true,
    confirm: '生成选股池与逐股信号（基于全市场预测）？',
    confirmDesc: '依赖可用的个股预测批次；当前指数模型不产生可信个股预测，执行可能为空或写入无效信号。',
  },
  {
    name: 'daily_pipeline',
    label: '执行每日流水线',
    icon: <SyncOutlined />,
    danger: true,
    confirm: '执行每日流水线（日线更新 + 回填实际值 + 预测 + 信号）？',
    confirmDesc: '预测步骤按配置 prediction_symbols（000300/510300）执行，与沪深300指数择时范围一致。',
  },
]

export default function MonitorPage() {
  const [monitor, setMonitor] = useState<SystemMonitor | null>(null)
  const [sources, setSources] = useState<DataSourceInfo[]>([])
  const [lossLogs, setLossLogs] = useState<TrainingLogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeJob, setActiveJob] = useState<BackgroundJob | null>(null)

  // ── 加载数据 ────────────────────────────────────────
  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [monData, srcData, logData] = await Promise.all([
        getSystemMonitor(),
        getDataSources(),
        getTrainingLogs(12),
      ])
      if (monData) setMonitor(monData)
      if (srcData) setSources(srcData)
      if (logData) setLossLogs(logData)
    } catch {
      setError('无法获取系统监控数据')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  useEffect(() => {
    if (!activeJob || !['accepted', 'running'].includes(activeJob.status)) return
    const id = window.setInterval(async () => {
      const status = await getAdminJobStatus(activeJob.job_id)
      if (status) {
        setActiveJob(status)
        if (status.status === 'completed') loadData()
      }
    }, 2000)
    return () => window.clearInterval(id)
  }, [activeJob, loadData])

  const startJob = async (name: AdminJobName) => {
    const job = await runAdminJob(name)
    if (job) setActiveJob(job)
  }

  // ── 自动刷新 (30s) ─────────────────────────────────
  useEffect(() => {
    const id = setInterval(loadData, 30_000)
    return () => clearInterval(id)
  }, [loadData])

  // ── 加载中 ──────────────────────────────────────────
  if (loading && !monitor) {
    return <LoadingSkeleton type="card" rows={4} />
  }

  // ── 错误 ────────────────────────────────────────────
  if (error && !monitor) {
    return <ErrorState message={error} onRetry={loadData} />
  }

  // ── 构建 Loss 数据 ─────────────────────────────────
  const lossMonths = lossLogs
    .filter((l) => l.best_val_loss != null)
    .map((l) => l.window_id)
    .reverse()
  const lossValues = lossLogs
    .filter((l) => l.best_val_loss != null)
    .map((l) => l.best_val_loss!)
    .reverse()

  // ── 数据源表格列 ────────────────────────────────────
  const srcColumns: ColumnsType<DataSourceInfo> = [
    {
      title: '标的',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
      render: (v: string) => <Typography.Text code>{v}</Typography.Text>,
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      width: 120,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 80,
    },
    {
      title: '数据范围',
      key: 'range',
      width: 180,
      render: (_: unknown, r: DataSourceInfo) =>
        r.start_date ? `${r.start_date} ~ ${r.end_date || '至今'}` : '--',
    },
    {
      title: '标的数',
      dataIndex: 'total_rows',
      key: 'total_rows',
      width: 90,
      align: 'right',
      render: (v: number | undefined) => v?.toLocaleString() ?? '--',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (v: string) => (
        <Tag color={STATUS_COLOR[v] || 'default'} icon={STATUS_ICON[v]}>
          {v}
        </Tag>
      ),
    },
    {
      title: '缺失天数',
      dataIndex: 'missing_dates',
      key: 'missing_dates',
      width: 90,
      align: 'right',
      render: (v: number | undefined) =>
        v != null ? <span style={{ color: v > 0 ? 'var(--danger)' : 'var(--success)' }}>{v}</span> : '--',
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>系统监控</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
            刷新
          </Button>
        </Space>
      </div>

      <Space wrap style={{ marginBottom: 16 }}>
        {JOB_BUTTONS.map((job) => {
          const busy = !!activeJob && ['accepted', 'running'].includes(activeJob.status)
          const button = (
            <Button
              type={job.type}
              danger={job.danger}
              icon={job.icon}
              disabled={busy}
              onClick={() => startJob(job.name)}
            >
              {job.label}
            </Button>
          )
          return job.confirm ? (
            <Popconfirm
              key={job.name}
              title={job.confirm}
              description={job.confirmDesc}
              okText="确认执行"
              cancelText="取消"
              okButtonProps={{ danger: true }}
              onConfirm={() => startJob(job.name)}
            >
              {button}
            </Popconfirm>
          ) : (
            <span key={job.name}>{button}</span>
          )
        })}
      </Space>
      {activeJob && (
        <Alert
          style={{ marginBottom: 16 }}
          type={activeJob.status === 'failed' ? 'error' : activeJob.status === 'completed' ? 'success' : 'info'}
          showIcon
          message={`${activeJob.name}: ${activeJob.status}`}
          description={activeJob.error || (activeJob.status === 'running' ? `进度 ${Math.round(activeJob.progress * 100)}%` : undefined)}
        />
      )}

      {/* ── 状态卡片 ───────────────────────────────── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <StatCard
            title="模型状态"
            value={monitor?.model_status ?? '--'}
            suffix=""
            precision={0}
          />
        </Col>
        <Col xs={12} sm={6}>
          <StatCard
            title="GPU 占用"
            value={monitor?.gpu_usage ?? 0}
            suffix="%"
            precision={1}
          />
        </Col>
        <Col xs={12} sm={6}>
          <StatCard
            title="数据状态"
            value={monitor?.data_status ?? '--'}
            suffix=""
            precision={0}
          />
        </Col>
        <Col xs={12} sm={6}>
          <StatCard
            title="选股池数量"
            value={monitor?.pool_count ?? 0}
            suffix=" 只"
            precision={0}
          />
        </Col>
      </Row>

      {/* ── 图表：Loss + GPU ────────────────────────── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="训练 Loss 趋势" size="small">
            {lossValues.length > 0 ? (
              <MonitorChart type="loss" months={lossMonths} data={lossValues} height={300} />
            ) : (
              <EmptyState message="暂无训练数据" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="GPU 利用率" size="small">
            {monitor?.gpu_usage != null ? (
              <MonitorChart
                type="gpu"
                hours={['当前']}
                gpuData={[monitor.gpu_usage ?? 0]}
                memData={[monitor.gpu_memory_used ?? 0]}
                height={300}
              />
            ) : (
              <EmptyState message="GPU 监控不可用" />
            )}
          </Card>
        </Col>
      </Row>

      {/* ── 系统信息 ───────────────────────────────── */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card size="small">
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>模型版本</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{monitor?.model_version || '--'}</div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>上次训练</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{monitor?.last_training_time || '--'}</div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>上次预测</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{monitor?.last_prediction_time || '--'}</div>
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>运行时间</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{monitor?.uptime || '--'}</div>
          </Card>
        </Col>
      </Row>

      {/* ── 数据源状态 ─────────────────────────────── */}
      <Card title="数据源状态" style={{ marginBottom: 16 }}>
        {sources.length === 0 ? (
          <EmptyState message="暂无数据源信息" />
        ) : (
          <Table<DataSourceInfo>
            columns={srcColumns}
            dataSource={sources}
            rowKey="symbol"
            size="small"
            pagination={{ pageSize: 15, showTotal: (t) => `共 ${t} 个数据源` }}
            scroll={{ x: 800 }}
            locale={{ emptyText: '暂无数据' }}
          />
        )}
      </Card>
    </div>
  )
}
