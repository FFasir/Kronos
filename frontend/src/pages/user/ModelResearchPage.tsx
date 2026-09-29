import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Row,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { DeleteOutlined, PlusOutlined, ReloadOutlined, SafetyCertificateOutlined, SwapOutlined } from '@ant-design/icons'
import EquityCurveChart from '@/components/charts/EquityCurveChart'
import { getResearchSummary } from '@/api/research'
import { activateRegisteredModel, deleteRegisteredModel, getRegisteredModels, registerCustomModel } from '@/api/admin'
import type { ModelComparison, ModelRegistryResponse, RegisteredModel, ResearchSummary } from '@/types/research'

import './model-research.css'

const { Title, Text, Paragraph } = Typography

const pct = (value: number | null | undefined) =>
  value == null ? '--' : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`

const number = (value: number | null | undefined, digits = 2) =>
  value == null ? '--' : value.toFixed(digits)

export default function ModelResearchPage() {
  const [data, setData] = useState<ResearchSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [registry, setRegistry] = useState<ModelRegistryResponse | null>(null)
  const [modelModalOpen, setModelModalOpen] = useState(false)
  const [modelAction, setModelAction] = useState<string | null>(null)
  const [form] = Form.useForm()

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    const [response, modelRegistry] = await Promise.all([getResearchSummary(), getRegisteredModels()])
    if (!response) setError('研究结果不可用，请检查后端产物与服务状态')
    setData(response)
    setRegistry(modelRegistry)
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load])

  const selected = data?.comparison.find((item) => item.selected)
  const equity = data?.selected_holdout_equity ?? []
  const chart = {
    dates: equity.map((item) => item.date),
    strategy: equity.map((item) => Number(item.equity.toFixed(4))),
    benchmark: equity.map((item) => Number(item.benchmark_equity.toFixed(4))),
  }

  const columns: ColumnsType<ModelComparison> = [
    {
      title: '模型',
      dataIndex: 'name',
      width: 190,
      fixed: 'left',
      render: (value, record) => (
        <Space size={6}>
          <Text strong={record.selected}>{value}</Text>
          {record.selected && <Tag color="success">生产候选</Tag>}
        </Space>
      ),
    },
    { title: '开发期净收益', render: (_, row) => pct(row.development.net_compounded_return), width: 130 },
    { title: '留出期净收益', render: (_, row) => <Text strong={row.selected}>{pct(row.holdout.net_compounded_return)}</Text>, width: 130 },
    { title: '留出期超额', render: (_, row) => pct(row.holdout.total_relative_excess_return), width: 120 },
    { title: '最大回撤', render: (_, row) => pct(row.holdout.max_drawdown), width: 110 },
    { title: 'Sharpe', render: (_, row) => number(row.holdout.sharpe_zero_rf, 3), width: 90 },
    { title: '方向准确率', render: (_, row) => pct(row.holdout.directional_accuracy), width: 120 },
    { title: '交易数', render: (_, row) => row.holdout.trade_count ?? '--', width: 80 },
  ]

  const activate = async (model: RegisteredModel) => {
    setModelAction(model.key)
    const result = await activateRegisteredModel(model.key)
    if (result?.success) {
      message.success(result.message)
      await load()
    } else {
      message.error('模型切换失败，请检查权重完整性和服务日志')
    }
    setModelAction(null)
  }

  const register = async () => {
    const values = await form.validateFields()
    setModelAction('register')
    const result = await registerCustomModel(values)
    if (result) {
      message.success('模型已登记，完成研究验证后再切换生产')
      setModelModalOpen(false)
      form.resetFields()
      await load()
    } else {
      message.error('登记失败，路径必须位于 models/ 且资产完整')
    }
    setModelAction(null)
  }

  const remove = async (model: RegisteredModel) => {
    setModelAction(model.key)
    try {
      await deleteRegisteredModel(model.key)
      message.success('登记已移除，权重文件未删除')
      await load()
    } catch {
      message.error('无法移除当前模型或服务拒绝了请求')
    }
    setModelAction(null)
  }

  const registryColumns: ColumnsType<RegisteredModel> = [
    {
      title: '模型',
      width: 210,
      render: (_, model) => <div><Text strong>{model.name}</Text><br /><Text type="secondary">{model.version}</Text></div>,
    },
    { title: '来源', dataIndex: 'kind', width: 90, render: (value) => value === 'builtin' ? '内置' : '自定义' },
    {
      title: '研究状态',
      width: 120,
      render: (_, model) => model.research_status === 'selected'
        ? <Tag color="success">当前推荐</Tag>
        : model.research_status === 'unverified' ? <Tag color="warning">未验证</Tag> : <Tag>未入选</Tag>,
    },
    {
      title: '资产状态',
      width: 110,
      render: (_, model) => model.ready ? <Tag color="success">完整</Tag> : <Tag color="error">缺失</Tag>,
    },
    {
      title: '权重路径',
      dataIndex: 'predictor_path',
      ellipsis: true,
      width: 360,
      render: (value) => <Text code title={value}>{value}</Text>,
    },
    {
      title: '生产状态',
      fixed: 'right',
      width: 180,
      render: (_, model) => model.active ? <Tag color="processing">使用中</Tag> : (
        <Space size={4}>
          <Popconfirm
            title="切换生产模型"
            description={model.research_status === 'selected' ? '后续预测任务将加载该权重。' : '该模型不是当前研究推荐，确认仍要切换？'}
            onConfirm={() => activate(model)}
          >
            <Button size="small" icon={<SwapOutlined />} disabled={!model.ready} loading={modelAction === model.key}>切换</Button>
          </Popconfirm>
          {model.kind === 'custom' && (
            <Popconfirm title="仅移除登记，不删除权重文件" onConfirm={() => remove(model)}>
              <Button size="small" danger aria-label="移除模型登记" icon={<DeleteOutlined />} />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  if (loading) return <div className="research-loading"><Spin size="large" /></div>
  if (error || !data || !selected) {
    return <Alert type="error" showIcon message={error || '研究结果不完整'} action={<Button icon={<ReloadOutlined />} onClick={load}>重试</Button>} />
  }

  const holdout = selected.holdout
  const stats = [
    ['留出期净收益', pct(holdout.net_compounded_return), 'positive'],
    ['相对 ETF 超额', pct(holdout.total_relative_excess_return), 'positive'],
    ['最大回撤', pct(holdout.max_drawdown), 'neutral'],
    ['Sharpe', number(holdout.sharpe_zero_rf, 3), 'neutral'],
    ['方向准确率', pct(holdout.directional_accuracy), 'neutral'],
    ['在场比例', pct(holdout.in_market_ratio), 'neutral'],
  ]

  return (
    <div className="research-page">
      <section className="research-heading">
        <div>
          <Space size={8} wrap>
            <Tag color={data.production_model.key === data.recommended_model_key ? 'success' : 'warning'} icon={<SafetyCertificateOutlined />}>
              {data.production_model.key === data.recommended_model_key ? '研究推荐已启用' : '非推荐模型使用中'}
            </Tag>
            <Text type="secondary">研究截止 {data.as_of}</Text>
          </Space>
          <Title level={3}>{data.production_model.name}</Title>
          <Paragraph>{data.production_model.purpose}</Paragraph>
        </div>
        <div className="model-version">
          <Text type="secondary">模型版本</Text>
          <Text strong>{data.production_model.version}</Text>
        </div>
      </section>

      <Alert
        showIcon
        type="warning"
        message="研究结果不等于实盘承诺"
        description={data.risk_notice}
      />

      <section className="research-section model-registry-section">
        <div className="section-title">
          <div>
            <Title level={4}>生产模型管理</Title>
            <Text type="secondary">切换只影响后续预测任务；正在执行的任务不会被中途替换</Text>
          </div>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModelModalOpen(true)}>登记模型</Button>
        </div>
        <Table
          rowKey="key"
          columns={registryColumns}
          dataSource={registry?.models ?? []}
          pagination={false}
          size="small"
          scroll={{ x: 1070 }}
          rowClassName={(row) => row.active ? 'active-model-row' : ''}
        />
      </section>

      <Row gutter={[12, 12]} className="research-stats">
        {stats.map(([label, value, tone]) => (
          <Col xs={12} md={8} xl={4} key={label}>
            <Card size="small" className="metric-card">
              <Text type="secondary">{label}</Text>
              <div className={`metric-value ${tone}`}>{value}</div>
            </Card>
          </Col>
        ))}
      </Row>

      <section className="research-section">
        <div className="section-title">
          <div>
            <Title level={4}>留出期净值</Title>
            <Text type="secondary">2025-10 至 2026-06，已计双边佣金与滑点</Text>
          </div>
          <Tag>{holdout.trading_days} 个交易日</Tag>
        </div>
        <div className="chart-surface">
          <EquityCurveChart
            dates={chart.dates}
            strategyValues={chart.strategy}
            benchmarkValues={chart.benchmark}
            benchmarkLabel="512100 买入持有"
            height={340}
          />
        </div>
      </section>

      <section className="research-section">
        <div className="section-title">
          <div>
            <Title level={4}>同口径模型比较</Title>
            <Text type="secondary">所有模型使用相同信号、成交、成本与前月滚动选参规则</Text>
          </div>
        </div>
        <Table
          rowKey="key"
          columns={columns}
          dataSource={data.comparison}
          pagination={false}
          size="middle"
          scroll={{ x: 1070 }}
          rowClassName={(row) => row.selected ? 'selected-model-row' : ''}
        />
      </section>

      <section className="research-section method-section">
        <Title level={4}>冻结实验口径</Title>
        <Descriptions column={{ xs: 1, md: 2 }} size="small" bordered>
          <Descriptions.Item label="信号标的">{data.experiment.signal_asset}</Descriptions.Item>
          <Descriptions.Item label="成交标的">{data.experiment.execution_asset}</Descriptions.Item>
          <Descriptions.Item label="开发期">{data.experiment.development_period}</Descriptions.Item>
          <Descriptions.Item label="留出期">{data.experiment.holdout_period}</Descriptions.Item>
          <Descriptions.Item label="滚动选参" span={2}>{data.experiment.selection}</Descriptions.Item>
          <Descriptions.Item label="推理规则" span={2}>{data.experiment.inference}</Descriptions.Item>
          <Descriptions.Item label="成交规则" span={2}>{data.experiment.execution}</Descriptions.Item>
          <Descriptions.Item label="成本" span={2}>{data.experiment.costs}</Descriptions.Item>
        </Descriptions>
      </section>

      <Modal
        title="登记已有 Kronos 模型"
        open={modelModalOpen}
        onCancel={() => setModelModalOpen(false)}
        onOk={register}
        confirmLoading={modelAction === 'register'}
        okText="校验并登记"
      >
        <Alert type="info" showIcon message="仅登记服务器项目 models/ 目录中的现有资产，不上传或复制权重。" style={{ marginBottom: 16 }} />
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="显示名称" rules={[{ required: true }]}><Input placeholder="例如：月度滚动模型 2026-07" /></Form.Item>
          <Form.Item name="version" label="版本标识" rules={[{ required: true }]}><Input placeholder="例如：rolling-202607" /></Form.Item>
          <Form.Item name="predictor_path" label="Predictor 目录" rules={[{ required: true }]}><Input placeholder="models/.../predictor/checkpoints/best_model" /></Form.Item>
          <Form.Item name="tokenizer_path" label="Tokenizer 目录" rules={[{ required: true }]}><Input placeholder="models/.../tokenizer/checkpoints/best_model" /></Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
