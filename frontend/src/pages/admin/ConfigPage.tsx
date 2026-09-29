/**
 * 系统配置页面
 * 对应旧版 admin_portal.html #page-config
 *
 * 功能：展示系统级配置参数（策略/回测/模型/超参搜索/服务器）
 * 数据来源：GET /api/v1/admin/config
 */

import { useEffect, useState, useCallback } from 'react'
import {
  Card,
  Typography,
  Row,
  Col,
  Descriptions,
  Tag,
  Button,
  Space,
  Table,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  ReloadOutlined,
  ControlOutlined,
  SettingOutlined,
  ExperimentOutlined,
  CodeOutlined,
  CloudServerOutlined,
} from '@ant-design/icons'
import { loadSystemConfig } from '@/api/admin'
import type { SystemConfigResponse } from '@/types/api'
import LoadingSkeleton from '@/components/common/LoadingSkeleton'
import ErrorState from '@/components/common/ErrorState'

const { Title, Text } = Typography

export default function ConfigPage() {
  const [config, setConfig] = useState<SystemConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // ── 加载配置 ────────────────────────────────────────
  const loadConfig = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await loadSystemConfig()
      if (data) {
        setConfig(data)
      } else {
        setError('无法加载系统配置，请确认管理会话和后端服务状态')
      }
    } catch {
      setError('无法加载系统配置')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadConfig() }, [loadConfig])

  // ── 加载中 ──────────────────────────────────────────
  if (loading) {
    return (
      <div>
        <Title level={4} style={{ marginBottom: 16 }}>系统配置</Title>
        <LoadingSkeleton type="card" rows={6} />
      </div>
    )
  }

  // ── 错误 ────────────────────────────────────────────
  if (error && !config) {
    return (
      <div>
        <Title level={4} style={{ marginBottom: 16 }}>系统配置</Title>
        <ErrorState message={error} onRetry={loadConfig} />
      </div>
    )
  }

  if (!config) {
    return (
      <div>
        <Title level={4} style={{ marginBottom: 16 }}>系统配置</Title>
        <Card>
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
            <ControlOutlined style={{ fontSize: 36, marginBottom: 8 }} />
            <p>暂无配置数据</p>
          </div>
        </Card>
      </div>
    )
  }

  // ── 超参搜索表格列 ──────────────────────────────────
  const paramCols: ColumnsType<{ key: string; values: string }> = [
    { title: '参数', dataIndex: 'key', key: 'key', width: 140 },
    { title: '候选值', dataIndex: 'values', key: 'values' },
  ]

  const paramRows = config.param_search
    ? [
        { key: 'T (temperature)', values: config.param_search.T_values?.join(', ') || '--' },
        { key: 'top_p', values: config.param_search.top_p_values?.join(', ') || '--' },
        { key: 'lookback', values: config.param_search.lookback_values?.join(', ') || '--' },
      ]
    : []

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>系统配置</Title>
        <Button icon={<ReloadOutlined />} onClick={loadConfig} loading={loading}>
          刷新
        </Button>
      </div>

      <Row gutter={[16, 16]}>
        {/* ── 策略参数 ─────────────────────────────── */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <SettingOutlined /> 策略参数
              </Space>
            }
            size="small"
          >
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="买入阈值">
                <Tag color="blue">{(config.strategy.buy_threshold * 100).toFixed(0)}%</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="持仓天数">
                {config.strategy.hold_days} 天
              </Descriptions.Item>
              <Descriptions.Item label="止损比例">
                <Tag color="red">{(config.strategy.stop_loss * 100).toFixed(0)}%</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="止盈比例">
                <Tag color="green">{(config.strategy.take_profit * 100).toFixed(0)}%</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="单只最大仓位">
                {(config.strategy.max_single_position * 100).toFixed(0)}%
              </Descriptions.Item>
              <Descriptions.Item label="最少持仓数">
                {config.strategy.min_positions} 只
              </Descriptions.Item>
              <Descriptions.Item label="仅做多">
                <Tag color={config.strategy.long_only ? 'green' : 'default'}>
                  {config.strategy.long_only ? '是' : '否'}
                </Tag>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        {/* ── 回测参数 ─────────────────────────────── */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <ExperimentOutlined /> 回测参数
              </Space>
            }
            size="small"
          >
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="初始资金">
                {(config.backtest.default_capital / 10000).toFixed(0)} 万
              </Descriptions.Item>
              <Descriptions.Item label="佣金">
                {(config.backtest.commission_rate * 10000).toFixed(1)}‱
              </Descriptions.Item>
              <Descriptions.Item label="印花税">
                {(config.backtest.stamp_tax_rate * 1000).toFixed(1)}‰（卖出）
              </Descriptions.Item>
              <Descriptions.Item label="滑点">
                {(config.backtest.slippage_rate * 100).toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="基准指数">
                <Text code>{config.backtest.benchmark_index || '000300'}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="无风险利率">
                {(config.backtest.risk_free_rate * 100).toFixed(2)}%
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        {/* ── 模型参数 ─────────────────────────────── */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <CodeOutlined /> 模型参数
              </Space>
            }
            size="small"
          >
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="Kronos 版本">
                {config.model.kronos_version}
              </Descriptions.Item>
              <Descriptions.Item label="设备">
                {config.model.device}
              </Descriptions.Item>
              <Descriptions.Item label="序列长度">
                {config.model.training.seq_len}
              </Descriptions.Item>
              <Descriptions.Item label="预测长度">
                {config.model.training.pred_len}
              </Descriptions.Item>
              <Descriptions.Item label="Epochs">
                {config.model.training.epochs}
              </Descriptions.Item>
              <Descriptions.Item label="Batch Size">
                {config.model.training.batch_size}
              </Descriptions.Item>
              <Descriptions.Item label="学习率">
                {config.model.training.learning_rate}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        {/* ── 超参搜索 + 服务器 ────────────────────── */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <CloudServerOutlined /> 超参搜索 & 服务器
              </Space>
            }
            size="small"
          >
            {paramRows.length > 0 && (
              <Table
                columns={paramCols}
                dataSource={paramRows}
                rowKey="key"
                size="small"
                pagination={false}
                style={{ marginBottom: 16 }}
              />
            )}
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="Host">
                {config.server?.host || '0.0.0.0'}
              </Descriptions.Item>
              <Descriptions.Item label="Port">
                {config.server?.port || 8000}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>
    </div>
  )
}
