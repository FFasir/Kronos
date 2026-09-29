/**
 * 策略设置页面
 * 对应旧版 user_portal.html #page-settings
 *
 * 功能：策略风格预设选择 → 选股阈值、风控参数、止损止盈比例配置
 * 数据保存：策略参数为全系统级配置（服务器 config/system.yaml 决定），
 * 普通用户端不提供服务器写入；此处只保存在本机浏览器并如实说明。
 */

import { useEffect, useState, useCallback } from 'react'
import {
  Card,
  Form,
  InputNumber,
  Slider,
  Button,
  Switch,
  Typography,
  Space,
  Divider,
  Spin,
  Tooltip,
  Row,
  Col,
  Radio,
} from 'antd'
import {
  SaveOutlined,
  ReloadOutlined,
  QuestionCircleOutlined,
  CloudUploadOutlined,
  SafetyCertificateOutlined,
  FundOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import type { StrategyConfig, StrategyPreset } from '@/types/api'
import { getStrategyPresets } from '@/api/prediction'
import Toast from '@/components/common/Toast'

const { Title, Text, Paragraph } = Typography

/** 默认策略参数 — 与 config/system.yaml 对齐 */
const DEFAULT_STRATEGY: StrategyConfig = {
  buy_threshold: 0.65,
  long_only: true,
  hold_days: 5,
  stop_loss: -0.08,
  take_profit: 0.15,
  max_single_position: 0.3,
  min_positions: 5,
}

/** 预设映射到表单值的本地降级（后端不可用时使用） */
const LOCAL_PRESETS: Record<string, Partial<StrategyConfig>> = {
  conservative: {
    buy_threshold: 0.75,
    long_only: true,
    hold_days: 3,
    stop_loss: -0.05,
    take_profit: 0.10,
    max_single_position: 0.15,
    min_positions: 8,
  },
  balanced: {
    buy_threshold: 0.65,
    long_only: true,
    hold_days: 5,
    stop_loss: -0.08,
    take_profit: 0.15,
    max_single_position: 0.3,
    min_positions: 5,
  },
  aggressive: {
    buy_threshold: 0.55,
    long_only: true,
    hold_days: 10,
    stop_loss: -0.15,
    take_profit: 0.30,
    max_single_position: 0.5,
    min_positions: 3,
  },
}

/** 预设显示信息 */
const PRESET_META: Record<string, { label: string; icon: React.ReactNode; desc: string; color: string }> = {
  conservative: {
    label: '保守',
    icon: <SafetyCertificateOutlined style={{ fontSize: 28 }} />,
    desc: '低风险、小仓位、严格止损，适合稳健投资者',
    color: '#22b07d',
  },
  balanced: {
    label: '均衡',
    icon: <FundOutlined style={{ fontSize: 28 }} />,
    desc: '中等风险与收益，适合大多数投资者',
    color: '#4f6ef6',
  },
  aggressive: {
    label: '激进',
    icon: <ThunderboltOutlined style={{ fontSize: 28 }} />,
    desc: '高仓位、宽止损、追求最大收益，适合风险偏好者',
    color: '#e74c5e',
  },
}

/** 参数推荐范围说明 */
const PARAM_RANGES: Record<string, string> = {
  buy_threshold: '推荐范围 0.55 ~ 0.80。值越高选股越严格，入选数量越少；值过低可能纳入低质量标的。',
  hold_days: '推荐范围 3 ~ 10 天。过短易频繁交易增加成本，过长可能错过止盈时机。',
  stop_loss: '推荐范围 -5% ~ -15%。过窄容易被正常波动止损出局，过宽则单笔亏损风险大。',
  take_profit: '推荐范围 10% ~ 30%。需与止损比例协调，一般止盈/止损 ≥ 2:1。',
  max_single_position: '推荐范围 15% ~ 50%。过小分散收益，过大集中风险。',
  min_positions: '推荐范围 3 ~ 10 只。过少集中风险，过多收益趋于平均。',
}

export default function SettingsPage() {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [savingBackend, setSavingBackend] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [presets, setPresets] = useState<StrategyPreset[]>([])

  // ── 加载配置 ────────────────────────────────────────
  const loadConfig = useCallback(async () => {
    setLoading(true)
    try {
      // 加载策略预设
      const presetResp = await getStrategyPresets()
      if (presetResp?.presets) {
        setPresets(presetResp.presets)
      }

      // 加载本地配置
      const raw = localStorage.getItem('kronos_user_config')
      const parsed = raw ? JSON.parse(raw) as Partial<StrategyConfig> : {}
      form.setFieldsValue({ ...DEFAULT_STRATEGY, ...parsed })
    } catch {
      form.setFieldsValue(DEFAULT_STRATEGY)
      Toast.warning('本地配置读取失败，已使用默认参数')
    } finally {
      setLoading(false)
    }
  }, [form])

  useEffect(() => { loadConfig() }, [loadConfig])

  // ── 选择预设 ────────────────────────────────────────
  const handlePresetSelect = useCallback((presetName: string) => {
    setSelectedPreset(presetName)

    // 优先使用后端预设数据，否则使用本地降级
    const backendPreset = presets.find((p) => p.name === presetName)
    if (backendPreset) {
      // 后端预设字段映射到表单
      form.setFieldsValue({
        buy_threshold: backendPreset.prob_up_threshold,
        // 后端预设不包含所有字段，其余保持默认
      })
      Toast.info(`已应用「${PRESET_META[presetName]?.label ?? presetName}」预设`)
      return
    }

    const localPreset = LOCAL_PRESETS[presetName]
    if (localPreset) {
      form.setFieldsValue({ ...DEFAULT_STRATEGY, ...localPreset })
      Toast.info(`已应用「${PRESET_META[presetName]?.label ?? presetName}」预设`)
    }
  }, [form, presets])

  // ── 保存配置到本地 ──────────────────────────────────
  const handleSave = useCallback(async () => {
    try {
      const values = await form.validateFields()
      setSaving(true)
      localStorage.setItem('kronos_user_config', JSON.stringify(values))
      Toast.success('策略参数已保存到当前浏览器')
    } catch {
      Toast.error('请检查表单填写')
    } finally {
      setSaving(false)
    }
  }, [form])

  // ── “保存到后端”已诚实化：策略参数为全系统级配置，普通用户端不提供服务器写入 ──
  // 后端 /admin/config 需管理员鉴权且仅做校验、并不真正持久化到 config/system.yaml，
  // 因此不再调用任何后端接口：只保存本机并如实说明规则。
  const handleSaveToBackend = useCallback(async () => {
    try {
      const values = await form.validateFields()
      setSavingBackend(true)
      localStorage.setItem('kronos_user_config', JSON.stringify(values))
      Toast.warning('已保存到本机浏览器。策略参数为全系统级配置（由服务器 config/system.yaml 决定），普通用户端不提供服务器写入；如需影响全系统，请由管理员在服务器上修改配置文件。')
    } catch {
      Toast.error('请检查表单填写')
    } finally {
      setSavingBackend(false)
    }
  }, [form])

  // ── 重置默认 ────────────────────────────────────────
  const handleReset = useCallback(() => {
    setSelectedPreset(null)
    form.setFieldsValue(DEFAULT_STRATEGY)
    Toast.info('已恢复默认参数')
  }, [form])

  // ── 加载中 ──────────────────────────────────────────
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 80 }}>
        <Spin size="large" tip="加载配置中…" />
      </div>
    )
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>策略设置</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={handleReset}>恢复默认</Button>
          <Button icon={<SaveOutlined />} onClick={handleSave} loading={saving}>
            保存本地
          </Button>
          <Button type="primary" icon={<CloudUploadOutlined />} onClick={handleSaveToBackend} loading={savingBackend}>
            保存设置(仅本机)
          </Button>
        </Space>
      </div>

      {/* ── 策略风格预设选择 ──────────────────────── */}
      <Card title="策略风格预设" style={{ marginBottom: 16 }}>
        <Radio.Group
          value={selectedPreset}
          onChange={(e) => handlePresetSelect(e.target.value)}
          style={{ width: '100%' }}
        >
          <Row gutter={[16, 16]}>
            {['conservative', 'balanced', 'aggressive'].map((key) => {
              const meta = PRESET_META[key]
              return (
                <Col xs={24} sm={8} key={key}>
                  <Card
                    hoverable
                    size="small"
                    style={{
                      textAlign: 'center',
                      border: selectedPreset === key ? `2px solid ${meta.color}` : '1px solid var(--border)',
                      background: selectedPreset === key ? `${meta.color}08` : 'var(--surface-1)',
                    }}
                    onClick={() => handlePresetSelect(key)}
                  >
                    <div style={{ color: meta.color, marginBottom: 8 }}>{meta.icon}</div>
                    <Radio value={key} style={{ fontSize: 15, fontWeight: 600 }}>
                      {meta.label}
                    </Radio>
                    <Paragraph
                      type="secondary"
                      style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}
                    >
                      {meta.desc}
                    </Paragraph>
                  </Card>
                </Col>
              )
            })}
          </Row>
        </Radio.Group>
      </Card>

      {/* ── 选股阈值 ───────────────────────────────── */}
      <Card title="选股参数" style={{ marginBottom: 16 }}>
        <Form form={form} layout="vertical" initialValues={DEFAULT_STRATEGY}>
          <Form.Item
            name="buy_threshold"
            label={
              <Space>
                上涨概率阈值
                <Tooltip title={PARAM_RANGES.buy_threshold}>
                  <QuestionCircleOutlined style={{ color: 'var(--text-muted)' }} />
                </Tooltip>
              </Space>
            }
          >
            <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
              <Slider
                style={{ flex: 1 }}
                min={0.5}
                max={0.95}
                step={0.01}
                marks={{ 0.5: '0.5', 0.65: '0.65', 0.8: '0.8', 0.95: '0.95' }}
                tooltip={{ formatter: (v) => `${((v ?? 0.65) * 100).toFixed(0)}%` }}
              />
              <Form.Item name="buy_threshold" noStyle>
                <InputNumber min={0.5} max={0.95} step={0.01} style={{ width: 80 }} />
              </Form.Item>
            </div>
          </Form.Item>

          <Form.Item name="long_only" label="仅做多" valuePropName="checked">
            <Switch />
            <Text type="secondary" style={{ marginLeft: 8 }}>开启后仅生成买入/持有信号，不生成卖出信号</Text>
          </Form.Item>

          <Form.Item
            name="hold_days"
            label={
              <Space>
                持仓天数
                <Tooltip title={PARAM_RANGES.hold_days}>
                  <QuestionCircleOutlined style={{ color: 'var(--text-muted)' }} />
                </Tooltip>
              </Space>
            }
          >
            <InputNumber min={1} max={30} step={1} addonAfter="天" style={{ width: 160 }} />
          </Form.Item>
        </Form>
      </Card>

      {/* ── 风控参数 ───────────────────────────────── */}
      <Card title="风控参数" style={{ marginBottom: 16 }}>
        <Form form={form} layout="vertical" initialValues={DEFAULT_STRATEGY}>
          <Form.Item
            name="stop_loss"
            label={
              <Space>
                止损比例
                <Tooltip title={PARAM_RANGES.stop_loss}>
                  <QuestionCircleOutlined style={{ color: 'var(--text-muted)' }} />
                </Tooltip>
              </Space>
            }
          >
            <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
              <Slider
                style={{ flex: 1 }}
                min={-0.2}
                max={-0.02}
                step={0.01}
                marks={{ [-0.2]: '-20%', [-0.08]: '-8%', [-0.05]: '-5%', [-0.02]: '-2%' }}
                tooltip={{ formatter: (v) => `${((v ?? -0.08) * 100).toFixed(0)}%` }}
              />
              <Form.Item name="stop_loss" noStyle>
                <InputNumber min={-0.2} max={-0.02} step={0.01} style={{ width: 80 }} />
              </Form.Item>
            </div>
          </Form.Item>

          <Form.Item
            name="take_profit"
            label={
              <Space>
                止盈比例
                <Tooltip title={PARAM_RANGES.take_profit}>
                  <QuestionCircleOutlined style={{ color: 'var(--text-muted)' }} />
                </Tooltip>
              </Space>
            }
          >
            <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
              <Slider
                style={{ flex: 1 }}
                min={0.05}
                max={0.5}
                step={0.01}
                marks={{ [0.05]: '5%', [0.15]: '15%', [0.3]: '30%', [0.5]: '50%' }}
                tooltip={{ formatter: (v) => `${((v ?? 0.15) * 100).toFixed(0)}%` }}
              />
              <Form.Item name="take_profit" noStyle>
                <InputNumber min={0.05} max={0.5} step={0.01} style={{ width: 80 }} />
              </Form.Item>
            </div>
          </Form.Item>

          <Divider />

          <Form.Item
            name="max_single_position"
            label={
              <Space>
                单只最大仓位
                <Tooltip title={PARAM_RANGES.max_single_position}>
                  <QuestionCircleOutlined style={{ color: 'var(--text-muted)' }} />
                </Tooltip>
              </Space>
            }
          >
            <InputNumber min={0.05} max={1} step={0.05} style={{ width: 160 }} />
          </Form.Item>

          <Form.Item
            name="min_positions"
            label={
              <Space>
                最少持仓数
                <Tooltip title={PARAM_RANGES.min_positions}>
                  <QuestionCircleOutlined style={{ color: 'var(--text-muted)' }} />
                </Tooltip>
              </Space>
            }
          >
            <InputNumber min={1} max={20} step={1} addonAfter="只" style={{ width: 140 }} />
          </Form.Item>
        </Form>
      </Card>

      {/* ── 底部操作 ───────────────────────────────── */}
      <div style={{ textAlign: 'right' }}>
        <Space>
          <Button onClick={handleReset}>恢复默认</Button>
          <Button icon={<SaveOutlined />} onClick={handleSave} loading={saving}>
            保存本地
          </Button>
          <Button type="primary" icon={<CloudUploadOutlined />} onClick={handleSaveToBackend} loading={savingBackend} size="large">
            保存设置(仅本机)
          </Button>
        </Space>
      </div>
    </div>
  )
}
