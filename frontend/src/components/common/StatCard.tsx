/**
 * StatCard — 统计卡片组件
 *
 * 替换旧版 .stat-card CSS 类，封装 Ant Design Card + Statistic。
 * 支持标题、数值、趋势箭头、加载骨架、空值占位。
 *
 * 用法:
 *   <StatCard title="上证指数" value={3368.52} change={1.25} loading={false} />
 */

import { Card, Statistic, Skeleton, Typography } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined } from '@ant-design/icons'

const { Text } = Typography

export interface StatCardProps {
  /** 卡片标题 */
  title: string
  /** 数值 */
  value: number | string | null | undefined
  /** 变化百分比 */
  change?: number | null
  /** 后缀 */
  suffix?: string
  /** 前缀 */
  prefix?: React.ReactNode
  /** 数值精度 */
  precision?: number
  /** 加载中 */
  loading?: boolean
  /** 额外内容（底部） */
  extra?: React.ReactNode
  /** 点击回调 */
  onClick?: () => void
}

export default function StatCard({
  title,
  value,
  change,
  suffix,
  prefix,
  precision = 2,
  loading = false,
  extra,
  onClick,
}: StatCardProps) {
  // 变化颜色（A 股约定：红涨绿跌）
  const changeColor =
    change == null ? undefined
    : change > 0 ? 'var(--danger)'
    : change < 0 ? 'var(--success)'
    : 'var(--text-muted)'

  const changeIcon =
    change == null ? null
    : change > 0 ? <ArrowUpOutlined />
    : change < 0 ? <ArrowDownOutlined />
    : <MinusOutlined />

  const changeText =
    change == null ? '--'
    : `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`

  return (
    <Card
      hoverable={!!onClick}
      onClick={onClick}
      style={{ background: 'var(--surface-1)', borderRadius: 'var(--radius)' }}
    >
      {loading ? (
        <Skeleton active paragraph={{ rows: 1 }} title={{ width: '60%' }} />
      ) : (
        <>
          <Text type="secondary" style={{ fontSize: 13, marginBottom: 8, display: 'block' }}>
            {title}
          </Text>
          <Statistic
            value={value ?? '--'}
            precision={precision}
            prefix={prefix}
            suffix={suffix}
            valueStyle={{ fontSize: 28, fontWeight: 600, color: 'var(--text)' }}
          />
          {change != null && (
            <span style={{ color: changeColor, fontSize: 13, fontWeight: 500 }}>
              {changeIcon} {changeText}
            </span>
          )}
          {extra && <div style={{ marginTop: 8 }}>{extra}</div>}
        </>
      )}
    </Card>
  )
}
