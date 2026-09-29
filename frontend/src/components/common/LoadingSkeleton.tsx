/**
 * LoadingSkeleton — 骨架屏组件
 *
 * 替换旧版 KronosUI.showLoading()，封装 Ant Design Skeleton。
 *
 * 用法:
 *   <LoadingSkeleton rows={5} />              // 表格占位
 *   <LoadingSkeleton type="card" />           // 卡片占位
 *   <LoadingSkeleton type="chart" />          // 图表占位
 */

import { Skeleton, Card } from 'antd'

export interface LoadingSkeletonProps {
  /** 骨架行数（type="table" 时有效） */
  rows?: number
  /** 类型: 表格 | 卡片 | 图表 */
  type?: 'table' | 'card' | 'chart'
  /** 是否显示 */
  active?: boolean
}

export default function LoadingSkeleton({ rows = 5, type = 'table', active = true }: LoadingSkeletonProps) {
  if (type === 'card') {
    return (
      <Card style={{ background: 'var(--surface-1)' }}>
        <Skeleton active={active} paragraph={{ rows: 3 }} />
      </Card>
    )
  }

  if (type === 'chart') {
    return (
      <div
        className="chart-container"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 300,
        }}
      >
        <Skeleton active={active} paragraph={{ rows: 4 }} title={{ width: '40%' }} />
      </div>
    )
  }

  // type === 'table'
  return (
    <div style={{ padding: '16px 0' }}>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton
          key={i}
          active={active}
          paragraph={false}
          title={{ width: `${70 + Math.floor(Math.random() * 30)}%` }}
          style={{ marginBottom: 12 }}
        />
      ))}
    </div>
  )
}
