/**
 * EmptyState — 空状态占位组件
 *
 * 替换旧版 KronosUI.showEmpty()，封装 Ant Design Empty。
 *
 * 用法:
 *   <EmptyState message="暂无数据" />
 *   <EmptyState message="搜索无结果" icon="search" />
 */

import { Empty } from 'antd'
import type { ReactNode } from 'react'

export interface EmptyStateProps {
  /** 提示文案 */
  message?: string
  /** 自定义图标 */
  icon?: ReactNode
  /** 图片样式 */
  imageStyle?: React.CSSProperties
  /** 额外操作按钮 */
  children?: ReactNode
}

export default function EmptyState({
  message = '暂无数据',
  icon,
  imageStyle = { height: 80 },
  children,
}: EmptyStateProps) {
  return (
    <div
      style={{
        textAlign: 'center',
        padding: '60px 0',
        color: 'var(--text-muted)',
      }}
    >
      <Empty
        description={message}
        image={icon || Empty.PRESENTED_IMAGE_SIMPLE}
        imageStyle={imageStyle}
      >
        {children}
      </Empty>
    </div>
  )
}
