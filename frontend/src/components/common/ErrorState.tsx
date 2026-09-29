/**
 * ErrorState — 错误状态组件（含重试按钮）
 *
 * 替换旧版 KronosUI.showError()，封装 Ant Design Result。
 *
 * 用法:
 *   <ErrorState message="加载失败" onRetry={() => refetch()} />
 */

import { Result, Button } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

export interface ErrorStateProps {
  /** 错误提示 */
  message?: string
  /** 详细信息 */
  detail?: string
  /** 重试回调 */
  onRetry?: () => void
}

export default function ErrorState({
  message = '加载失败，请检查网络连接',
  detail,
  onRetry,
}: ErrorStateProps) {
  return (
    <Result
      status="warning"
      title={message}
      subTitle={detail}
      extra={
        onRetry && (
          <Button type="primary" icon={<ReloadOutlined />} onClick={onRetry}>
            重试
          </Button>
        )
      }
    />
  )
}
