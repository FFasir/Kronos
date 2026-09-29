/**
 * Toast — 轻量消息通知
 *
 * 替换旧版 KronosUI.toast()，封装 Ant Design message。
 *
 * 用法:
 *   Toast.success('操作成功')
 *   Toast.error('操作失败')
 *   Toast.warning('请先选择股票')
 */

import { message } from 'antd'
import type { ReactNode } from 'react'

const DEFAULT_DURATION = 3

const Toast = {
  success: (content: ReactNode, duration = DEFAULT_DURATION) =>
    message.success(content, duration),

  error: (content: ReactNode, duration = DEFAULT_DURATION) =>
    message.error(content, duration),

  warning: (content: ReactNode, duration = DEFAULT_DURATION) =>
    message.warning(content, duration),

  info: (content: ReactNode, duration = DEFAULT_DURATION) =>
    message.info(content, duration),

  loading: (content: ReactNode, duration = DEFAULT_DURATION) =>
    message.loading(content, duration),

  destroy: () => message.destroy(),
}

export default Toast
