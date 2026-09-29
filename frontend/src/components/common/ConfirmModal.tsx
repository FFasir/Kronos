/**
 * ConfirmModal — 确认弹窗组件
 *
 * 替换旧版 `confirm()` 调用，封装 Ant Design Modal.confirm。
 *
 * 用法:
 *   ConfirmModal.show({ title: '确认删除？', onOk: () => handleDelete() })
 */

import { Modal } from 'antd'
import { ExclamationCircleOutlined } from '@ant-design/icons'

export interface ConfirmOptions {
  title: string
  content?: string
  okText?: string
  cancelText?: string
  danger?: boolean
  onOk: () => void | Promise<void>
  onCancel?: () => void
}

const ConfirmModal = {
  show: ({
    title,
    content,
    okText = '确认',
    cancelText = '取消',
    danger = false,
    onOk,
    onCancel,
  }: ConfirmOptions) => {
    Modal.confirm({
      title,
      content,
      icon: <ExclamationCircleOutlined />,
      okText,
      cancelText,
      okButtonProps: danger ? { danger: true } : { type: 'primary' },
      onOk: async () => {
        await onOk()
      },
      onCancel,
    })
  },

  delete: (title: string, onOk: () => void | Promise<void>) => {
    ConfirmModal.show({
      title,
      content: '此操作不可撤销',
      okText: '删除',
      danger: true,
      onOk,
    })
  },
}

export default ConfirmModal
