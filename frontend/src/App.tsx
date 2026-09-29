import { useEffect } from 'react'
import { ConfigProvider, theme as antdTheme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { RouterProvider } from 'react-router-dom'
import { useAppStore } from '@/stores/useAppStore'
import { useApi } from '@/hooks/useApi'
import { useTheme } from '@/hooks/useTheme'
import { router } from '@/router'

/**
 * Kronos 股票分析策略系统 — React App 根组件
 *
 * 职责:
 *  1. 初始化 API 可达性探测（useApi）
 *  2. 初始化暗色模式检测 + 系统偏好跟随（useTheme）
 *  3. Ant Design ConfigProvider (主题 + 中文)
 *  4. React Router v7 RouterProvider
 */
export default function App() {
  const theme = useAppStore((s) => s.theme)

  // 初始化 API 探测
  useApi()

  // 初始化暗色模式跟随
  useTheme()

  // Ensure data-theme attribute is set on mount and on theme change
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme === 'dark' ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: {
          colorPrimary: '#4f6ef6',
          borderRadius: 10,
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif",
        },
      }}
    >
      <RouterProvider router={router} />
    </ConfigProvider>
  )
}
