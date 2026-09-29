import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Layout, Switch, Typography, Tag, Button, Space, Tooltip } from 'antd'
import { MoonOutlined, SunOutlined, ClockCircleOutlined, LogoutOutlined, MenuOutlined } from '@ant-design/icons'
import { useAppStore } from '@/stores/useAppStore'
import { useTheme } from '@/hooks/useTheme'
import { adminLogout } from '@/api/admin'

const { Header: AntHeader } = Layout
const { Text } = Typography

/**
 * 页面标题映射
 */
const pageTitles: Record<string, string> = {
  '/user/dashboard': '仪表盘',
  '/user/stock': '个股看板',
  '/user/market': 'A股行情',
  '/user/watchlist': '我的自选',
  '/user/pool': '选股池',
  '/user/backtest': '回测中心',
  '/admin/monitor': '系统监控',
  '/admin/config': '系统配置',
  '/admin/logs': '运行日志',
  '/admin/research': '模型与策略',
}

/**
 * 顶栏
 * - 当前页面标题
 * - 实时时钟
 * - API 连接状态指示
 * - 暗色模式切换（useTheme hook → localStorage 持久化）
 */
interface HeaderProps {
  portal: 'user' | 'admin'
  mobile?: boolean
  onMenuClick?: () => void
}

export default function Header({ portal, mobile = false, onMenuClick }: HeaderProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const { isDark, toggle } = useTheme()
  const apiStatus = useAppStore((s) => s.apiStatus)
  const logout = useAppStore((s) => s.logout)
  const [clock, setClock] = useState('')

  // Clock update
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setClock(
        now.toLocaleTimeString('zh-CN', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
      )
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  // Extract title from pathname (handle sub-paths like /user/stock/:code)
  const basePath =
    '/' +
    location.pathname
      .split('/')
      .slice(1, 3)
      .filter(Boolean)
      .join('/')
  const title = pageTitles[basePath] || pageTitles[location.pathname] || 'Kronos'

  const handleLogout = async () => {
    await adminLogout()
    logout()
    navigate('/admin/login', { replace: true })
  }

  const apiTag =
    apiStatus === 'available' ? (
      <Tag color="success" style={{ margin: 0 }}>
        已连接
      </Tag>
    ) : apiStatus === 'checking' ? (
      <Tag color="warning" style={{ margin: 0 }}>
        检测中
      </Tag>
    ) : (
      <Tag color="error" style={{ margin: 0 }}>
        离线
      </Tag>
    )

  return (
    <AntHeader
      style={{
        height: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
      }}
    >
      {/* Left: page title */}
      <Space size={10}>
        {mobile && <Button type="text" icon={<MenuOutlined />} aria-label="打开导航" onClick={onMenuClick} />}
        <Text strong style={{ fontSize: 16 }}>{title}</Text>
      </Space>

      {/* Right: clock + API status + dark mode switch */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {!mobile && <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, color: 'var(--text-muted)' }}>
          <ClockCircleOutlined />
          {clock}
        </span>}

        {!mobile && <span style={{ fontSize: 13 }}>{apiTag}</span>}

        {portal === 'admin' && (
          <Tooltip title={mobile ? '退出登录' : undefined}>
            <Button type="text" icon={<LogoutOutlined />} aria-label="退出登录" onClick={handleLogout}>
              {mobile ? null : '退出登录'}
            </Button>
          </Tooltip>
        )}

        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {!mobile && <SunOutlined style={{ fontSize: 13, color: 'var(--text-muted)' }} />}
          <Switch
            size="small"
            checked={isDark}
            onChange={toggle}
          />
          {!mobile && <MoonOutlined style={{ fontSize: 13, color: 'var(--text-muted)' }} />}
        </span>
      </div>
    </AntHeader>
  )
}
