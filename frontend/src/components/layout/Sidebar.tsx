import { useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import type { MenuProps } from 'antd'
import {
  DashboardOutlined,
  AppstoreOutlined,
  ExperimentOutlined,
  MonitorOutlined,
  ControlOutlined,
  FileTextOutlined,
  FundOutlined,
  AuditOutlined,
  StarOutlined,
} from '@ant-design/icons'

const { Sider } = Layout

type MenuItem = Required<MenuProps>['items'][number]

const userMenuItems: MenuItem[] = [
  {
    key: 'group-user',
    type: 'group',
    label: '股民策略看板',
    children: [
      { key: '/user/dashboard', icon: <DashboardOutlined />, label: '仪表盘' },
      { key: '/user/market', icon: <FundOutlined />, label: 'A股行情' },
      { key: '/user/watchlist', icon: <StarOutlined />, label: '我的自选' },
      { key: '/user/pool', icon: <AppstoreOutlined />, label: '选股池' },
      { key: '/user/backtest', icon: <ExperimentOutlined />, label: '回测中心' },
    ],
  },
]

const adminMenuItems: MenuItem[] = [
  { key: '/admin/monitor', icon: <MonitorOutlined />, label: '系统监控' },
  { key: '/admin/research', icon: <AuditOutlined />, label: '模型与策略' },
  { key: '/admin/config', icon: <ControlOutlined />, label: '系统配置' },
  { key: '/admin/logs', icon: <FileTextOutlined />, label: '运行日志' },
]

interface SidebarProps {
  portal: 'user' | 'admin'
  embedded?: boolean
  onNavigate?: () => void
}

/**
 * 左侧导航侧边栏
 * - 固定 230px 宽
 * - Menu 根据当前路由自动高亮
 * - Logo 区域 + 导航菜单
 */
export default function Sidebar({ portal, embedded = false, onNavigate }: SidebarProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const isAdmin = portal === 'admin'

  const handleClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key)
    onNavigate?.()
  }

  // Determine selected key: match the longest path prefix
  const selectedKey = [location.pathname]

  return (
    <Sider
      width={230}
      style={{
        position: embedded ? 'relative' : 'fixed',
        top: 0,
        left: 0,
        bottom: embedded ? undefined : 0,
        minHeight: '100vh',
        zIndex: 100,
        overflow: 'auto',
      }}
    >
      {/* Logo */}
      <div
        style={{
          height: 56,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 20px',
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div
          style={{
            width: 34,
            height: 34,
            background: 'var(--primary)',
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontSize: 18,
            fontWeight: 700,
            flexShrink: 0,
          }}
        >
          <FundOutlined />
        </div>
        <div style={{ lineHeight: 1.3 }}>
          <div style={{ fontSize: 15, fontWeight: 600 }}>
            {isAdmin ? 'Kronos 管理中心' : 'Kronos'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
            {isAdmin ? '系统运维后台' : '股票分析策略系统'}
          </div>
        </div>
      </div>

      {/* Navigation */}
      <Menu
        mode="inline"
        selectedKeys={selectedKey}
        items={isAdmin ? adminMenuItems : userMenuItems}
        onClick={handleClick}
        style={{ borderInlineEnd: 'none', marginTop: 8 }}
      />
    </Sider>
  )
}
