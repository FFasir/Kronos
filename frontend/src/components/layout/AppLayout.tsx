import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Drawer, Grid, Layout } from 'antd'
import Sidebar from './Sidebar'
import Header from './Header'

const { Content } = Layout

/**
 * 整体布局壳
 * 左侧固定侧边栏 + 右侧 (顶栏 + 内容区)
 */
interface AppLayoutProps {
  portal: 'user' | 'admin'
}

export default function AppLayout({ portal }: AppLayoutProps) {
  const screens = Grid.useBreakpoint()
  const mobile = !screens.md
  const [navigationOpen, setNavigationOpen] = useState(false)

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {mobile ? (
        <Drawer
          placement="left"
          width={230}
          open={navigationOpen}
          onClose={() => setNavigationOpen(false)}
          closable={false}
          styles={{ body: { padding: 0 } }}
        >
          <Sidebar portal={portal} embedded onNavigate={() => setNavigationOpen(false)} />
        </Drawer>
      ) : <Sidebar portal={portal} />}
      <Layout style={{ marginLeft: mobile ? 0 : 230 }}>
        <Header portal={portal} mobile={mobile} onMenuClick={() => setNavigationOpen(true)} />
        <Content
          style={{
            padding: mobile ? '14px 12px' : '20px 24px',
            background: 'var(--bg)',
            minHeight: 'calc(100vh - 56px)',
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
