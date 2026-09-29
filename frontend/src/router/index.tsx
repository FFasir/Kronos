import { createBrowserRouter, Navigate } from 'react-router-dom'
import AppLayout from '@/components/layout/AppLayout'
import AdminGuard from '@/components/layout/AdminGuard'
import DashboardPage from '@/pages/user/DashboardPage'
import StockDetailPage from '@/pages/user/StockDetailPage'
import StockPoolPage from '@/pages/user/StockPoolPage'
import WatchlistPage from '@/pages/user/WatchlistPage'
import BacktestPage from '@/pages/user/BacktestPage'
import MarketPage from '@/pages/user/MarketPage'
import ModelResearchPage from '@/pages/user/ModelResearchPage'
import MonitorPage from '@/pages/admin/MonitorPage'
import ConfigPage from '@/pages/admin/ConfigPage'
import LogsPage from '@/pages/admin/LogsPage'
import AdminLoginPage from '@/pages/admin/AdminLoginPage'

/**
 * Kronos 路由配置
 *
 * /user/*   → 股民策略看板（5 页面，免登录）
 * /admin/*  → 管理后台（3 页面，AdminGuard 登录守卫）
 * /         → 重定向到 /user/dashboard
 */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <Navigate to="/user/dashboard" replace />,
  },
  {
    element: <AppLayout portal="user" />,
    children: [
      // ── 股民端（免登录）──────────────────
      { path: 'user/dashboard', element: <DashboardPage /> },
      // 无代码的个股页改为跳行情页（个股详情从行情/自选/选股池行点击进入）
      { path: 'user/stock', element: <Navigate to="/user/market" replace /> },
      { path: 'user/stock/:code', element: <StockDetailPage /> },
      { path: 'user/market', element: <MarketPage /> },
      { path: 'user/watchlist', element: <WatchlistPage /> },
      { path: 'user/pool', element: <StockPoolPage /> },
      { path: 'user/backtest', element: <BacktestPage /> },
    ],
  },
  // 管理后台拥有独立入口和独立布局，不在用户端展示入口。
  { path: 'admin/login', element: <AdminLoginPage /> },
  {
    element: <AdminGuard />,
    children: [
      {
        element: <AppLayout portal="admin" />,
        children: [
          { path: 'admin', element: <Navigate to="/admin/monitor" replace /> },
          { path: 'admin/monitor', element: <MonitorPage /> },
          { path: 'admin/config', element: <ConfigPage /> },
          { path: 'admin/logs', element: <LogsPage /> },
          { path: 'admin/research', element: <ModelResearchPage /> },
        ],
      },
    ],
  },
  // 404 catch-all
  {
    path: '*',
    element: <Navigate to="/user/dashboard" replace />,
  },
])
