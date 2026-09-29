/**
 * AdminGuard — 管理后台登录守卫
 *
 * 职责:
 *   1. 拦截 /admin/* 路径，检查是否已认证
 *   2. 未认证 → 跳转独立登录页，阻止访问子页面
 *   3. 已认证 → 正常渲染 <Outlet />
 *
 * 登录成功后 token 存入 useAppStore（内存，刷新页面需重新登录）。
 */

import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAppStore } from '@/stores/useAppStore'

export default function AdminGuard() {
  const location = useLocation()
  const isAdminAuthenticated = useAppStore((s) => s.isAdminAuthenticated)

  if (!isAdminAuthenticated) {
    return <Navigate to="/admin/login" replace state={{ from: location.pathname }} />
  }

  return <Outlet />
}
