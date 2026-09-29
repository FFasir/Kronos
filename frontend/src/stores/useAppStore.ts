import { create } from 'zustand'

/** 主题模式 */
export type ThemeMode = 'light' | 'dark'

/** API 连接状态 */
export type ApiStatus = 'checking' | 'available' | 'unavailable'

/** 全局应用状态 */
interface AppState {
  // ── API ─────────────────────────────────
  apiStatus: ApiStatus
  lastProbeTime: number
  setApiStatus: (status: ApiStatus) => void

  // ── 主题 ───────────────────────────────
  theme: ThemeMode
  toggleTheme: () => void
  setTheme: (theme: ThemeMode) => void

  // ── 认证（管理后台）────────────────────
  adminToken: string | null
  isAdminAuthenticated: boolean
  login: (token: string) => void
  logout: () => void

  // ── 页面标题 ───────────────────────────
  pageTitle: string
  setPageTitle: (title: string) => void
}

export const useAppStore = create<AppState>((set) => ({
  // API
  apiStatus: 'checking',
  lastProbeTime: 0,
  setApiStatus: (apiStatus) => set({ apiStatus, lastProbeTime: Date.now() }),

  // 主题 (默认跟随系统，fallback light；DOM data-theme 属性由 App.tsx 统一设置)
  theme: (() => {
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark'
    }
    return 'light'
  })(),
  toggleTheme: () =>
    set((s) => ({ theme: s.theme === 'light' ? 'dark' : 'light' })),
  setTheme: (theme) => set({ theme }),

  // 认证
  adminToken: typeof window !== 'undefined' ? sessionStorage.getItem('kronos_admin_token') : null,
  isAdminAuthenticated:
    typeof window !== 'undefined' && Boolean(sessionStorage.getItem('kronos_admin_token')),
  login: (adminToken) => {
    sessionStorage.setItem('kronos_admin_token', adminToken)
    set({ adminToken, isAdminAuthenticated: true })
  },
  logout: () => {
    sessionStorage.removeItem('kronos_admin_token')
    set({ adminToken: null, isAdminAuthenticated: false })
  },

  // 页面标题
  pageTitle: '仪表盘',
  setPageTitle: (pageTitle) => set({ pageTitle }),
}))
