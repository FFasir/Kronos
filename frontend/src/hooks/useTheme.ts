/**
 * useTheme — 暗色模式跟随 Hook
 *
 * 替换旧版 CSS @media (prefers-color-scheme: dark) 检测逻辑。
 *
 * 职责:
 *   1. 初始化时检测系统主题偏好 (matchMedia)
 *   2. 监听系统主题变化，自动同步到 useAppStore
 *   3. 手动切换时覆盖系统偏好
 *
 * 用法:
 *   const { theme, isDark, toggle } = useTheme()
 */

import { useEffect, useCallback } from 'react'
import { useAppStore } from '@/stores/useAppStore'

/** localStorage key — 用户手动覆盖的主题偏好 */
const USER_PREF_KEY = 'kronos_theme'

export function useTheme() {
  const theme = useAppStore((s) => s.theme)
  const setTheme = useAppStore((s) => s.setTheme)
  const toggleTheme = useAppStore((s) => s.toggleTheme)

  const isDark = theme === 'dark'

  // ── 初始化：检查用户保存的偏好 ─────────────────────
  useEffect(() => {
    try {
      const saved = localStorage.getItem(USER_PREF_KEY)
      if (saved === 'dark' || saved === 'light') {
        setTheme(saved)
        return
      }
    } catch {
      // localStorage 不可用
    }

    // 跟随系统
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    setTheme(mq.matches ? 'dark' : 'light')

    const handler = (e: MediaQueryListEvent) => {
      // 仅在用户未手动覆盖时跟随系统
      try {
        if (!localStorage.getItem(USER_PREF_KEY)) {
          setTheme(e.matches ? 'dark' : 'light')
        }
      } catch {
        setTheme(e.matches ? 'dark' : 'light')
      }
    }

    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [setTheme])

  // ── 手动切换（覆盖系统偏好）────────────────────────
  const toggle = useCallback(() => {
    toggleTheme()
    // 保存用户偏好
    try {
      const next = theme === 'light' ? 'dark' : 'light'
      localStorage.setItem(USER_PREF_KEY, next)
    } catch {
      // ignore
    }
  }, [theme, toggleTheme])

  return { theme, isDark, toggle, setTheme }
}
