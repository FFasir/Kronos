/**
 * 自选股 store（当前用 localStorage 本地持久化，无需登录）
 *
 * 多用户升级路径：待系统引入登录后，将这里的读写替换为按用户存储的服务端
 * API（codes 内容不变），前端各使用方无需改动。
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface WatchlistState {
  /** 自选股票代码列表（6 位） */
  codes: string[]
  add: (code: string) => void
  remove: (code: string) => void
  toggle: (code: string) => void
  clear: () => void
}

export const useWatchlistStore = create<WatchlistState>()(
  persist(
    (set) => ({
      codes: [],
      add: (code) =>
        set((s) => (s.codes.includes(code) ? s : { codes: [...s.codes, code] })),
      remove: (code) =>
        set((s) => ({ codes: s.codes.filter((c) => c !== code) })),
      toggle: (code) =>
        set((s) => ({
          codes: s.codes.includes(code)
            ? s.codes.filter((c) => c !== code)
            : [...s.codes, code],
        })),
      clear: () => set({ codes: [] }),
    }),
    { name: 'kronos-watchlist' },
  ),
)
