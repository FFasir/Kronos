import { create } from 'zustand'
import type { BacktestProgress, BacktestResult } from '@/types/backtest'
import type { BacktestRequest } from '@/types/api'

/** 回测状态机: idle → submitted → running → completed | failed */
type BacktestPhase = 'idle' | 'submitted' | 'running' | 'completed' | 'failed'

/** 回测状态 */
interface BacktestState {
  // ── 状态机 ─────────────────────────────
  phase: BacktestPhase
  setPhase: (phase: BacktestPhase) => void

  // ── 进度 ───────────────────────────────
  progress: BacktestProgress | null
  setProgress: (progress: BacktestProgress) => void

  // ── 结果 ───────────────────────────────
  result: BacktestResult | null
  setResult: (result: BacktestResult | null) => void

  // ── 配置缓存 ───────────────────────────
  lastConfig: BacktestRequest | null
  setLastConfig: (config: BacktestRequest) => void

  // ── 重置 ───────────────────────────────
  reset: () => void
}

export const useBacktestStore = create<BacktestState>((set) => ({
  phase: 'idle',
  setPhase: (phase) => set({ phase }),

  progress: null,
  setProgress: (progress) => set({ progress }),

  result: null,
  setResult: (result) => set({ result }),

  lastConfig: null,
  setLastConfig: (lastConfig) => set({ lastConfig }),

  reset: () =>
    set({
      phase: 'idle',
      progress: null,
      result: null,
    }),
}))
