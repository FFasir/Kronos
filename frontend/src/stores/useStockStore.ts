import { create } from 'zustand'
import type { Stock, MarketOverview } from '@/types/stock'
import type { StockDetail, PredictionResult } from '@/types/api'

/** 个股 + 选股池状态 */
interface StockState {
  // ── 大盘 ───────────────────────────────
  overview: MarketOverview | null
  setOverview: (overview: MarketOverview) => void

  // ── 选股池 ─────────────────────────────
  pool: PredictionResult[]
  poolLoading: boolean
  setPool: (pool: PredictionResult[]) => void
  setPoolLoading: (loading: boolean) => void

  // ── 个股详情 ───────────────────────────
  currentStock: StockDetail | null
  stockLoading: boolean
  setCurrentStock: (stock: StockDetail | null) => void
  setStockLoading: (loading: boolean) => void

  // ── 搜索 ───────────────────────────────
  searchResults: Stock[]
  searchLoading: boolean
  setSearchResults: (results: Stock[]) => void
  setSearchLoading: (loading: boolean) => void
}

export const useStockStore = create<StockState>((set) => ({
  overview: null,
  setOverview: (overview) => set({ overview }),

  pool: [],
  poolLoading: false,
  setPool: (pool) => set({ pool, poolLoading: false }),
  setPoolLoading: (poolLoading) => set({ poolLoading }),

  currentStock: null,
  stockLoading: false,
  setCurrentStock: (currentStock) => set({ currentStock, stockLoading: false }),
  setStockLoading: (stockLoading) => set({ stockLoading }),

  searchResults: [],
  searchLoading: false,
  setSearchResults: (searchResults) => set({ searchResults, searchLoading: false }),
  setSearchLoading: (searchLoading) => set({ searchLoading }),
}))
