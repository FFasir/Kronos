/**
 * SearchBox — 股票搜索框组件
 *
 * 封装 Ant Design AutoComplete，支持 300ms 防抖、键盘回车选中。
 *
 * 两种模式:
 *   1. 受控模式 — 传入 results + onSearch，由外部提供候选（预留给全市场搜索）
 *   2. 默认模式 — 不传 onSearch 时，自动从 GET /index/list 加载标的列表，
 *      按代码/名称本地过滤（当前系统共 12 个指数+ETF）
 *
 * 用法:
 *   <SearchBox onSelect={(stock) => navigate(`/user/stock/${stock.code}`)} />
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { AutoComplete, Input } from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { getIndexList } from '@/api/index'
import type { Stock } from '@/types/stock'

export interface SearchBoxProps {
  /** 搜索结果列表（受控模式） */
  results?: Stock[]
  /** 搜索中 */
  loading?: boolean
  /** 占位文案 */
  placeholder?: string
  /** 选中回调 */
  onSelect?: (stock: Stock) => void
  /** 输入变化回调（受控模式；不传则用内置标的列表本地过滤） */
  onSearch?: (keyword: string) => void
  /** 防抖延迟 ms */
  debounceMs?: number
}

const TYPE_LABELS: Record<string, string> = { index: '指数', etf: 'ETF', stock: '股票' }

export default function SearchBox({
  results,
  loading = false,
  placeholder = '搜索股票代码或名称...',
  onSelect,
  onSearch,
  debounceMs = 300,
}: SearchBoxProps) {
  const [open, setOpen] = useState(false)
  const [localResults, setLocalResults] = useState<Stock[]>([])
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const isControlled = !!onSearch

  useEffect(() => () => clearTimeout(timerRef.current), [])

  // ── 默认模式：加载标的列表并本地过滤 ────────────────
  const localSearch = useCallback(async (keyword: string) => {
    const data = await getIndexList('stock', keyword, 20)
    setLocalResults((data?.indices ?? []).map((item) => ({
      code: item.code,
      name: item.name,
      industry: TYPE_LABELS[item.type] ?? item.type,
      market_cap: null,
    })))
    setOpen(true)
  }, [])

  const handleSearch = useCallback(
    (value: string) => {
      const kw = value.trim()
      if (!kw) {
        setOpen(false)
        setLocalResults([])
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => {
        if (isControlled) {
          onSearch?.(kw)
          setOpen(true)
        } else {
          localSearch(kw)
        }
      }, debounceMs)
    },
    [isControlled, onSearch, localSearch, debounceMs],
  )

  const displayResults = useMemo(
    () => (isControlled ? (results ?? []) : localResults),
    [isControlled, results, localResults],
  )

  const handleSelect = useCallback(
    (value: string, option: { data?: Stock }) => {
      const stock = displayResults.find((s) => s.code === value) || option.data
      if (stock) {
        onSelect?.(stock)
        setOpen(false)
      }
    },
    [displayResults, onSelect],
  )

  const options = displayResults.map((s) => ({
    value: s.code,
    label: `${s.code} — ${s.name}${s.industry ? `（${s.industry}）` : ''}`,
    data: s,
  }))

  return (
    <AutoComplete
      open={open}
      onOpenChange={setOpen}
      options={options}
      onSelect={handleSelect}
      onSearch={handleSearch}
      style={{ width: 320 }}
      notFoundContent="未找到匹配的标的"
    >
      <Input
        prefix={<SearchOutlined style={{ color: 'var(--text-muted)' }} />}
        placeholder={placeholder}
        allowClear
        suffix={loading ? <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>搜索中...</span> : undefined}
      />
    </AutoComplete>
  )
}
