/**
 * useApi — API 可达性探测 Hook
 *
 * 替换旧版 KronosApp.isApiAvailable() 探测逻辑。
 *
 * 职责:
 *   1. 应用初始化时探测后端 API 是否可达
 *   2. 监听浏览器 online/offline 事件自动重试
 *   3. 同步结果到 useAppStore.apiStatus
 *
 * 用法:
 *   const { status, probe } = useApi()
 *   useEffect(() => { probe() }, [])
 */

import { useEffect, useCallback, useRef } from 'react'
import { useAppStore } from '@/stores/useAppStore'
import { apiProbe } from '@/api/client'

/** 探测间隔 (ms) — 离线时每 30s 重试 */
const RETRY_INTERVAL = 30_000

/** 初始启动延时 (ms) — 给 FastAPI 一点启动时间 */
const INITIAL_DELAY = 1_500

export function useApi() {
  const { apiStatus, setApiStatus } = useAppStore()
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const destroyedRef = useRef(false)

  // ── 探测 ────────────────────────────────────────────
  const probe = useCallback(async () => {
    if (destroyedRef.current) return
    setApiStatus('checking')
    const reachable = await apiProbe()
    if (destroyedRef.current) return
    setApiStatus(reachable ? 'available' : 'unavailable')
  }, [setApiStatus])

  // ── 初始化探测 ──────────────────────────────────────
  useEffect(() => {
    destroyedRef.current = false
    const id = setTimeout(probe, INITIAL_DELAY)

    return () => {
      destroyedRef.current = true
      clearTimeout(id)
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [probe])

  // ── 离线重试 ────────────────────────────────────────
  useEffect(() => {
    if (apiStatus === 'unavailable' && !destroyedRef.current) {
      timerRef.current = setInterval(probe, RETRY_INTERVAL)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [apiStatus, probe])

  // ── 浏览器 online/offline ───────────────────────────
  useEffect(() => {
    const handleOnline = () => probe()
    const handleOffline = () => setApiStatus('unavailable')

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [probe, setApiStatus])

  return { status: apiStatus, probe }
}
