/**
 * Kronos API 客户端 — Axios 封装
 *
 * 功能：
 * - 统一 baseURL + 超时 10s
 * - 请求拦截器：自动附加 admin token
 * - 响应拦截器：统一错误处理
 * - 简单内存缓存（Map + TTL）
 * - 失败自动重试（最多 2 次）
 * - API 可达性探测
 *
 * 注意：请求失败时返回 null（不抛异常），调用方需显式判空。
 *
 * 用法：
 *   import { apiClient, apiProbe } from '@/api/client'
 *   const data = await apiClient.get<IndexListResponse>('/index/list', { cache: true })
 */

import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosError } from 'axios'
import { useAppStore } from '@/stores/useAppStore'

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

/** 后端 API 基础地址 */
const API_BASE: string =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE) ||
  '/api/v1'

/** 请求超时 (ms) */
const TIMEOUT_MS = 10_000

/** 最大重试次数 */
const MAX_RETRIES = 2

/** 默认缓存 TTL (ms) */
const DEFAULT_CACHE_TTL = 30_000

// ═══════════════════════════════════════════════════════════════════
// Memory Cache
// ═══════════════════════════════════════════════════════════════════

interface CacheEntry {
  data: unknown
  expires: number
}

const _cache = new Map<string, CacheEntry>()

/** 清除所有缓存 */
export function clearCache(): void {
  _cache.clear()
}

/** 清除匹配前缀的缓存 */
export function clearCacheByPrefix(prefix: string): void {
  for (const key of _cache.keys()) {
    if (key.startsWith(prefix)) {
      _cache.delete(key)
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// Axios Instance
// ═══════════════════════════════════════════════════════════════════

const _axios: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: TIMEOUT_MS,
  headers: { 'Content-Type': 'application/json' },
})

// ── 请求拦截器：附加 admin token ────────────────────────────

_axios.interceptors.request.use((config) => {
  // 管理后台请求附加 token
  if (config.url?.startsWith('/admin/')) {
    const token = useAppStore.getState().adminToken
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
  }
  return config
})

// ── 响应拦截器：统一错误处理 ────────────────────────────────

_axios.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // 管理后台 token 过期 → 自动登出
      useAppStore.getState().logout()
    }
    return Promise.reject(error)
  },
)

// ═══════════════════════════════════════════════════════════════════
// Extended Request Options
// ═══════════════════════════════════════════════════════════════════

export interface RequestOptions {
  /** 是否缓存响应 */
  cache?: boolean
  /** 缓存 TTL (ms)，默认 30s */
  cacheTtl?: number
  /** 重试次数，默认 MAX_RETRIES */
  retries?: number
  /** 超时时间 (ms)，默认 TIMEOUT_MS */
  timeout?: number
  /** 静默模式：失败时不打印 warn */
  silent?: boolean
}

// ═══════════════════════════════════════════════════════════════════
// Core Request Function
// ═══════════════════════════════════════════════════════════════════

/**
 * 发送 GET 请求（带缓存 + 重试 + mock 降级）。
 *
 * @returns 响应 data，失败时返回 null
 */
async function _get<T>(
  url: string,
  opts: RequestOptions = {},
): Promise<T | null> {
  const {
    cache = false,
    cacheTtl = DEFAULT_CACHE_TTL,
    retries = MAX_RETRIES,
    timeout = TIMEOUT_MS,
    silent = false,
  } = opts

  const cacheKey = `GET:${url}`

  // 命中缓存
  if (cache && _cache.has(cacheKey)) {
    const entry = _cache.get(cacheKey)!
    if (Date.now() < entry.expires) {
      return entry.data as T
    }
    _cache.delete(cacheKey)
  }

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const config: AxiosRequestConfig = { timeout }
      const resp = await _axios.get<T>(url, config)

      // 写入缓存
      if (cache) {
        _cache.set(cacheKey, { data: resp.data, expires: Date.now() + cacheTtl })
      }

      return resp.data
    } catch (err) {
      lastError = err as Error
      if (attempt < retries) {
        if (!silent) {
          console.warn(`[API] retry ${attempt + 1}/${retries} → ${url}`)
        }
        await _sleep(500 * (attempt + 1))
      }
    }
  }

  if (!silent && lastError) {
    const axiosErr = lastError as AxiosError
    if (axiosErr.code !== 'ERR_CANCELED') {
      console.warn(`[API] 不可达 (${url}):`, lastError.message)
    }
  }

  return null
}

/**
 * 发送 POST 请求（带重试）。
 */
async function _post<T>(
  url: string,
  body?: unknown,
  opts: RequestOptions = {},
): Promise<T | null> {
  const { retries = 1, timeout = TIMEOUT_MS, silent = false } = opts

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const config: AxiosRequestConfig = { timeout }
      const resp = await _axios.post<T>(url, body, config)
      return resp.data
    } catch (err) {
      lastError = err as Error
      if (attempt < retries) {
        if (!silent) {
          console.warn(`[API] POST retry ${attempt + 1}/${retries} → ${url}`)
        }
        await _sleep(500 * (attempt + 1))
      }
    }
  }

  if (!silent && lastError) {
    console.warn(`[API] POST 失败 (${url}):`, lastError.message)
  }

  return null
}

// ═══════════════════════════════════════════════════════════════════
// API Availability Probe
// ═══════════════════════════════════════════════════════════════════

/**
 * 检测后端是否可达。
 * 使用快速 HEAD 请求（3s 超时）探测 `/index/list` 端点。
 */
export async function apiProbe(): Promise<boolean> {
  try {
    await _axios.get('/index/list', { timeout: 3000 })
    return true
  } catch {
    return false
  }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

export const apiClient = {
  /** GET 请求（带缓存 + 重试） */
  get: _get,
  /** POST 请求（带重试，不缓存） */
  post: _post,
  /** 清除所有缓存 */
  clearCache,
  /** 清除匹配前缀的缓存 */
  clearCacheByPrefix,
  /** Axios 原始实例（仅在需要自定义配置时使用） */
  instance: _axios,
}

export type { AxiosError }
