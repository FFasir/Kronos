/**
 * 管理后台 API — api_admin.py
 *
 * 端点:
 *   POST /api/v1/admin/login         — 管理员登录
 *   GET  /api/v1/admin/monitor       — 系统监控状态
 *   GET  /api/v1/admin/training/log  — 训练日志
 *   GET  /api/v1/admin/data/sources  — 数据源状态
 *   GET  /api/v1/admin/config        — 读取系统配置
 *   POST /api/v1/admin/config        — 保存用户配置
 */

import { apiClient } from './client'
import type {
  AdminLoginResponse,
  SystemMonitor,
  SystemStatus,
  TrainingLogEntry,
  DataSourceInfo,
  SystemConfigResponse,
  BackgroundJob,
} from '@/types/api'
import type { ModelRegistryResponse, RegisteredModel } from '@/types/research'

// ═══════════════════════════════════════════════════════════════════
// Login
// ═══════════════════════════════════════════════════════════════════

/**
 * 管理员登录。
 * POST /api/v1/admin/login
 */
export async function adminLogin(
  username: string,
  password: string,
): Promise<AdminLoginResponse | null> {
  return apiClient.post<AdminLoginResponse>(
    '/admin/login',
    { username, password },
    { timeout: 8_000 },
  )
}

/** 注销当前管理会话。 */
export async function adminLogout(): Promise<void> {
  await apiClient.post<{ success: boolean }>('/admin/logout', undefined, {
    retries: 0,
    silent: true,
  })
}

// ═══════════════════════════════════════════════════════════════════
// System Monitor
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取系统监控状态。
 * GET /api/v1/admin/monitor
 */
export async function getSystemMonitor(): Promise<SystemMonitor | null> {
  return apiClient.get<SystemMonitor>('/admin/monitor', {
    cache: true,
    cacheTtl: 15_000,
  })
}

/**
 * 获取免登录的系统状态摘要。
 * GET /api/v1/system/status（无需管理员鉴权）
 */
export async function getSystemStatus(): Promise<SystemStatus | null> {
  return apiClient.get<SystemStatus>('/system/status', {
    cache: true,
    cacheTtl: 30_000,
  })
}

/**
 * 获取训练日志（最近 N 条）。
 * GET /api/v1/admin/training/log?limit=20
 */
export async function getTrainingLogs(
  limit = 20,
): Promise<TrainingLogEntry[] | null> {
  return apiClient.get<TrainingLogEntry[]>(
    `/admin/training/log?limit=${limit}`,
  )
}

/**
 * 获取所有标的数据源状态。
 * GET /api/v1/admin/data/sources
 */
export async function getDataSources(): Promise<DataSourceInfo[] | null> {
  return apiClient.get<DataSourceInfo[]>('/admin/data/sources', {
    cache: true,
    cacheTtl: 60_000,
  })
}

export type AdminJobName = 'data_update' | 'timing_data_update' | 'preprocess' | 'timing_signal' | 'prediction' | 'rolling_training' | 'selection_signal' | 'stock_hfq_update' | 'daily_pipeline'

export async function runAdminJob(name: AdminJobName): Promise<BackgroundJob | null> {
  return apiClient.post<BackgroundJob>(`/admin/jobs/${name}`, undefined, { timeout: 10_000 })
}

export async function getAdminJobStatus(jobId: string): Promise<BackgroundJob | null> {
  return apiClient.get<BackgroundJob>(`/admin/jobs/status/${jobId}`, { cache: false })
}

// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════

/**
 * 获取系统配置（策略参数等）。
 * GET /api/v1/admin/config
 */
export async function loadSystemConfig(): Promise<SystemConfigResponse | null> {
  return apiClient.get<SystemConfigResponse>('/admin/config', { cache: true })
}

/**
 * 保存用户配置到后端。
 * POST /api/v1/admin/config
 *
 * 失败时自动降级到 localStorage。
 */
export async function saveAdminConfig(
  config: Record<string, unknown>,
): Promise<boolean> {
  const result = await apiClient.post<{ success: boolean }>(
    '/admin/config',
    config,
  )
  if (result?.success) return true

  return false
}

export function getRegisteredModels() {
  return apiClient.get<ModelRegistryResponse>('/admin/models', { cache: false })
}

export function registerCustomModel(payload: {
  name: string
  version: string
  predictor_path: string
  tokenizer_path: string
}) {
  return apiClient.post<RegisteredModel>('/admin/models/register', payload, { retries: 0 })
}

export function activateRegisteredModel(key: string) {
  return apiClient.post<{ success: boolean; active_key: string; message: string }>(
    `/admin/models/${key}/activate`,
    undefined,
    { retries: 0, timeout: 30_000 },
  )
}

export function deleteRegisteredModel(key: string) {
  return apiClient.instance.delete<{ success: boolean }>(`/admin/models/${key}`)
}
