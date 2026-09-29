/**
 * 回测 API — api_backtest.py
 *
 * 端点:
 *   POST /api/v1/backtest            — 提交回测任务
 *   GET  /api/v1/backtest/{task_id}  — 查询回测任务状态/结果
 *   GET  /api/v1/backtest/history    — 历史回测任务列表
 */

import { apiClient } from './client'
import type {
  BacktestRequest,
  BacktestSubmitResponse,
  BacktestResult,
  BacktestHistoryItem,
  BacktestProgress,
  BacktestAvailability,
  OptimizeRequest,
  OptimizeResult,
  CompareRequest,
  CompareResult,
} from '@/types/api'

// ═══════════════════════════════════════════════════════════════════
// Endpoints
// ═══════════════════════════════════════════════════════════════════

/**
 * 提交回测任务。
 * POST /api/v1/backtest
 */
export async function submitBacktest(
  params: BacktestRequest,
): Promise<BacktestSubmitResponse | null> {
  return apiClient.post<BacktestSubmitResponse>('/backtest', params, {
    timeout: 15_000,
  })
}

/**
 * 查询回测任务状态/结果。
 * GET /api/v1/backtest/{task_id}
 */
export async function getBacktestResult(
  taskId: string,
): Promise<BacktestResult | null> {
  return apiClient.get<BacktestResult>(
    `/backtest/${encodeURIComponent(taskId)}`,
    { cache: false },
  )
}

/**
 * 获取历史回测任务列表。
 * GET /api/v1/backtest/history
 */
export async function getBacktestHistory(): Promise<BacktestHistoryItem[] | null> {
  return apiClient.get<BacktestHistoryItem[]>('/backtest/history')
}

/**
 * 获取回测可用性诊断（各 scope 的信号样本）。
 * GET /api/v1/backtest/availability
 */
export async function getBacktestAvailability(): Promise<BacktestAvailability | null> {
  return apiClient.get<BacktestAvailability>('/backtest/availability', {
    cache: true,
    cacheTtl: 30_000,
  })
}

// ═══════════════════════════════════════════════════════════════════
// Polling (WebSocket / SSE 降级)
// ═══════════════════════════════════════════════════════════════════

/**
 * 轮询回测任务直到完成。
 *
 * @param taskId      - 任务 ID
 * @param onProgress  - 进度回调 ({ status, progress, message }) => void
 * @param intervalMs  - 轮询间隔 (ms)，默认 2000
 * @param timeoutMs   - 总超时 (ms)，默认 300000 (5 分钟)
 * @returns 最终结果，超时返回 { status: 'timeout' }
 */
export async function pollBacktest(
  taskId: string,
  onProgress?: (progress: Partial<BacktestProgress>) => void,
  intervalMs = 2000,
  timeoutMs = 300_000,
): Promise<Partial<BacktestProgress> | null> {
  const start = Date.now()

  while (Date.now() - start < timeoutMs) {
    const result = await getBacktestResult(taskId)

    if (!result) {
      onProgress?.({ status: 'failed', progress: 0 })
      return null
    }

    if (result.status === 'completed' || result.status === 'failed') {
      onProgress?.({
        task_id: result.task_id,
        status: result.status,
        progress: 1,
        message: result.message,
      })
      return { task_id: taskId, status: result.status, progress: 1 }
    }

    // 运行中：透传后端实时进度
    onProgress?.({
      task_id: result.task_id,
      status: result.status,
      progress: result.progress ?? 0,
      message: result.message,
    })

    await _sleep(intervalMs)
  }

  return { task_id: taskId, status: 'timeout', progress: 0 }
}

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// ═══════════════════════════════════════════════════════════════════
// Strategy Optimization
// ═══════════════════════════════════════════════════════════════════

/**
 * 提交参数优化任务。
 * POST /api/v1/backtest/optimize
 */
export async function submitOptimize(
  params: OptimizeRequest,
): Promise<BacktestSubmitResponse | null> {
  return apiClient.post<BacktestSubmitResponse>('/backtest/optimize', params, {
    timeout: 15_000,
  })
}

/**
 * 查询优化任务状态/结果。
 * GET /api/v1/backtest/optimize/{task_id}
 */
export async function getOptimizeResult(
  taskId: string,
): Promise<OptimizeResult | null> {
  return apiClient.get<OptimizeResult>(
    `/backtest/optimize/${encodeURIComponent(taskId)}`,
    { cache: false },
  )
}

/**
 * 轮询优化任务直到完成。
 */
export async function pollOptimize(
  taskId: string,
  onProgress?: (progress: Partial<OptimizeResult>) => void,
  intervalMs = 3000,
  timeoutMs = 600_000,
): Promise<OptimizeResult | null> {
  const start = Date.now()

  while (Date.now() - start < timeoutMs) {
    const result = await getOptimizeResult(taskId)

    if (!result) {
      onProgress?.({ task_id: taskId, status: 'failed' as any, mode: '', metric: '', total_combinations: 0, progress: 0, results: [], walk_forward_windows: [] })
      return null
    }

    if (result.status === 'completed' || result.status === 'failed') {
      onProgress?.(result)
      return result
    }

    onProgress?.(result)
    await _sleep(intervalMs)
  }

  return null
}

// ═══════════════════════════════════════════════════════════════════
// Multi-Strategy Compare
// ═══════════════════════════════════════════════════════════════════

/**
 * 提交多策略对比任务。
 * POST /api/v1/backtest/compare
 */
export async function submitCompare(
  params: CompareRequest,
): Promise<BacktestSubmitResponse | null> {
  return apiClient.post<BacktestSubmitResponse>('/backtest/compare', params, {
    timeout: 15_000,
  })
}

/**
 * 查询对比任务状态/结果。
 * GET /api/v1/backtest/compare/{task_id}
 */
export async function getCompareResult(
  taskId: string,
): Promise<CompareResult | null> {
  return apiClient.get<CompareResult>(
    `/backtest/compare/${encodeURIComponent(taskId)}`,
    { cache: false },
  )
}
