import { apiClient } from './client'
import type { ResearchSummary } from '@/types/research'

export function getResearchSummary() {
  return apiClient.get<ResearchSummary>('/admin/research/summary', {
    cache: true,
    cacheTtl: 60_000,
  })
}
