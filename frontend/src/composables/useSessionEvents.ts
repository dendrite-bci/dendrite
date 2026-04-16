import { ref } from 'vue'
import { apiFetchOrNull } from '../utils/api'

interface EventEntry { id: number; label: string }

export function useSessionEvents() {
  const events = ref<number[]>([])
  const mapping = ref<Record<number, string>>({})

  async function fetchEvents() {
    const data = await apiFetchOrNull<{ events?: number[]; event_mapping?: Record<number, string> }>(
      '/api/pipeline/session-events',
    )
    if (!data) return
    events.value = data.events ?? []
    mapping.value = data.event_mapping ?? {}
  }

  function buildEntries(): EventEntry[] {
    return events.value.map(code => ({
      id: code,
      label: mapping.value[code] ?? '',
    }))
  }

  return { events, mapping, buildEntries, fetchEvents }
}
