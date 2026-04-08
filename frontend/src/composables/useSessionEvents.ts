import { ref } from 'vue'

interface EventEntry { id: number; label: string }

export function useSessionEvents() {
  const events = ref<number[]>([])
  const mapping = ref<Record<number, string>>({})

  async function fetchEvents() {
    try {
      const res = await fetch('/api/pipeline/session-events')
      if (res.ok) {
        const data = await res.json()
        events.value = data.events ?? []
        mapping.value = data.event_mapping ?? {}
      }
    } catch { /* endpoint may not exist if pipeline isn't running */ }
  }

  function buildEntries(): EventEntry[] {
    return events.value.map(code => ({
      id: code,
      label: mapping.value[code] ?? '',
    }))
  }

  return { events, mapping, buildEntries, fetchEvents }
}
