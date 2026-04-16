import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useWebSocket } from '../composables/useWebSocket'
import type { TelemetryData } from '../types/api'
import { apiFetch } from '../utils/api'

const SPARKLINE_HISTORY = 30

export const useTelemetryStore = defineStore('telemetry', () => {
  const data = ref<TelemetryData | null>(null)

  // Sparkline histories (last N samples per stream type)
  const latencyHistory = ref<Record<string, number[]>>({})

  function handleMessage(msg: TelemetryData) {
    data.value = msg

    // Update sparkline histories
    for (const stream of msg.streams) {
      if (!latencyHistory.value[stream.type]) {
        latencyHistory.value[stream.type] = []
      }
      const hist = latencyHistory.value[stream.type]!
      hist!.push(stream.latency_ms)
      if (hist!.length > SPARKLINE_HISTORY) {
        hist!.shift()
      }
    }
  }

  const ws = useWebSocket('/ws/telemetry', {
    onMessage: handleMessage,
  })

  const connected = computed(() => ws.connected.value)

  /** Toggle a channel's manual bad flag. Sends to backend, which merges with auto-detected. */
  async function toggleChannelFlag(modality: string, index: number) {
    // Read current flags from telemetry data
    const qc = data.value?.channel_quality
    const currentFlagged = { ...(qc?.manual_flags ?? {}) }
    const currentUnflagged = { ...(qc?.manual_unflagged ?? {}) }
    const autoBad = qc?.bad_channels?.[modality] ?? []

    const flagSet = new Set(currentFlagged[modality] ?? [])
    const unflagSet = new Set(currentUnflagged[modality] ?? [])

    if (flagSet.has(index)) {
      // Remove manual flag
      flagSet.delete(index)
    } else if (autoBad.includes(index) && !unflagSet.has(index)) {
      // Auto-detected bad → unflag it (operator override)
      unflagSet.add(index)
    } else if (unflagSet.has(index)) {
      // Was unflagged → remove override (restore auto-detected)
      unflagSet.delete(index)
    } else {
      // Good channel → manually flag as bad
      flagSet.add(index)
    }

    currentFlagged[modality] = [...flagSet].sort((a, b) => a - b)
    currentUnflagged[modality] = [...unflagSet].sort((a, b) => a - b)

    // Clean up empty arrays
    if (currentFlagged[modality].length === 0) delete currentFlagged[modality]
    if (currentUnflagged[modality].length === 0) delete currentUnflagged[modality]

    try {
      await apiFetch('/api/pipeline/channel-flags', {
        method: 'PUT',
        json: { flagged: currentFlagged, unflagged: currentUnflagged },
      })
    } catch {
      // toast surfaced by apiFetch
    }
  }

  return { data, connected, latencyHistory, toggleChannelFlag }
})
