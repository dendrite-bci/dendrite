import { defineStore } from 'pinia'
import { ref } from 'vue'

interface ManagedStream {
  id: string
  source: string
  progress: number
  running: boolean
}

interface FinishedStream {
  id: string
  source: string
  label: string
  config: Record<string, any>
}

interface MoabbDataset {
  name: string
  paradigm: string
  n_subjects: number
  subjects: number[]
  events: Record<string, number>
}

export interface FileInfo {
  path: string
  duration_s: number
  sample_rate: number
  n_channels: number
  channel_names: string[]
  n_events: number
  event_id: Record<string, number> | null
}

export const useStreamManagerStore = defineStore('streamManager', () => {
  const streams = ref<ManagedStream[]>([])
  const finishedStreams = ref<FinishedStream[]>([])
  const streamLabels = ref<Record<string, string>>({})
  const streamConfigs = ref<Record<string, Record<string, any>>>({})
  const moabbDatasets = ref<MoabbDataset[]>([])
  const loading = ref(false)
  let pollTimer: number | null = null
  let prevIds = new Set<string>()

  async function fetchStatus() {
    const res = await fetch('/api/stream-manager/status')
    if (!res.ok) return
    const data = await res.json()
    const active: ManagedStream[] = data.streams ?? []
    const activeIds = new Set(active.map(s => s.id))

    // Hydrate configs from backend for streams we don't know about (e.g. after page refresh)
    for (const raw of (data.streams ?? []) as Record<string, any>[]) {
      if (raw.id && !streamConfigs.value[raw.id]) {
        const { id: _id, running: _r, progress: _p, ...cfg } = raw
        streamConfigs.value[raw.id] = cfg
      }
    }

    // Detect streams that just finished (were active, now gone)
    for (const id of prevIds) {
      if (!activeIds.has(id) && !finishedStreams.value.some(f => f.id === id)) {
        finishedStreams.value.push({
          id,
          source: streamConfigs.value[id]?.source ?? 'file',
          label: streamLabels.value[id] ?? 'Stream',
          config: streamConfigs.value[id] ?? {},
        })
      }
    }
    prevIds = activeIds
    streams.value = active
  }

  async function startStream(config: Record<string, any>, label?: string) {
    loading.value = true
    try {
      const res = await fetch('/api/stream-manager/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const data = await res.json()
      if (data.id) {
        if (label) streamLabels.value[data.id] = label
        streamConfigs.value[data.id] = config
      }
      await fetchStatus()
      startPolling()
      return data.id
    } finally {
      loading.value = false
    }
  }

  async function stopStream(id: string) {
    await fetch(`/api/stream-manager/stop/${id}`, { method: 'POST' })
    delete streamLabels.value[id]
    await fetchStatus()
  }

  async function restartFinished(id: string) {
    const idx = finishedStreams.value.findIndex(f => f.id === id)
    if (idx < 0) return
    const finished = finishedStreams.value[idx]!
    finishedStreams.value.splice(idx, 1)
    return startStream(finished!.config, finished!.label)
  }

  function dismissFinished(id: string) {
    finishedStreams.value = finishedStreams.value.filter(f => f.id !== id)
  }

  function clearFinished() {
    finishedStreams.value = []
  }

  async function fetchMoabb() {
    loading.value = true
    try {
      const res = await fetch('/api/stream-manager/moabb')
      if (!res.ok) { moabbDatasets.value = []; return }
      const data = await res.json()
      moabbDatasets.value = data.datasets
    } catch {
      moabbDatasets.value = []
    } finally {
      loading.value = false
    }
  }

  async function fetchFileInfo(path: string): Promise<FileInfo | null> {
    const res = await fetch('/api/stream-manager/file-info', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    })
    if (!res.ok) return null
    return await res.json()
  }

  function startPolling() {
    if (pollTimer) return
    pollTimer = window.setInterval(async () => {
      await fetchStatus()
      if (streams.value.length === 0 && pollTimer) {
        window.clearInterval(pollTimer)
        pollTimer = null
      }
    }, 1000)
  }

  function stopPolling() {
    if (pollTimer) {
      window.clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    streams, finishedStreams, streamLabels, moabbDatasets, loading,
    fetchStatus, startStream, stopStream, restartFinished, dismissFinished, clearFinished,
    fetchMoabb, fetchFileInfo, startPolling, stopPolling,
  }
})
