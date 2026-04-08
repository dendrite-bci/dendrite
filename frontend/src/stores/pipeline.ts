import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { PipelineStatus, PreflightResult } from '../types/api'

export const usePipelineStore = defineStore('pipeline', () => {
  const status = ref<PipelineStatus>({
    recording: false,
    recording_id: null,
    elapsed_seconds: 0,
    log_file: null,
    mode_pids: {},
    system_pids: {},
    component_states: {},
  })
  const loading = ref(false)
  const error = ref<string | null>(null)

  const preflight = ref<PreflightResult | null>(null)
  const preflightLoading = ref(false)

  const canStart = computed(() =>
    preflight.value?.ready === true && !status.value.recording && !loading.value
  )

  let pollTimer: ReturnType<typeof setInterval> | null = null

  async function fetchStatus() {
    try {
      const res = await fetch('/api/pipeline/status')
      status.value = await res.json()
    } catch (e: any) {
      error.value = e.message
    }
  }

  async function fetchPreflight() {
    preflightLoading.value = true
    try {
      const res = await fetch('/api/pipeline/preflight')
      preflight.value = await res.json()
    } catch {
      // Preflight is advisory — don't block on failure
    } finally {
      preflightLoading.value = false
    }
  }

  async function start() {
    loading.value = true
    error.value = null

    // Refresh preflight before attempting start
    await fetchPreflight()
    if (!preflight.value?.ready) {
      error.value = 'Pre-start checks failed. Review the checklist.'
      loading.value = false
      return
    }

    try {
      const res = await fetch('/api/pipeline/start', { method: 'POST' })
      if (!res.ok) {
        const data = await res.json()
        const detail = data.detail
        if (typeof detail === 'object' && detail.message) {
          throw new Error(detail.message)
        }
        throw new Error(detail || 'Failed to start pipeline')
      }
      startPolling()
      await fetchStatus()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function stop() {
    loading.value = true
    error.value = null
    try {
      await fetch('/api/pipeline/stop', { method: 'POST' })
      stopPolling()
      await fetchStatus()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  function startPolling() {
    stopPolling()
    pollTimer = setInterval(fetchStatus, 2000)
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  // Restore state on init — resume polling if already recording
  fetchStatus().then(() => {
    if (status.value.recording) startPolling()
  })
  fetchPreflight()

  return {
    status, loading, error,
    preflight, preflightLoading, canStart,
    start, stop, fetchStatus, fetchPreflight,
  }
})
