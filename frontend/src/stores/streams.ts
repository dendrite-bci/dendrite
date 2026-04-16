import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { StreamMetadata, StreamModalities } from '../types/api'
import { apiFetch, apiFetchOrNull } from '../utils/api'
import { usePipelineStore } from './pipeline'

export const useStreamsStore = defineStore('streams', () => {
  const discoveredStreams = ref<Record<string, StreamMetadata>>({})
  const configuredStreams = ref<Record<string, StreamMetadata>>({})
  const modalitiesByStream = ref<Record<string, StreamModalities>>({})
  const isDiscovering = ref(false)
  const liveness = ref<Record<string, boolean>>({})
  const isCheckingLiveness = ref(false)

  const hasStreams = computed(() => Object.keys(configuredStreams.value).length > 0)

  /** All unique modality keys across all streams. */
  const allModalities = computed(() => {
    const mods = new Set<string>()
    for (const entry of Object.values(modalitiesByStream.value)) {
      for (const mod of Object.keys(entry.modalities)) mods.add(mod)
    }
    return [...mods]
  })

  /** Get stream UIDs that have a given modality. */
  function streamsForModality(modality: string): string[] {
    return Object.entries(modalitiesByStream.value)
      .filter(([, entry]) => modality in entry.modalities)
      .map(([uid]) => uid)
  }

  /** Get sample rate for a modality by finding its stream. */
  function getSampleRate(modality: string): number | null {
    for (const entry of Object.values(modalitiesByStream.value)) {
      if (modality in entry.modalities && entry.sample_rate) return entry.sample_rate
    }
    return null
  }

  async function discover() {
    if (usePipelineStore().status.recording) return
    isDiscovering.value = true
    try {
      const data = await apiFetch('/api/streams/discover', { method: 'POST' })
      discoveredStreams.value = data.streams
      configuredStreams.value = {}
      modalitiesByStream.value = {}
      liveness.value = {}
      stopLivenessPolling()
    } catch {
      // toast surfaced by apiFetch
    } finally {
      isDiscovering.value = false
    }
  }

  async function configure(selectedUids: string[], channelOverrides?: Record<string, any>) {
    try {
      const data = await apiFetch('/api/streams/configure', {
        method: 'POST',
        json: { selected_uids: selectedUids, channel_overrides: channelOverrides || {} },
      })
      configuredStreams.value = data.configured
      modalitiesByStream.value = data.modalities_by_stream
      liveness.value = {}
      usePipelineStore().fetchPreflight()
      startLivenessPolling()
    } catch {
      // toast surfaced by apiFetch
    }
  }

  async function fetchConfigured() {
    const data = await apiFetchOrNull<{ streams: Record<string, StreamMetadata>; modalities_by_stream: Record<string, StreamModalities> }>('/api/streams')
    if (!data) return
    configuredStreams.value = data.streams
    modalitiesByStream.value = data.modalities_by_stream
    usePipelineStore().fetchPreflight()
    if (Object.keys(data.streams).length > 0) {
      startLivenessPolling()
    }
  }

  let livenessTimer: ReturnType<typeof setInterval> | null = null

  async function checkLiveness() {
    if (!hasStreams.value) return
    isCheckingLiveness.value = true
    const data = await apiFetchOrNull<{ liveness: Record<string, boolean> }>('/api/streams/liveness')
    if (data) liveness.value = data.liveness
    isCheckingLiveness.value = false
  }

  function startLivenessPolling() {
    stopLivenessPolling()
    checkLiveness()
    livenessTimer = setInterval(checkLiveness, 30_000)
  }

  function stopLivenessPolling() {
    if (livenessTimer) {
      clearInterval(livenessTimer)
      livenessTimer = null
    }
  }

  fetchConfigured()

  return {
    discoveredStreams, configuredStreams, modalitiesByStream,
    allModalities, streamsForModality, getSampleRate,
    isDiscovering, hasStreams, liveness, isCheckingLiveness,
    discover, configure, fetchConfigured, checkLiveness,
  }
})
