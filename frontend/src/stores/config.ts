import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ConfigFile, GeneralConfig, ProtocolAvailability, ProtocolFieldError } from '../types/api'
import { usePipelineStore } from './pipeline'

const GENERAL_DEFAULTS: GeneralConfig = {
  study_name: 'default_study',
  subject_id: '01',
  session_id: '01',
  recording_name: 'recording',
}

export const useConfigStore = defineStore('config', () => {
  const general = ref<GeneralConfig>({ ...GENERAL_DEFAULTS })
  const nextRun = ref<number | null>(null)
  let _runTimer: ReturnType<typeof setTimeout> | null = null

  const output = ref<Record<string, any>>({})
  const outputAvailability = ref<ProtocolAvailability>({})
  const outputDefaults = ref<Record<string, Record<string, any>>>({})
  const outputErrors = ref<Record<string, ProtocolFieldError[]>>({})

  const availableConfigs = ref<ConfigFile[]>([])
  const knownStudyNames = ref<string[]>([])
  const validationErrors = ref<Record<string, string>>({})

  async function fetchGeneral() {
    const res = await fetch('/api/config/general')
    if (res.ok) general.value = await res.json()
  }

  async function updateGeneral(config: Partial<GeneralConfig>) {
    const merged = { ...general.value, ...config }
    // Fill empty required fields with defaults to pass BIDS validation
    if (!merged.study_name) merged.study_name = GENERAL_DEFAULTS.study_name
    if (!merged.subject_id) merged.subject_id = GENERAL_DEFAULTS.subject_id
    if (!merged.session_id) merged.session_id = GENERAL_DEFAULTS.session_id
    if (!merged.recording_name) merged.recording_name = GENERAL_DEFAULTS.recording_name
    const res = await fetch('/api/config/general', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(merged),
    })
    if (res.ok) {
      general.value = await res.json()
      validationErrors.value = {}
      usePipelineStore().fetchPreflight()
      debouncedFetchNextRun()
    } else {
      const err = await res.json()
      if (err.detail && Array.isArray(err.detail)) {
        const errors: Record<string, string> = {}
        for (const e of err.detail) {
          const field = e.loc?.[1] || 'unknown'
          errors[field] = e.msg
        }
        validationErrors.value = errors
      }
    }
  }

  async function fetchNextRun() {
    const { subject_id, session_id, recording_name } = general.value
    if (!subject_id || !session_id || !recording_name) return
    const params = new URLSearchParams({ subject_id, session_id, recording_name })
    const res = await fetch(`/api/config/next-run?${params}`)
    if (res.ok) nextRun.value = (await res.json()).run_number
  }

  function debouncedFetchNextRun() {
    if (_runTimer) clearTimeout(_runTimer)
    _runTimer = setTimeout(fetchNextRun, 300)
  }

  async function fetchOutput() {
    const res = await fetch('/api/config/output')
    if (!res.ok) return
    const data = await res.json()
    output.value = data.output || {}
  }

  async function updateOutput(protocols: Record<string, any>) {
    const res = await fetch('/api/config/output', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ protocols }),
    })
    if (res.ok) {
      const data = await res.json()
      output.value = data.output || {}
      outputErrors.value = {}
      usePipelineStore().fetchPreflight()
    } else {
      const err = await res.json()
      if (err.detail?.protocol_errors) {
        outputErrors.value = err.detail.protocol_errors
      }
    }
  }

  async function fetchOutputAvailability() {
    const res = await fetch('/api/config/output/availability')
    if (res.ok) outputAvailability.value = await res.json()
  }

  async function fetchOutputDefaults() {
    const res = await fetch('/api/config/output/defaults')
    if (res.ok) outputDefaults.value = await res.json()
  }

  async function listConfigs() {
    const res = await fetch('/api/config/list')
    if (res.ok) {
      const data = await res.json()
      availableConfigs.value = data.configs
      knownStudyNames.value = data.study_names ?? []
    }
  }

  async function loadConfig(filePath: string) {
    const res = await fetch(`/api/config/load?file_path=${encodeURIComponent(filePath)}`, {
      method: 'POST',
    })
    if (res.ok) {
      await fetchGeneral()
      await fetchOutput()
      // Refresh modes and streams from backend (restored by load_configuration)
      const { useModesStore } = await import('./modes')
      const { useStreamsStore } = await import('./streams')
      await useModesStore().fetchAll()
      await useStreamsStore().fetchConfigured()
      usePipelineStore().fetchPreflight()
    }
    return res.ok
  }

  async function saveConfig(filePath?: string): Promise<string | null> {
    const url = filePath
      ? `/api/config/save?file_path=${encodeURIComponent(filePath)}`
      : '/api/config/save'
    const res = await fetch(url, { method: 'POST' })
    if (res.ok) return (await res.json()).file_path
    return null
  }

  // Restore all config from backend on init
  fetchGeneral().then(fetchNextRun)
  fetchOutput()
  fetchOutputAvailability()
  fetchOutputDefaults()

  return {
    general, nextRun, output, outputAvailability, outputDefaults, outputErrors,
    availableConfigs, knownStudyNames, validationErrors,
    fetchGeneral, updateGeneral, fetchNextRun,
    fetchOutput, updateOutput, fetchOutputAvailability, fetchOutputDefaults,
    listConfigs, loadConfig, saveConfig,
  }
})
