import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useToast } from '../composables/useToast'
import type { ConfigFile, GeneralConfig, ProtocolAvailability, ProtocolFieldError } from '../types/api'
import { apiFetch, apiFetchOrNull, ApiError } from '../utils/api'
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
    const data = await apiFetchOrNull<GeneralConfig>('/api/config/general')
    if (data) general.value = data
  }

  async function updateGeneral(config: Partial<GeneralConfig>) {
    const merged = { ...general.value, ...config }
    // Fill empty required fields with defaults to pass BIDS validation
    if (!merged.study_name) merged.study_name = GENERAL_DEFAULTS.study_name
    if (!merged.subject_id) merged.subject_id = GENERAL_DEFAULTS.subject_id
    if (!merged.session_id) merged.session_id = GENERAL_DEFAULTS.session_id
    if (!merged.recording_name) merged.recording_name = GENERAL_DEFAULTS.recording_name
    try {
      general.value = await apiFetch('/api/config/general', {
        method: 'PUT',
        json: merged,
        silent: true,
      })
      validationErrors.value = {}
      usePipelineStore().fetchPreflight()
      debouncedFetchNextRun()
    } catch (e) {
      if (e instanceof ApiError && Array.isArray(e.detail)) {
        const errors: Record<string, string> = {}
        for (const item of e.detail) {
          const field = item.loc?.[1] || 'unknown'
          errors[field] = item.msg
        }
        validationErrors.value = errors
      }
    }
  }

  async function fetchNextRun() {
    const { subject_id, session_id, recording_name } = general.value
    if (!subject_id || !session_id || !recording_name) return
    const params = new URLSearchParams({ subject_id, session_id, recording_name })
    const data = await apiFetchOrNull<{ run_number: number }>(`/api/config/next-run?${params}`)
    if (data) nextRun.value = data.run_number
  }

  function debouncedFetchNextRun() {
    if (_runTimer) clearTimeout(_runTimer)
    _runTimer = setTimeout(fetchNextRun, 300)
  }

  async function fetchOutput() {
    const data = await apiFetchOrNull<{ output?: Record<string, any> }>('/api/config/output')
    if (data) output.value = data.output || {}
  }

  async function updateOutput(protocols: Record<string, any>) {
    try {
      const data = await apiFetch('/api/config/output', {
        method: 'PUT',
        json: { protocols },
        silent: true,
      })
      output.value = data.output || {}
      outputErrors.value = {}
      usePipelineStore().fetchPreflight()
    } catch (e) {
      if (e instanceof ApiError && e.detail?.protocol_errors) {
        outputErrors.value = e.detail.protocol_errors
      }
    }
  }

  async function fetchOutputAvailability() {
    const data = await apiFetchOrNull<ProtocolAvailability>('/api/config/output/availability')
    if (data) outputAvailability.value = data
  }

  async function fetchOutputDefaults() {
    const data = await apiFetchOrNull<Record<string, Record<string, any>>>('/api/config/output/defaults')
    if (data) outputDefaults.value = data
  }

  async function listConfigs() {
    const data = await apiFetchOrNull<{ configs: ConfigFile[]; study_names?: string[] }>('/api/config/list')
    if (data) {
      availableConfigs.value = data.configs
      knownStudyNames.value = data.study_names ?? []
    }
  }

  async function _refreshAfterApply() {
    await fetchGeneral()
    await fetchOutput()
    const { useModesStore } = await import('./modes')
    const { useStreamsStore } = await import('./streams')
    const { useDataStore } = await import('./data')
    await useModesStore().fetchAll()
    await useStreamsStore().fetchConfigured()
    // Backend auto-creates the study row on load/upload; refresh so the
    // data explorer reflects the (possibly new) study.
    await useDataStore().fetchStudies()
    usePipelineStore().fetchPreflight()
  }

  async function loadConfig(filePath: string) {
    try {
      await apiFetch(`/api/config/load?file_path=${encodeURIComponent(filePath)}`, {
        method: 'POST',
      })
      await _refreshAfterApply()
      return true
    } catch {
      return false
    }
  }

  async function uploadConfig(file: File): Promise<boolean> {
    let cfg: unknown
    try {
      cfg = JSON.parse(await file.text())
    } catch {
      useToast().error('File is not valid JSON')
      return false
    }
    try {
      await apiFetch('/api/config/apply', {
        method: 'POST',
        json: cfg,
        fallbackMessage: 'Failed to apply configuration',
      })
      await _refreshAfterApply()
      await listConfigs()
      return true
    } catch {
      return false
    }
  }

  async function saveConfig(filePath?: string): Promise<string | null> {
    const url = filePath
      ? `/api/config/save?file_path=${encodeURIComponent(filePath)}`
      : '/api/config/save'
    const data = await apiFetchOrNull<{ file_path: string }>(url, { method: 'POST' })
    return data?.file_path ?? null
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
    listConfigs, loadConfig, uploadConfig, saveConfig,
  }
})
