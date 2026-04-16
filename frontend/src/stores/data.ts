import { defineStore } from 'pinia'
import { ref, computed, type Ref } from 'vue'
import type { Study, StudyDetail, Recording, Decoder, DecoderMetadata, H5FileInfo, ChannelInfo, SignalPreview, ERPPreview, EventSummary, SessionSummary, RecordingTelemetry, ModePerformance } from '../types/api'
import { apiFetch, apiFetchOrNull } from '../utils/api'

export const useDataStore = defineStore('data', () => {
  // --- State ---
  const studies = ref<Study[]>([])
  const recordings = ref<Recording[]>([])
  const decoders = ref<Decoder[]>([])

  const expandedStudyId = ref<number | null>(null)
  const selectedStudyDetail = ref<StudyDetail | null>(null)
  const selectedRecording = ref<Recording | null>(null)
  const selectedDecoder = ref<Decoder | null>(null)
  const decoderMetadata = ref<DecoderMetadata | null>(null)

  const searchQuery = ref('')
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Detail panel state
  const recordingFileInfo = ref<H5FileInfo | null>(null)
  const recordingChannels = ref<ChannelInfo | null>(null)
  const signalPreview = ref<SignalPreview | null>(null)
  const erpPreview = ref<ERPPreview | null>(null)
  const eventSummary = ref<EventSummary | null>(null)
  const sessionSummary = ref<SessionSummary | null>(null)
  const recordingTelemetry = ref<RecordingTelemetry | null>(null)
  const modePerformance = ref<ModePerformance | null>(null)

  // --- Derived ---

  const recordingDecoders = computed(() => {
    const rid = selectedRecording.value?.recording_id
    if (rid == null) return []
    return decoders.value.filter(d => {
      if (!d.training_recording_ids) return false
      try {
        const ids: number[] = JSON.parse(d.training_recording_ids)
        return ids.includes(rid)
      } catch { return false }
    })
  })

  // --- Studies ---

  async function fetchStudies() {
    const data = await apiFetchOrNull<Study[]>('/api/data/studies')
    if (data) studies.value = data
  }

  async function fetchStudyDetail(studyId: number) {
    const data = await apiFetchOrNull<StudyDetail>(`/api/data/studies/${studyId}`)
    if (data) selectedStudyDetail.value = data
  }

  async function createStudy(name: string, description?: string) {
    const data = await apiFetch('/api/data/studies', {
      method: 'POST',
      json: { study_name: name, description },
      fallbackMessage: 'Failed to create study',
    })
    await fetchStudies()
    return data
  }

  async function updateStudy(studyId: number, description?: string) {
    try {
      await apiFetch(`/api/data/studies/${studyId}`, {
        method: 'PUT',
        json: { description },
      })
      await fetchStudies()
      await fetchStudyDetail(studyId)
      return true
    } catch {
      return false
    }
  }

  async function deleteStudy(studyId: number) {
    try {
      await apiFetch(`/api/data/studies/${studyId}`, { method: 'DELETE' })
      await fetchStudies()
      if (selectedStudyDetail.value?.study_id === studyId) {
        selectedStudyDetail.value = null
      }
      if (expandedStudyId.value === studyId) {
        expandedStudyId.value = null
        recordings.value = []
      }
      return true
    } catch {
      return false
    }
  }

  // --- Study expand (tree) ---

  async function expandStudy(studyId: number) {
    if (expandedStudyId.value === studyId) {
      expandedStudyId.value = null
      recordings.value = []
      decoders.value = []
      return
    }
    expandedStudyId.value = studyId
    const data = await apiFetchOrNull<Recording[]>(`/api/data/recordings?study_id=${studyId}`)
    if (data) recordings.value = data
    fetchDecoders(studyId)
  }

  function selectStudy(studyId: number) {
    selectedRecording.value = null
    selectedDecoder.value = null
    decoderMetadata.value = null
    resetRecordingDetail()
    fetchStudyDetail(studyId)
    if (expandedStudyId.value !== studyId) {
      expandStudy(studyId)
    }
  }

  // --- Recordings ---

  function resetRecordingDetail() {
    recordingFileInfo.value = null
    recordingChannels.value = null
    signalPreview.value = null
    erpPreview.value = null
    eventSummary.value = null
    sessionSummary.value = null
    recordingTelemetry.value = null
    modePerformance.value = null
  }

  async function selectRecording(recordingId: number) {
    selectedStudyDetail.value = null
    selectedDecoder.value = null
    decoderMetadata.value = null
    selectedRecording.value = await apiFetchOrNull<Recording>(`/api/data/recordings/${recordingId}`)
    resetRecordingDetail()
    if (selectedRecording.value) {
      const id = selectedRecording.value.recording_id
      const sid = selectedRecording.value.study_id
      fetchSessionSummary(id)
      fetchRecordingChannels(id)
      fetchDecoders(sid)
    }
  }

  async function deleteRecording(recordingId: number) {
    try {
      await apiFetch(`/api/data/recordings/${recordingId}`, { method: 'DELETE' })
      if (selectedRecording.value?.recording_id === recordingId) {
        selectedRecording.value = null
        resetRecordingDetail()
      }
      if (expandedStudyId.value) {
        const data = await apiFetchOrNull<Recording[]>(`/api/data/recordings?study_id=${expandedStudyId.value}`)
        if (data) recordings.value = data
      }
      return true
    } catch {
      return false
    }
  }

  async function _fetchDetail(url: string, target: Ref<any>) {
    loading.value = true
    const data = await apiFetchOrNull(url)
    if (data !== null) target.value = data
    loading.value = false
  }

  const fetchRecordingFileInfo = (id: number) => _fetchDetail(`/api/data/recordings/${id}/file-info`, recordingFileInfo)
  const fetchRecordingChannels = (id: number) => _fetchDetail(`/api/data/recordings/${id}/channels`, recordingChannels)
  const fetchSignalPreview = (id: number) => _fetchDetail(`/api/data/recordings/${id}/signal-preview`, signalPreview)
  const fetchEventSummary = (id: number) => _fetchDetail(`/api/data/recordings/${id}/event-summary`, eventSummary)
  const fetchSessionSummary = (id: number) => _fetchDetail(`/api/data/recordings/${id}/summary`, sessionSummary)
  const fetchRecordingTelemetry = (id: number) => _fetchDetail(`/api/data/recordings/${id}/telemetry`, recordingTelemetry)
  const fetchModePerformance = (id: number) => _fetchDetail(`/api/data/recordings/${id}/mode-performance`, modePerformance)

  async function fetchErpPreview(recordingId: number, params?: {
    epoch_tmin?: number, epoch_tmax?: number, lowcut?: number, highcut?: number, apply_rereferencing?: boolean
  }) {
    loading.value = true
    const q = new URLSearchParams()
    if (params?.epoch_tmin != null) q.set('epoch_tmin', String(params.epoch_tmin))
    if (params?.epoch_tmax != null) q.set('epoch_tmax', String(params.epoch_tmax))
    if (params?.lowcut != null) q.set('lowcut', String(params.lowcut))
    if (params?.highcut != null) q.set('highcut', String(params.highcut))
    if (params?.apply_rereferencing != null) q.set('apply_rereferencing', String(params.apply_rereferencing))
    const qs = q.toString() ? `?${q}` : ''
    const data = await apiFetchOrNull<ERPPreview>(`/api/data/recordings/${recordingId}/erp${qs}`)
    if (data) erpPreview.value = data
    loading.value = false
  }

  // --- Decoders ---

  async function fetchDecoders(studyId?: number) {
    const params = new URLSearchParams()
    if (studyId) params.set('study_id', String(studyId))
    const qs = params.toString()
    const data = await apiFetchOrNull<Decoder[]>(`/api/data/decoders${qs ? '?' + qs : ''}`)
    if (data) decoders.value = data
  }

  async function selectDecoder(decoderId: number) {
    selectedStudyDetail.value = null
    selectedRecording.value = null
    decoderMetadata.value = null
    resetRecordingDetail()
    const decoder = await apiFetchOrNull<Decoder>(`/api/data/decoders/${decoderId}`)
    if (!decoder) return
    selectedDecoder.value = decoder
    decoderMetadata.value = await apiFetchOrNull<DecoderMetadata>(`/api/data/decoders/${decoderId}/metadata`)
    // Ensure recordings are loaded for provenance links
    if (decoder.study_id) {
      const recs = await apiFetchOrNull<Recording[]>(`/api/data/recordings?study_id=${decoder.study_id}`)
      if (recs) recordings.value = recs
    }
  }

  async function deleteDecoder(decoderId: number) {
    try {
      await apiFetch(`/api/data/decoders/${decoderId}`, { method: 'DELETE' })
      if (selectedDecoder.value?.decoder_id === decoderId) {
        selectedDecoder.value = null
        decoderMetadata.value = null
      }
      // Refresh decoders if viewing a study
      if (selectedStudyDetail.value) {
        await fetchDecoders(selectedStudyDetail.value.study_id)
      }
      return true
    } catch {
      return false
    }
  }

  // --- Helpers ---

  function clearSelection() {
    selectedStudyDetail.value = null
    selectedRecording.value = null
    selectedDecoder.value = null
    decoderMetadata.value = null
    resetRecordingDetail()
  }

  async function importStudyFolder(config: { folder_path: string; study_name: string; description?: string }) {
    const result = await apiFetch('/api/data/studies/import-folder', {
      method: 'POST',
      json: config,
      fallbackMessage: 'Failed to import study folder',
    })
    await fetchStudies()
    return result
  }

  async function pickFolder(): Promise<string | null> {
    const data = await apiFetchOrNull<{ path?: string }>('/api/data/studies/pick-folder', { method: 'POST' })
    return data?.path ?? null
  }

  return {
    // State
    studies, recordings, decoders,
    expandedStudyId, selectedStudyDetail, selectedRecording, selectedDecoder, decoderMetadata,
    searchQuery, loading, error,
    recordingFileInfo, recordingChannels, signalPreview, erpPreview, eventSummary, sessionSummary, recordingTelemetry, modePerformance,
    // Derived
    recordingDecoders,
    // Studies
    fetchStudies, fetchStudyDetail, createStudy, updateStudy, deleteStudy, expandStudy, selectStudy,
    // Recordings
    selectRecording, deleteRecording,
    fetchRecordingFileInfo, fetchRecordingChannels, fetchSignalPreview, fetchErpPreview, fetchEventSummary, fetchSessionSummary, fetchRecordingTelemetry, fetchModePerformance,
    // Decoders
    fetchDecoders, selectDecoder, deleteDecoder,
    // Helpers
    clearSelection, importStudyFolder, pickFolder,
  }
})
