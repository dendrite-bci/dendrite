import { defineStore } from 'pinia'
import { ref, computed, type Ref } from 'vue'
import type { Study, StudyDetail, Recording, Decoder, DecoderMetadata, H5FileInfo, ChannelInfo, SignalPreview, ERPPreview, EventSummary, SessionSummary, RecordingTelemetry, ModePerformance } from '../types/api'

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
    const res = await fetch('/api/data/studies')
    if (res.ok) studies.value = await res.json()
  }

  async function fetchStudyDetail(studyId: number) {
    const res = await fetch(`/api/data/studies/${studyId}`)
    if (res.ok) selectedStudyDetail.value = await res.json()
  }

  async function createStudy(name: string, description?: string) {
    const res = await fetch('/api/data/studies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ study_name: name, description }),
    })
    if (res.ok) {
      await fetchStudies()
      return await res.json()
    }
    return null
  }

  async function updateStudy(studyId: number, description?: string) {
    const res = await fetch(`/api/data/studies/${studyId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description }),
    })
    if (res.ok) {
      await fetchStudies()
      await fetchStudyDetail(studyId)
    }
    return res.ok
  }

  async function deleteStudy(studyId: number) {
    const res = await fetch(`/api/data/studies/${studyId}`, { method: 'DELETE' })
    if (res.ok) {
      await fetchStudies()
      if (selectedStudyDetail.value?.study_id === studyId) {
        selectedStudyDetail.value = null
      }
      if (expandedStudyId.value === studyId) {
        expandedStudyId.value = null
        recordings.value = []
      }
    }
    return res.ok
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
    const res = await fetch(`/api/data/recordings?study_id=${studyId}`)
    if (res.ok) recordings.value = await res.json()
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
    const res = await fetch(`/api/data/recordings/${recordingId}`)
    if (res.ok) selectedRecording.value = await res.json()
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
    const res = await fetch(`/api/data/recordings/${recordingId}`, { method: 'DELETE' })
    if (res.ok) {
      if (selectedRecording.value?.recording_id === recordingId) {
        selectedRecording.value = null
        resetRecordingDetail()
      }
      // Refresh recordings for expanded study
      if (expandedStudyId.value) {
        const r = await fetch(`/api/data/recordings?study_id=${expandedStudyId.value}`)
        if (r.ok) recordings.value = await r.json()
      }
    }
    return res.ok
  }

  async function _fetchDetail(url: string, target: Ref<any>) {
    loading.value = true
    try {
      const res = await fetch(url)
      if (res.ok) target.value = await res.json()
    } finally {
      loading.value = false
    }
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
    try {
      const q = new URLSearchParams()
      if (params?.epoch_tmin != null) q.set('epoch_tmin', String(params.epoch_tmin))
      if (params?.epoch_tmax != null) q.set('epoch_tmax', String(params.epoch_tmax))
      if (params?.lowcut != null) q.set('lowcut', String(params.lowcut))
      if (params?.highcut != null) q.set('highcut', String(params.highcut))
      if (params?.apply_rereferencing != null) q.set('apply_rereferencing', String(params.apply_rereferencing))
      const qs = q.toString() ? `?${q}` : ''
      const res = await fetch(`/api/data/recordings/${recordingId}/erp${qs}`)
      if (res.ok) erpPreview.value = await res.json()
    } finally {
      loading.value = false
    }
  }

  // --- Decoders ---

  async function fetchDecoders(studyId?: number) {
    const params = new URLSearchParams()
    if (studyId) params.set('study_id', String(studyId))
    const qs = params.toString()
    const res = await fetch(`/api/data/decoders${qs ? '?' + qs : ''}`)
    if (res.ok) decoders.value = await res.json()
  }

  async function selectDecoder(decoderId: number) {
    selectedStudyDetail.value = null
    selectedRecording.value = null
    decoderMetadata.value = null
    resetRecordingDetail()
    const res = await fetch(`/api/data/decoders/${decoderId}`)
    if (res.ok) {
      selectedDecoder.value = await res.json()
      // Load rich metadata from the decoder JSON file
      const metaRes = await fetch(`/api/data/decoders/${decoderId}/metadata`)
      if (metaRes.ok) decoderMetadata.value = await metaRes.json()
      // Ensure recordings are loaded for provenance links
      const sid = selectedDecoder.value?.study_id
      if (sid) {
        const recRes = await fetch(`/api/data/recordings?study_id=${sid}`)
        if (recRes.ok) recordings.value = await recRes.json()
      }
    }
  }

  async function deleteDecoder(decoderId: number) {
    const res = await fetch(`/api/data/decoders/${decoderId}`, { method: 'DELETE' })
    if (res.ok) {
      if (selectedDecoder.value?.decoder_id === decoderId) {
        selectedDecoder.value = null
        decoderMetadata.value = null
      }
      // Refresh decoders if viewing a study
      if (selectedStudyDetail.value) {
        await fetchDecoders(selectedStudyDetail.value.study_id)
      }
    }
    return res.ok
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
    const res = await fetch('/api/data/studies/import-folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Import failed' }))
      throw new Error(err.detail || 'Import failed')
    }
    const result = await res.json()
    await fetchStudies()
    return result
  }

  async function pickFolder(): Promise<string | null> {
    const res = await fetch('/api/data/studies/pick-folder', { method: 'POST' })
    if (!res.ok) return null
    const { path } = await res.json()
    return path ?? null
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
