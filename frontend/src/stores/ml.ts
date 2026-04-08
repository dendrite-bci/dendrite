import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  ModelInfo,
  TrainingJob,
  TrainingProgress,
  MoabbDataset,
  LoadedDataInfo,
  Recording,
} from '../types/api'
import { useToast } from '../composables/useToast'

/** Non-reactive dirty flag for high-frequency eval timeline updates (same pattern as vizDirty) */
export const evalDirty = { timelineChanged: false }

export const useMLStore = defineStore('ml', () => {
  // --- Global State ---
  const loading = ref(false)

  // --- Data Tab ---
  const moabbDatasets = ref<MoabbDataset[]>([])
  const moabbLoading = ref(false)
  const loadedData = ref<LoadedDataInfo | null>(null)
  const evalData = ref<LoadedDataInfo | null>(null)
  const dataLoading = ref(false)
  const dataLoadingStep = ref<string | null>(null)
  const selectedMoabbDataset = ref<MoabbDataset | null>(null)
  const dataSourceTab = ref<'moabb' | 'recordings'>('recordings')

  // Recording browser
  const recordings = ref<Recording[]>([])
  const selectedStudyId = ref<number | null>(null)
  const recordingRoles = ref<Record<number, 'train' | 'eval'>>({})
  const recordingEventSummaries = ref<Record<number, Record<string, number>>>({})

  // Preprocessing config
  const dataPreproc = ref({
    lowcut: null as number | null,
    highcut: null as number | null,
    apply_rereferencing: false,
    subject: 1,
    epoch_tmin: -0.2,
    epoch_tmax: 0.8,
    selected_events: null as string[] | null,
    selected_channels: null as number[] | null,
    use_epoch_qc: true,
    include_background: false,
    eval_split: 0.2,
    use_paradigm_epochs: false,
  })

  // Event groups (persisted globally in localStorage)
  const eventGroups = ref<{ name: string; events: string[] }[]>([])
  try {
    const raw = localStorage.getItem('dendrite_event_groups')
    if (raw) eventGroups.value = JSON.parse(raw)
  } catch { /* ignore corrupt data */ }

  function _persistEventGroups() {
    localStorage.setItem('dendrite_event_groups', JSON.stringify(eventGroups.value))
  }

  function saveEventGroup(name: string) {
    const events = dataPreproc.value.selected_events
    if (!events || events.length === 0) return
    eventGroups.value = [...eventGroups.value.filter(g => g.name !== name), { name, events: [...events] }]
    _persistEventGroups()
  }

  function deleteEventGroup(name: string) {
    eventGroups.value = eventGroups.value.filter(g => g.name !== name)
    _persistEventGroups()
  }

  function applyEventGroup(events: string[] | null) {
    dataPreproc.value.selected_events = events ? [...events] : null
  }

  // --- Training Tab ---
  const models = ref<ModelInfo[]>([])
  const jobs = ref<TrainingJob[]>([])
  const selectedJob = ref<TrainingJob | null>(null)
  const trainingProgress = ref<Record<number, TrainingProgress>>({})
  const modelSchema = ref<Record<string, any> | null>(null)
  const searchCategories = ref<Record<string, { label: string; params: string[] }>>({})
  const searchTotalParams = ref(0)

  const trainingConfig = ref({
    model_type: 'EEGNet',
    pipeline_steps: null as string[] | null,
    study_id: null as number | null,
    num_classes: 2,
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2,
    use_early_stopping: true,
    early_stopping_patience: 10,
    holdout_ratio: 0.0,
    optuna_enabled: false,
    optuna_n_trials: 30,
    search_categories: ['optimizer'] as string[],
    use_loaded_data: false,
    model_params: {} as Record<string, any>,
    optimizer_type: 'Adam' as 'Adam' | 'AdamW',
    weight_decay: 0.0,
    use_augmentation: false,
    aug_strategy: 'moderate' as 'light' | 'moderate' | 'aggressive',
    use_class_weights: true,
    use_lr_scheduler: true,
    lr_scheduler_type: 'OneCycleLR',
    loss_type: 'cross_entropy' as 'cross_entropy' | 'focal',
    label_smoothing_factor: 0.0,
    mixup_alpha: 0.0,
  })

  // --- Evaluation Tab ---
  const evalConfig = ref({
    job_id: null as number | null,
    mode: 'sliding_window' as 'epoch' | 'sliding_window',
    step_size_ms: 100,
  })
  const evalGate = ref({
    detection_strategy: 'dwell' as 'dwell' | 'majority',
    dwell_n: 3,
    confidence_threshold: 0,
  })
  const evalMetrics = ref<Record<string, any> | null>(null)
  const evalRunning = ref(false)
  const evalJobId = ref<number | null>(null)
  const liveEval = ref<{
    timeline: Array<{ time_s: number; confidence: number; correct: boolean; prediction: number; trial_idx: number }>,
    trials: Array<Record<string, any>>,
    step: number, total: number,
  } | null>(null)

  // --- Benchmark Tab ---
  const benchConfig = ref({
    model_types: [] as string[],
    n_folds: 5,
  })
  const benchResults = ref<any[]>([])
  const benchRunning = ref(false)
  const benchJobId = ref<number | null>(null)

  // WebSocket
  const toast = useToast()
  let ws: WebSocket | null = null

  // ============================================================
  // Data Tab Actions
  // ============================================================

  async function discoverMoabb() {
    moabbLoading.value = true
    try {
      const res = await fetch('/api/ml/moabb/datasets')
      if (res.ok) moabbDatasets.value = await res.json()
      else toast.error('Failed to discover MOABB datasets')
    } catch (e: any) {
      toast.error(e.message)
    } finally {
      moabbLoading.value = false
    }
  }

  function _preprocBody() {
    const pp = dataPreproc.value
    return {
      lowcut: pp.lowcut,
      highcut: pp.highcut,
      apply_rereferencing: pp.apply_rereferencing,
      epoch_tmin: pp.epoch_tmin,
      epoch_tmax: pp.epoch_tmax,
      selected_events: pp.selected_events,
      selected_channels: pp.selected_channels,
      use_epoch_qc: pp.use_epoch_qc,
      include_background: pp.include_background,
    }
  }

  async function loadMoabbDataset(code: string) {
    dataLoading.value = true
    try {
      const res = await fetch('/api/ml/moabb/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_code: code,
          subject: dataPreproc.value.subject,
          paradigm: selectedMoabbDataset.value?.paradigm ?? 'MotorImagery',
          lowcut: dataPreproc.value.lowcut,
          highcut: dataPreproc.value.highcut,
          apply_rereferencing: dataPreproc.value.apply_rereferencing,
          eval_split: dataPreproc.value.eval_split,
          use_paradigm_epochs: dataPreproc.value.use_paradigm_epochs,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        evalData.value = data.eval ?? null
        delete data.eval
        loadedData.value = data
        _onDataLoaded()
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Failed to load dataset')
      }
    } catch (e: any) {
      toast.error(e.message)
    } finally {
      dataLoading.value = false
      dataLoadingStep.value = null
    }
  }

  function _onDataLoaded() {
    dataPreproc.value.selected_channels = null
    const nClasses = loadedData.value?.metadata?.class_names?.length
    if (nClasses && nClasses >= 2) {
      trainingConfig.value.num_classes = nClasses
      trainingConfig.value.use_loaded_data = true
    }
  }

  async function fetchRecordings(studyId?: number) {
    try {
      const params = new URLSearchParams()
      if (studyId != null) params.set('study_id', String(studyId))
      const qs = params.toString()
      const res = await fetch(`/api/data/recordings${qs ? '?' + qs : ''}`)
      if (res.ok) {
        recordings.value = await res.json()
      }
    } catch { /* network error — silently degrade */ }
  }

  const _inflight = new Set<number>()
  async function fetchRecordingEvents(recordingIds: number[]) {
    const uncached = recordingIds.filter(
      id => !(id in recordingEventSummaries.value) && !_inflight.has(id)
    )
    if (uncached.length === 0) return
    for (const id of uncached) _inflight.add(id)
    const results = await Promise.allSettled(
      uncached.map(async (id) => {
        const res = await fetch(`/api/data/recordings/${id}/event-summary`)
        if (res.ok) {
          const data = await res.json()
          return { id, events: data.event_types as Record<string, number> }
        }
        return null
      })
    )
    let changed = false
    for (const r of results) {
      if (r.status === 'fulfilled' && r.value) {
        recordingEventSummaries.value[r.value.id] = r.value.events
        changed = true
      }
    }
    for (const id of uncached) _inflight.delete(id)
    if (changed) recordingEventSummaries.value = { ...recordingEventSummaries.value }
  }

  function saveStudyState() {
    const sid = selectedStudyId.value
    if (sid == null) return
    localStorage.setItem(`ml_study_${sid}`, JSON.stringify({
      recordingRoles: recordingRoles.value,
      dataPreproc: dataPreproc.value,
    }))
  }

  function restoreStudyState(studyId: number) {
    const raw = localStorage.getItem(`ml_study_${studyId}`)
    if (!raw) return
    try {
      const saved = JSON.parse(raw)
      if (saved.recordingRoles) {
        recordingRoles.value = saved.recordingRoles
        fetchRecordingEvents(Object.keys(saved.recordingRoles).map(Number))
      }
      if (saved.dataPreproc) Object.assign(dataPreproc.value, saved.dataPreproc)
    } catch { /* ignore corrupt data */ }
  }

  async function loadPool() {
    const trainIds = Object.entries(recordingRoles.value)
      .filter(([, role]) => role === 'train')
      .map(([id]) => Number(id))
    const evalIds = Object.entries(recordingRoles.value)
      .filter(([, role]) => role === 'eval')
      .map(([id]) => Number(id))

    if (trainIds.length === 0) return
    dataLoading.value = true
    try {
      const res = await fetch('/api/ml/load-recording', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recording_ids: trainIds,
          eval_recording_ids: evalIds.length > 0 ? evalIds : undefined,
          eval_split: evalIds.length === 0 ? dataPreproc.value.eval_split : undefined,
          ..._preprocBody(),
        }),
      })
      if (res.ok) {
        const data = await res.json()
        evalData.value = data.eval ?? null
        delete data.eval
        loadedData.value = data
        _onDataLoaded()
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Failed to load recordings')
      }
    } catch (e: any) {
      toast.error(e.message)
    } finally {
      dataLoading.value = false
      dataLoadingStep.value = null
    }
  }

  // ============================================================
  // Training Tab Actions
  // ============================================================

  async function fetchModels() {
    const res = await fetch('/api/ml/models')
    if (res.ok) models.value = await res.json()
  }

  async function fetchModelSchema(modelType: string) {
    const res = await fetch(`/api/ml/models/${encodeURIComponent(modelType)}/schema`)
    if (res.ok) {
      modelSchema.value = await res.json()
    } else {
      modelSchema.value = null
    }
  }

  function applySearchParams(params: Record<string, any>) {
    const tc = trainingConfig.value
    const knownKeys = new Set(Object.keys(tc))
    for (const [key, val] of Object.entries(params)) {
      if (key === 'model_type') continue
      if (knownKeys.has(key)) {
        ;(tc as any)[key] = val
      }
      // Unknown keys are ignored — don't dump into model_params.
      // Model architecture params are applied separately via the schema UI.
    }
  }

  async function fetchSearchCategories(decoderType: string) {
    const res = await fetch(`/api/ml/search-categories/${encodeURIComponent(decoderType)}`)
    if (res.ok) {
      const data = await res.json()
      searchCategories.value = data.categories ?? {}
      searchTotalParams.value = data.total_params ?? 0
      // Reset selected categories to only available ones
      const available = Object.keys(searchCategories.value)
      trainingConfig.value.search_categories = trainingConfig.value.search_categories.filter(
        c => available.includes(c),
      )
      if (trainingConfig.value.search_categories.length === 0 && available.length > 0) {
        trainingConfig.value.search_categories = [available[0]!]
      }
    }
  }

  async function fetchJobs(studyId?: number, jobType?: string) {
    const params = new URLSearchParams()
    if (studyId != null) params.set('study_id', String(studyId))
    if (jobType) params.set('job_type', jobType)
    const qs = params.toString()
    const res = await fetch(`/api/ml/jobs${qs ? '?' + qs : ''}`)
    if (res.ok) jobs.value = await res.json()
  }

  async function selectJob(job: TrainingJob) {
    selectedJob.value = job
    // Lazy-fetch full result_json (excluded from list query for efficiency)
    if (job.status === 'completed' && !job.result_json) {
      const res = await fetch(`/api/ml/jobs/${job.job_id}`)
      if (res.ok) {
        const full = await res.json()
        if (full.result_json) {
          job = { ...job, result_json: full.result_json }
          selectedJob.value = job
        }
      }
    }
    // Restore eval results into evalMetrics so EvalResultsPanel displays them
    if (job.job_type === 'evaluation' && job.status === 'completed' && job.result_json) {
      try {
        const result = JSON.parse(job.result_json)
        result._job_id = job.job_id
        evalMetrics.value = result
        evalJobId.value = job.job_id
      } catch (e) { console.warn('Malformed eval result_json', e) }
    }
  }

  async function startTraining() {
    if (!selectedStudyId.value) {
      toast.error('Select a study before training')
      return
    }
    loading.value = true
    try {
      // Derive primary modality from selected channels' types
      const types = loadedData.value?.channel_types ?? []
      const sel = dataPreproc.value.selected_channels
      const selectedTypes = sel ? sel.map(i => types[i]?.toLowerCase() ?? 'eeg') : types.map(t => t?.toLowerCase() ?? 'eeg')
      const modality = selectedTypes[0] ?? 'eeg'

      const res = await fetch('/api/ml/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...trainingConfig.value,
          modality,
          study_id: selectedStudyId.value,
          selected_channels: dataPreproc.value.selected_channels,
          selected_events: dataPreproc.value.selected_events,
          lowcut: dataPreproc.value.lowcut,
          highcut: dataPreproc.value.highcut,
          apply_rereferencing: dataPreproc.value.apply_rereferencing,
          epoch_tmin: dataPreproc.value.epoch_tmin,
          epoch_tmax: dataPreproc.value.epoch_tmax,
          use_epoch_qc: dataPreproc.value.use_epoch_qc,
          include_background: dataPreproc.value.include_background,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        await fetchJobs()
        const newJob = jobs.value.find(j => j.job_id === data.job_id)
        if (newJob) selectedJob.value = newJob
        return data
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Training failed to start')
      }
    } catch (e: any) {
      toast.error(e.message)
    } finally {
      loading.value = false
    }
    return null
  }

  async function cancelTraining(jobId: number) {
    const res = await fetch(`/api/ml/train/${jobId}/cancel`, { method: 'POST' })
    if (res.ok) await fetchJobs()
    return res.ok
  }

  async function saveDecoder(jobId: number, name: string, description?: string) {
    const res = await fetch(`/api/ml/jobs/${jobId}/save-decoder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decoder_name: name, description }),
    })
    if (res.ok) {
      await fetchJobs()
      return await res.json()
    }
    return null
  }

  async function deleteJob(jobId: number) {
    const res = await fetch(`/api/ml/jobs/${jobId}`, { method: 'DELETE' })
    if (res.ok) {
      if (selectedJob.value?.job_id === jobId) selectedJob.value = null
      await fetchJobs()
    }
    return res.ok
  }

  // ============================================================
  // Evaluation & Benchmark
  // ============================================================

  async function startEvaluation() {
    evalRunning.value = true
    evalMetrics.value = null
    evalJobId.value = null
    liveEval.value = { timeline: [], trials: [], step: 0, total: 0 }
    try {
      const res = await fetch('/api/ml/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...evalConfig.value,
          ...evalGate.value,
          channel_indices: dataPreproc.value.selected_channels,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        toast.error(err.detail || 'Evaluation failed to start')
        evalRunning.value = false
      } else {
        const data = await res.json()
        evalJobId.value = data.job_id ?? null
      }
    } catch (e: any) {
      toast.error(e.message)
      evalRunning.value = false
    }
  }

  async function reaggregateEval() {
    if (!evalJobId.value && !evalMetrics.value) return
    const jobId = evalJobId.value ?? (evalMetrics.value as any)?._job_id
    if (!jobId) return
    try {
      const res = await fetch(`/api/ml/jobs/${jobId}/reaggregate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(evalGate.value),
      })
      if (res.ok) {
        const data = await res.json()
        // Merge aggregated metrics (now includes updated event_markers, ttd, itr)
        if (evalMetrics.value) {
          evalMetrics.value = { ...evalMetrics.value, ...data }
          evalDirty.timelineChanged = true
        }
      }
    } catch (e: any) {
      toast.error(e.message)
    }
  }

  async function startBenchmark() {
    benchRunning.value = true
    benchResults.value = []
    try {
      const res = await fetch('/api/ml/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...benchConfig.value,
          channel_indices: dataPreproc.value.selected_channels,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        benchJobId.value = data.job_id ?? null
      } else {
        const err = await res.json()
        toast.error(err.detail || 'Benchmark failed to start')
        benchRunning.value = false
      }
    } catch (e: any) {
      toast.error(e.message)
      benchRunning.value = false
    }
  }

  // ============================================================
  // WebSocket
  // ============================================================

  function connectWS() {
    if (ws) return
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    ws = new WebSocket(`${protocol}//${host}/ws/training`)

    ws.onmessage = (event) => {
      const msg: TrainingProgress = JSON.parse(event.data)
      trainingProgress.value = { ...trainingProgress.value, [msg.job_id]: msg }

      // Terminal states → refresh job list + toast
      const isTerminal = msg.type === 'complete' || msg.type === 'failed' || msg.type === 'cancelled'
      if (isTerminal) {
        fetchJobs()
        const { [msg.job_id]: _, ...rest } = trainingProgress.value
        trainingProgress.value = rest
      }
      if (msg.type === 'complete') toast.success('Training completed')
      else if (msg.type === 'failed') toast.error(msg.error || 'Training failed')

      // Update selected job progress
      if (selectedJob.value?.job_id === msg.job_id) {
        selectedJob.value = { ...selectedJob.value, progress: msg }
        if (msg.type === 'complete') {
          selectedJob.value.status = 'completed'
          if (msg.result) {
            selectedJob.value.result_json = JSON.stringify(msg.result)
          }
        } else if (msg.type === 'failed') {
          selectedJob.value.status = 'failed'
          selectedJob.value.error_message = msg.error || null
        }
      }

      // Evaluation results
      if (msg.type === 'eval_step') {
        if (liveEval.value) {
          const m = msg as any
          liveEval.value.step = m.step ?? 0
          liveEval.value.total = m.total ?? 0
          if (m.trial) liveEval.value.trials.push(m.trial)
          if (m.timeline?.length) liveEval.value.timeline.push(...m.timeline)
          evalDirty.timelineChanged = true  // non-reactive dirty flag for rAF loop
          liveEval.value = { ...liveEval.value }  // reactive trigger for v-if/computed
        }
      } else if (msg.type === 'eval_metrics') {
        const result = (msg as any).result ?? null
        if (result) result._job_id = msg.job_id
        evalMetrics.value = result
        evalRunning.value = false
        liveEval.value = null
        const { [msg.job_id]: _ep, ...restProgress } = trainingProgress.value
        trainingProgress.value = restProgress
        fetchJobs()
        toast.success('Evaluation completed')
      }
      if (msg.type === 'failed' && msg.job_id === evalJobId.value) {
        evalRunning.value = false
        evalJobId.value = null
        liveEval.value = null
        toast.error(msg.error || 'Evaluation failed')
      }

      // Benchmark progress
      if (msg.type === 'bench_model_complete' && msg.result && msg.job_id === benchJobId.value) {
        benchResults.value.push(msg.result)
      }
      if (isTerminal && msg.job_id === benchJobId.value) {
        benchRunning.value = false
        benchJobId.value = null
        if (msg.type === 'complete') toast.success('Benchmark completed')
        else if (msg.type === 'failed') toast.error(msg.error || 'Benchmark failed')
      }

      // Data loading step progress
      if (msg.type === 'data_loading_step') {
        dataLoadingStep.value = (msg as any).step ?? null
      }

    }

    ws.onclose = () => { ws = null }
  }

  function disconnectWS() {
    if (ws) {
      ws.close()
      ws = null
    }
  }

  return {
    // Global
    loading,
    // Data
    moabbDatasets, moabbLoading, loadedData, evalData, dataLoading, dataLoadingStep,
    selectedMoabbDataset, dataSourceTab, dataPreproc,
    recordings, selectedStudyId, recordingRoles, recordingEventSummaries,
    discoverMoabb, loadMoabbDataset, fetchRecordings, fetchRecordingEvents, loadPool,
    saveStudyState, restoreStudyState,
    eventGroups, saveEventGroup, deleteEventGroup, applyEventGroup,
    // Training
    models, jobs, selectedJob, trainingProgress, trainingConfig,
    modelSchema, searchCategories,
    fetchModels, fetchModelSchema, fetchSearchCategories, applySearchParams,
    fetchJobs, selectJob, startTraining, cancelTraining,
    saveDecoder, deleteJob,
    // Evaluation
    evalConfig, evalGate, evalMetrics, evalRunning, liveEval,
    startEvaluation, reaggregateEval,
    // Benchmark
    benchConfig, benchResults, benchRunning, startBenchmark,
    // WS
    connectWS, disconnectWS,
  }
})
