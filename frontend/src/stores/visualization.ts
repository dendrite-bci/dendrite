import { defineStore } from 'pinia'
import { ref, shallowRef, computed } from 'vue'
import { useWebSocket } from '../composables/useWebSocket'
import { RingBuffer } from '../composables/useRingBuffer'
import { apiFetchOrNull } from '../utils/api'
import { useTelemetryStore } from './telemetry'
import { useModesStore } from './modes'

/**
 * Non-reactive hot-path state. Exported directly (not through Pinia)
 * to avoid Vue proxy overhead on 30fps+ WS handlers and rAF loops.
 */
export const vizDirty = {
  dataVersion: 0,
  sampleCounter: 0,
  eventsChanged: false,
  psdVersion: 0,
}

const DEFAULT_TIME_WINDOW = 10.0 // seconds
const DEFAULT_CHANNEL_HEIGHT = 60 // px
const COMPACT_CHANNEL_HEIGHT = 40 // px
const DEFAULT_SAMPLE_RATE = 250
const MAX_METRICS_HISTORY = 200
const HIGH_FREQ_NOTIFY_MS = 100  // ~4Hz throttle for metrics + predictions
const LOW_FREQ_NOTIFY_MS = 500   // ~2Hz throttle for ERPs + band power
// NF rolling window: 30s @ 4Hz (default step_size_ms=250). Buffer is a fixed
// sample count, so a longer step shows a longer real-time window (we read the
// step from the mode config to label the x-axis correctly).
const BAND_POWER_BUFFER_SAMPLES = 120
const DEFAULT_BAND_POWER_STEP_SEC = 0.25

// Pre-allocated scratch buffer for de-interleaving batched channel data (avoids per-frame GC)
let _scratchBuf = new Float32Array(64)

// Non-reactive staging buffer for events — flushed to reactive ref by render loop
const _pendingEvents: EventEntry[] = []

interface EventEntry {
  time: number
  value: number
  timeAxisPos: number
}

interface ModeMetrics {
  accuracy: number[]
  confidence: number[]
  chanceLevel: number[]
  kappa: number[]
}

/**
 * Unpack msgpack binary or array data into a numeric array.
 * msgpackr returns Uint8Array views into a shared buffer at arbitrary
 * offsets — slice() copies to a new aligned ArrayBuffer for Float32Array.
 */
function unpackChannelData(raw: any): Float32Array | number[] {
  if (raw?.bytes) return new Float32Array(raw.bytes.slice().buffer)
  if (Array.isArray(raw)) return raw
  return [raw as number]
}

export const useVisualizationStore = defineStore('visualization', () => {
  // EEG buffers
  const eegBuffers = ref<RingBuffer[]>([])
  const eegLabels = ref<string[]>([])
  const sampleRate = ref(250)
  const bufferSize = ref(2500) // 10s * 250Hz

  // Time axis (shared across all time-series plots)
  const timeAxis = ref<number[]>([])

  // Modality buffers (EMG, EOG, etc.)
  const modalityBuffers = ref<Record<string, { buffers: RingBuffer[]; labels: string[] }>>({})

  // PSD data per modality (updated ~1Hz from backend Welch computation)
  const psdData = shallowRef<Record<string, { freqs: number[]; power: number[] }>>({})

  // Events
  const eventHistory = ref<EventEntry[]>([])
  const MAX_EVENTS = 300

  // Mode data — shallowRef + triggerRef for explicit notification (no timer)
  const modeMetrics = shallowRef<Record<string, ModeMetrics>>({})
  const modeNamesList = shallowRef<string[]>([])

  // Mode data — ERP (running average per mode per event class)
  interface ERPAccum {
    sum: number[]           // grand-average sum (for compact view)
    channelSums: number[][] // per-channel sums [nCh][nTimes]
    channelLabels: string[] // channel names
    count: number
    nTimes: number
    sampleRate: number
    startOffsetMs: number
  }
  const modeERPs = shallowRef<Record<string, Record<string, ERPAccum>>>({})

  // Mode data — band power rolling history per (channel, band) for time-series plots.
  // Buffers are mutated in place via .append(); the throttled notify spreads
  // the top-level object so Vue picks up a new reference each tick.
  const modeBandPowerHistory = shallowRef<Record<string, {
    channels: Record<string, Record<string, RingBuffer>>
    bandNames: string[]
    stepSec: number
  }>>({})

  // Mode data — latest prediction per mode
  const modePredictions = shallowRef<Record<string, {
    eventName: string; confidence: number; timestamp: number
  }>>({})

  // Mode data — prediction history (for async timeline plot)
  const modePredictionHistory = shallowRef<Record<string, Array<{
    eventName: string; confidence: number; ts: number; detected: boolean
  }>>>({})

  // IAF calibration results per mode
  const modeIAF = shallowRef<Record<string, {
    iafHz: number; offsetHz: number; shiftedBands: Record<string, number[]>
  }>>({})

  // Mode types (e.g., "asynchronous", "synchronous", "neurofeedback")
  const modeTypes = shallowRef<Record<string, string>>({})

  // State
  const initialized = ref(false)
  const connected = computed(() => ws.connected.value)

  // Channel display density
  const channelHeight = ref(DEFAULT_CHANNEL_HEIGHT)
  const channelsPerPage = ref(12) // default, updated by container resize
  const compact = ref(false)

  function setDensity(mode: 'compact' | 'normal') {
    compact.value = mode === 'compact'
    channelHeight.value = compact.value ? COMPACT_CHANNEL_HEIGHT : DEFAULT_CHANNEL_HEIGHT
  }

  function setChannelsPerPage(containerHeight: number) {
    // Account for 1px gap between grid rows: n * height + (n-1) * 1 <= container
    channelsPerPage.value = Math.max(1, Math.floor((containerHeight + 1) / (channelHeight.value + 1)))
  }

  // Viz preprocessing (mutable during recording)
  const vizPreproc = ref<Record<string, Record<string, any>>>({
    eeg: { filter_low: 0.5, filter_high: 50.0, apply_rereferencing: true },
  })
  let _vizPreprocDebounce: ReturnType<typeof setTimeout> | null = null

  async function updateVizPreproc(config: Record<string, Record<string, any>>) {
    vizPreproc.value = config
    // Debounce API call — pipeline may not be running
    if (_vizPreprocDebounce) clearTimeout(_vizPreprocDebounce)
    _vizPreprocDebounce = setTimeout(() => {
      apiFetchOrNull('/api/pipeline/viz-preprocessing', { method: 'PUT', json: config })
    }, 500)
  }

  // Channel pagination
  const currentPage = ref(0)
  const totalEegChannels = computed(() => eegLabels.value.length)
  const totalPages = computed(() => Math.max(1, Math.ceil(totalEegChannels.value / channelsPerPage.value)))
  const visibleChannelRange = computed(() => {
    const start = currentPage.value * channelsPerPage.value
    const end = Math.min(start + channelsPerPage.value, totalEegChannels.value)
    return { start, end }
  })

  function initBuffers(rate: number, labels: Record<string, string[]>) {
    sampleRate.value = rate
    bufferSize.value = Math.round(DEFAULT_TIME_WINDOW * rate)

    // Time axis
    const ta = new Array(bufferSize.value)
    for (let i = 0; i < bufferSize.value; i++) {
      ta[i] = (i / bufferSize.value) * DEFAULT_TIME_WINDOW
    }
    timeAxis.value = ta

    // EEG buffers
    const eegCh = labels['eeg'] || []
    eegLabels.value = eegCh
    eegBuffers.value = eegCh.map(() => new RingBuffer(bufferSize.value))

    // Modality buffers
    const modBufs: Record<string, { buffers: RingBuffer[]; labels: string[] }> = {}
    for (const [mod, chLabels] of Object.entries(labels)) {
      if (mod === 'eeg' || mod === 'markers' || mod === 'events') continue
      modBufs[mod] = {
        buffers: chLabels.map(() => new RingBuffer(bufferSize.value)),
        labels: chLabels,
      }
    }
    modalityBuffers.value = modBufs

    // Purge all stale session data
    eventHistory.value = []
    modeMetrics.value = {}
    modeERPs.value = {}
    modeBandPowerHistory.value = {}
    modeIAF.value = {}
    modeNamesList.value = []
    modePredictions.value = {}
    modePredictionHistory.value = {}
    modeTypes.value = {}
    psdData.value = {}
    vizDirty.sampleCounter = 0
    vizDirty.dataVersion = 0

    initialized.value = true
    currentPage.value = 0
  }

  function handleMessage(msg: any) {
    if (!msg) return

    // Handle raw data
    if (msg.ch === 'raw_data' || msg.type === 'raw_data' || msg.output_type === 'raw_data') {
      handleRawData(msg)
      return
    }

    // Handle PSD (1Hz from backend Welch computation)
    if (msg.ch === 'psd') {
      handlePSD(msg)
      return
    }

    // Handle mode data
    if (msg.ch === 'mode_history' || msg.type === 'mode_history') {
      handleModeData(msg)
      return
    }
  }

  function handleRawData(msg: any) {
    const data = msg.d || msg.data
    if (!data) return

    // Initialize on first data if needed
    if (!initialized.value) {
      const rate = msg.meta?.sample_rate || msg.sample_rate || msg._viz_sample_rate || DEFAULT_SAMPLE_RATE
      const labels = msg.meta?.channel_labels || msg.channel_labels || {}
      initBuffers(rate, labels)
    }

    const batchSize = msg.meta?.batch || 1

    // Push EEG data (batched: shape [nSamples, nChannels])
    const eegData = data.eeg
    if (eegData) {
      const raw = unpackChannelData(eegData)
      // Auto-create EEG buffers if another stream initialized first
      if (eegBuffers.value.length === 0) {
        const nCh = batchSize > 1 ? raw.length / batchSize : raw.length
        const labels = (msg.channel_labels?.eeg as string[])
          || Array.from({ length: nCh }, (_, i) => `EEG${i + 1}`)
        eegLabels.value = labels
        eegBuffers.value = labels.map(() => new RingBuffer(bufferSize.value))
      }
      const nCh = eegBuffers.value.length
      if (batchSize > 1 && raw.length === batchSize * nCh) {
        // Batched: interleaved [s0_ch0, s0_ch1, ..., s1_ch0, s1_ch1, ...]
        // Reuse a single scratch buffer to avoid per-channel Float32Array allocations
        if (_scratchBuf.length < batchSize) _scratchBuf = new Float32Array(batchSize)
        for (let ch = 0; ch < nCh; ch++) {
          for (let s = 0; s < batchSize; s++) _scratchBuf[s] = raw[s * nCh + ch]!
          eegBuffers.value[ch]!.appendBatch(_scratchBuf, batchSize)
        }
      } else {
        // Single sample fallback
        for (let i = 0; i < Math.min(raw.length, nCh); i++) {
          eegBuffers.value[i]!.append(raw[i]!)
        }
      }
    }

    // Push modality data (batched), auto-creating buffers for late-arriving streams
    for (const [mod, modData] of Object.entries(data)) {
      if (mod === 'eeg' || mod === 'markers' || mod === 'events') continue
      if (mod.startsWith('_')) continue

      const raw = unpackChannelData(modData as any)
      // Auto-create buffers for modalities from other streams (e.g., EMG)
      if (!modalityBuffers.value[mod]) {
        const nCh = batchSize > 1 ? raw.length / batchSize : raw.length
        const labels = (msg.channel_labels?.[mod] as string[])
          || Array.from({ length: nCh }, (_, i) => `${mod.toUpperCase()}${i + 1}`)
        modalityBuffers.value[mod] = {
          buffers: labels.map(() => new RingBuffer(bufferSize.value)),
          labels,
        }
      }

      const modBuf = modalityBuffers.value[mod]
      const nCh = modBuf.buffers.length
      if (batchSize > 1 && raw.length === batchSize * nCh) {
        if (_scratchBuf.length < batchSize) _scratchBuf = new Float32Array(batchSize)
        for (let ch = 0; ch < nCh; ch++) {
          for (let s = 0; s < batchSize; s++) _scratchBuf[s] = raw[s * nCh + ch]!
          modBuf.buffers[ch]!.appendBatch(_scratchBuf, batchSize)
        }
      } else {
        for (let i = 0; i < Math.min(raw.length, nCh); i++) {
          modBuf.buffers[i]!.append(raw[i]!)
        }
      }
    }

    // Push events/markers — collect into non-reactive staging buffer,
    // flushed to reactive ref on rAF via vizDirty.eventsChanged
    const markers = data.markers ?? data.events
    if (markers !== undefined && markers !== null) {
      const vals = Array.isArray(markers) ? markers : [markers]
      for (const m of vals) {
        const val = typeof m === 'number' ? m : Number(m)
        if (val !== 0 && !isNaN(val) && _pendingEvents.length < MAX_EVENTS) {
          _pendingEvents.push({
            time: msg.ts || Date.now() / 1000,
            value: val,
            timeAxisPos: vizDirty.sampleCounter / sampleRate.value,
          })
        }
      }
      if (_pendingEvents.length > 0) vizDirty.eventsChanged = true
    }

    // Only advance timeline for EEG (primary) stream — prevents 2x speed with multi-stream
    if (eegData) vizDirty.sampleCounter += batchSize

    vizDirty.dataVersion++
  }

  function handlePSD(msg: any) {
    const d = msg.d
    if (!d) return
    // Mutate in place + bump version — avoids replacing the entire shallowRef
    // which would trigger re-renders of every PSD panel on every 1Hz update.
    const current = psdData.value
    let changed = false
    for (const [mod, modPsd] of Object.entries(d)) {
      const p = modPsd as any
      const n = p.n as number
      const freqsBuf = new Float32Array(p.freqs.slice().buffer)
      const powerBuf = new Float32Array(p.power.slice().buffer)
      current[mod] = {
        freqs: Array.from(freqsBuf.subarray(0, n)),
        power: Array.from(powerBuf.subarray(0, n)),
      }
      changed = true
    }
    if (changed) vizDirty.psdVersion++
  }

  function handleModeData(msg: any) {
    const name = msg.mode_name || msg.name || 'unknown'
    const type = msg.type || ''
    const d = msg.data || msg

    // Ensure mode is registered
    if (!modeNamesList.value.includes(name)) {
      modeNamesList.value.push(name)
      _namesChanged = true
    }

    // Track mode type
    if (msg.mode_type && !modeTypes.value[name]) {
      modeTypes.value[name] = msg.mode_type
      _namesChanged = true
    }

    // Route by output type
    if (type === 'erp') {
      _handleERP(name, d)
      _scheduleLowFreqNotify()
    } else if (type === 'neurofeedback' || type === 'neurofeedback_features') {
      _handleBandPower(name, d)
      _scheduleLowFreqNotify()
    } else if (type === 'iaf_result') {
      _handleIAF(name, d)
      _scheduleLowFreqNotify()
    } else if (type === 'prediction') {
      _handlePrediction(name, d)
      _scheduleHighFreqNotify()
    } else if (type === 'performance') {
      _handlePerformance(name, d)
      _scheduleHighFreqNotify()
    } else {
      _handlePerformance(name, d)
      _scheduleHighFreqNotify()
    }
    // Names/types change rarely — only clone when actually added
    if (_namesChanged) {
      _namesChanged = false
      modeNamesList.value = [...modeNamesList.value]
      modeTypes.value = { ...modeTypes.value }
    }
  }

  // Throttled Vue reactivity triggers — avoids proxy overhead on hot WS paths
  function _makeThrottledNotify(delayMs: number, flush: () => void) {
    let pending = false
    return () => {
      if (pending) return
      pending = true
      setTimeout(() => { pending = false; flush() }, delayMs)
    }
  }

  const _scheduleHighFreqNotify = _makeThrottledNotify(HIGH_FREQ_NOTIFY_MS, () => {
    modeMetrics.value = { ...modeMetrics.value }
    modePredictions.value = { ...modePredictions.value }
    modePredictionHistory.value = { ...modePredictionHistory.value }
  })

  const _scheduleLowFreqNotify = _makeThrottledNotify(LOW_FREQ_NOTIFY_MS, () => {
    modeERPs.value = { ...modeERPs.value }
    modeBandPowerHistory.value = { ...modeBandPowerHistory.value }
  })

  let _namesChanged = false

  function _handlePerformance(name: string, d: any) {
    if (!modeMetrics.value[name]) {
      modeMetrics.value[name] = { accuracy: [], confidence: [], chanceLevel: [], kappa: [] }
    }
    const m = modeMetrics.value[name]
    const acc = d.accuracy ?? d.balanced_accuracy
    if (acc !== undefined) m.accuracy.push(acc)
    if (d.confidence !== undefined) m.confidence.push(d.confidence)
    if (d.chance_level !== undefined) m.chanceLevel.push(d.chance_level)
    if (d.cohens_kappa !== undefined) m.kappa.push(d.cohens_kappa)
    for (const arr of [m.accuracy, m.confidence, m.chanceLevel, m.kappa]) {
      if (arr.length > MAX_METRICS_HISTORY) arr.splice(0, arr.length - MAX_METRICS_HISTORY)
    }
    // Async mode sends event_name + confidence in performance messages — feed prediction history
    if (d.event_name && d.confidence !== undefined) {
      _handlePrediction(name, d)
    }
  }

  function _getModeChannelLabels(modeName: string, nCh: number): string[] {
    // Derive channel labels by indexing channel_selection into eegLabels.
    // Both are modality-relative (per-modality local_index from stream_service).
    const modes = useModesStore()
    const sel = modes.instances[modeName]?.channel_selection as Record<string, number[]> | undefined
    if (sel) {
      const indices = Object.values(sel)[0]
      if (indices?.length === nCh) {
        return indices.map(i => eegLabels.value[i] ?? `ch${i}`)
      }
    }
    return Array.from({ length: nCh }, (_, i) => `ch${i}`)
  }

  function _handleERP(name: string, d: any) {
    const eventType = d.event_type as string
    if (!eventType || !d.data) return

    if (!modeERPs.value[name]) modeERPs.value[name] = {}
    const erps = modeERPs.value[name]

    const raw = unpackChannelData(d.data)
    const shape = d.data.shape as number[] | undefined  // [n_channels, n_times]
    const nCh = shape?.[0] ?? 1
    const nTimes = shape?.[1] ?? raw.length

    // Get bad channels from telemetry (exclude from grand average)
    const telemetry = useTelemetryStore()
    const badList: number[] = (telemetry.data?.channel_quality as any)?.bad_channels?.eeg ?? []
    const badSet = new Set(badList)
    const goodCount = nCh - badSet.size

    // Grand-average across good channels only (row-major layout)
    const avg = new Float64Array(nTimes)
    if (goodCount > 0) {
      for (let ch = 0; ch < nCh; ch++) {
        if (badSet.has(ch)) continue
        const offset = ch * nTimes
        for (let t = 0; t < nTimes; t++) avg[t] = avg[t]! + raw[offset + t]!
      }
      for (let t = 0; t < nTimes; t++) avg[t] = avg[t]! / goodCount
    }

    if (!erps[eventType]) {
      erps[eventType] = {
        sum: new Array(nTimes).fill(0),
        channelSums: Array.from({ length: nCh }, () => new Array(nTimes).fill(0)),
        channelLabels: _getModeChannelLabels(name, nCh),
        count: 0,
        nTimes,
        sampleRate: d.sample_rate || DEFAULT_SAMPLE_RATE,
        startOffsetMs: d.start_offset_ms || 0,
      }
    }

    const acc = erps[eventType]
    // Accumulate grand average
    for (let i = 0; i < Math.min(nTimes, acc.sum.length); i++) {
      acc.sum[i]! += avg[i]!
    }
    // Accumulate per-channel
    for (let ch = 0; ch < Math.min(nCh, acc.channelSums.length); ch++) {
      const offset = ch * nTimes
      const chSum = acc.channelSums[ch]!
      for (let t = 0; t < Math.min(nTimes, chSum!.length); t++) {
        chSum![t]! += raw[offset + t]!
      }
    }
    acc.count++
  }

  function _handleBandPower(name: string, d: any) {
    if (!d.channel_powers) return

    let hist = modeBandPowerHistory.value[name]
    if (!hist) {
      const modes = useModesStore()
      const stepMs = modes.instances[name]?.step_size_ms as number | undefined
      hist = {
        channels: {},
        bandNames: [],
        stepSec: stepMs ? stepMs / 1000 : DEFAULT_BAND_POWER_STEP_SEC,
      }
      modeBandPowerHistory.value[name] = hist
    }
    for (const [chName, bandPowers] of Object.entries(
      d.channel_powers as Record<string, Record<string, number>>,
    )) {
      let chBufs = hist.channels[chName]
      if (!chBufs) {
        chBufs = {}
        hist.channels[chName] = chBufs
      }
      for (const [bandName, power] of Object.entries(bandPowers)) {
        let buf = chBufs[bandName]
        if (!buf) {
          buf = new RingBuffer(BAND_POWER_BUFFER_SAMPLES)
          chBufs[bandName] = buf
          if (!hist.bandNames.includes(bandName)) hist.bandNames.push(bandName)
        }
        buf.append(power)
      }
    }
  }

  function _handleIAF(name: string, d: any) {
    modeIAF.value = {
      ...modeIAF.value,
      [name]: {
        iafHz: d.iaf_hz,
        offsetHz: d.offset_hz,
        shiftedBands: d.shifted_bands || {},
      },
    }
  }

  function _handlePrediction(name: string, d: any) {
    const eventName = d.event_name || String(d.prediction ?? '?')
    const confidence = d.confidence ?? 0
    modePredictions.value[name] = {
      eventName,
      confidence,
      timestamp: Date.now(),
    }
    // Append to history for async timeline
    if (!modePredictionHistory.value[name]) {
      modePredictionHistory.value[name] = []
    }
    const hist = modePredictionHistory.value[name]
    hist.push({ eventName, confidence, ts: Date.now(), detected: !!d.detected })
    if (hist.length > MAX_METRICS_HISTORY) hist.splice(0, hist.length - MAX_METRICS_HISTORY)
  }

  /** Flush staged events to reactive ref — call from rAF render loop, not WS handler. */
  function flushEvents() {
    if (_pendingEvents.length === 0) return
    eventHistory.value.push(..._pendingEvents)
    _pendingEvents.length = 0
    if (eventHistory.value.length > MAX_EVENTS) {
      eventHistory.value = eventHistory.value.slice(-MAX_EVENTS)
    }
    vizDirty.eventsChanged = false
  }

  function nextPage() {
    if (currentPage.value < totalPages.value - 1) currentPage.value++
  }

  function prevPage() {
    if (currentPage.value > 0) currentPage.value--
  }

  // WebSocket connections — raw EEG and mode data on separate channels
  const ws = useWebSocket('/ws/visualization', {
    binary: true,
    onMessage: handleMessage,
    onClose: () => { initialized.value = false },
  })
  useWebSocket('/ws/mode_data', {
    binary: true,
    onMessage: handleMessage,
  })

  return {
    // EEG
    eegBuffers, eegLabels, sampleRate, bufferSize, timeAxis,
    currentPage, totalEegChannels, totalPages, visibleChannelRange,
    channelHeight, channelsPerPage, compact,
    // Modalities
    modalityBuffers, psdData,
    // Events
    eventHistory,
    // Mode data
    modeMetrics, modeNamesList,
    modeERPs, modeBandPowerHistory, modeIAF, modePredictions, modePredictionHistory, modeTypes,
    // State
    initialized, connected,
    // Viz preprocessing
    vizPreproc, updateVizPreproc,
    // Actions
    initBuffers, flushEvents, nextPage, prevPage, setDensity, setChannelsPerPage,
  }
})
