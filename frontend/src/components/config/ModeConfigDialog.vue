<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useStreamsStore } from '../../stores/streams'
import { useModesStore } from '../../stores/modes'
import { useMLStore } from '../../stores/ml'
import { usePipelineStore } from '../../stores/pipeline'
import { useConfigStore } from '../../stores/config'
import { useDecoderPicker } from '../../composables/useDecoderPicker'
import { useSessionEvents } from '../../composables/useSessionEvents'
import type { ModeInstance } from '../../types/api'
import { getModeColor } from '../../utils/colors'
import DecoderPickerDialog from './DecoderPickerDialog.vue'
import NumberInput from '../common/NumberInput.vue'
import PillSelector from '../common/PillSelector.vue'
import ToggleSwitch from '../common/ToggleSwitch.vue'
import EventMappingEditor from './EventMappingEditor.vue'
import SessionEventHint from './SessionEventHint.vue'
import EpochWindowViz from './EpochWindowViz.vue'
import PipelineBuilder from './PipelineBuilder.vue'

const props = defineProps<{
  instanceName: string
  instance: ModeInstance
}>()

const emit = defineEmits<{
  close: []
}>()

const streams = useStreamsStore()
const modes = useModesStore()
const ml = useMLStore()
const pipeline = usePipelineStore()

function onEsc(e: KeyboardEvent) { if (e.key === 'Escape') emit('close') }
onMounted(() => window.addEventListener('keydown', onEsc))
onUnmounted(() => window.removeEventListener('keydown', onEsc))

// Ensure decoder list is loaded from backend registry
if (ml.models.length === 0) ml.fetchModels()

const name = ref(props.instanceName)
const saving = ref(false)
const saveWarning = ref('')

const panelClass = 'flex-1 min-w-0 overflow-y-auto border border-border rounded-lg p-4 bg-bg-elevated/40'
const sectionHeadingClass = 'text-xs font-semibold text-text-muted uppercase tracking-wide'

// --- Mode helpers ---
const inst = props.instance

const isSynchronous = computed(() => inst.mode === 'synchronous')
const isAsynchronous = computed(() => inst.mode === 'asynchronous')
const isNeurofeedback = computed(() => inst.mode === 'neurofeedback')

const statusDotColor = computed(() => {
  if (!pipeline.status.recording) return 'var(--color-text-disabled)'
  switch (modes.modeStates[props.instanceName]) {
    case 'running':  return 'var(--color-status-ok)'
    case 'error':    return 'var(--color-status-error)'
    case 'starting':
    case 'stopping': return 'var(--color-status-warn)'
    default:         return 'var(--color-text-disabled)'
  }
})

// --- Per-mode preprocessing (defined early — used by channel selection + decoder picker) ---
const PREPROC_DEFAULTS: Record<string, Record<string, any>> = {
  eeg: { lowcut: 0.5, highcut: 50.0, apply_rereferencing: true },
  emg: { lowcut: 20.0, highcut: 200.0, line_freq: 50 },
  eog: { lowcut: 0.1, highcut: 10.0 },
}
const NFB_DEFAULTS: Record<string, Record<string, any>> = {
  eeg: { lowcut: 1.0, highcut: 45.0, apply_rereferencing: true },
}

function getDefaults() {
  return isNeurofeedback.value ? { ...PREPROC_DEFAULTS, ...NFB_DEFAULTS } : PREPROC_DEFAULTS
}

const modePreproc = ref<Record<string, Record<string, any>>>(
  inst.mode_preprocessing
    ? JSON.parse(JSON.stringify(inst.mode_preprocessing))
    : JSON.parse(JSON.stringify(getDefaults()))
)


// --- Channel selection (single modality per mode, scoped to one stream) ---
const channelSelection = ref<Record<string, number[]>>(
  inst.channel_selection ? JSON.parse(JSON.stringify(inst.channel_selection)) : {}
)
const sourceStream = ref<string>(inst.source_stream ?? '')

// Unique modality keys across all streams
const modalities = computed(() =>
  streams.allModalities.map(key => {
    const streamUids = streams.streamsForModality(key)
    let totalCount = 0
    for (const uid of streamUids) {
      totalCount += streams.modalitiesByStream[uid]?.modalities[key]?.length ?? 0
    }
    return { key, label: key.toUpperCase(), totalCount }
  })
)

const selectedModality = ref<string>(
  Object.keys(channelSelection.value)[0] ?? modalities.value[0]?.key ?? ''
)

// Stream UIDs that have the selected modality
const streamsWithModality = computed(() =>
  streams.streamsForModality(selectedModality.value)
)

// Auto-select stream: use saved source_stream key, else first stream with this modality
const selectedStreamUid = ref<string>((() => {
  if (sourceStream.value) {
    const match = Object.entries(streams.modalitiesByStream)
      .find(([, e]) => e.stream_key === sourceStream.value && selectedModality.value in e.modalities)
    if (match) return match[0]
  }
  return streamsWithModality.value[0] ?? ''
})())

// Channels for the selected stream + modality
const selectedChannels = computed(() => {
  const entry = streams.modalitiesByStream[selectedStreamUid.value]
  return entry?.modalities[selectedModality.value] ?? []
})

function switchModality(key: string) {
  if (selectedModality.value && selectedModality.value !== key) {
    delete channelSelection.value[selectedModality.value]
    delete modePreproc.value[selectedModality.value]
    if (!modePreproc.value[key]) {
      modePreproc.value[key] = { ...(getDefaults()[key] ?? {}) }
    }
  }
  selectedModality.value = key
  // Reset to first stream with this modality
  const uids = streams.streamsForModality(key)
  selectedStreamUid.value = uids[0] ?? ''
  channelSelection.value[key] = []
}

function switchStream(uid: string) {
  selectedStreamUid.value = uid
  // Clear channel selection when switching streams (indices change)
  channelSelection.value[selectedModality.value] = []
}

function isChannelSelected(modality: string, index: number): boolean {
  return channelSelection.value[modality]?.includes(index) ?? false
}
function toggleChannel(modality: string, index: number) {
  if (!channelSelection.value[modality]) channelSelection.value[modality] = []
  const arr = channelSelection.value[modality]
  const idx = arr.indexOf(index)
  if (idx >= 0) arr.splice(idx, 1)
  else arr.push(index)
}
function selectAll(modality: string) {
  channelSelection.value[modality] = selectedChannels.value.map((_, i) => i)
}
function selectNone(modality: string) {
  channelSelection.value[modality] = []
}

// --- Preprocessing computed ---
const currentModalityPreproc = computed(() => {
  if (!selectedModality.value) return null
  const defaults = getDefaults()[selectedModality.value] ?? {}
  return { ...defaults, ...(modePreproc.value[selectedModality.value] ?? {}) }
})

function updatePreproc(field: string, value: any) {
  if (!selectedModality.value) return
  if (!modePreproc.value[selectedModality.value]) {
    modePreproc.value[selectedModality.value] = { ...(getDefaults()[selectedModality.value] ?? {}) }
  }
  modePreproc.value[selectedModality.value]![field] = value
}

// --- Unified state (shared by sync + async) ---
const decoderCfg = inst.decoder_config ?? {}
const modelType = ref((decoderCfg.model_config ?? {}).model_type ?? 'EEGNet')
const pipelineSteps = ref<string[] | null>(decoderCfg.pipeline_steps ?? null)
const epochTmin = ref<number>(inst.epoch_tmin ?? 0)
const epochTmax = ref<number>(inst.epoch_tmax ?? 2.0)

function parseEventMapping(
  mapping: Record<string, string> | undefined,
  fallback: { id: number; label: string }[]
) {
  return mapping
    ? Object.entries(mapping).map(([id, label]) => ({ id: Number(id), label: label as string }))
    : fallback
}

const eventMapping = ref(parseEventMapping(
  inst.event_mapping,
  inst.mode === 'synchronous' ? [{ id: 1, label: 'Left' }, { id: 2, label: 'Right' }] : [],
))

function addEvent() {
  const maxId = eventMapping.value.reduce((m, e) => Math.max(m, e.id), 0)
  eventMapping.value.push({ id: maxId + 1, label: '' })
}
function removeEvent(index: number) {
  eventMapping.value.splice(index, 1)
}

// --- Decoder picker composable (async mode — selection, metadata, restore) ---
const {
  source: decoderSource,
  path: decoderPath,
  id: decoderId,
  showPicker: showDecoderPicker,
  selectedInfo: selectedDecoderInfo,
  decoderEventMapping,
  sourceMode,
  numClasses: decoderNumClasses,
  onSelected: onDecoderSelected,
  applyMappings: applyDecoderMappings,
  clear: clearSelectedDecoder,
} = useDecoderPicker(inst, {
  eventMapping, epochTmin, epochTmax, modelType,
  channelSelection, selectedModality, modePreproc,
  selectedChannels, switchModality,
})

// --- Session events composable ---
const {
  events: sessionEvents,
  mapping: sessionEventMapping,
  buildEntries: buildSessionEntries,
  fetchEvents: fetchSessionEvents,
} = useSessionEvents()

if (pipeline.status.recording) fetchSessionEvents()

function applySessionEvents() {
  eventMapping.value = buildSessionEntries()
}

// --- Synchronous-specific ---
const trainingInterval = ref(inst.training_interval ?? 10)
const useEpochQc = ref(inst.use_epoch_qc ?? true)
const includeBackground = ref(inst.include_background ?? false)
const useStudyHistory = ref(inst.use_study_history ?? false)
const studyHistoryRecordingIds = ref<number[]>(inst.study_history_recording_ids ?? [])
const studyRecordings = ref<any[]>([])

watch(useStudyHistory, async (on) => {
  if (on && studyRecordings.value.length === 0) {
    const cfg = useConfigStore()
    const studiesRes = await fetch('/api/data/studies')
    if (!studiesRes.ok) return
    const studies = await studiesRes.json()
    const study = studies.find((s: any) => s.study_name === cfg.general.study_name)
    if (study) {
      const res = await fetch(`/api/data/recordings?study_id=${study.study_id}`)
      if (res.ok) studyRecordings.value = await res.json()
    }
  }
}, { immediate: true })

watch(includeBackground, (on) => {
  const hasRest = eventMapping.value.some(e => e.id === 0 && e.label === 'rest')
  if (on && !hasRest) eventMapping.value.push({ id: 0, label: 'rest' })
  if (!on && hasRest) eventMapping.value = eventMapping.value.filter(e => !(e.id === 0 && e.label === 'rest'))
})

// --- Asynchronous-specific ---
const asyncStepSize = ref(inst.step_size_ms ?? 50)

// --- Neurofeedback ---
const nfbWindowLength = ref(inst.window_length_sec ?? 1.0)
const nfbStepSize = ref(inst.step_size_ms ?? 250)
const featureCfg = inst.feature_config ?? {}
const useRelativePower = ref(featureCfg.use_relative_power ?? true)
const useClusterMode = ref(featureCfg.use_cluster_mode ?? false)

const targetBands = ref<Array<{ name: string; low: number; high: number }>>(
  featureCfg.target_bands
    ? Object.entries(featureCfg.target_bands).map(([name, range]) => ({
        name,
        low: (range as number[])[0] ?? 0,
        high: (range as number[])[1] ?? 0,
      }))
    : [{ name: 'alpha', low: 8.0, high: 12.0 }]
)

function addBand() { targetBands.value.push({ name: '', low: 0, high: 0 }) }
function removeBand(i: number) { targetBands.value.splice(i, 1) }

// IAF calibration
const iafEnabled = ref(!!featureCfg.iaf_event_id)
const iafEventId = ref<number>(featureCfg.iaf_event_id ?? 99)
const iafBaselineSec = ref<number>(featureCfg.iaf_baseline_sec ?? 5.0)

// --- Available decoders (filtered by selected modality) ---
const availableDecoders = computed(() => {
  const mod = selectedModality.value?.toLowerCase() || ''
  return ml.models.filter(m =>
    m.modalities.some(d => d.toLowerCase() === 'any' || d.toLowerCase() === mod)
  )
})

// --- Validation ---
function validate(): string | null {
  const hasChannels = Object.values(channelSelection.value).some(arr => arr.length > 0)
  if (!hasChannels) return 'No channels selected'

  if (isSynchronous.value) {
    if (eventMapping.value.length < 2)
      return `Need at least 2 event classes (have ${eventMapping.value.length})`
    if (epochTmax.value <= epochTmin.value)
      return 'Epoch end must be greater than epoch start'
  }

  if (isAsynchronous.value) {
    if (epochTmax.value <= epochTmin.value)
      return 'Epoch end must be greater than epoch start'
    if (decoderSource.value === 'database' && !sourceMode.value && !decoderPath.value)
      return 'Select a decoder file or link to a synchronous mode'
  }

  if (isNeurofeedback.value) {
    const validBands = targetBands.value.filter(b => b.name)
    if (validBands.length === 0) return 'Add at least one frequency band'
    for (const b of validBands) {
      if (b.low >= b.high) return `Band "${b.name}": low (${b.low}) must be < high (${b.high})`
    }
  }

  return null
}

const toEventMappingDict = () =>
  Object.fromEntries(eventMapping.value.map((e) => [e.id, e.label]))

// --- Save ---
async function save() {
  saveWarning.value = ''
  const error = validate()
  if (error) {
    saveWarning.value = error
    return
  }
  saving.value = true
  try {
    const streamEntry = streams.modalitiesByStream[selectedStreamUid.value]
    const config: Record<string, any> = {
      channel_selection: channelSelection.value,
      source_stream: streamEntry?.stream_key ?? '',
    }

    // Add per-mode preprocessing overrides (only non-empty entries)
    const cleanPreproc: Record<string, any> = {}
    for (const [mod, cfg] of Object.entries(modePreproc.value)) {
      if (cfg && Object.keys(cfg).length > 0) cleanPreproc[mod] = { ...cfg }
    }
    if (Object.keys(cleanPreproc).length > 0) {
      config.mode_preprocessing = cleanPreproc
    }

    if (isSynchronous.value) {
      config.event_mapping = toEventMappingDict()
      config.epoch_tmin = epochTmin.value
      config.epoch_tmax = epochTmax.value
      config.training_interval = trainingInterval.value
      config.use_epoch_qc = useEpochQc.value
      config.include_background = includeBackground.value
      config.use_study_history = useStudyHistory.value
      if (useStudyHistory.value && studyHistoryRecordingIds.value.length > 0) {
        config.study_history_recording_ids = studyHistoryRecordingIds.value
      }
      config.decoder_config = {
        decoder_type: 'Decoder',
        pipeline_steps: pipelineSteps.value,
        model_config: { model_type: modelType.value },
      }
    }

    if (isAsynchronous.value) {
      config.step_size_ms = asyncStepSize.value
      config.decoder_source = decoderSource.value
      if (sourceMode.value) {
        config.source_mode = sourceMode.value
      } else if (decoderSource.value === 'database') {
        config.decoder_config = {
          decoder_type: 'Decoder',
          decoder_id: decoderId.value,
          decoder_path: decoderPath.value || null,
          model_config: { model_type: modelType.value, num_classes: decoderNumClasses.value },
        }
      }
      if (eventMapping.value.length > 0) {
        config.event_mapping = toEventMappingDict()
      }
    }

    if (isNeurofeedback.value) {
      config.window_length_sec = nfbWindowLength.value
      config.step_size_ms = nfbStepSize.value
      config.feature_config = {
        target_bands: Object.fromEntries(
          targetBands.value.filter(b => b.name).map(b => [b.name, [b.low, b.high]])
        ),
        use_relative_power: useRelativePower.value,
        use_cluster_mode: useClusterMode.value,
        ...(iafEnabled.value ? {
          iaf_event_id: iafEventId.value,
          iaf_baseline_sec: iafBaselineSec.value,
        } : {}),
      }
    }

    if (name.value !== props.instanceName) {
      await modes.renameInstance(props.instanceName, name.value)
    }
    const result = await modes.updateInstance(name.value, props.instance.mode, config)
    if (result.error) {
      saveWarning.value = result.error
      return
    }
    if (result.data?.enabled === false && config.enabled !== false) {
      const reason = result.data._disable_reason || 'decoder is incompatible with current configuration'
      saveWarning.value = `Mode auto-disabled: ${reason}`
      setTimeout(() => emit('close'), 3500)
      return
    }
    emit('close')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl shadow-black/40 w-[1400px] max-w-[95vw] h-[80vh] flex flex-col">
        <!-- Header -->
        <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <div class="flex items-center gap-3">
            <span
              class="w-2.5 h-2.5 rounded-full shrink-0"
              :class="pipeline.status.recording && (modes.modeStates[props.instanceName] === 'running') ? 'animate-pulse' : ''"
              :style="{ backgroundColor: statusDotColor }"
            />
            <span class="text-xs font-semibold uppercase tracking-wider px-2 py-1 rounded"
                  :style="{ color: getModeColor(instance.mode), backgroundColor: getModeColor(instance.mode) + '15' }">
              {{ instance.mode }}
            </span>
            <input
              v-model="name"
              class="text-sm font-semibold text-text-main bg-transparent border-b border-transparent
                     hover:border-border focus:border-accent focus:outline-none px-1 py-0.5 transition-colors"
            />
          </div>
          <button @click="emit('close')" class="text-text-disabled hover:text-text-main transition-colors p-1">
            <i class="pi pi-times" />
          </button>
        </div>

        <!-- Content: row layout -->
        <div class="flex-1 flex flex-col p-6 gap-5 overflow-hidden">
          <!-- Row 1: Input -->
          <div class="shrink-0 border border-border rounded-lg p-4 bg-bg-elevated/40">
          <div class="flex gap-5">
          <!-- Modality sidebar (vertical, when multiple modalities) -->
          <div v-if="modalities.length > 1" class="flex flex-col bg-bg-input rounded-lg p-0.5 shrink-0 mr-1">
            <button v-for="m in modalities" :key="m.key"
              @click="switchModality(m.key)"
              class="flex items-center gap-2 px-2.5 py-1.5 text-xs rounded transition-colors text-left"
              :class="selectedModality === m.key
                ? 'bg-bg-hover text-text-main font-semibold'
                : 'text-text-muted hover:text-text-main'"
            >
              <span>{{ m.label }}</span>
              <span class="text-[10px] text-text-disabled tabular-nums">{{ m.totalCount }}</span>
            </button>
          </div>

          <div class="flex-1 min-w-0 max-w-[55%]">
              <div v-if="modalities.length === 0" class="text-xs text-text-disabled text-center py-8">
                <i class="pi pi-info-circle text-lg block mb-2" />
                Configure streams first to select channels
              </div>
              <template v-else>
                <!-- Stream picker (when modality exists in multiple streams) -->
                <div v-if="streamsWithModality.length > 1" class="flex items-center gap-2 mb-2">
                  <span class="text-[11px] text-text-muted">Stream:</span>
                  <button v-for="uid in streamsWithModality" :key="uid"
                    @click="switchStream(uid)"
                    class="px-2 py-0.5 text-xs rounded border transition-colors"
                    :class="selectedStreamUid === uid
                      ? 'border-accent bg-accent/15 text-accent'
                      : 'border-border text-text-muted hover:border-accent'"
                  >{{ streams.modalitiesByStream[uid]?.stream_name }}</button>
                </div>

                <!-- Channel grid for selected modality -->
                <div v-if="selectedChannels.length > 0">
                  <div class="flex items-center gap-2 mb-2">
                    <span class="text-xs font-medium text-text-muted uppercase tracking-wide">
                      {{ selectedModality.toUpperCase() }} ({{ channelSelection[selectedModality]?.length ?? 0 }}/{{ selectedChannels.length }})
                    </span>
                    <button @click="selectAll(selectedModality)"
                      class="text-[11px] px-2 py-0.5 rounded bg-bg-input text-text-muted hover:text-text-main transition-colors">All</button>
                    <button @click="selectNone(selectedModality)"
                      class="text-[11px] px-2 py-0.5 rounded bg-bg-input text-text-muted hover:text-text-main transition-colors">None</button>
                  </div>
                  <div class="grid grid-cols-10 gap-1">
                    <button
                      v-for="(ch, i) in selectedChannels" :key="i"
                      @click="toggleChannel(selectedModality, i)"
                      class="px-1.5 py-1 text-xs rounded border transition-colors text-center truncate"
                      :class="isChannelSelected(selectedModality, i)
                        ? 'border-accent bg-accent/15 text-accent'
                        : 'border-border bg-bg-input text-text-muted hover:border-accent'"
                      :title="ch.label"
                    >{{ ch.label }}</button>
                  </div>
                </div>
              </template>

            </div>

            <!-- Separator -->
            <div v-if="selectedModality && currentModalityPreproc" class="border-l border-border" />

            <!-- Preprocessing -->
            <div v-if="selectedModality && currentModalityPreproc" class="flex-1 min-w-0 space-y-2.5">
              <h4 :class="sectionHeadingClass">Preprocessing</h4>
              <div class="flex gap-3">
                <label class="flex-1">
                  <span class="text-[11px] text-text-muted block mb-1">Lowcut (Hz)</span>
                  <NumberInput
                    :model-value="currentModalityPreproc.lowcut"
                    @update:model-value="updatePreproc('lowcut', $event)"
                    :step="0.1" :min="0"
                    class="w-full font-mono" />
                </label>
                <label class="flex-1">
                  <span class="text-[11px] text-text-muted block mb-1">Highcut (Hz)</span>
                  <NumberInput
                    :model-value="currentModalityPreproc.highcut"
                    @update:model-value="updatePreproc('highcut', $event)"
                    :step="0.5" :min="0"
                    class="w-full font-mono" />
                </label>
              </div>
              <!-- EEG: CAR toggle -->
              <div v-if="selectedModality === 'eeg'" class="flex items-center justify-between">
                <span class="text-xs text-text-muted">CAR</span>
                <ToggleSwitch
                  :model-value="currentModalityPreproc.apply_rereferencing"
                  @update:model-value="updatePreproc('apply_rereferencing', $event)"
                />
              </div>
              <!-- EMG: notch frequency -->
              <div v-if="selectedModality === 'emg'" class="flex items-center justify-between">
                <span class="text-xs text-text-muted">Notch</span>
                <div class="flex gap-1">
                  <button v-for="freq in [50, 60]" :key="freq"
                    @click="updatePreproc('line_freq', freq)"
                    class="px-2 py-0.5 text-xs rounded border transition-colors"
                    :class="currentModalityPreproc.line_freq === freq
                      ? 'border-accent/40 bg-accent/10 text-accent'
                      : 'border-border text-text-muted hover:text-text-main'">
                    {{ freq }} Hz
                  </button>
                </div>
              </div>
            </div>
          </div>
          </div>

          <!-- ==================== Synchronous ==================== -->
          <div v-if="isSynchronous" class="flex-1 min-h-0 flex gap-5">
            <div :class="[panelClass, 'space-y-0']">
              <h4 :class="[sectionHeadingClass, 'mb-3']">Decoder Pipeline</h4>
              <PipelineBuilder
                :model-type="modelType"
                :models="availableDecoders"
                :pipeline-steps="pipelineSteps"
                @update:model-type="modelType = $event"
                @update:pipeline-steps="pipelineSteps = $event"
              />

              <!-- Epoch Window -->
              <div class="border-t border-border pt-5 mt-5">
                <h4 :class="[sectionHeadingClass, 'mb-3']">Epoch Window</h4>
                <div class="flex gap-3">
                  <label class="flex-1">
                    <span class="text-sm text-text-muted block mb-1">Start Offset (s)</span>
                    <NumberInput v-model="epochTmin" :step="0.1" :max="epochTmax - 0.1"
                      class="w-full font-mono" />
                  </label>
                  <label class="flex-1">
                    <span class="text-sm text-text-muted block mb-1">End Offset (s)</span>
                    <NumberInput v-model="epochTmax" :step="0.1" :min="epochTmin + 0.1"
                      class="w-full font-mono" />
                  </label>
                </div>
                <EpochWindowViz v-if="epochTmin < epochTmax" :tmin="epochTmin" :tmax="epochTmax" />
              </div>

              <!-- Training -->
              <div class="border-t border-border pt-5 mt-5">
                <h4 :class="[sectionHeadingClass, 'mb-3']">Training</h4>
                <label class="block mb-3">
                  <span class="text-sm text-text-muted block mb-1">Auto-train Interval (epochs)</span>
                  <NumberInput v-model="trainingInterval" :min="1"
                    class="w-full font-mono" />
                </label>

                <div class="flex items-center justify-between mb-3">
                  <span class="text-sm text-text-main">Filter Bad Epochs</span>
                  <ToggleSwitch v-model="useEpochQc" />
                </div>

                <div class="flex items-center justify-between mb-3">
                  <span class="text-sm text-text-main">Train Rest Class</span>
                  <ToggleSwitch v-model="includeBackground" />
                </div>

                <div class="flex items-center justify-between mb-3">
                  <span class="text-sm text-text-main">Use Study History</span>
                  <ToggleSwitch v-model="useStudyHistory" />
                </div>
                <div v-if="useStudyHistory && studyRecordings.length > 0" class="mb-3 ml-1">
                  <span class="text-[11px] text-text-muted block mb-1.5">Select recordings to augment training</span>
                  <div class="max-h-32 overflow-y-auto space-y-1">
                    <label v-for="rec in studyRecordings" :key="rec.recording_id"
                      class="flex items-center gap-2 text-xs text-text-main cursor-pointer hover:text-text-bright">
                      <input type="checkbox" :value="rec.recording_id"
                        v-model="studyHistoryRecordingIds"
                        class="rounded text-accent" />
                      {{ rec.recording_name }}
                    </label>
                  </div>
                </div>
                <div v-if="useStudyHistory && studyRecordings.length === 0"
                  class="mb-3 text-xs text-text-disabled ml-1">
                  No recordings found in study
                </div>

                <div v-if="pipeline.status.recording" class="border-t border-border pt-3 mt-3">
                  <div class="flex items-center gap-2 text-xs text-text-muted">
                    <i class="pi pi-info-circle" />
                    Training triggers automatically every {{ trainingInterval }} epochs
                  </div>
                </div>
              </div>
            </div>

            <div :class="panelClass">
              <h4 :class="[sectionHeadingClass, 'mb-3']">Event Mapping</h4>

              <SessionEventHint :events="sessionEvents" :event-mapping="sessionEventMapping" @apply="applySessionEvents" />

              <EventMappingEditor :events="eventMapping"
                :lockedIds="includeBackground ? new Set([0]) : undefined"
                @add="addEvent"
                @remove="removeEvent($event)" />
            </div>
          </div>

          <!-- ==================== Asynchronous ==================== -->
          <div v-if="isAsynchronous" class="flex-1 min-h-0 flex gap-5">
            <div :class="panelClass">
              <h4 :class="[sectionHeadingClass, 'mb-3']">Decoder Source</h4>

              <PillSelector v-model="decoderSource"
                :options="[{ label: 'Database', value: 'database' }, { label: 'Online', value: 'online' }]"
                class="mb-4" />

              <label class="block mb-4">
                <span class="text-sm text-text-muted block mb-1">Prediction Step (ms)</span>
                <NumberInput v-model="asyncStepSize" :step="10" :min="10" :max="1000"
                  class="w-full font-mono" />
                <span class="text-xs text-text-disabled mt-0.5 block">Interval between decoder predictions</span>
              </label>

              <!-- Online mode -->
              <div v-if="decoderSource === 'online'" class="space-y-3">
                <div class="bg-accent/5 rounded p-2.5">
                  <p v-if="sourceMode" class="text-xs text-text-muted">
                    <span class="text-sm text-text-main font-medium">Linked to {{ sourceMode }}</span><br />
                    Decoder, preprocessing, and epoch window are inherited automatically.
                  </p>
                  <p v-else class="text-xs text-text-muted">
                    Waiting for a decoder from a linked sync mode.
                  </p>
                </div>
              </div>

              <!-- Database decoder picker -->
              <div v-else class="space-y-3">
                <div v-if="selectedDecoderInfo" class="bg-accent/5 rounded-lg p-3">
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-xs font-medium text-text-main">{{ selectedDecoderInfo.decoder_name }}</span>
                    <button @click="clearSelectedDecoder" class="text-text-disabled hover:text-text-main transition-colors p-0.5">
                      <i class="pi pi-times text-xs" />
                    </button>
                  </div>
                  <div class="flex items-center gap-1.5">
                    <span class="text-xs font-semibold text-accent uppercase">{{ selectedDecoderInfo.model_type }}</span>
                    <template v-if="selectedDecoderInfo.num_classes">
                      <span class="text-xs text-text-disabled">&middot;</span>
                      <span class="text-xs text-text-muted">{{ selectedDecoderInfo.num_classes }} classes</span>
                    </template>
                    <template v-if="selectedDecoderInfo.training_accuracy != null">
                      <span class="text-xs text-text-disabled">&middot;</span>
                      <span class="text-xs text-status-ok">{{ (selectedDecoderInfo.training_accuracy * 100).toFixed(1) }}%</span>
                    </template>
                  </div>
                  <div class="text-xs text-text-disabled mt-1 font-mono truncate">{{ selectedDecoderInfo.decoder_path }}</div>
                  <div class="mt-2 pt-2 border-t border-border/30">
                    <PipelineBuilder
                      :model-type="selectedDecoderInfo.model_type"
                      :models="ml.models"
                      readonly
                    />
                  </div>
                </div>

                <div v-else-if="decoderPath" class="bg-bg-elevated rounded-lg p-3">
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-xs font-medium text-text-main">Loaded Decoder</span>
                    <button @click="clearSelectedDecoder" class="text-text-disabled hover:text-text-main transition-colors p-0.5">
                      <i class="pi pi-times text-xs" />
                    </button>
                  </div>
                  <div class="text-xs text-text-disabled font-mono truncate">{{ decoderPath }}</div>
                </div>

                <button @click="showDecoderPicker = true"
                  class="w-full px-3 py-2.5 rounded-lg border border-dashed border-border text-text-muted
                         hover:border-accent hover:text-accent transition-colors text-xs">
                  <i class="pi pi-search mr-1.5" />
                  {{ selectedDecoderInfo || decoderPath ? 'Change Decoder' : 'Browse Decoders...' }}
                </button>
              </div>
            </div>

            <div :class="panelClass">
              <h4 :class="[sectionHeadingClass, 'mb-3']">Event Mapping</h4>

              <!-- Linked: show inherited events read-only -->
              <template v-if="sourceMode">
                <div v-if="eventMapping.length > 0" class="space-y-1.5">
                  <div v-for="e in eventMapping" :key="e.id"
                    class="flex items-center gap-3 px-3 py-1.5 bg-bg-input rounded text-sm">
                    <span class="font-mono text-accent w-8">{{ e.id }}</span>
                    <span class="text-text-main">{{ e.label || '—' }}</span>
                  </div>
                </div>
                <p v-else class="text-xs text-text-disabled">
                  Events will be inherited from {{ sourceMode }} when decoder loads.
                </p>
              </template>

              <!-- Standalone / Database: editable -->
              <template v-else>
                <SessionEventHint :events="sessionEvents" :event-mapping="sessionEventMapping" @apply="applySessionEvents" />

                <EventMappingEditor :events="eventMapping"
                  :decoder-mapping-count="decoderEventMapping ? Object.keys(decoderEventMapping).length : 0"
                  @add="addEvent"
                  @remove="removeEvent($event)"
                  @import-decoder="applyDecoderMappings" />
              </template>
            </div>
          </div>

          <!-- ==================== Neurofeedback: Features ==================== -->
          <div v-if="isNeurofeedback" class="flex-1 min-h-0 flex gap-5">
            <!-- Left: Window Parameters + Feature Options -->
            <div :class="[panelClass, 'space-y-4']">
              <div>
                <h4 :class="[sectionHeadingClass, 'mb-3']">Window Parameters</h4>
                <div class="flex gap-3">
                  <label class="flex-1">
                    <span class="text-[11px] text-text-muted block mb-1">Window Length (s)</span>
                    <NumberInput v-model="nfbWindowLength" :step="0.1" :min="0.1"
                      class="w-full font-mono" />
                  </label>
                  <label class="flex-1">
                    <span class="text-[11px] text-text-muted block mb-1">Step Size (ms)</span>
                    <NumberInput v-model="nfbStepSize" :step="10" :min="1"
                      class="w-full font-mono" />
                  </label>
                </div>
              </div>

              <div class="border-t border-border pt-4">
                <h4 :class="[sectionHeadingClass, 'mb-2']">Power Calculation</h4>
                <PillSelector v-model="useRelativePower"
                  :options="[{ label: 'Relative', value: true }, { label: 'Absolute', value: false }]" />
              </div>

              <div class="border-t border-border pt-4">
                <h4 :class="[sectionHeadingClass, 'mb-2']">Channel Output</h4>
                <PillSelector v-model="useClusterMode"
                  :options="[{ label: 'Individual', value: false }, { label: 'Clustered', value: true }]" />
              </div>

            </div>

            <!-- Right: Target Frequency Bands -->
            <div :class="panelClass">
              <h4 :class="[sectionHeadingClass, 'mb-4']">Target Frequency Bands</h4>

              <!-- Band table header -->
              <div class="grid grid-cols-[1fr_80px_80px_28px] gap-2 mb-2 px-1">
                <span class="text-xs text-text-disabled font-medium">Band Name</span>
                <span class="text-xs text-text-disabled font-medium">Low (Hz)</span>
                <span class="text-xs text-text-disabled font-medium">High (Hz)</span>
                <span></span>
              </div>

              <!-- Band rows -->
              <div class="space-y-1.5 mb-3 max-h-[150px] overflow-y-auto">
                <div v-for="(band, i) in targetBands" :key="i" class="grid grid-cols-[1fr_80px_80px_28px] gap-2 items-center">
                  <input v-model="band.name"
                    placeholder="Band name" />
                  <NumberInput v-model="band.low" :step="0.5" :min="0"
                    class="font-mono" />
                  <NumberInput v-model="band.high" :step="0.5" :min="0"
                    class="font-mono" />
                  <button @click="removeBand(i)" class="w-6 h-6 rounded flex items-center justify-center text-text-disabled hover:text-status-error hover:bg-status-error/10 transition-colors justify-self-center">
                    <i class="pi pi-times text-xs" />
                  </button>
                </div>
              </div>

              <div class="flex gap-2">
                <button @click="addBand" class="text-xs text-accent hover:text-accent-hover transition-colors">
                  <i class="pi pi-plus mr-1" />Add Band
                </button>
              </div>

              <p class="text-xs text-text-disabled italic mt-3">
                Each band produces one feature per selected channel.
              </p>

              <!-- IAF: shift bands to individual alpha frequency -->
              <div class="border-t border-border pt-3 mt-3 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-[11px] text-text-muted">Shift bands to Individual Alpha Frequency</span>
                  <ToggleSwitch v-model="iafEnabled" />
                </div>
                <div v-if="iafEnabled" class="flex items-center gap-3">
                  <label class="flex items-center gap-1.5">
                    <span class="text-[11px] text-text-muted">Event ID</span>
                    <NumberInput v-model="iafEventId" :min="1" :step="1" class="w-16 font-mono" />
                  </label>
                  <label class="flex items-center gap-1.5">
                    <span class="text-[11px] text-text-muted">Baseline</span>
                    <NumberInput v-model="iafBaselineSec" :min="1" :max="30" :step="0.5" class="w-16 font-mono" />
                    <span class="text-[11px] text-text-disabled">s</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div v-if="saveWarning" class="px-5 py-2 bg-status-warn/10 border-t border-status-warn/30">
          <div class="flex items-center gap-2 text-xs text-status-warn">
            <i class="pi pi-exclamation-triangle text-xs" />
            {{ saveWarning }}
          </div>
        </div>
        <div class="flex justify-end gap-2 px-5 py-3 border-t border-border shrink-0">
          <button @click="emit('close')"
            class="px-4 py-1.5 text-xs rounded border border-border text-text-muted hover:text-text-main hover:border-text-muted transition-colors">
            Cancel
          </button>
          <button @click="save" :disabled="saving"
            class="px-4 py-1.5 text-xs rounded bg-accent text-white hover:bg-accent-hover disabled:opacity-30 transition-colors">
            <i v-if="saving" class="pi pi-spin pi-spinner mr-1" />
            Save
          </button>
        </div>
      </div>
    </div>

    <!-- Decoder Picker (nested dialog) -->
    <DecoderPickerDialog
      v-if="showDecoderPicker"
      @select="onDecoderSelected"
      @close="showDecoderPicker = false"
    />
  </Teleport>
</template>
