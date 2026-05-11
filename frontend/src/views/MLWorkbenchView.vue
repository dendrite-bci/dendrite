<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useMLStore } from '../stores/ml'
import NumberInput from '../components/common/NumberInput.vue'
import ToggleSwitch from '../components/common/ToggleSwitch.vue'
import TrainingConfigPanel from '../components/ml/TrainingConfigPanel.vue'
import RecordingBrowser from '../components/ml/RecordingBrowser.vue'
import MoabbBrowser from '../components/ml/MoabbBrowser.vue'
import DataSummaryDetail from '../components/ml/DataSummaryDetail.vue'
import TrainingResultsPanel from '../components/ml/TrainingResultsPanel.vue'
import EvaluationPanel from '../components/ml/EvaluationPanel.vue'
import EvalResultsPanel from '../components/ml/EvalResultsPanel.vue'
import BenchmarkPanel from '../components/ml/BenchmarkPanel.vue'
import BenchmarkResultsPanel from '../components/ml/BenchmarkResultsPanel.vue'
import JobHistoryList from '../components/ml/JobHistoryList.vue'

const ml = useMLStore()

// --- Workspace tab ---
type WorkspaceTab = 'training' | 'evaluation' | 'benchmark'
const workspaceTab = ref<WorkspaceTab>('training')

// --- Section collapse state ---
const recordingsOpen = ref(true)
const eventsOpen = ref(true)
const preprocOpen = ref(true)

// --- Event groups ---
const creatingGroup = ref(false)
const newGroupName = ref('')

const activeGroupName = computed(() => {
  const sel = ml.dataPreproc.selected_events
  if (sel === null) return 'All'
  const selSet = new Set(sel)
  for (const g of ml.eventGroups) {
    if (g.events.length === selSet.size && g.events.every(e => selSet.has(e))) return g.name
  }
  return null
})

function saveGroup() {
  const name = newGroupName.value.trim()
  if (!name) return
  ml.saveEventGroup(name)
  newGroupName.value = ''
  creatingGroup.value = false
}

function applyGroup(name: string) {
  if (name === 'All') {
    ml.applyEventGroup(null)
  } else {
    const group = ml.eventGroups.find(g => g.name === name)
    if (group) ml.applyEventGroup(group.events)
  }
}

// --- Data source ---
const sourceOptions = [
  { label: 'Recordings', value: 'recordings' },
  { label: 'MOABB', value: 'moabb' },
]

// --- Counts ---
const selectedCount = computed(() => Object.keys(ml.recordingRoles).length)
const trainCount = computed(() =>
  Object.values(ml.recordingRoles).filter(r => r === 'train').length
)
const evalCount = computed(() =>
  Object.values(ml.recordingRoles).filter(r => r === 'eval').length
)

const selectedRecordings = computed(() => {
  const ids = Object.keys(ml.recordingRoles).map(Number)
  return ml.recordings.filter(r => ids.includes(r.recording_id))
})

function toggleEval(recordingId: number) {
  ml.recordingRoles[recordingId] = ml.recordingRoles[recordingId] === 'eval' ? 'train' : 'eval'
}

const loadButtonText = computed(() => {
  if (ml.dataLoading) return 'Loading...'
  const evalSuffix = evalCount.value > 0 ? `, ${evalCount.value} eval` : ''
  return `Load Data (${trainCount.value} train${evalSuffix})`
})

// --- Event selection (from pool or loaded data) ---
const poolEvents = computed(() => {
  const merged: Record<string, number> = {}
  for (const [id] of Object.entries(ml.recordingRoles)) {
    const summary = ml.recordingEventSummaries[Number(id)]
    if (summary) {
      for (const [name, count] of Object.entries(summary)) {
        merged[name] = (merged[name] ?? 0) + count
      }
    }
  }
  return merged
})

const availableEvents = computed(() => {
  const eventId = ml.loadedData?.metadata?.event_id
  if (eventId && typeof eventId === 'object' && Object.keys(eventId).length > 0) {
    return Object.entries(eventId)
      .map(([name, code]) => ({ name, code: Number(code), count: poolEvents.value[name] ?? null }))
      .sort((a, b) => a.code - b.code)
  }
  const pe = poolEvents.value
  if (Object.keys(pe).length === 0) return []
  return Object.entries(pe)
    .map(([name, count]) => ({ name, code: null as number | null, count }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

function isEventSelected(name: string): boolean {
  const sel = ml.dataPreproc.selected_events
  return sel === null || sel.includes(name)
}

function toggleEvent(name: string) {
  const allNames = availableEvents.value.map(e => e.name)
  if (ml.dataPreproc.selected_events === null) {
    ml.dataPreproc.selected_events = allNames.filter(n => n !== name)
  } else if (ml.dataPreproc.selected_events.includes(name)) {
    ml.dataPreproc.selected_events = ml.dataPreproc.selected_events.filter(n => n !== name)
  } else {
    ml.dataPreproc.selected_events = [...ml.dataPreproc.selected_events, name]
  }
  if (ml.dataPreproc.selected_events?.length === allNames.length) {
    ml.dataPreproc.selected_events = null
  }
}

// --- Channel type selection ---
const allChannels = computed(() => ml.loadedData?.channel_names ?? [])
const allTypes = computed(() => ml.loadedData?.channel_types ?? [])

const typeGroups = computed(() => {
  const groups: Record<string, number[]> = {}
  for (let i = 0; i < allTypes.value.length; i++) {
    const t = (allTypes.value[i] || 'unknown').toUpperCase()
    if (!groups[t]) groups[t] = []
    groups[t].push(i)
  }
  return groups
})

const enabledTypes = computed(() => {
  const sel = ml.dataPreproc.channel_indices
  const result: Record<string, boolean> = {}
  for (const [type, indices] of Object.entries(typeGroups.value)) {
    result[type] = !sel || indices.some(i => sel.includes(i))
  }
  return result
})

const selectedChannelCount = computed(() => ml.dataPreproc.channel_indices?.length ?? allChannels.value.length)

function toggleType(type: string) {
  const indices = typeGroups.value[type] || []
  const current = ml.dataPreproc.channel_indices
  if (!current) {
    const allIndices = allChannels.value.map((_: string, i: number) => i)
    ml.dataPreproc.channel_indices = allIndices.filter((i: number) => !indices.includes(i))
  } else if (enabledTypes.value[type]) {
    ml.dataPreproc.channel_indices = current.filter((i: number) => !indices.includes(i))
  } else {
    ml.dataPreproc.channel_indices = [...current, ...indices].sort((a: number, b: number) => a - b)
  }
}

// --- Section summaries ---
const eventsSummary = computed(() => {
  const parts: string[] = []
  const sel = ml.dataPreproc.selected_events
  if (sel) {
    parts.push(sel.join(', '))
  } else if (availableEvents.value.length > 0) {
    parts.push(availableEvents.value.map(e => e.name).join(', '))
  }
  if (Object.keys(typeGroups.value).length > 0) {
    const types = Object.entries(typeGroups.value)
      .filter(([type]) => enabledTypes.value[type])
      .map(([type, idx]) => `${idx.length} ${type}`)
    if (types.length) parts.push(types.join(', '))
  }
  return parts.join(' · ') || 'No events'
})

const preprocSummary = computed(() => {
  if (ml.dataPreproc.use_paradigm_epochs && ml.selectedMoabbDataset) {
    const bp = ml.selectedMoabbDataset.paradigm_bandpass
    return bp ? `Paradigm: ${bp[0]}–${bp[1]} Hz` : 'Paradigm preprocessing'
  }
  const pp = ml.dataPreproc
  const parts: string[] = []
  if (pp.lowcut || pp.highcut) {
    parts.push(`${pp.lowcut ?? '—'}–${pp.highcut ?? '—'} Hz`)
  }
  if (pp.apply_rereferencing) parts.push('CAR')
  if (pp.use_epoch_qc) parts.push('QC')
  if (ml.dataSourceTab === 'recordings') {
    parts.push(`${pp.epoch_tmin ?? 0}–${pp.epoch_tmax ?? 2}s`)
  }
  return parts.join(' · ') || 'None'
})

// --- Lifecycle ---
onMounted(() => {
  ml.connectWS()
  ml.fetchModels()
  ml.fetchJobs()
})
onUnmounted(() => { ml.disconnectWS() })
</script>

<template>
  <div class="flex h-full relative">
    <!-- Centered empty state overlay -->
    <div v-if="workspaceTab === 'training' && !ml.selectedJob && !ml.loadedData"
         class="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <div class="text-center">
        <i class="pi pi-database text-4xl text-text-disabled mb-4 block" />
        <p class="text-text-muted text-sm">No data loaded</p>
        <p class="text-text-disabled text-xs mt-1">Select recordings and load data from the left panel</p>
      </div>
    </div>
    <div v-else-if="workspaceTab === 'evaluation' && !ml.evalMetrics && !(ml.liveEval && ml.liveEval.timeline.length > 0)"
         class="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <div class="text-center">
        <i class="pi pi-bolt text-4xl text-text-disabled mb-4 block" />
        <p class="text-text-muted text-sm">No evaluation results</p>
        <p class="text-text-disabled text-xs mt-1">Select a training job and run evaluation</p>
      </div>
    </div>
    <div v-else-if="workspaceTab === 'benchmark' && ml.benchResults.length === 0"
         class="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <div class="text-center">
        <i class="pi pi-table text-4xl text-text-disabled mb-4 block" />
        <p class="text-text-muted text-sm">No benchmark results</p>
        <p class="text-text-disabled text-xs mt-1">Select models and run a benchmark</p>
      </div>
    </div>

    <!-- ==================== DATA PANEL ==================== -->
    <div class="w-[560px] max-w-[560px] flex flex-col border-r border-border bg-bg-panel">

      <!-- Source selector header -->
      <div class="flex items-center px-4 border-b border-border shrink-0 h-[42px]">
        <button
          v-for="tab in sourceOptions" :key="tab.value"
          @click="ml.dataSourceTab = tab.value as 'recordings' | 'moabb'"
          class="px-3 h-full text-sm font-medium transition-colors border-b-2 -mb-px"
          :class="ml.dataSourceTab === tab.value
            ? 'text-text-main border-accent'
            : 'text-text-muted hover:text-text-main border-transparent'"
        >{{ tab.label }}</button>
      </div>

      <!-- Scrollable sections -->
      <div class="flex-1 overflow-y-auto p-3 space-y-2">

        <!-- Section 1: Recordings -->
        <div class="rounded-lg border border-border/30 bg-bg-elevated/30 overflow-hidden">
          <button
            @click="recordingsOpen = !recordingsOpen"
            class="w-full flex items-center justify-between px-3 py-2 hover:bg-bg-elevated/50 transition-colors"
          >
            <div class="flex items-center gap-2">
              <i class="pi text-[10px] text-text-disabled" :class="recordingsOpen ? 'pi-chevron-down' : 'pi-chevron-right'" />
              <span class="text-xs font-semibold text-text-label uppercase tracking-wide">Recordings</span>
            </div>
            <div class="flex gap-2 text-xs">
              <span v-if="trainCount > 0" class="text-data-train font-medium">{{ trainCount }} train</span>
              <span v-if="evalCount > 0" class="text-data-eval font-medium">{{ evalCount }} eval</span>
            </div>
          </button>
          <div v-if="recordingsOpen" class="px-3 pb-3 border-t border-border/20">
            <RecordingBrowser v-if="ml.dataSourceTab === 'recordings'" />
            <template v-else>
              <MoabbBrowser />
              <label v-if="ml.selectedMoabbDataset" class="flex items-center gap-2 text-xs text-text-muted cursor-pointer mt-2 pt-2 border-t border-border/20">
                <ToggleSwitch v-model="ml.dataPreproc.use_paradigm_epochs" compact />
                Use paradigm preprocessing
                <span class="text-text-disabled">(matches published benchmarks)</span>
              </label>
            </template>
          </div>
        </div>

        <!-- Section 2: Events & Channels -->
        <div
          v-if="availableEvents.length > 0 || Object.keys(typeGroups).length > 0"
          class="rounded-lg border border-border/30 bg-bg-elevated/30 overflow-hidden"
        >
          <button
            @click="eventsOpen = !eventsOpen"
            class="w-full flex items-center justify-between px-3 py-2 hover:bg-bg-elevated/50 transition-colors"
          >
            <div class="flex items-center gap-2">
              <i class="pi text-[10px] text-text-disabled" :class="eventsOpen ? 'pi-chevron-down' : 'pi-chevron-right'" />
              <span class="text-xs font-semibold text-text-label uppercase tracking-wide">Events & Channels</span>
            </div>
            <span v-if="!eventsOpen" class="text-xs text-text-disabled truncate max-w-[240px]">{{ eventsSummary }}</span>
          </button>
          <div v-if="eventsOpen" class="px-3 pb-3 border-t border-border/20">
            <!-- Event groups -->
            <div v-if="availableEvents.length > 0" class="mt-2 mb-3">
              <div class="flex items-center gap-1.5 mb-2 flex-wrap">
                <!-- All chip -->
                <button
                  @click="applyGroup('All')"
                  class="px-2 py-0.5 rounded text-xs font-semibold transition-colors"
                  :class="activeGroupName === 'All'
                    ? 'bg-accent/20 text-accent ring-1 ring-accent/30'
                    : 'bg-bg-input text-text-muted hover:text-text-main'"
                >All</button>
                <!-- User groups -->
                <div v-for="g in ml.eventGroups" :key="g.name" class="relative group/chip">
                  <button
                    @click="applyGroup(g.name)"
                    class="px-2 py-0.5 rounded text-xs font-semibold transition-colors pr-5"
                    :class="activeGroupName === g.name
                      ? 'bg-accent/20 text-accent ring-1 ring-accent/30'
                      : 'bg-bg-input text-text-muted hover:text-text-main'"
                    :title="g.events.join(', ')"
                  >{{ g.name }}</button>
                  <button
                    @click.stop="ml.deleteEventGroup(g.name)"
                    class="absolute right-0.5 top-1/2 -translate-y-1/2 w-4 h-4 rounded-full flex items-center justify-center
                           text-text-disabled hover:text-status-error opacity-0 group-hover/chip:opacity-100 transition-opacity"
                  >&times;</button>
                </div>
                <!-- Save current selection as group -->
                <template v-if="!creatingGroup">
                  <button
                    v-if="ml.dataPreproc.selected_events && ml.dataPreproc.selected_events.length > 0"
                    @click="creatingGroup = true"
                    class="px-1.5 py-0.5 rounded text-xs text-text-disabled hover:text-text-muted transition-colors"
                    title="Save current selection as group"
                  >+</button>
                </template>
                <form v-else @submit.prevent="saveGroup" class="flex items-center gap-1">
                  <input
                    v-model="newGroupName"
                    ref="groupNameInput"
                    placeholder="Group name"
                    class="w-20 text-xs px-1.5 py-0.5 rounded"
                    @keydown.escape="creatingGroup = false"
                    @vue:mounted="($event: any) => $event.el.focus()"
                  />
                  <button type="submit" class="text-xs text-accent hover:text-accent/80">Save</button>
                  <button type="button" @click="creatingGroup = false" class="text-xs text-text-disabled">&times;</button>
                </form>
              </div>
              <!-- Individual event toggles -->
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="evt in availableEvents" :key="evt.name"
                  @click="toggleEvent(evt.name)"
                  class="px-2 py-0.5 rounded text-xs font-medium transition-colors"
                  :class="isEventSelected(evt.name)
                    ? 'bg-accent/15 text-accent'
                    : 'bg-bg-input text-text-disabled line-through'"
                >
                  {{ evt.name }}
                  <span v-if="evt.count != null" class="opacity-60 ml-0.5">×{{ evt.count }}</span>
                </button>
              </div>
              <p v-if="ml.dataPreproc.selected_events" class="text-xs text-text-disabled mt-1">
                {{ ml.dataPreproc.selected_events.length }}/{{ availableEvents.length }} selected
              </p>
            </div>
            <!-- Channel types -->
            <div v-if="Object.keys(typeGroups).length > 0">
              <div class="flex items-center justify-between mb-1.5">
                <span class="text-xs font-medium text-text-muted">
                  Channels ({{ selectedChannelCount }}/{{ allChannels.length }})
                </span>
                <div class="flex gap-1">
                  <button @click="ml.dataPreproc.channel_indices = null"
                    class="text-xs px-2 py-0.5 rounded bg-bg-input text-text-muted hover:text-text-main transition-colors">All</button>
                  <button @click="ml.dataPreproc.channel_indices = []"
                    class="text-xs px-2 py-0.5 rounded bg-bg-input text-text-muted hover:text-text-main transition-colors">None</button>
                </div>
              </div>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="(indices, type) in typeGroups" :key="type"
                  @click="toggleType(type as string)"
                  class="px-2 py-0.5 text-xs font-medium rounded transition-colors"
                  :class="enabledTypes[type as string]
                    ? 'bg-accent/15 text-accent'
                    : 'bg-bg-input text-text-disabled'"
                >
                  {{ type }} <span class="font-mono ml-1 opacity-70">{{ indices.length }}</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Section 3: Preprocessing -->
        <div class="rounded-lg border border-border/30 bg-bg-elevated/30 overflow-hidden">
          <button
            @click="preprocOpen = !preprocOpen"
            class="w-full flex items-center justify-between px-3 py-2 hover:bg-bg-elevated/50 transition-colors"
          >
            <div class="flex items-center gap-2">
              <i class="pi text-[10px] text-text-disabled" :class="preprocOpen ? 'pi-chevron-down' : 'pi-chevron-right'" />
              <span class="text-xs font-semibold text-text-label uppercase tracking-wide">Preprocessing</span>
            </div>
            <span v-if="!preprocOpen" class="text-xs text-text-disabled truncate max-w-[240px]">{{ preprocSummary }}</span>
          </button>
          <div v-if="preprocOpen" class="px-3 pb-3 border-t border-border/20">
            <!-- Paradigm preprocessing info (read-only) -->
            <div v-if="ml.dataPreproc.use_paradigm_epochs && ml.selectedMoabbDataset" class="mt-2 mb-3 p-2 rounded bg-bg-input/50 text-xs text-text-muted space-y-1">
              <div class="font-medium text-text-main">Paradigm preprocessing (MOABB)</div>
              <div v-if="ml.selectedMoabbDataset.paradigm_bandpass">
                Bandpass: {{ ml.selectedMoabbDataset.paradigm_bandpass[0] }}–{{ ml.selectedMoabbDataset.paradigm_bandpass[1] }} Hz
              </div>
              <div v-if="ml.selectedMoabbDataset.interval">
                Epoch: {{ ml.selectedMoabbDataset.interval[0] }}–{{ ml.selectedMoabbDataset.interval[1] }} s
              </div>
              <div v-if="ml.selectedMoabbDataset.events">
                Events: {{ Object.keys(ml.selectedMoabbDataset.events).join(', ') }}
              </div>
            </div>
            <!-- Custom preprocessing controls -->
            <template v-if="!ml.dataPreproc.use_paradigm_epochs">
            <div class="grid grid-cols-2 gap-3 mt-2 mb-3">
              <div>
                <label class="text-sm text-text-muted block mb-1">Low Cut (Hz)</label>
                <NumberInput v-model="ml.dataPreproc.lowcut" :min="0" :max="100" :step="0.5"
                  placeholder="None" class="w-full placeholder-text-disabled" />
              </div>
              <div>
                <label class="text-sm text-text-muted block mb-1">High Cut (Hz)</label>
                <NumberInput v-model="ml.dataPreproc.highcut" :min="0" :max="500" :step="0.5"
                  placeholder="None" class="w-full placeholder-text-disabled" />
              </div>
            </div>
            <div class="flex gap-4 mb-3">
              <label class="flex items-center gap-2 text-xs text-text-muted cursor-pointer">
                <ToggleSwitch v-model="ml.dataPreproc.apply_rereferencing" compact />
                CAR
              </label>
              <label class="flex items-center gap-2 text-xs text-text-muted cursor-pointer">
                <ToggleSwitch v-model="ml.dataPreproc.use_epoch_qc" compact />
                Epoch QC
              </label>
              <label class="flex items-center gap-2 text-xs text-text-muted cursor-pointer">
                <ToggleSwitch v-model="ml.dataPreproc.include_background" compact />
                Rest Class
              </label>
            </div>
            </template>
            <div v-if="ml.dataSourceTab === 'recordings'">
              <h4 class="text-xs font-medium text-text-muted mb-2">Epoch Window</h4>
              <div class="grid grid-cols-2 gap-3">
                <div>
                  <label class="text-sm text-text-muted block mb-1">tmin (s)</label>
                  <NumberInput v-model="ml.dataPreproc.epoch_tmin" :min="-2" :max="0" :step="0.1" class="w-full" />
                </div>
                <div>
                  <label class="text-sm text-text-muted block mb-1">tmax (s)</label>
                  <NumberInput v-model="ml.dataPreproc.epoch_tmax" :min="0" :max="5" :step="0.1" class="w-full" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Pinned footer -->
      <div class="shrink-0 border-t border-border px-4 py-3 space-y-2.5">
        <!-- MOABB: subject selector -->
        <div v-if="ml.dataSourceTab === 'moabb' && ml.selectedMoabbDataset" class="flex items-center gap-3">
          <label class="text-[11px] text-text-muted shrink-0">Subject</label>
          <NumberInput
            v-model="ml.dataPreproc.subject"
            :min="1"
            :max="ml.selectedMoabbDataset.n_subjects || 100"
            class="w-[72px] text-xs"
          />
          <span class="text-[10px] text-text-disabled">of {{ ml.selectedMoabbDataset.n_subjects }}</span>
        </div>

        <!-- Recordings: eval strategy depends on selection count -->
        <template v-if="ml.dataSourceTab === 'recordings' && selectedCount > 0">
          <!-- Single recording: eval split slider -->
          <div v-if="selectedCount === 1" class="flex items-center gap-3">
            <label class="text-[11px] text-text-muted shrink-0 w-14">Eval split</label>
            <input v-model.number="ml.dataPreproc.eval_split" type="range" min="0" max="0.5" step="0.05" class="flex-1" />
            <span class="text-[11px] font-mono text-text-muted w-8 text-right">{{ (ml.dataPreproc.eval_split * 100).toFixed(0) }}%</span>
          </div>

          <!-- Multiple recordings: click to mark as eval -->
          <div v-else>
            <div class="flex items-center gap-2 mb-1.5">
              <span class="text-[11px] text-text-muted">Click to hold out for eval:</span>
            </div>
            <div class="flex flex-wrap gap-1">
              <button
                v-for="rec in selectedRecordings" :key="rec.recording_id"
                @click="toggleEval(rec.recording_id)"
                class="text-[10px] px-2 py-0.5 rounded font-medium transition-colors"
                :class="ml.recordingRoles[rec.recording_id] === 'eval'
                  ? 'bg-data-eval/20 text-data-eval ring-1 ring-data-eval/30'
                  : 'bg-data-train/10 text-data-train/70 hover:bg-data-eval/10 hover:text-data-eval'"
              >{{ rec.recording_name }}</button>
            </div>
          </div>
        </template>

        <!-- MOABB: eval split slider -->
        <div v-if="ml.dataSourceTab === 'moabb'" class="flex items-center gap-3">
          <label class="text-[11px] text-text-muted shrink-0 w-14">Eval split</label>
          <input v-model.number="ml.dataPreproc.eval_split" type="range" min="0" max="0.5" step="0.05" class="flex-1" />
          <span class="text-[11px] font-mono text-text-muted w-8 text-right">{{ (ml.dataPreproc.eval_split * 100).toFixed(0) }}%</span>
        </div>

        <!-- Load button -->
        <button
          v-if="ml.dataSourceTab === 'recordings'"
          @click="ml.loadPool()"
          :disabled="selectedCount === 0 || ml.dataLoading"
          class="w-full px-3.5 py-1.5 text-xs font-semibold rounded transition-colors"
          :class="selectedCount > 0 && !ml.dataLoading
            ? 'bg-accent text-white hover:bg-accent/80'
            : 'bg-bg-input text-text-disabled cursor-not-allowed'"
        >{{ loadButtonText }}</button>
        <button
          v-else
          @click="ml.selectedMoabbDataset && ml.loadMoabbDataset(ml.selectedMoabbDataset.code)"
          :disabled="!ml.selectedMoabbDataset || ml.dataLoading"
          class="w-full px-3.5 py-1.5 text-xs font-semibold rounded transition-colors"
          :class="ml.selectedMoabbDataset && !ml.dataLoading
            ? 'bg-accent text-white hover:bg-accent/80'
            : 'bg-bg-input text-text-disabled cursor-not-allowed'"
        >{{ ml.dataLoading ? 'Loading...' : 'Load Dataset' }}</button>
        <p v-if="ml.dataLoading && ml.dataLoadingStep" class="text-xs text-text-muted text-center">
          {{ ml.dataLoadingStep }}
        </p>
      </div>
    </div>

    <!-- ==================== WORKSPACE ==================== -->
    <div class="flex-1 flex flex-col min-w-0 bg-bg-main">

      <!-- Header bar -->
      <div class="flex items-center justify-between px-4 border-b border-border shrink-0 bg-bg-panel h-[42px]">
        <div class="flex bg-bg-main/50 rounded-lg p-1 gap-1">
          <button
            v-for="tab in ([
              { key: 'training', label: 'Training', icon: 'pi pi-chart-line' },
              { key: 'evaluation', label: 'Evaluation', icon: 'pi pi-bolt' },
              { key: 'benchmark', label: 'Benchmark', icon: 'pi pi-table' },
            ] as const)"
            :key="tab.key"
            @click="workspaceTab = tab.key"
            class="flex items-center justify-center gap-1.5 px-4 py-2 rounded-md
                   text-sm font-medium transition-colors"
            :class="workspaceTab === tab.key
              ? 'bg-bg-elevated text-text-main'
              : 'text-text-muted hover:text-text-main'"
          >
            <i :class="tab.icon" class="text-sm" />
            {{ tab.label }}
          </button>
        </div>
        <JobHistoryList :dropdown="true" />
      </div>

      <!-- Config strip -->
      <div class="shrink-0 border-b border-border px-4 py-2.5 bg-bg-panel/50">
        <TrainingConfigPanel v-if="workspaceTab === 'training'" />
        <EvaluationPanel v-else-if="workspaceTab === 'evaluation'" />
        <BenchmarkPanel v-else />
      </div>

      <!-- Results area -->
      <div class="flex-1 overflow-y-auto p-4">
        <DataSummaryDetail v-if="ml.loadedData" />

        <template v-if="workspaceTab === 'training'">
          <TrainingResultsPanel v-if="ml.selectedJob" />
        </template>

        <template v-if="workspaceTab === 'evaluation'">
          <EvalResultsPanel v-if="ml.evalMetrics || (ml.liveEval && ml.liveEval.timeline.length > 0)" />
        </template>

        <template v-if="workspaceTab === 'benchmark'">
          <BenchmarkResultsPanel v-if="ml.benchResults.length > 0" />
        </template>
      </div>
    </div>
  </div>
</template>
