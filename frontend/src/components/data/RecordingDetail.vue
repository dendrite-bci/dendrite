<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useDataStore } from '../../stores/data'
import { usePipelineStore } from '../../stores/pipeline'
import { useToast } from '../../composables/useToast'
import ConfirmDialog from '../common/ConfirmDialog.vue'
import H5FileViewer from './H5FileViewer.vue'
import SignalPreviewPlot from './SignalPreviewPlot.vue'
import EventSummaryPanel from './EventSummaryPanel.vue'
import TelemetryPanel from './TelemetryPanel.vue'
import ModePerformancePanel from './ModePerformancePanel.vue'
import SignalQCPanel from './SignalQCPanel.vue'
import NumberInput from '../common/NumberInput.vue'
import ERPPreviewPanel from './ERPPreviewPanel.vue'
import { formatDate, formatPercent } from '../../utils/format'
import { getModalityColor } from '../../utils/colors'

const data = useDataStore()
const pipeline = usePipelineStore()
const toast = useToast()

const isActiveRecording = computed(() =>
  pipeline.status.recording &&
  pipeline.status.recording_id === data.selectedRecording?.recording_id
)

const showDeleteConfirm = ref(false)
async function confirmDelete() {
  if (!data.selectedRecording) return
  const ok = await data.deleteRecording(data.selectedRecording.recording_id)
  if (!ok) toast.error('Cannot delete — recording file is locked')
  showDeleteConfirm.value = false
}
const activeSection = ref<'overview' | 'analysis'>('overview')
const showH5 = ref(false)

// Shared analysis preprocessing
const aLowcut = ref(0.5)
const aHighcut = ref(40)
const aReref = ref(true)
const aEpochTmin = ref(-0.2)
const aEpochTmax = ref(0.8)
const analysisKey = ref(0)

function applyAnalysis() {
  analysisKey.value++
}

const hiddenDatasets = ref(new Set<string>())

const availableDatasets = computed(() => {
  if (data.signalPreview) return Object.keys(data.signalPreview)
  return data.sessionSummary?.datasets?.filter(d => !['Event', 'Event_Clean', 'events'].includes(d)) ?? []
})

const visibleDatasets = computed(() =>
  availableDatasets.value.filter(d => !hiddenDatasets.value.has(d))
)

function toggleDataset(name: string) {
  const s = new Set(hiddenDatasets.value)
  if (s.has(name)) s.delete(name)
  else s.add(name)
  hiddenDatasets.value = s
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = (seconds % 60).toFixed(0)
  return `${m}m ${s}s`
}

const hasMetricsData = computed(() => {
  const t = data.recordingTelemetry
  const p = data.modePerformance
  const hasTelemetry = t && (
    Object.keys(t.latencies).length > 0 ||
    Object.keys(t.mode_metrics).length > 0 ||
    Object.keys(t.bandwidth).length > 0
  )
  const hasPerformance = p && Object.keys(p).length > 0
  return hasTelemetry || hasPerformance
})

async function loadOverviewData() {
  if (!data.selectedRecording) return
  const id = data.selectedRecording.recording_id
  if (!data.signalPreview) data.fetchSignalPreview(id)
  if (!data.eventSummary) data.fetchEventSummary(id)
}

async function loadMetrics() {
  if (!data.selectedRecording) return
  const id = data.selectedRecording.recording_id
  if (!data.recordingTelemetry) await data.fetchRecordingTelemetry(id)
  if (!data.modePerformance) await data.fetchModePerformance(id)
}

async function toggleH5() {
  showH5.value = !showH5.value
  if (showH5.value && !data.recordingFileInfo && data.selectedRecording) {
    await data.fetchRecordingFileInfo(data.selectedRecording.recording_id)
  }
}

watch(() => data.selectedRecording, () => {
  activeSection.value = 'overview'
  showH5.value = false
  loadOverviewData()
}, { immediate: true })
</script>

<template>
  <div v-if="data.selectedRecording" class="space-y-5">
    <!-- Header panel -->
    <div class="bg-bg-elevated rounded-lg px-4 py-3">
      <div class="flex items-start justify-between mb-2">
        <h2 class="text-sm font-semibold text-text-main">{{ data.selectedRecording.recording_name }}</h2>
        <div class="flex items-center gap-2 shrink-0 ml-3">
          <span class="text-xs text-text-disabled">
            {{ data.selectedRecording.study_name }} &middot; {{ formatDate(data.selectedRecording.session_timestamp) }}
          </span>
          <button
            v-if="!isActiveRecording"
            @click="showDeleteConfirm = true"
            class="w-5 h-5 flex items-center justify-center text-text-disabled hover:text-status-error transition-colors rounded"
            title="Delete recording"
          >
            <i class="pi pi-trash text-xs" />
          </button>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-x-5 gap-y-1">
        <span v-if="data.sessionSummary" class="text-xs">
          <span class="text-text-disabled">Duration</span>
          <span class="text-text-main font-mono ml-1">{{ formatDuration(data.sessionSummary.duration_seconds) }}</span>
        </span>
        <span v-if="data.sessionSummary" class="text-xs">
          <span class="text-text-disabled">Rate</span>
          <span class="text-text-main font-mono ml-1">{{ data.sessionSummary.sample_rate }} Hz</span>
        </span>
        <span v-if="data.recordingChannels" class="text-xs">
          <span class="text-text-disabled">Channels</span>
          <span class="text-text-main font-mono ml-1">{{ data.recordingChannels.count }}</span>
          <span v-if="data.recordingChannels.count <= 6" class="text-text-muted ml-0.5">({{ data.recordingChannels.labels.join(', ') }})</span>
        </span>
        <span v-if="data.selectedRecording.subject_id" class="text-xs">
          <span class="text-text-disabled">Subject</span>
          <span class="text-text-main font-mono ml-1">{{ data.selectedRecording.subject_id }}</span>
        </span>
        <span v-if="data.selectedRecording.session_id" class="text-xs">
          <span class="text-text-disabled">Session</span>
          <span class="text-text-main font-mono ml-1">{{ data.selectedRecording.session_id }}</span>
        </span>
        <span class="text-xs">
          <span class="text-text-disabled">Run</span>
          <span class="text-text-main font-mono ml-1">{{ data.selectedRecording.run_number }}</span>
        </span>
        <span v-if="data.recordingChannels" class="text-xs text-text-disabled">
          {{ data.recordingChannels.n_samples.toLocaleString() }} samples
        </span>
      </div>
    </div>

    <!-- Tabs -->
    <div class="flex items-center">
      <div class="flex">
        <button
          v-for="tab in [
            { key: 'overview', label: 'Overview' },
            { key: 'analysis', label: 'Analysis' },
          ]"
          :key="tab.key"
          @click="activeSection = tab.key as any"
          class="px-3 py-2 text-sm font-medium transition-colors border-b-2"
          :class="activeSection === tab.key
            ? 'border-accent text-accent'
            : 'border-transparent text-text-muted hover:text-text-main'"
        >{{ tab.label }}</button>
      </div>
    </div>

    <!-- ==================== Overview ==================== -->
    <div v-if="activeSection === 'overview'" class="space-y-6">
      <!-- Channel badges (only when > 6, otherwise shown inline in header) -->
      <div v-if="data.recordingChannels && data.recordingChannels.count > 6">
        <span class="text-xs font-semibold text-text-label block mb-2">
          Channels ({{ data.recordingChannels.count }})
        </span>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="(label, i) in data.recordingChannels.labels"
            :key="i"
            class="px-2 py-0.5 text-xs bg-bg-elevated rounded text-text-main"
          >{{ label }}</span>
        </div>
      </div>

      <!-- Modes -->
      <div v-if="data.sessionSummary?.modes?.length" class="flex flex-wrap gap-1.5">
        <span class="text-xs font-semibold text-text-label block w-full mb-1">Modes</span>
        <span
          v-for="mode in data.sessionSummary.modes"
          :key="mode"
          class="px-2 py-0.5 text-xs bg-accent/10 text-accent rounded"
        >{{ mode }}</span>
      </div>

      <!-- Signals -->
      <div v-if="availableDatasets.length > 0">
        <div class="flex items-center gap-2 mb-3">
          <span class="text-xs font-semibold text-text-label uppercase tracking-wider">Datasets</span>
          <div class="flex flex-wrap gap-1">
            <button
              v-for="ds in availableDatasets"
              :key="ds"
              @click="toggleDataset(ds)"
              class="px-2 py-0.5 text-xs rounded transition-colors"
              :class="hiddenDatasets.has(ds) ? 'opacity-30' : ''"
              :style="{ backgroundColor: getModalityColor(ds.toLowerCase()) + '25', color: getModalityColor(ds.toLowerCase()) }"
            >{{ ds }}</button>
          </div>
        </div>
        <div class="space-y-4">
          <template v-for="mod in visibleDatasets" :key="mod">
            <SignalPreviewPlot
              v-if="data.signalPreview?.[mod]"
              :modality="mod"
              :preview="data.signalPreview![mod]!"
            />
          </template>
        </div>
      </div>

      <!-- Events (inline) -->
      <div v-if="data.eventSummary">
        <span class="text-xs font-semibold text-text-label uppercase tracking-wider block mb-2">Events</span>
        <EventSummaryPanel :summary="data.eventSummary" />
      </div>

      <!-- Associated decoders -->
      <div v-if="data.recordingDecoders.length > 0">
        <span class="text-xs font-semibold text-text-label block mb-2">
          Trained Decoders ({{ data.recordingDecoders.length }})
        </span>
        <div class="space-y-1.5">
          <div
            v-for="dec in data.recordingDecoders"
            :key="dec.decoder_id"
            class="px-3 py-2 bg-bg-elevated rounded"
          >
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-text-main">{{ dec.decoder_name }}</span>
              <span class="text-[10px] text-accent font-semibold uppercase">{{ dec.model_type }}</span>
            </div>
            <div v-if="dec.training_accuracy != null" class="text-xs text-text-disabled mt-0.5">
              Accuracy: {{ formatPercent(dec.training_accuracy) }}
            </div>
          </div>
        </div>
      </div>

      <!-- Performance metrics (lazy) -->
      <div v-if="data.sessionSummary?.has_metrics" class="space-y-4">
        <div v-if="!data.recordingTelemetry && !data.modePerformance">
          <button
            @click="loadMetrics"
            :disabled="data.loading"
            class="text-xs text-accent hover:text-accent/80 transition-colors"
          >
            <i v-if="data.loading" class="pi pi-spin pi-spinner mr-1" />
            <i v-else class="pi pi-chart-bar mr-1" />
            Load performance metrics
          </button>
        </div>
        <template v-else>
          <ModePerformancePanel v-if="data.modePerformance" :performance="data.modePerformance" />
          <TelemetryPanel v-if="data.recordingTelemetry" :telemetry="data.recordingTelemetry" />
          <p v-if="!hasMetricsData && !data.loading" class="text-xs text-text-disabled">
            No metrics data available.
          </p>
        </template>
      </div>

      <!-- H5 Structure (collapsible) -->
      <div class="pt-2">
        <button
          @click="toggleH5"
          class="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-main transition-colors"
        >
          <i class="pi text-xs" :class="showH5 ? 'pi-chevron-down' : 'pi-chevron-right'" />
          H5 File Structure
          <i v-if="data.loading && showH5 && !data.recordingFileInfo" class="pi pi-spin pi-spinner ml-1" />
        </button>
        <div v-if="showH5 && data.recordingFileInfo" class="mt-3">
          <H5FileViewer :info="data.recordingFileInfo" />
        </div>
      </div>
    </div>

    <!-- ==================== Analysis ==================== -->
    <div v-if="activeSection === 'analysis' && data.selectedRecording" class="space-y-4">
      <!-- Shared preprocessing controls -->
      <div class="flex items-center gap-3 flex-wrap px-3 py-2 bg-bg-elevated rounded-lg">
        <label class="flex items-center gap-1.5 text-xs text-text-muted">
          Bandpass
          <NumberInput v-model="aLowcut" :min="0.01" :max="aHighcut - 0.1" :step="0.5" compact class="w-16" />
          <span class="text-text-disabled">&ndash;</span>
          <NumberInput v-model="aHighcut" :min="aLowcut + 0.1" :max="200" :step="1" compact class="w-16" />
          <span class="text-text-disabled">Hz</span>
        </label>
        <span class="text-border">|</span>
        <label class="flex items-center gap-1.5 text-xs text-text-muted">
          <input type="checkbox" v-model="aReref" class="accent-accent" /> CAR
        </label>
        <span class="text-border">|</span>
        <label class="flex items-center gap-1.5 text-xs text-text-muted">
          Epoch
          <NumberInput v-model="aEpochTmin" :min="-2" :max="0" :step="0.1" compact class="w-16" />
          <span class="text-text-disabled">to</span>
          <NumberInput v-model="aEpochTmax" :min="0.1" :max="5" :step="0.1" compact class="w-16" />
          <span class="text-text-disabled">s</span>
        </label>
        <span class="flex-1" />
        <button
          @click="applyAnalysis"
          class="px-3 py-1 text-xs font-medium bg-accent/20 text-accent rounded hover:bg-accent/30 transition-colors"
        >Apply</button>
      </div>

      <!-- ERP -->
      <ERPPreviewPanel
        :key="'erp-' + analysisKey"
        :recordingId="data.selectedRecording.recording_id"
        :isLive="isActiveRecording"
        :lowcut="aLowcut" :highcut="aHighcut" :apply-rereferencing="aReref"
        :epochTmin="aEpochTmin" :epochTmax="aEpochTmax"
      />

      <!-- Signal QC -->
      <SignalQCPanel
        :key="'qc-' + analysisKey"
        :recordingId="data.selectedRecording.recording_id"
        :lowcut="aLowcut" :highcut="aHighcut" :apply-rereferencing="aReref"
      />
    </div>

    <!-- Loading -->
    <div v-if="data.loading" class="text-xs text-text-muted text-center py-4">
      <i class="pi pi-spin pi-spinner mr-1" /> Loading...
    </div>

    <ConfirmDialog
      v-if="showDeleteConfirm"
      title="Delete Recording"
      :message="`Delete recording &quot;${data.selectedRecording.recording_name}&quot;? This cannot be undone.`"
      @confirm="confirmDelete"
      @cancel="showDeleteConfirm = false"
    />
  </div>
</template>
