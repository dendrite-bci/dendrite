<script setup lang="ts">
import { ref, computed } from 'vue'
import { predClassColor } from '../utils/colors'
import { usePipelineStore } from '../stores/pipeline'
import { useTelemetryStore } from '../stores/telemetry'
import { useVisualizationStore } from '../stores/visualization'
import EEGTimeSeries from '../components/dashboard/EEGTimeSeries.vue'
import EventRaster from '../components/dashboard/EventRaster.vue'
import PerformancePlot from '../components/dashboard/PerformancePlot.vue'
import ERPPlot from '../components/dashboard/ERPPlot.vue'
import BandPowerPlot from '../components/dashboard/BandPowerPlot.vue'
import AsyncPredictionPlot from '../components/dashboard/AsyncPredictionPlot.vue'
import ModalityTimeSeries from '../components/dashboard/ModalityTimeSeries.vue'
import PSDSidePanel from '../components/dashboard/PSDSidePanel.vue'
import ModeDetailDialog from '../components/dashboard/ModeDetailDialog.vue'

const pipeline = usePipelineStore()
const telemetry = useTelemetryStore()
const viz = useVisualizationStore()

const modeNames = computed(() => viz.modeNamesList)

function textLevel(value: number, low: number, high: number): string {
  if (value < low) return 'text-level-ok'
  if (value < high) return 'text-level-warn'
  return 'text-level-danger'
}

function bgLevel(value: number, low: number, high: number): string {
  if (value < low) return 'bg-level-ok'
  if (value < high) return 'bg-level-warn'
  return 'bg-level-danger'
}

function formatElapsed(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

const qualitySummary = computed(() => {
  const qc = (telemetry.data as any)?.channel_quality
  if (!qc?.channels) return null
  const total = qc.channels.length
  const bad = qc.channels.filter((c: any) => c.status === 'bad').length
  return { total, good: total - bad, bad }
})

const modeTelemetryMap = computed(() => {
  const map: Record<string, any> = {}
  for (const md of telemetry.data?.modes ?? []) map[md.name] = md
  return map
})

const modeDisplayData = computed(() =>
  modeNames.value.map(name => ({
    name,
    isAsync: viz.modeTypes[name] === 'asynchronous',
    telemetry: modeTelemetryMap.value[name] ?? null,
  }))
)

function modeLatest(name: string) {
  const m = viz.modeMetrics[name]
  if (!m || m.accuracy.length === 0) return null
  return {
    accuracy: m.accuracy[m.accuracy.length - 1],
    confidence: m.confidence.length ? m.confidence[m.confidence.length - 1] : null,
    kappa: m.kappa.length ? m.kappa[m.kappa.length - 1] : null,
    trials: m.accuracy.length,
  }
}

function modePrediction(name: string) {
  return viz.modePredictions[name] ?? null
}

// Collapsible sections
const showEvents = ref(true)
const expandedModalities = ref<Record<string, boolean>>({})
const expandedMode = ref<string | null>(null)
const showEEG = ref(true)
const hiddenModalities = ref<Set<string>>(new Set())
const psdPanels = ref<Set<string>>(new Set())
const psdWidth = ref(220)

function togglePSD(mod: string) {
  const s = psdPanels.value
  if (s.has(mod)) s.delete(mod)
  else s.add(mod)
  psdPanels.value = new Set(s)
}

function startDragPSD(e: MouseEvent) {
  e.preventDefault()
  const startX = e.clientX
  const startW = psdWidth.value
  const onMove = (ev: MouseEvent) => {
    const delta = startX - ev.clientX
    psdWidth.value = Math.max(120, Math.min(400, startW + delta))
  }
  const onUp = () => {
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}

function toggleModality(mod: string) {
  expandedModalities.value[mod] = !expandedModalities.value[mod]
}

function toggleModalityVisibility(mod: string) {
  const s = hiddenModalities.value
  if (s.has(mod)) s.delete(mod)
  else s.add(mod)
  hiddenModalities.value = new Set(s)
}

const allModalities = computed(() =>
  ['eeg', ...Object.keys(viz.modalityBuffers).filter(k => k !== 'eeg')]
)
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- Dashboard -->
    <template v-if="pipeline.status.recording">
      <!-- Status bar -->
      <div class="bg-bg-panel border-b border-border shrink-0 text-xs h-[42px] flex flex-col justify-center">
        <!-- Row 1: Signal context -->
        <div class="flex items-center gap-4 px-4 py-1">
          <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full" :class="viz.connected ? 'bg-status-ok' : 'bg-status-error'" />
            <span class="font-mono text-text-muted">{{ formatElapsed(pipeline.status.elapsed_seconds) }}</span>
          </div>
          <span v-if="viz.initialized" class="text-text-disabled">
            {{ viz.totalEegChannels }} ch · {{ viz.sampleRate }} Hz
          </span>
          <div v-if="viz.initialized" class="flex items-center gap-1">
            <button
              v-for="mod in allModalities" :key="mod"
              @click="mod === 'eeg' ? showEEG = !showEEG : toggleModalityVisibility(mod)"
              class="px-2 py-0.5 text-xs font-semibold uppercase rounded transition-colors"
              :class="(mod === 'eeg' ? showEEG : !hiddenModalities.has(mod))
                ? 'bg-accent/20 text-accent'
                : 'bg-bg-input text-text-disabled hover:text-text-muted'"
            >{{ mod }}</button>
          </div>
          <div class="flex-1" />
          <!-- Stream latencies -->
          <template v-if="telemetry.data?.streams">
            <span v-for="s in telemetry.data.streams" :key="s.type" class="text-text-muted">
              <span class="text-text-disabled">{{ s.type }}</span> <span class="font-mono" :class="textLevel(s.latency_ms, 10, 30)">{{ s.latency_ms.toFixed(1) }}ms</span>
            </span>
          </template>
          <span class="text-text-disabled">{{ viz.eventHistory.length }} events</span>
        </div>
        <!-- Row 2: Health & resources -->
        <div v-if="telemetry.data" class="flex items-center gap-4 px-4 py-1">
          <!-- Channel quality -->
          <div v-if="qualitySummary" class="flex items-center gap-1.5 text-text-muted">
            <span class="text-text-disabled">Quality</span>
            <span class="font-mono text-level-ok">{{ qualitySummary.good }}/{{ qualitySummary.total }}</span>
            <span v-if="qualitySummary.bad > 0" class="font-mono text-level-danger">{{ qualitySummary.bad }} bad</span>
          </div>
          <div class="w-px h-3 bg-white/[0.06]" />
          <!-- Per-process health dots + hover popover -->
          <div v-if="telemetry.data.system.processes.length > 0" class="relative group flex items-center gap-1.5 cursor-default">
            <span class="text-text-disabled">Processes</span>
            <div
              v-for="proc in telemetry.data.system.processes" :key="proc.pid"
              class="w-2.5 h-2.5 rounded-full transition-colors"
              :class="bgLevel(proc.cpu_percent, 50, 80)"
            />
            <!-- Popover on hover -->
            <div class="absolute left-0 top-full mt-1.5 z-50 hidden group-hover:block">
              <div class="bg-bg-elevated border border-border rounded-lg shadow-xl p-2.5 min-w-[200px] space-y-1">
                <div
                  v-for="proc in telemetry.data.system.processes" :key="'pop_' + proc.pid"
                  class="flex items-center gap-2 text-xs"
                >
                  <div class="w-2 h-2 rounded-full shrink-0" :class="bgLevel(proc.cpu_percent, 50, 80)" />
                  <span class="text-text-label w-[80px] truncate shrink-0">{{ proc.name }}</span>
                  <div class="w-[50px] h-1.5 bg-bg-input rounded-full overflow-hidden shrink-0">
                    <div
                      class="h-full rounded-full"
                      :class="bgLevel(proc.cpu_percent, 50, 80)"
                      :style="{ width: `${Math.min(proc.cpu_percent, 100)}%` }"
                    />
                  </div>
                  <span class="font-mono text-text-muted w-8 text-right shrink-0">{{ proc.cpu_percent.toFixed(0) }}%</span>
                  <span class="font-mono text-text-disabled">{{ proc.memory_mb.toFixed(0) }}M</span>
                </div>
              </div>
            </div>
          </div>
          <div class="flex-1" />
          <!-- System totals -->
          <span class="text-text-muted">
            <span class="text-text-disabled">CPU</span> <span class="font-mono" :class="textLevel(telemetry.data.system.cpu_percent || 0, 50, 80)">{{ (telemetry.data.system.cpu_percent || 0).toFixed(0) }}%</span>
          </span>
          <span class="text-text-muted">
            <span class="text-text-disabled">RAM</span> <span class="font-mono">{{ (telemetry.data.system.memory_used_gb || 0).toFixed(1) }}/{{ (telemetry.data.system.memory_total_gb || 0).toFixed(0) }}G</span>
          </span>
        </div>
      </div>

      <!-- Main content -->
      <div class="flex flex-1 min-h-0">
        <!-- Left: Signals (75%) -->
        <div class="flex-[3] flex flex-col border-r border-border min-h-0 overflow-y-auto">
          <!-- EEG section -->
          <div v-if="viz.initialized && showEEG" class="shrink-0">
            <div class="flex items-center">
              <div class="flex-1 flex items-center gap-1.5 px-3 py-1.5 text-xs text-text-muted">
                <span class="font-medium">EEG</span>
                <span class="text-text-disabled">({{ viz.totalEegChannels }} ch)</span>
              </div>
              <button
                v-if="viz.psdData['eeg']"
                @click.stop="togglePSD('eeg')"
                class="px-2 py-1 text-xs transition-colors"
                :class="psdPanels.has('eeg')
                  ? 'text-accent'
                  : 'text-text-disabled hover:text-text-muted'"
                title="Toggle PSD"
              >PSD</button>
            </div>
            <div class="px-2 pb-2">
              <div class="flex bg-bg-panel rounded-md border border-border/30 overflow-hidden" style="min-height: 300px; height: 60vh;">
                <div class="flex-1 min-w-0 min-h-0 overflow-hidden">
                  <EEGTimeSeries />
                </div>
                <template v-if="psdPanels.has('eeg') && viz.psdData['eeg']">
                  <PSDSidePanel modality="eeg" :width="psdWidth" @drag-start="startDragPSD" />
                </template>
              </div>
            </div>
          </div>
          <div v-else class="h-[200px] flex items-center justify-center text-xs text-text-disabled">
            Waiting for data...
          </div>

          <!-- Modality panels (EMG, EOG, etc.) — collapsible, stacked below EEG -->
          <div
            v-for="(modData, modName) in viz.modalityBuffers"
            :key="modName"
            v-show="!hiddenModalities.has(modName as string)"
            class="shrink-0 border-t border-border"
          >
            <div class="flex items-center">
              <button
                @click="toggleModality(modName as string)"
                class="flex-1 flex items-center gap-1.5 px-3 py-1.5 text-xs text-text-muted hover:text-text-main transition-colors"
              >
                <i class="pi text-xs" :class="expandedModalities[modName as string] ? 'pi-chevron-down' : 'pi-chevron-right'" />
                <span class="font-medium">{{ (modName as string).toUpperCase() }}</span>
                <span class="text-text-disabled">({{ modData.labels.length }} ch)</span>
              </button>
              <button
                v-if="viz.psdData[modName as string] && expandedModalities[modName as string]"
                @click.stop="togglePSD(modName as string)"
                class="px-2 py-1 text-xs transition-colors"
                :class="psdPanels.has(modName as string)
                  ? 'text-accent'
                  : 'text-text-disabled hover:text-text-muted'"
                title="Toggle PSD"
              >PSD</button>
            </div>
            <div v-if="expandedModalities[modName as string]" class="flex px-2 pb-2">
              <div class="flex-1 min-w-0 overflow-hidden">
                <ModalityTimeSeries :modality="modName as string" />
              </div>
              <template v-if="psdPanels.has(modName as string) && viz.psdData[modName as string]">
                <PSDSidePanel :modality="modName as string" :width="psdWidth" @drag-start="startDragPSD" />
              </template>
            </div>
          </div>

          <!-- Events section (collapsible) -->
          <div v-if="viz.eventHistory.length > 0 || showEvents" class="shrink-0 border-t border-border">
            <button
              @click="showEvents = !showEvents"
              class="w-full flex items-center gap-1.5 px-3 py-1 text-xs text-text-muted hover:text-text-main transition-colors"
            >
              <i class="pi text-xs" :class="showEvents ? 'pi-chevron-down' : 'pi-chevron-right'" />
              Events
            </button>
            <div v-if="showEvents" class="px-2 pb-2">
              <EventRaster />
            </div>
          </div>
        </div>

        <!-- Right: Mode Decoding (25%) -->
        <div class="flex-1 p-3 space-y-3 overflow-y-auto">
          <h3 class="text-xs font-medium text-text-muted uppercase tracking-wide px-1">
            Modes
          </h3>

          <div v-if="modeDisplayData.length === 0" class="text-center py-4">
            <i class="pi pi-brain text-xl text-text-disabled block mb-1" />
            <p class="text-xs text-text-disabled">No mode data yet</p>
          </div>

          <div v-for="md in modeDisplayData" :key="md.name"
               class="bg-bg-elevated rounded-md border border-border/30 overflow-hidden">
            <div class="px-3 py-2 border-b border-border/30 bg-white/[0.02]">
              <div class="flex items-center justify-between mb-1">
                <div class="flex items-center gap-2">
                  <span class="text-sm font-semibold text-text-main">{{ md.name }}</span>
                  <button
                    @click="expandedMode = md.name"
                    class="w-5 h-5 rounded flex items-center justify-center text-text-disabled hover:text-text-main hover:bg-bg-input transition-colors"
                    title="Expand"
                  >
                    <i class="pi pi-external-link text-xs" />
                  </button>
                </div>
                <span v-if="modePrediction(md.name)"
                  class="text-xs font-mono font-semibold px-2 py-0.5 rounded"
                  :style="{ color: predClassColor(modePrediction(md.name)!.eventName), backgroundColor: predClassColor(modePrediction(md.name)!.eventName) + '20' }">
                  {{ modePrediction(md.name)!.eventName }} {{ (modePrediction(md.name)!.confidence * 100).toFixed(0) }}%
                </span>
              </div>
              <div class="flex items-center gap-4 text-xs">
                <span v-if="modeLatest(md.name)?.accuracy != null" class="text-text-muted">
                  Acc <span class="font-mono font-semibold text-level-ok">{{ (modeLatest(md.name)!.accuracy! * 100).toFixed(0) }}%</span>
                </span>
                <span v-if="modeLatest(md.name)?.confidence != null" class="text-text-muted">
                  Conf <span class="font-mono text-text-label">{{ (modeLatest(md.name)!.confidence! * 100).toFixed(0) }}%</span>
                </span>
                <span v-if="modeLatest(md.name)?.kappa != null" class="text-text-muted">
                  K <span class="font-mono text-text-label">{{ modeLatest(md.name)!.kappa!.toFixed(2) }}</span>
                </span>
                <span v-if="md.telemetry?.internal_ms != null" class="text-text-muted">
                  Proc <span class="font-mono" :class="textLevel(md.telemetry.internal_ms, 2, 10)">{{ md.telemetry.internal_ms.toFixed(0) }}ms</span>
                </span>
                <span v-if="md.telemetry?.inference_ms != null" class="text-text-muted">
                  Inf <span class="font-mono" :class="textLevel(md.telemetry.inference_ms, 10, 30)">{{ md.telemetry.inference_ms.toFixed(0) }}ms</span>
                </span>
                <span v-if="modeLatest(md.name)?.trials" class="text-text-disabled">
                  {{ modeLatest(md.name)!.trials }} trials
                </span>
              </div>
            </div>
            <div class="p-2 space-y-2">
              <AsyncPredictionPlot v-if="md.isAsync" :mode-name="md.name" />
              <PerformancePlot v-else :mode-name="md.name" />
              <ERPPlot :mode-name="md.name" />
              <BandPowerPlot :mode-name="md.name" />
            </div>
          </div>
        </div>
      </div>
    </template>

    <!-- Mode Detail Dialog -->
    <ModeDetailDialog :mode-name="expandedMode" @close="expandedMode = null" />
  </div>
</template>
