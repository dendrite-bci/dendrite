<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_HIDDEN, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import NumberInput from '../common/NumberInput.vue'
import type { QCPreview, QCChannelQuality } from '../../types/api'
import { apiFetch } from '../../utils/api'

const props = defineProps<{
  recordingId: number
  lowcut?: number
  highcut?: number
  applyRereferencing?: boolean
}>()

const BAD_COLOR = '#ef4444'
const GOOD_COLOR = '#20c8a0'
const RAW_COLOR = '#2a8be8'
const PREPROC_COLOR = '#20c8a0'
const CH_HEIGHT = 50
const PAGE_SIZE = 8

const MOTOR_CHANNELS = new Set([
  'C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
])

// Controls (use props as defaults)
const badChannelMode = ref<'none' | 'exclude' | 'interpolate'>('exclude')
const channelOffset = ref(0)

// State
const loading = ref(false)
const error = ref('')
const qcData = ref<QCPreview | null>(null)

// Playback
const viewStart = ref(0)
const windowSec = ref(10)
const playing = ref(false)
const speed = ref(1)
let animFrame = 0
let lastTick = 0

// Per-channel charts
const rawContainers = ref<HTMLElement[]>([])
const preprocContainers = ref<HTMLElement[]>([])
const rawPlots: uPlot[] = []
const preprocPlots: uPlot[] = []
let resizeObs: ResizeObserver | null = null

const totalChannels = computed(() => qcData.value?.total_channels ?? 0)
const maxOffset = computed(() => Math.max(0, totalChannels.value - PAGE_SIZE))
const pageLabel = computed(() => {
  const start = channelOffset.value + 1
  const end = Math.min(channelOffset.value + PAGE_SIZE, totalChannels.value)
  return `${start}–${end} / ${totalChannels.value}`
})

const totalDuration = computed(() => {
  if (!qcData.value || qcData.value.sample_rate <= 0) return 0
  return qcData.value.total_samples / qcData.value.sample_rate
})
const maxStart = computed(() => Math.max(0, totalDuration.value - windowSec.value))
const timeDisplay = computed(() => {
  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
  return `${fmt(viewStart.value)} / ${fmt(totalDuration.value)}`
})
const progress = computed(() =>
  totalDuration.value <= 0 ? '0%' : `${Math.round(viewStart.value / totalDuration.value * 100)}%`
)

// --- Fetch ---

async function fetchQC() {
  loading.value = true
  error.value = ''
  try {
    const indices = Array.from({ length: PAGE_SIZE }, (_, i) => channelOffset.value + i)
    const params = new URLSearchParams({
      lowcut: (props.lowcut ?? 0.5).toString(),
      highcut: (props.highcut ?? 50).toString(),
      apply_rereferencing: (props.applyRereferencing ?? true).toString(),
      bad_channel_mode: badChannelMode.value,
      channels: indices.join(','),
    })
    const prevCount = qcData.value?.raw.channels.length ?? 0
    qcData.value = await apiFetch<QCPreview>(
      `/api/data/recordings/${props.recordingId}/qc-preview?${params}`,
      { fallbackMessage: 'Failed to load QC preview' },
    )
    await nextTick()
    const newCount = qcData.value?.raw.channels.length ?? 0
    if (newCount === prevCount && rawPlots.length === newCount) {
      refreshData()
    } else {
      createPlots()
    }
  } catch (e: any) {
    error.value = e.message || 'Failed to load QC preview'
  } finally {
    loading.value = false
  }
}

// --- Per-channel uPlot ---

function buildPlot(el: HTMLElement, color: string, isBottom: boolean): uPlot {
  return new uPlot({
    width: el.clientWidth,
    height: CH_HEIGHT,
    cursor: CURSOR_HIDDEN,
    legend: LEGEND_HIDDEN,
    series: [{}, { stroke: color, width: 1, points: { show: false } }],
    axes: [
      makeAxis({
        show: isBottom,
        size: isBottom ? 24 : 0,
        values: (_, ticks) => ticks.map(v => v.toFixed(0) + 's'),
      }),
      makeAxis({ show: false, size: 0 }),
    ],
    scales: { x: { time: false } },
  }, [[0], [0]], el)
}

function destroyAllPlots() {
  rawPlots.forEach(p => p.destroy())
  preprocPlots.forEach(p => p.destroy())
  rawPlots.length = 0
  preprocPlots.length = 0
}

function createPlots() {
  destroyAllPlots()
  resizeObs?.disconnect()

  if (!qcData.value) return
  const d = qcData.value
  const nCh = d.raw.channels.length
  if (nCh === 0) return

  const rawEls = rawContainers.value
  const preprocEls = preprocContainers.value
  const time = new Float64Array(d.raw.time)

  for (let i = 0; i < nCh; i++) {
    const isBottom = i === nCh - 1
    const isBad = d.raw.channels[i]!.is_bad
    if (rawEls[i]) {
      const p = buildPlot(rawEls[i]!, isBad ? BAD_COLOR : RAW_COLOR, isBottom)
      p.setData([time, new Float64Array(d.raw.channels[i]!.data)])
      rawPlots.push(p)
    }
    if (preprocEls[i]) {
      const p = buildPlot(preprocEls[i]!, isBad ? BAD_COLOR : PREPROC_COLOR, isBottom)
      p.setData([time, new Float64Array(d.preprocessed.channels[i]!.data)])
      preprocPlots.push(p)
    }
  }
  updateView()

  if (rawEls[0]) {
    resizeObs = new ResizeObserver(() => {
      for (let i = 0; i < rawPlots.length; i++) if (rawEls[i]) rawPlots[i]!.setSize({ width: rawEls[i]!.clientWidth, height: CH_HEIGHT })
      for (let i = 0; i < preprocPlots.length; i++) if (preprocEls[i]) preprocPlots[i]!.setSize({ width: preprocEls[i]!.clientWidth, height: CH_HEIGHT })
    })
    resizeObs.observe(rawEls[0]!)
  }
}

function refreshData() {
  if (!qcData.value) return
  const d = qcData.value
  const time = new Float64Array(d.raw.time)
  for (let i = 0; i < rawPlots.length && i < d.raw.channels.length; i++)
    rawPlots[i]!.setData([time, new Float64Array(d.raw.channels[i]!.data)])
  for (let i = 0; i < preprocPlots.length && i < d.preprocessed.channels.length; i++)
    preprocPlots[i]!.setData([time, new Float64Array(d.preprocessed.channels[i]!.data)])
  updateView()
}

function updateView() {
  const min = viewStart.value, max = min + windowSec.value
  for (const p of rawPlots) p.setScale('x', { min, max })
  for (const p of preprocPlots) p.setScale('x', { min, max })
}

// --- Channel paging ---

function prevPage() {
  channelOffset.value = Math.max(0, channelOffset.value - PAGE_SIZE)
  fetchQC()
}
function nextPage() {
  channelOffset.value = Math.min(maxOffset.value, channelOffset.value + PAGE_SIZE)
  fetchQC()
}
function setMotorChannels() {
  if (!qcData.value) return
  const motorIdx = qcData.value.quality.channels.filter(ch => MOTOR_CHANNELS.has(ch.label)).map(ch => ch.index)
  if (motorIdx.length === 0) return
  channelOffset.value = motorIdx[0]!
  fetchQC()
}
function resetChannels() {
  channelOffset.value = 0
  fetchQC()
}

// --- Playback ---

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    if (viewStart.value >= maxStart.value) viewStart.value = 0
    startPlayback()
  } else {
    stopPlayback()
  }
}
function startPlayback() {
  lastTick = performance.now()
  function tick(now: number) {
    const dt = (now - lastTick) / 1000 * speed.value
    lastTick = now
    viewStart.value = Math.min(viewStart.value + dt, maxStart.value)
    updateView()
    if (viewStart.value >= maxStart.value) { playing.value = false; return }
    if (playing.value) animFrame = requestAnimationFrame(tick)
  }
  animFrame = requestAnimationFrame(tick)
}
function stopPlayback() {
  playing.value = false
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = 0 }
}
function onSliderInput(e: Event) {
  viewStart.value = parseFloat((e.target as HTMLInputElement).value)
  updateView()
}
function onWindowChange() {
  if (viewStart.value > maxStart.value) viewStart.value = maxStart.value
  updateView()
}

// --- Helpers ---

function statusColor(s: QCChannelQuality['status']): string {
  if (s === 'bad') return BAD_COLOR
  if (s === 'warning') return '#e8a020'
  return GOOD_COLOR
}
function formatNum(v: number): string {
  if (Math.abs(v) < 0.01) return v.toExponential(1)
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  return v.toFixed(2)
}
function setRawRef(el: any, i: number) { if (el) rawContainers.value[i] = el as HTMLElement }
function setPreprocRef(el: any, i: number) { if (el) preprocContainers.value[i] = el as HTMLElement }

watch(() => props.recordingId, () => { viewStart.value = 0; channelOffset.value = 0; fetchQC() })
onMounted(() => fetchQC())
onUnmounted(() => {
  stopPlayback()
  destroyAllPlots()
  resizeObs?.disconnect()
})
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center gap-3">
      <span class="text-xs font-semibold text-text-label uppercase tracking-wider">Signal QC</span>
      <label class="flex items-center gap-1.5 text-xs text-text-muted">
        Bad ch
        <select v-model="badChannelMode" @change="stopPlayback(); fetchQC()" class="text-xs py-0.5 px-1.5">
          <option value="none">None</option>
          <option value="exclude">Exclude from CAR</option>
          <option value="interpolate">Interpolate</option>
        </select>
      </label>
    </div>

    <div v-if="error" class="text-xs text-status-error px-2">{{ error }}</div>
    <div v-if="loading && !qcData" class="text-xs text-text-muted text-center py-8">
      <i class="pi pi-spin pi-spinner mr-1" /> Processing...
    </div>

    <template v-if="qcData">
      <!-- Channel nav + window -->
      <div class="flex items-center gap-2 px-3 py-1.5 bg-bg-elevated rounded-lg">
        <button @click="prevPage" :disabled="channelOffset <= 0 || loading"
          class="px-2 py-0.5 text-xs text-text-muted bg-bg-input rounded hover:text-text-main disabled:opacity-30 transition-colors">&lt;</button>
        <span class="text-xs text-text-muted font-mono min-w-[70px] text-center">{{ pageLabel }}</span>
        <button @click="nextPage" :disabled="channelOffset >= maxOffset || loading"
          class="px-2 py-0.5 text-xs text-text-muted bg-bg-input rounded hover:text-text-main disabled:opacity-30 transition-colors">&gt;</button>
        <button @click="setMotorChannels" class="px-2 py-0.5 text-xs text-text-muted bg-bg-input rounded hover:text-text-main transition-colors">Motor</button>
        <button @click="resetChannels" class="px-2 py-0.5 text-xs text-text-muted bg-bg-input rounded hover:text-text-main transition-colors">Reset</button>
        <span class="flex-1" />
        <label class="flex items-center gap-1.5 text-xs text-text-muted">
          Window
          <NumberInput v-model="windowSec" :min="1" :max="60" :step="1" compact class="w-14" @update:modelValue="onWindowChange" />
          <span class="text-text-disabled">s</span>
        </label>
        <i v-if="loading" class="pi pi-spin pi-spinner text-xs text-accent" />
      </div>

      <!-- Raw panel -->
      <div>
        <span class="text-xs font-semibold px-1" :style="{ color: RAW_COLOR }">Raw</span>
        <div class="grid grid-cols-1 gap-px rounded-lg overflow-hidden mt-0.5">
          <div v-for="(ch, i) in qcData.raw.channels" :key="'r' + i" class="relative" :style="{ height: CH_HEIGHT + 'px' }">
            <span class="absolute top-0.5 left-1 z-10 flex items-center gap-1">
              <span class="w-1.5 h-1.5 rounded-full shrink-0" :style="{ backgroundColor: ch.is_bad ? BAD_COLOR : GOOD_COLOR }" />
              <span class="text-xs text-text-disabled font-mono">{{ ch.label }}</span>
            </span>
            <div :ref="(el) => setRawRef(el, i)" class="w-full h-full bg-bg-panel" />
          </div>
        </div>
      </div>

      <!-- Filtered panel -->
      <div>
        <span class="text-xs font-semibold px-1" :style="{ color: PREPROC_COLOR }">
          Filtered
          <span class="text-text-muted font-normal">
            {{ qcData.preprocessing.lowcut }}&ndash;{{ qcData.preprocessing.highcut }} Hz
            <template v-if="qcData.preprocessing.apply_rereferencing"> + CAR</template>
          </span>
        </span>
        <div class="grid grid-cols-1 gap-px rounded-lg overflow-hidden mt-0.5">
          <div v-for="(ch, i) in qcData.preprocessed.channels" :key="'f' + i" class="relative" :style="{ height: CH_HEIGHT + 'px' }">
            <span class="absolute top-0.5 left-1 z-10 flex items-center gap-1">
              <span class="w-1.5 h-1.5 rounded-full shrink-0" :style="{ backgroundColor: ch.is_bad ? BAD_COLOR : GOOD_COLOR }" />
              <span class="text-xs text-text-disabled font-mono">{{ ch.label }}</span>
            </span>
            <div :ref="(el) => setPreprocRef(el, i)" class="w-full h-full bg-bg-panel" />
          </div>
        </div>
      </div>

      <!-- Playback -->
      <div class="flex items-center gap-3 px-3 py-2 bg-bg-elevated rounded-lg">
        <button @click="togglePlay" class="w-16 py-1 text-xs font-medium rounded transition-colors"
          :class="playing ? 'bg-status-error/20 text-status-error' : 'bg-accent/20 text-accent'">
          {{ playing ? 'Pause' : 'Play' }}
        </button>
        <select v-model.number="speed" class="text-xs py-0.5 px-1.5 w-14">
          <option :value="1">1x</option>
          <option :value="2">2x</option>
          <option :value="5">5x</option>
          <option :value="10">10x</option>
        </select>
        <span class="text-xs text-text-muted font-mono w-24 text-center">{{ timeDisplay }}</span>
        <input type="range" :min="0" :max="maxStart" :step="0.1" :value="viewStart" @input="onSliderInput" class="flex-1 accent-accent h-1" />
        <span class="text-xs text-text-muted font-mono w-10 text-right">{{ progress }}</span>
      </div>

      <!-- Channel quality -->
      <div class="bg-bg-elevated border border-border rounded-lg p-3">
        <span class="text-xs font-semibold text-text-label block mb-2">Channel Quality</span>
        <div class="overflow-x-auto max-h-64 overflow-y-auto">
          <table class="w-full text-xs">
            <thead class="sticky top-0 bg-bg-elevated">
              <tr class="text-text-muted text-left">
                <th class="pb-1 pr-3 font-medium">Channel</th>
                <th class="pb-1 pr-3 font-medium">Status</th>
                <th class="pb-1 pr-3 font-medium text-right">Variance</th>
                <th class="pb-1 pr-3 font-medium text-right">Std</th>
                <th class="pb-1 font-medium text-right">Max Deriv</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="ch in qcData.quality.channels" :key="ch.index" class="border-t border-border/50">
                <td class="py-1 pr-3 text-text-main font-mono">{{ ch.label }}</td>
                <td class="py-1 pr-3">
                  <span class="inline-block px-1.5 py-0.5 rounded text-xs font-medium"
                    :style="{ background: statusColor(ch.status) + '20', color: statusColor(ch.status) }">{{ ch.status }}</span>
                </td>
                <td class="py-1 pr-3 text-right text-text-muted font-mono">{{ formatNum(ch.variance) }}</td>
                <td class="py-1 pr-3 text-right text-text-muted font-mono">{{ formatNum(ch.std) }}</td>
                <td class="py-1 text-right text-text-muted font-mono">{{ formatNum(ch.max_deriv) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>
