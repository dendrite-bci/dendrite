<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, computed, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { useVisualizationStore } from '../../stores/visualization'
import { useTelemetryStore } from '../../stores/telemetry'

const props = withDefaults(defineProps<{
  modeName: string
  detail?: boolean
}>(), { detail: false })

const viz = useVisualizationStore()
const telemetry = useTelemetryStore()
const container = ref<HTMLElement | null>(null)
let plot: uPlot | null = null
let resizeObserver: ResizeObserver | null = null

const EVENT_COLORS = ['#66B2FF', '#FF6B6B', '#4ECB71', '#E8B44C', '#B088F9', '#FF85B3']
const CH_COLORS = [
  '#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#fb923c',
  '#38bdf8', '#e879f9', '#4ade80', '#f87171', '#818cf8', '#facc15',
]

const erpData = computed(() => viz.modeERPs[props.modeName] || {})
const hasData = computed(() => Object.keys(erpData.value).length > 0)

// Channel selection (detail mode only)
const selectedChannels = ref<Set<number>>(new Set())
const selectedEvent = ref<string>('')

const availableChannels = computed(() => {
  const first = Object.values(erpData.value)[0]
  return first?.channelLabels ?? []
})

const badChannelSet = computed(() => {
  const list: number[] = (telemetry.data?.channel_quality as any)?.bad_channels?.eeg ?? []
  return new Set(list)
})

// Auto-select first event when data appears
watch(hasData, (v) => {
  if (v && !selectedEvent.value) {
    selectedEvent.value = Object.keys(erpData.value)[0] ?? ''
  }
})

function toggleChannel(idx: number) {
  const s = selectedChannels.value
  if (s.has(idx)) s.delete(idx)
  else s.add(idx)
  selectedChannels.value = new Set(s) // trigger reactivity
  rebuildNeeded = true
}

function selectAllChannels() {
  selectedChannels.value = new Set(availableChannels.value.map((_, i) => i))
  rebuildNeeded = true
}

function clearChannels() {
  selectedChannels.value = new Set()
  rebuildNeeded = true
}

// --- Grand average mode (compact) ---
function buildGrandAvgData() {
  const classes = Object.keys(erpData.value)
  const first = erpData.value[classes[0]!]
  if (!first || first.count === 0) return null

  const nTimes = first.nTimes
  const dt = 1000 / first.sampleRate
  const timeAxis = Array.from({ length: nTimes }, (_, i) => first.startOffsetMs + i * dt)

  const seriesData: number[][] = [timeAxis]
  for (const cls of classes) {
    const acc = erpData.value[cls]
    if (!acc || acc.count === 0) { seriesData.push([]); continue }
    seriesData.push(acc.sum.map(v => v / acc.count))
  }
  return { labels: classes, seriesData }
}

// --- Per-channel mode (detail) ---
function buildChannelData() {
  const evt = selectedEvent.value
  const acc = erpData.value[evt]
  if (!acc || acc.count === 0) return null

  const nTimes = acc.nTimes
  const dt = 1000 / acc.sampleRate
  const timeAxis = Array.from({ length: nTimes }, (_, i) => acc.startOffsetMs + i * dt)

  const labels: string[] = []
  const seriesData: number[][] = [timeAxis]

  // Grand average always shown as first series
  labels.push('Grand Avg')
  seriesData.push(acc.sum.map(v => v / acc.count))

  // Selected channels
  for (const chIdx of Array.from(selectedChannels.value).sort((a, b) => a - b)) {
    if (chIdx >= acc.channelSums.length) continue
    labels.push(acc.channelLabels[chIdx] ?? `ch${chIdx}`)
    seriesData.push(acc.channelSums[chIdx]!.map(v => v / acc.count))
  }

  return { labels, seriesData }
}

let rebuildNeeded = true

function seriesColor(index: number): string {
  if (props.detail) {
    return index === 0 ? '#888' : CH_COLORS[(index - 1) % CH_COLORS.length]!
  }
  return EVENT_COLORS[index % EVENT_COLORS.length]!
}

function buildPlot() {
  if (!container.value || !hasData.value) return
  if (plot) { plot.destroy(); plot = null }

  const result = props.detail && selectedEvent.value ? buildChannelData() : buildGrandAvgData()
  if (!result) return
  const { labels, seriesData } = result

  const seriesOpts: uPlot.Series[] = [{}]
  for (let i = 0; i < labels.length; i++) {
    const isGrandAvg = props.detail && i === 0
    seriesOpts.push({
      label: labels[i]!,
      stroke: seriesColor(i),
      width: isGrandAvg ? 2 : 1.5,
      dash: isGrandAvg ? [6, 3] : undefined,
    })
  }

  const opts: uPlot.Options = {
    width: container.value.clientWidth,
    height: container.value.clientHeight,
    cursor: { show: props.detail },
    legend: LEGEND_HIDDEN,
    series: seriesOpts,
    axes: [
      makeAxis({ size: 22, values: (_, ticks) => ticks.map(v => v.toFixed(0)), label: props.detail ? 'ms' : '' }),
      makeAxis({ size: 35 }),
    ],
    scales: { x: { time: false } },
  }

  plot = new uPlot(opts, seriesData as any, container.value)
  lastSeriesCount = seriesData.length
  rebuildNeeded = false
}

let lastSeriesCount = 0

function updatePlot() {
  if (rebuildNeeded || !plot || !hasData.value) { buildPlot(); return }
  const result = props.detail && selectedEvent.value ? buildChannelData() : buildGrandAvgData()
  if (!result) return
  // Rebuild if series count changed (new event class appeared)
  if (result.seriesData.length !== lastSeriesCount) { buildPlot(); return }
  try { plot.setData(result.seriesData as any) } catch { buildPlot() }
}

// Reactive: store replaces modeERPs.value with a new Record on change
watch(() => viz.modeERPs, () => {
  if (hasData.value) nextTick(updatePlot)
})

onMounted(() => {
  if (hasData.value) {
    selectedEvent.value = selectedEvent.value || Object.keys(erpData.value)[0] || ''
    nextTick(buildPlot)
  }
  if (container.value) {
    resizeObserver = new ResizeObserver(() => {
      if (plot && container.value) {
        plot.setSize({ width: container.value.clientWidth, height: container.value.clientHeight })
      }
    })
    resizeObserver.observe(container.value)
  }
})
onUnmounted(() => {
  resizeObserver?.disconnect()
  plot?.destroy()
})
</script>

<template>
  <div v-if="hasData">
    <div class="text-xs text-text-muted uppercase tracking-wide mb-1">ERP</div>

    <!-- Detail mode: event selector + channel checkboxes -->
    <div v-if="detail" class="mb-2 space-y-1.5">
      <!-- Event selector -->
      <div class="flex items-center gap-2">
        <span class="text-xs text-text-muted">Event:</span>
        <button v-for="cls in Object.keys(erpData)" :key="cls"
          @click="selectedEvent = cls; rebuildNeeded = true"
          class="px-2 py-0.5 text-xs rounded transition-colors"
          :class="selectedEvent === cls
            ? 'bg-accent/20 text-accent font-semibold'
            : 'bg-bg-input text-text-disabled hover:text-text-muted'"
        >{{ cls }} (n={{ erpData[cls]?.count || 0 }})</button>
      </div>
      <!-- Channel selector -->
      <div class="flex items-center gap-1 flex-wrap">
        <span class="text-xs text-text-muted mr-1">Channels:</span>
        <button @click="selectAllChannels()" class="px-1.5 py-0.5 text-xs text-text-disabled hover:text-text-muted">All</button>
        <button @click="clearChannels()" class="px-1.5 py-0.5 text-xs text-text-disabled hover:text-text-muted">None</button>
        <button v-for="(label, idx) in availableChannels" :key="idx"
          @click="toggleChannel(idx)"
          class="px-1.5 py-0.5 text-xs rounded transition-colors"
          :class="[
            selectedChannels.has(idx) ? 'font-semibold' : 'text-text-disabled hover:text-text-muted',
            badChannelSet.has(idx) ? 'line-through opacity-50' : '',
          ]"
          :style="selectedChannels.has(idx)
            ? { color: CH_COLORS[idx % CH_COLORS.length], backgroundColor: CH_COLORS[idx % CH_COLORS.length] + '20' }
            : {}"
          :title="badChannelSet.has(idx) ? `${label} (bad)` : label"
        >{{ label }}</button>
      </div>
    </div>

    <div ref="container" :class="detail ? 'h-[300px]' : 'h-[120px]'" class="bg-bg-panel rounded border border-border/50" />

    <!-- Compact mode: event legend -->
    <div v-if="!detail" class="flex gap-1.5 mt-1 flex-wrap">
      <span v-for="(cls, i) in Object.keys(erpData)" :key="cls"
        class="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs text-text-muted"
        :style="{ backgroundColor: EVENT_COLORS[i % EVENT_COLORS.length] + '25' }">
        <span class="w-2 h-2 rounded-full" :style="{ backgroundColor: EVENT_COLORS[i % EVENT_COLORS.length] }" />
        {{ cls }} &times;{{ erpData[cls]?.count || 0 }}
      </span>
    </div>
  </div>
</template>
