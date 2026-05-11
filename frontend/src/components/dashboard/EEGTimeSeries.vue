<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_HIDDEN, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { getModalityColor } from '../../utils/colors'
import { useVisualizationStore, vizDirty } from '../../stores/visualization'
import { useTelemetryStore } from '../../stores/telemetry'
import NumberInput from '../common/NumberInput.vue'
import ToggleSwitch from '../common/ToggleSwitch.vue'

const viz = useVisualizationStore()
const telemetry = useTelemetryStore()

const QC_COLORS: Record<string, string> = {
  good: '#40E8C0',
  warning: '#F5A623',
  bad: '#E84040',
  unknown: '#555',
}

function channelQuality(chIdx: number): string {
  const qc = telemetry.data?.channel_quality as any
  if (!qc?.channels) return 'unknown'
  const ch = qc.channels.find((c: any) => c.index === chIdx)
  return ch?.status ?? 'unknown'
}

function isManualFlag(chIdx: number): boolean {
  const flags = telemetry.data?.channel_quality?.manual_flags?.eeg
  return flags?.includes(chIdx) ?? false
}

function isManualUnflag(chIdx: number): boolean {
  const unflagged = telemetry.data?.channel_quality?.manual_unflagged?.eeg
  return unflagged?.includes(chIdx) ?? false
}

function qcDotStyle(chIdx: number): Record<string, string> {
  const status = channelQuality(chIdx)
  const manual = isManualFlag(chIdx)
  const unflagged = isManualUnflag(chIdx)

  if (manual) {
    // Manually flagged: ring outline, no fill
    return { border: `2px solid ${QC_COLORS.bad}`, backgroundColor: 'transparent' }
  }
  if (unflagged) {
    // Auto-detected but operator unflagged: dashed green
    return { border: `1.5px dashed ${QC_COLORS.good}`, backgroundColor: 'transparent' }
  }
  return { backgroundColor: (QC_COLORS[status] || QC_COLORS['unknown'])! }
}

function onChannelClick(chIdx: number) {
  telemetry.toggleChannelFlag('eeg', chIdx)
}

const wrapperEl = ref<HTMLElement | null>(null)
const containers = ref<HTMLElement[]>([])
const plots: uPlot[] = []
let animFrame = 0
let lastDataVersion = 0
let resizeObs: ResizeObserver | null = null

const EEG_COLOR = getModalityColor('eeg')

function buildPlot(el: HTMLElement, isBottom: boolean): uPlot {
  const opts: uPlot.Options = {
    width: el.clientWidth,
    height: viz.channelHeight,
    cursor: CURSOR_HIDDEN,
    legend: LEGEND_HIDDEN,
    series: [
      {},
      { stroke: EEG_COLOR, width: 1 },
    ],
    axes: [
      makeAxis({
        show: isBottom,
        size: isBottom ? 24 : 0,
        values: (_, ticks) => ticks.map(v => v.toFixed(0) + 's'),
      }),
      makeAxis({ show: false, size: 0 }),
    ],
    scales: { x: { time: false } },
  }
  return new uPlot(opts, [[0], [0]], el)
}

function render() {
  // Flush staged events to reactive ref (batched, once per frame)
  if (vizDirty.eventsChanged) viz.flushEvents()

  if (vizDirty.dataVersion === lastDataVersion || !viz.initialized) {
    animFrame = requestAnimationFrame(render)
    return
  }

  const { start, end } = viz.visibleChannelRange
  const ta = viz.timeAxis

  for (let i = 0; i < plots.length; i++) {
    const chIdx = start + i
    if (chIdx >= end || chIdx >= viz.eegBuffers.length) continue
    const buf = viz.eegBuffers[chIdx]!
    const data = buf.toArray()
    const t = ta.length === data.length ? ta : ta.slice(ta.length - data.length)
    plots[i]!.setData([t, data])
  }

  lastDataVersion = vizDirty.dataVersion
  animFrame = requestAnimationFrame(render)
}

function createPlots() {
  plots.forEach(p => p.destroy())
  plots.length = 0

  const { start, end } = viz.visibleChannelRange
  const els = containers.value

  const nVisible = end - start
  for (let i = 0; i < nVisible; i++) {
    if (!els[i]) continue
    const isBottom = i === nVisible - 1
    plots.push(buildPlot(els[i]!, isBottom))
  }
}

function updateChannelsPerPage() {
  if (wrapperEl.value) {
    viz.setChannelsPerPage(wrapperEl.value.clientHeight)
  }
}

watch(() => viz.visibleChannelRange, () => nextTick(createPlots), { deep: true })
watch(() => viz.initialized, (v) => { if (v) nextTick(createPlots) })
watch(() => viz.channelHeight, () => {
  updateChannelsPerPage()
  nextTick(createPlots)
})

onMounted(() => {
  if (viz.initialized) nextTick(createPlots)
  animFrame = requestAnimationFrame(render)

  // Auto-size channels to fill container
  if (wrapperEl.value) {
    updateChannelsPerPage()
    resizeObs = new ResizeObserver(() => {
      updateChannelsPerPage()
      // Resize existing plots to match new container width
      for (let i = 0; i < plots.length; i++) {
        const el = containers.value[i]
        if (el) plots[i]!.setSize({ width: el.clientWidth, height: viz.channelHeight })
      }
    })
    resizeObs.observe(wrapperEl.value)
  }
})

onUnmounted(() => {
  cancelAnimationFrame(animFrame)
  plots.forEach(p => p.destroy())
  if (resizeObs) resizeObs.disconnect()
})

function setRef(el: any, i: number) {
  if (el) containers.value[i] = el as HTMLElement
}

const showPagination = computed(() => viz.totalPages > 1)

// Viz preprocessing controls
function vizPreprocField<T>(field: string, defaultVal: T) {
  return computed({
    get: () => viz.vizPreproc.eeg?.[field] ?? defaultVal,
    set: (v: T) => {
      viz.updateVizPreproc({ ...viz.vizPreproc, eeg: { ...viz.vizPreproc.eeg, [field]: v } })
    },
  })
}
const eegLow = vizPreprocField('filter_low', 0.5)
const eegHigh = vizPreprocField('filter_high', 50.0)
const eegCAR = vizPreprocField('apply_rereferencing', true)
</script>

<template>
  <div class="flex flex-col h-full">
    <!-- Controls -->
    <div class="flex items-center justify-between mb-1 px-1 shrink-0">
      <span class="text-xs text-text-muted">
        {{ viz.visibleChannelRange.start + 1 }}–{{ viz.visibleChannelRange.end }}
        / {{ viz.totalEegChannels }} ch
      </span>
      <div class="flex gap-1 items-center">
        <!-- Viz preprocessing -->
        <label class="flex items-center gap-0.5" title="Lowcut filter (Hz)">
          <span class="text-xs text-text-muted">Lowcut</span>
          <NumberInput v-model="eegLow" :step="0.1" :min="0" :max="20" compact class="w-20" />
        </label>
        <label class="flex items-center gap-0.5" title="Highcut filter (Hz)">
          <span class="text-xs text-text-muted">Highcut</span>
          <NumberInput v-model="eegHigh" :step="1" :min="1" :max="500" compact class="w-20" />
        </label>
        <label class="flex items-center gap-1" title="Common Average Reference">
          <span class="text-xs text-text-muted">CAR</span>
          <ToggleSwitch v-model="eegCAR" compact />
        </label>
        <div class="w-px h-4 bg-border mx-1" />
        <!-- Density toggle -->
        <button
          @click="viz.setDensity(viz.compact ? 'normal' : 'compact')"
          class="px-1.5 py-0.5 text-xs rounded transition-colors border"
          :class="viz.compact
            ? 'bg-accent/15 text-accent border-accent/40'
            : 'bg-bg-input text-text-disabled border-border'"
          title="Toggle compact/normal channel height"
        >
          {{ viz.compact ? 'Compact' : 'Normal' }}
        </button>
        <!-- Pagination (only if needed) -->
        <template v-if="showPagination">
          <button
            @click="viz.prevPage()"
            :disabled="viz.currentPage === 0"
            class="px-1.5 py-0.5 text-xs rounded bg-bg-input text-text-muted hover:text-text-main
                   disabled:opacity-30 transition-colors border border-border"
          >
            <i class="pi pi-chevron-left text-xs" />
          </button>
          <span class="text-xs text-text-disabled px-1">
            {{ viz.currentPage + 1 }}/{{ viz.totalPages }}
          </span>
          <button
            @click="viz.nextPage()"
            :disabled="viz.currentPage >= viz.totalPages - 1"
            class="px-1.5 py-0.5 text-xs rounded bg-bg-input text-text-muted hover:text-text-main
                   disabled:opacity-30 transition-colors border border-border"
          >
            <i class="pi pi-chevron-right text-xs" />
          </button>
        </template>
      </div>
    </div>

    <!-- Plot grid (fills remaining height) -->
    <div ref="wrapperEl" class="flex-1 min-h-0 overflow-hidden">
      <div class="grid grid-cols-1 gap-px h-full" :style="{ gridAutoRows: viz.channelHeight + 'px' }">
        <div
          v-for="i in (viz.visibleChannelRange.end - viz.visibleChannelRange.start)"
          :key="viz.visibleChannelRange.start + i - 1"
          class="relative min-h-0"
        >
          <!-- Channel label + QC dot (click to toggle bad flag) -->
          <button
            class="absolute top-0.5 left-1 z-10 flex items-center gap-1 cursor-pointer rounded px-0.5 hover:bg-text-main/5"
            @click="onChannelClick(viz.visibleChannelRange.start + i - 1)"
            :title="`Click to flag/unflag channel ${viz.eegLabels[viz.visibleChannelRange.start + i - 1] || 'CH' + (viz.visibleChannelRange.start + i)}`"
          >
            <div
              class="w-1.5 h-1.5 rounded-full shrink-0"
              :style="qcDotStyle(viz.visibleChannelRange.start + i - 1)"
            />
            <span class="text-xs text-text-disabled font-mono leading-none">
              {{ viz.eegLabels[viz.visibleChannelRange.start + i - 1] || `CH${viz.visibleChannelRange.start + i}` }}
            </span>
          </button>
          <div
            :ref="(el) => setRef(el, i - 1)"
            class="w-full h-full"
          />
        </div>
      </div>
    </div>
  </div>
</template>
