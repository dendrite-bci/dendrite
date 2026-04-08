<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { useVisualizationStore } from '../../stores/visualization'

export interface PerfData {
  accuracy: number[]
  confidence: number[]
  chanceLevel: number[]
}

const props = defineProps<{
  modeName: string
  data?: PerfData
}>()

const viz = useVisualizationStore()
const container = ref<HTMLElement>()
let plot: uPlot | null = null
let resizeObserver: ResizeObserver | null = null

function getMetrics(): PerfData | null {
  if (props.data) return props.data
  const m = viz.modeMetrics[props.modeName]
  return m && m.accuracy.length > 0 ? m : null
}

const hasData = computed(() => !!getMetrics())

function createPlot() {
  if (!container.value) return
  const el = container.value

  const opts: uPlot.Options = {
    width: el.clientWidth,
    height: el.clientHeight,
    cursor: { show: false },
    legend: LEGEND_HIDDEN,
    series: [
      {},
      { stroke: '#34d399', width: 2, label: 'Accuracy', fill: '#34d39918' },
      { stroke: '#fbbf24', width: 2, label: 'Confidence', fill: '#fbbf2418' },
      { stroke: '#8A8A8A', width: 1, dash: [5, 3], label: 'Chance' },
    ],
    axes: [
      { stroke: '#666', grid: { stroke: '#1e1e1e', width: 1 }, size: 28, font: '10px Inter, sans-serif', label: 'Trial' },
      { stroke: '#666', grid: { stroke: '#1e1e1e', width: 1 }, size: 45, font: '10px Inter, sans-serif', values: (_, ticks) => ticks.map(v => (v * 100).toFixed(0) + '%') },
    ],
    scales: {
      x: { time: false },
      y: { range: [0, 1.05] },
    },
  }

  plot = new uPlot(opts, [[0], [0], [0], [0]], el)
}

function updatePlot() {
  if (!plot) {
    createPlot()
    if (!plot) return
  }
  const metrics = getMetrics()
  if (!metrics) return

  const len = metrics.accuracy.length
  const xAxis = new Array(len)
  for (let i = 0; i < len; i++) xAxis[i] = i + 1
  plot.setData([
    xAxis,
    metrics.accuracy,
    metrics.confidence,
    metrics.chanceLevel.length === len ? metrics.chanceLevel : new Array(len).fill(0.5),
  ])
}

// Reactive: store or props change
watch(() => viz.modeMetrics, () => {
  if (hasData.value) nextTick(updatePlot)
})
watch(() => props.data, () => {
  if (hasData.value) nextTick(updatePlot)
})

onMounted(() => {
  createPlot()
  if (hasData.value) updatePlot()
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
    <div class="flex items-center justify-between mb-1">
      <span class="text-xs text-text-muted uppercase tracking-wide">Performance</span>
      <div class="flex items-center gap-3">
        <span class="flex items-center gap-1 text-xs text-text-muted">
          <span class="inline-block w-2.5 h-0.5 rounded-full bg-[#34d399]" />Accuracy
        </span>
        <span class="flex items-center gap-1 text-xs text-text-muted">
          <span class="inline-block w-2.5 h-0.5 rounded-full bg-[#fbbf24]" />Confidence
        </span>
        <span class="flex items-center gap-1 text-xs text-text-disabled">
          <span class="inline-block w-2.5 h-0.5 rounded-full bg-[#8A8A8A] opacity-60" style="border-top: 1px dashed #8A8A8A" />Chance
        </span>
      </div>
    </div>
    <div ref="container" class="h-[180px] bg-bg-panel rounded border border-border/30" />
  </div>
</template>
