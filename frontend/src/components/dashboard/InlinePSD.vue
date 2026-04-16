<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_HIDDEN, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { useUPlot } from '../../composables/useUPlot'
import { getModalityColor } from '../../utils/colors'
import { useVisualizationStore, vizDirty } from '../../stores/visualization'

const props = defineProps<{ modality: string }>()

const viz = useVisualizationStore()
const container = ref<HTMLDivElement | null>(null)
const { create, setData } = useUPlot(container)

const color = getModalityColor(props.modality)

function createPlot() {
  create(({ width, height }) => ({
    width,
    height,
    cursor: CURSOR_HIDDEN,
    legend: LEGEND_HIDDEN,
    series: [
      {},
      { stroke: color, width: 2, label: 'PSD', fill: color + '18' },
    ],
    axes: [
      makeAxis({ size: 20 }),
      makeAxis({ size: 32 }),
    ],
    scales: {
      x: { time: false },
      y: {
        range: (_u: uPlot, dataMin: number, dataMax: number) => {
          // Ensure minimum 30dB span, pad by 5dB, round to 10dB grid
          const MIN_SPAN = 30
          const pad = 5
          let lo = dataMin - pad
          let hi = dataMax + pad
          const span = hi - lo
          if (span < MIN_SPAN) {
            const mid = (lo + hi) / 2
            lo = mid - MIN_SPAN / 2
            hi = mid + MIN_SPAN / 2
          }
          lo = Math.floor(lo / 10) * 10
          hi = Math.ceil(hi / 10) * 10
          return [lo, hi] as uPlot.Range.MinMax
        },
      },
    },
  }), [[0], [0]])
}

function updatePlot() {
  const d = viz.psdData[props.modality]
  if (!d || d.freqs.length === 0) return
  setData([d.freqs, d.power])
}

let lastPsdVersion = 0
let pollTimer: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  createPlot()
  updatePlot()
  // Poll dirty flag instead of watching reactive ref (avoids 1Hz re-render flicker)
  pollTimer = setInterval(() => {
    const v = vizDirty.psdVersion || 0
    if (v !== lastPsdVersion) {
      lastPsdVersion = v
      updatePlot()
    }
  }, 500)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <div class="h-full flex rounded overflow-hidden border border-border/30">
    <div class="w-[3px] shrink-0" :style="{ backgroundColor: color }" />
    <div ref="container" class="flex-1 bg-bg-panel" />
  </div>
</template>
