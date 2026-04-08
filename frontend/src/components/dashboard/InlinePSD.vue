<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_HIDDEN, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { getModalityColor } from '../../utils/colors'
import { useVisualizationStore, vizDirty } from '../../stores/visualization'

const props = defineProps<{ modality: string }>()

const viz = useVisualizationStore()
const container = ref<HTMLElement>()
let plot: uPlot | null = null
let resizeObserver: ResizeObserver | null = null

const color = getModalityColor(props.modality)

function createPlot() {
  if (!container.value) return
  const el = container.value

  const opts: uPlot.Options = {
    width: el.clientWidth,
    height: el.clientHeight,
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
  }

  plot = new uPlot(opts, [[0], [0]], el)
}

function updatePlot() {
  if (!plot) return
  const d = viz.psdData[props.modality]
  if (!d || d.freqs.length === 0) return
  plot.setData([d.freqs, d.power])
}

let lastPsdVersion = 0
let pollTimer: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  createPlot()
  updatePlot()
  if (container.value) {
    resizeObserver = new ResizeObserver(() => {
      if (plot && container.value) {
        plot.setSize({ width: container.value.clientWidth, height: container.value.clientHeight })
      }
    })
    resizeObserver.observe(container.value)
  }
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
  resizeObserver?.disconnect()
  plot?.destroy()
})
</script>

<template>
  <div class="h-full flex rounded overflow-hidden border border-border/30">
    <div class="w-[3px] shrink-0" :style="{ backgroundColor: color }" />
    <div ref="container" class="flex-1 bg-bg-panel" />
  </div>
</template>
