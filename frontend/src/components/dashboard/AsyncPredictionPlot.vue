<script setup lang="ts">
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { predClassColor } from '../../utils/colors'
import { useVisualizationStore } from '../../stores/visualization'
import { useUPlot } from '../../composables/useUPlot'
import { makeAxis, LEGEND_HIDDEN } from '../../utils/chartDefaults'

const props = defineProps<{
  modeName: string
}>()

const viz = useVisualizationStore()
const container = ref<HTMLDivElement | null>(null)
const { create, setData, getPlot } = useUPlot(container)

const hasData = computed(() => {
  const h = viz.modePredictionHistory[props.modeName]
  return !!h && h.length > 0
})

// Discover unique class names from history
const classNames = computed(() => {
  const h = viz.modePredictionHistory[props.modeName]
  if (!h) return []
  const seen = new Set<string>()
  for (const p of h) seen.add(p.eventName)
  return Array.from(seen).sort()
})

function classColor(className: string): string {
  return predClassColor(className)
}

function createPlot() {
  // Draw plugin: detection markers as vertical lines
  const drawPlugin: uPlot.Plugin = {
    hooks: {
      draw: [
        (u: uPlot) => {
          const hist = viz.modePredictionHistory[props.modeName]
          if (!hist || hist.length === 0) return
          const ctx = u.ctx
          const { left, top, width: plotW, height: plotH } = u.bbox

          ctx.save()
          ctx.beginPath()
          ctx.rect(left, top, plotW, plotH)
          ctx.clip()

          ctx.lineWidth = 2
          for (let i = 0; i < hist.length; i++) {
            if (!hist[i]!.detected) continue
            const x = u.valToPos(i + 1, 'x', true)
            if (x < left || x > left + plotW) continue
            const color = predClassColor(hist[i]!.eventName)
            ctx.strokeStyle = color + 'B3' // 70% opacity
            ctx.beginPath()
            ctx.moveTo(x, top)
            ctx.lineTo(x, top + plotH)
            ctx.stroke()
          }
          ctx.restore()
        },
      ],
    },
  }

  create(({ width, height }) => ({
    width,
    height,
    cursor: { show: false },
    legend: LEGEND_HIDDEN,
    series: [
      {},
      {
        stroke: '#34d399',
        width: 2,
        label: 'Confidence',
        fill: '#34d39918',
        points: { show: false },
      },
    ],
    axes: [
      makeAxis({ label: '', size: 28 }),
      makeAxis({ size: 45, values: (_, ticks) => ticks.map(v => (v * 100).toFixed(0) + '%') }),
    ],
    scales: {
      x: { time: false },
      y: { range: [0, 1.05] },
    },
    plugins: [drawPlugin],
  }), [[0], [0]])
}

function updatePlot() {
  let plot = getPlot()
  if (!plot) {
    createPlot()
    plot = getPlot()
    if (!plot) return
  }
  const hist = viz.modePredictionHistory[props.modeName]
  if (!hist || hist.length === 0) return

  const len = hist.length
  const latest = hist[len - 1]!
  const color = classColor(latest.eventName)
  plot.series[1]!.stroke = () => color
  plot.series[1]!.fill = () => color + '18'

  const xAxis = new Array(len)
  const conf = new Array(len)
  for (let i = 0; i < len; i++) { xAxis[i] = i + 1; conf[i] = hist[i]!.confidence }
  setData([xAxis, conf])
}

// Reactive: store replaces modePredictionHistory.value with a new Record on change
watch(() => viz.modePredictionHistory, () => {
  if (hasData.value) nextTick(updatePlot)
})

onMounted(() => {
  createPlot()
  if (hasData.value) updatePlot()
})
</script>

<template>
  <div v-if="hasData">
    <div class="text-xs text-text-muted uppercase tracking-wide mb-1">Predictions</div>
    <!-- Confidence chart -->
    <div ref="container" class="h-[140px] bg-bg-panel rounded border border-border/30" />
    <!-- Class raster -->
    <div class="mt-1 flex gap-px h-5 rounded overflow-hidden border border-border/30">
      <div
        v-for="(p, i) in viz.modePredictionHistory[modeName]?.slice(-100) ?? []"
        :key="i"
        class="flex-1 min-w-0"
        :style="{
          backgroundColor: classColor(p.eventName),
          opacity: 0.3 + p.confidence * 0.7,
          borderTop: p.detected ? '2px solid white' : 'none',
        }"
        :title="`${p.eventName} (${(p.confidence * 100).toFixed(0)}%)${p.detected ? ' ✓ detected' : ''}`"
      />
    </div>
    <!-- Legend -->
    <div class="mt-1 flex gap-3 flex-wrap">
      <span
        v-for="cn in classNames"
        :key="cn"
        class="flex items-center gap-1 text-xs text-text-muted"
      >
        <span class="w-2 h-2 rounded-full" :style="{ backgroundColor: classColor(cn) }" />
        {{ cn }}
      </span>
    </div>
  </div>
</template>
