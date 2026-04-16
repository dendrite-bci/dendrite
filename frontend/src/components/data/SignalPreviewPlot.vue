<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_INTERACTIVE, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { useUPlot } from '../../composables/useUPlot'
import type { ModalitySignalPreview } from '../../types/api'
import { getModalityColor } from '../../utils/colors'

const props = defineProps<{
  modality: string
  preview: ModalitySignalPreview
}>()

const chartEl = ref<HTMLDivElement | null>(null)
const { create } = useUPlot(chartEl)

function chartHeight(n: number): number {
  return Math.max(200, n * 50 + 60)
}

function buildChart() {
  if (props.preview.channels.length === 0) return

  const channels = props.preview.channels
  const time = props.preview.time

  const stds = channels.map(ch => {
    const arr = ch.data
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length
    const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length
    return Math.sqrt(variance)
  })
  const medianStd = stds.slice().sort((a, b) => a - b)[Math.floor(stds.length / 2)] || 1
  const separation = medianStd * 4

  const data: uPlot.AlignedData = [
    new Float64Array(time),
    ...channels.map((ch, i) => {
      const offset = i * separation
      return new Float64Array(ch.data.map(v => v + offset))
    }),
  ]

  const baseColor = getModalityColor(props.modality.toLowerCase())

  const series: uPlot.Series[] = [
    { label: 'Time (s)' },
    ...channels.map((ch) => ({
      label: ch.label,
      stroke: baseColor,
      width: 1,
    })),
  ]

  create(({ width }) => ({
    width,
    height: chartHeight(channels.length),
    cursor: CURSOR_INTERACTIVE,
    legend: LEGEND_HIDDEN,
    series,
    axes: [
      makeAxis({ label: 'Time (s)', size: 30 }),
      makeAxis({ size: 50, values: () => [] }),
    ],
    scales: { x: { time: false } },
    hooks: {
      draw: [
        (u: uPlot) => {
          const ctx = u.ctx
          ctx.save()
          ctx.font = '10px Inter, sans-serif'
          ctx.textAlign = 'right'
          ctx.textBaseline = 'middle'
          const yScale = u.scales.y
          if (!yScale?.min && yScale?.min !== 0) { ctx.restore(); return }
          for (let i = 0; i < channels.length; i++) {
            const yVal = i * separation
            const yPos = u.valToPos(yVal, 'y')
            ctx.fillStyle = baseColor
            ctx.fillText(channels[i]!.label, u.bbox.left / devicePixelRatio - 4, yPos)
          }
          ctx.restore()
        },
      ],
    },
  }), data)
}

onMounted(() => nextTick(buildChart))
</script>

<template>
  <div>
    <div class="flex items-center justify-between px-1 mb-1">
      <span class="text-xs font-semibold text-text-label">{{ modality }}
        <span class="text-text-muted font-normal ml-1">{{ preview.channels.length }} ch &middot; {{ preview.sample_rate }} Hz</span>
      </span>
      <span class="text-xs text-text-disabled">
        {{ preview.display_samples.toLocaleString() }} / {{ preview.total_samples.toLocaleString() }} samples
      </span>
    </div>
    <div
      ref="chartEl"
      class="bg-bg-elevated rounded-lg p-1"
      :style="{ minHeight: chartHeight(preview.channels.length) + 'px' }"
    />
  </div>
</template>
