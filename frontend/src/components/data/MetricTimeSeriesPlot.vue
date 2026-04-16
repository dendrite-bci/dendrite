<script setup lang="ts">
import { ref, reactive, onMounted, nextTick, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_INTERACTIVE, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { useUPlot } from '../../composables/useUPlot'
import type { PlotSeries } from '../../utils/metrics'
import { CHART_COLORS } from '../../utils/colors'

const CHART_HEIGHT = 280

const props = defineProps<{
  title: string
  series: PlotSeries[]
}>()

const chartEl = ref<HTMLDivElement | null>(null)
const { create, getPlot } = useUPlot(chartEl)

const hidden = reactive(new Set<number>())

function seriesColor(i: number): string {
  return props.series[i]?.color ?? CHART_COLORS[i % CHART_COLORS.length] ?? ''
}

function toggleSeries(idx: number) {
  if (hidden.has(idx)) hidden.delete(idx)
  else hidden.add(idx)
  // uPlot series index is +1 because index 0 is the time axis
  getPlot()?.setSeries(idx + 1, { show: !hidden.has(idx) })
}

function buildChart() {
  if (props.series.length === 0) return

  const timeArr = new Float64Array(props.series[0]!.time)
  const data: uPlot.AlignedData = [
    timeArr,
    ...props.series.map(s => new Float64Array(s.values)),
  ]

  const uSeries: uPlot.Series[] = [
    { label: 'Time (s)' },
    ...props.series.map((s, i) => ({
      label: s.label,
      stroke: seriesColor(i),
      width: 1.5,
      show: !hidden.has(i),
    })),
  ]

  create(({ width }) => ({
    width,
    height: CHART_HEIGHT,
    cursor: CURSOR_INTERACTIVE,
    legend: LEGEND_HIDDEN,
    series: uSeries,
    axes: [
      makeAxis({ label: 'Time (s)', size: 30 }),
      makeAxis({ size: 50 }),
    ],
    scales: { x: { time: false } },
  }), data)
}

onMounted(() => nextTick(buildChart))
watch(() => props.series, () => {
  hidden.clear()
  nextTick(buildChart)
})
</script>

<template>
  <div class="space-y-1.5">
    <div class="flex items-center justify-between px-1">
      <span class="text-xs font-semibold text-text-label">{{ title }}</span>
    </div>

    <!-- Series toggle pills -->
    <div v-if="series.length > 1" class="flex flex-wrap gap-1.5 px-1">
      <button
        v-for="(s, i) in series"
        :key="s.label"
        @click="toggleSeries(i)"
        class="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs transition-all border"
        :class="hidden.has(i)
          ? 'opacity-35 border-border text-text-disabled'
          : 'border-transparent text-text-main'"
        :style="!hidden.has(i) ? { backgroundColor: seriesColor(i) + '25' } : {}"
      >
        <span
          class="w-2 h-2 rounded-full shrink-0"
          :style="{ backgroundColor: hidden.has(i) ? '#666' : seriesColor(i) }"
        />
        {{ s.label }}
      </button>
    </div>

    <div
      ref="chartEl"
      class="bg-bg-elevated rounded-lg p-1"
      :style="{ minHeight: CHART_HEIGHT + 'px' }"
    />
  </div>
</template>
