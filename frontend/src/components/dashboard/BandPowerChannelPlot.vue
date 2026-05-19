<script setup lang="ts">
import { ref, watch, onMounted, nextTick } from 'vue'
import 'uplot/dist/uPlot.min.css'
import { LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { getBandColor } from '../../utils/colors'
import { useUPlot } from '../../composables/useUPlot'
import { useVisualizationStore } from '../../stores/visualization'

const props = defineProps<{
  modeName: string
  channelName: string
  bandNames: string[]
}>()

const viz = useVisualizationStore()
const container = ref<HTMLDivElement | null>(null)
const { create, setData, getPlot } = useUPlot(container)
const currentValues = ref<number[]>([])

function formatValue(v: number | undefined): string {
  if (v === undefined || !isFinite(v)) return '—'
  return v.toPrecision(2)
}

function createPlot() {
  create(({ width, height }) => ({
    width,
    height,
    cursor: { show: false },
    legend: LEGEND_HIDDEN,
    series: [
      {},
      ...props.bandNames.map(b => ({
        stroke: getBandColor(b),
        width: 1.5,
      })),
    ],
    axes: [{ show: false }, { show: false }],
    scales: { x: { time: false }, y: { auto: true } },
    padding: [4, 2, 2, 2],
  }), [[0], ...props.bandNames.map(() => [0])])
}

function updatePlot() {
  const hist = viz.modeBandPowerHistory[props.modeName]
  const chBufs = hist?.channels[props.channelName]
  if (!hist || !chBufs) return
  if (!getPlot()) createPlot()

  const yData = props.bandNames.map(b => chBufs[b]?.toArray() ?? [])
  const len = yData[0]?.length ?? 0
  if (len === 0) return

  const step = hist.stepSec
  const x = new Array(len)
  for (let i = 0; i < len; i++) x[i] = -(len - 1 - i) * step

  setData([x, ...yData])
  currentValues.value = yData.map(arr => arr[arr.length - 1] ?? 0)
}

watch(() => viz.modeBandPowerHistory, () => nextTick(updatePlot))

onMounted(() => {
  createPlot()
  updatePlot()
})
</script>

<template>
  <div class="flex items-stretch gap-1">
    <div class="relative flex-1">
      <div ref="container" class="h-[60px]" />
      <span class="absolute top-0.5 left-1 text-[10px] text-text-disabled font-mono pointer-events-none">
        {{ channelName }}
      </span>
    </div>
    <div class="w-12 flex flex-col justify-center text-[10px] font-mono leading-tight">
      <div v-for="(b, i) in bandNames" :key="b" :style="{ color: getBandColor(b) }">
        {{ formatValue(currentValues[i]) }}
      </div>
    </div>
  </div>
</template>
