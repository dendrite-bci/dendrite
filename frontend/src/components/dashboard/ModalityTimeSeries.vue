<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { CURSOR_HIDDEN, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { getModalityColor } from '../../utils/colors'
import { useVisualizationStore, vizDirty } from '../../stores/visualization'

const props = defineProps<{
  modality: string
}>()

const viz = useVisualizationStore()
const wrapperEl = ref<HTMLElement | null>(null)
const containers = ref<HTMLElement[]>([])
const plots: uPlot[] = []
let animFrame = 0
let lastDataVersion = 0
let resizeObs: ResizeObserver | null = null

const lineColor = getModalityColor(props.modality)
const channelHeight = 50

function buildPlot(el: HTMLElement): uPlot {
  const opts: uPlot.Options = {
    width: el.clientWidth,
    height: channelHeight,
    cursor: CURSOR_HIDDEN,
    legend: LEGEND_HIDDEN,
    series: [
      {},
      { stroke: lineColor, width: 1 },
    ],
    axes: [
      { show: false, size: 0 },
      { show: false, size: 0 },
    ],
    scales: { x: { time: false } },
  }
  return new uPlot(opts, [[0], [0]], el)
}

function createPlots() {
  plots.forEach(p => p.destroy())
  plots.length = 0
  const modData = viz.modalityBuffers[props.modality]
  if (!modData) return
  const els = containers.value
  for (let i = 0; i < modData.buffers.length; i++) {
    if (!els[i]) continue
    plots.push(buildPlot(els[i]!))
  }
}

function render() {
  if (vizDirty.dataVersion === lastDataVersion || !viz.initialized) {
    animFrame = requestAnimationFrame(render)
    return
  }

  const modData = viz.modalityBuffers[props.modality]
  if (!modData) {
    animFrame = requestAnimationFrame(render)
    return
  }

  const ta = viz.timeAxis
  for (let i = 0; i < plots.length; i++) {
    if (i >= modData.buffers.length) continue
    const data = modData.buffers[i]!.toArray()
    const t = ta.length === data.length ? ta : ta.slice(ta.length - data.length)
    plots[i]!.setData([t, data])
  }

  lastDataVersion = vizDirty.dataVersion
  animFrame = requestAnimationFrame(render)
}

function setRef(el: any, i: number) {
  if (el) containers.value[i] = el as HTMLElement
}

watch(() => viz.initialized, (v) => { if (v) nextTick(createPlots) })
watch(() => viz.modalityBuffers[props.modality]?.buffers.length, () => nextTick(createPlots))

function resizePlots() {
  for (let i = 0; i < plots.length; i++) {
    const el = containers.value[i]
    if (el) plots[i]!.setSize({ width: el.clientWidth, height: channelHeight })
  }
}

onMounted(() => {
  if (viz.initialized) nextTick(createPlots)
  animFrame = requestAnimationFrame(render)
  if (wrapperEl.value) {
    resizeObs = new ResizeObserver(() => resizePlots())
    resizeObs.observe(wrapperEl.value)
  }
})

onUnmounted(() => {
  cancelAnimationFrame(animFrame)
  resizeObs?.disconnect()
  plots.forEach(p => p.destroy())
})
</script>

<template>
  <div v-if="viz.modalityBuffers[modality]" ref="wrapperEl" class="flex flex-col">
    <div
      class="grid grid-cols-1 gap-px"
      :style="{ gridAutoRows: channelHeight + 'px' }"
    >
      <div
        v-for="(label, i) in viz.modalityBuffers[modality]!.labels"
        :key="i"
        class="relative min-h-0"
      >
        <div class="absolute top-0.5 left-1 z-10 flex items-center gap-1">
          <span
            class="w-1.5 h-1.5 rounded-full shrink-0"
            :style="{ backgroundColor: lineColor }"
          />
          <span class="text-xs text-text-disabled font-mono leading-none">
            {{ label }}
          </span>
        </div>
        <div
          :ref="(el) => setRef(el, i)"
          class="w-full h-full bg-bg-panel"
        />
      </div>
    </div>
  </div>
</template>
