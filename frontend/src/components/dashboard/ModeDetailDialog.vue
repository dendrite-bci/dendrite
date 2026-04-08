<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { predClassColor } from '../../utils/colors'
import { useVisualizationStore } from '../../stores/visualization'
import PerformancePlot from './PerformancePlot.vue'
import AsyncPredictionPlot from './AsyncPredictionPlot.vue'
import ERPPlot from './ERPPlot.vue'
import BandPowerPlot from './BandPowerPlot.vue'

const props = defineProps<{
  modeName: string | null
}>()
const emit = defineEmits<{
  close: []
}>()

const viz = useVisualizationStore()

const isAsync = computed(() => {
  if (!props.modeName) return false
  return viz.modeTypes[props.modeName] === 'asynchronous'
})

const latest = computed(() => {
  if (!props.modeName) return null

  const m = viz.modeMetrics[props.modeName]
  if (!m || m.accuracy.length === 0) return null
  return {
    accuracy: m.accuracy[m.accuracy.length - 1],
    confidence: m.confidence.length ? m.confidence[m.confidence.length - 1] : null,
    kappa: m.kappa.length ? m.kappa[m.kappa.length - 1] : null,
    trials: m.accuracy.length,
  }
})

const prediction = computed(() => {
  if (!props.modeName) return null

  return viz.modePredictions[props.modeName] ?? null
})

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}
onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <Teleport to="body">
    <div
      v-if="modeName"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      @click.self="emit('close')"
    >
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl w-[90vw] max-w-[1200px] max-h-[85vh] flex flex-col">
        <!-- Header -->
        <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <div class="flex items-center gap-4">
            <h2 class="text-base font-semibold text-text-main">{{ modeName }}</h2>
            <span v-if="prediction"
              class="text-xs font-mono font-semibold px-2 py-0.5 rounded"
              :style="{ color: predClassColor(prediction.eventName), backgroundColor: predClassColor(prediction.eventName) + '20' }">
              {{ prediction.eventName }} {{ (prediction.confidence * 100).toFixed(0) }}%
            </span>
          </div>
          <div class="flex items-center gap-4">
            <!-- Stats -->
            <div class="flex items-center gap-5 text-xs">
              <span v-if="latest?.accuracy != null" class="text-text-muted">
                Acc <span class="font-mono font-semibold text-level-ok">{{ (latest.accuracy * 100).toFixed(0) }}%</span>
              </span>
              <span v-if="latest?.confidence != null" class="text-text-muted">
                Conf <span class="font-mono text-text-label">{{ (latest.confidence * 100).toFixed(0) }}%</span>
              </span>
              <span v-if="latest?.kappa != null" class="text-text-muted">
                K <span class="font-mono text-text-label">{{ latest.kappa.toFixed(2) }}</span>
              </span>
              <span v-if="latest?.trials" class="text-text-disabled">
                {{ latest.trials }} trials
              </span>
            </div>
            <button
              @click="emit('close')"
              class="w-7 h-7 rounded flex items-center justify-center text-text-muted hover:text-text-main hover:bg-bg-elevated transition-colors"
            >
              <i class="pi pi-times text-sm" />
            </button>
          </div>
        </div>

        <!-- Content -->
        <div class="flex-1 overflow-y-auto p-6 space-y-5">
          <AsyncPredictionPlot v-if="isAsync" :mode-name="modeName" />
          <PerformancePlot v-else :mode-name="modeName" />
          <ERPPlot :mode-name="modeName" detail />
          <BandPowerPlot :mode-name="modeName" />
        </div>
      </div>
    </div>
  </Teleport>
</template>
