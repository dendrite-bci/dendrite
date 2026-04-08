<script setup lang="ts">
import { computed } from 'vue'
import type { ModePerformance } from '../../types/api'
import type { MetricSeries } from '../../types/api'
import type { PerfData } from '../dashboard/PerformancePlot.vue'
import PerformancePlot from '../dashboard/PerformancePlot.vue'
import MetricTimeSeriesPlot from './MetricTimeSeriesPlot.vue'
import { metricsToPlotSeries } from '../../utils/metrics'

const props = defineProps<{
  performance: ModePerformance
}>()

const PERF_KEYS = new Set(['accuracy', 'balanced_accuracy', 'confidence', 'chance_level'])

const modeNames = computed(() => Object.keys(props.performance))

const perMode = computed(() => {
  const out: Record<string, { perf: PerfData | null; rest: ReturnType<typeof metricsToPlotSeries> }> = {}
  for (const mode of modeNames.value) {
    const metrics = props.performance[mode] ?? {}
    const acc = metrics['accuracy'] ?? metrics['balanced_accuracy']
    const perf = acc && acc.values.length > 0
      ? {
          accuracy: Array.from(acc.values),
          confidence: metrics['confidence'] ? Array.from(metrics['confidence'].values) : [],
          chanceLevel: metrics['chance_level'] ? Array.from(metrics['chance_level'].values) : new Array(acc.values.length).fill(0.5),
        }
      : null
    const remaining: Record<string, MetricSeries> = {}
    for (const [k, v] of Object.entries(metrics)) {
      if (!PERF_KEYS.has(k)) remaining[k] = v
    }
    out[mode] = { perf, rest: metricsToPlotSeries(remaining) }
  }
  return out
})
</script>

<template>
  <div v-if="modeNames.length > 0" class="space-y-4">
    <div v-for="mode in modeNames" :key="mode">
      <span class="text-xs font-semibold text-text-label block mb-1.5">{{ mode }}</span>

      <!-- Accuracy/confidence plot (dashboard style) -->
      <PerformancePlot
        v-if="perMode[mode]?.perf"
        :modeName="mode"
        :data="perMode[mode]!.perf!"
      />

      <!-- Any remaining metrics (inference_ms, gpu_mb, etc.) -->
      <MetricTimeSeriesPlot
        v-if="perMode[mode]?.rest.length"
        :title="''"
        :series="perMode[mode]!.rest"
      />
    </div>
  </div>
  <p v-else class="text-xs text-text-disabled text-center py-6">
    No mode performance data in this recording.
  </p>
</template>
