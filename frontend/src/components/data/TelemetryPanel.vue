<script setup lang="ts">
import { computed } from 'vue'
import type { RecordingTelemetry } from '../../types/api'
import MetricTimeSeriesPlot from './MetricTimeSeriesPlot.vue'
import { metricsToPlotSeries } from '../../utils/metrics'

const props = defineProps<{
  telemetry: RecordingTelemetry
}>()

const hasLatencies = computed(() => Object.keys(props.telemetry.latencies).length > 0)
const hasModeMetrics = computed(() => Object.keys(props.telemetry.mode_metrics).length > 0)
const hasBandwidth = computed(() => Object.keys(props.telemetry.bandwidth).length > 0)
const hasAny = computed(() => hasLatencies.value || hasModeMetrics.value || hasBandwidth.value)
</script>

<template>
  <div v-if="hasAny" class="space-y-4">
    <MetricTimeSeriesPlot
      v-if="hasLatencies"
      title="Latencies"
      :series="metricsToPlotSeries(telemetry.latencies)"
    />
    <MetricTimeSeriesPlot
      v-if="hasModeMetrics"
      title="Mode Metrics"
      :series="metricsToPlotSeries(telemetry.mode_metrics)"
    />
    <MetricTimeSeriesPlot
      v-if="hasBandwidth"
      title="Bandwidth"
      :series="metricsToPlotSeries(telemetry.bandwidth)"
    />
  </div>
  <p v-else class="text-xs text-text-disabled text-center py-6">
    No telemetry data in this recording.
  </p>
</template>
