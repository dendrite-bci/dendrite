<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, getMutedStroke } from '../../utils/chartDefaults'
import { useUPlot } from '../../composables/useUPlot'
import { useDataStore } from '../../stores/data'
import { CHART_COLORS } from '../../utils/colors'

const props = defineProps<{
  recordingId: number
  isLive?: boolean
  lowcut?: number
  highcut?: number
  applyRereferencing?: boolean
  epochTmin?: number
  epochTmax?: number
}>()

const data = useDataStore()
const MAX_CHANNELS = 8

// State
const loading = ref(false)
const error = ref('')
const selectedEvent = ref<string | null>(null)
const channelPage = ref(0)

// Chart
const chartEl = ref<HTMLDivElement | null>(null)
const { create } = useUPlot(chartEl)

const erp = computed(() => data.erpPreview)
const eventNames = computed(() => erp.value ? Object.keys(erp.value.erp_by_event) : [])
const currentErp = computed(() => {
  if (!erp.value || !selectedEvent.value) return null
  return erp.value.erp_by_event[selectedEvent.value]
})
const totalChannels = computed(() => currentErp.value?.labels.length ?? 0)
const maxPage = computed(() => Math.max(0, Math.ceil(totalChannels.value / MAX_CHANNELS) - 1))
const visibleLabels = computed(() => {
  if (!currentErp.value) return []
  const start = channelPage.value * MAX_CHANNELS
  return currentErp.value.labels.slice(start, start + MAX_CHANNELS)
})

async function fetchERP() {
  loading.value = true
  error.value = ''
  try {
    await data.fetchErpPreview(props.recordingId, {
      epoch_tmin: props.epochTmin ?? -0.2,
      epoch_tmax: props.epochTmax ?? 0.8,
      lowcut: props.lowcut ?? 0.5,
      highcut: props.highcut ?? 30,
      apply_rereferencing: props.applyRereferencing ?? false,
    })
    if (erp.value && eventNames.value.length > 0 && !selectedEvent.value) {
      selectedEvent.value = eventNames.value[0] ?? null
    }
    await nextTick()
    buildChart()
  } catch (e: any) {
    error.value = e.message || 'Failed to load ERP'
  } finally {
    loading.value = false
  }
}

function buildChart() {
  if (!erp.value || !currentErp.value) return

  const timeAxis = erp.value.time_axis
  const start = channelPage.value * MAX_CHANNELS
  const channels: number[][] = currentErp.value.channels.slice(start, start + MAX_CHANNELS)
  const labels = visibleLabels.value

  const series: uPlot.Series[] = [{ label: 'Time (s)' }]
  labels.forEach((label: string, i: number) => {
    series.push({
      label,
      stroke: CHART_COLORS[i % CHART_COLORS.length],
      width: 1.5,
    })
  })

  const uData: uPlot.AlignedData = [
    new Float64Array(timeAxis),
    ...channels.map(ch => new Float64Array(ch)),
  ]

  create(({ width }) => ({
    width,
    height: 300,
    cursor: { show: true, drag: { x: true, y: false } },
    scales: { x: { time: false }, y: { auto: true } },
    axes: [
      makeAxis({ label: 'Time (s)', size: 40 }),
      makeAxis({ label: 'Amplitude', size: 55 }),
    ],
    series,
    plugins: [zeroLinePlugin()],
  }), uData)
}

function zeroLinePlugin(): uPlot.Plugin {
  return {
    hooks: {
      draw: [
        (u: uPlot) => {
          const ctx = u.ctx
          const x0 = u.valToPos(0, 'x', true)
          if (x0 >= u.bbox.left && x0 <= u.bbox.left + u.bbox.width) {
            ctx.save()
            ctx.globalAlpha = 0.4
            ctx.strokeStyle = getMutedStroke()
            ctx.lineWidth = 1
            ctx.setLineDash([4, 4])
            ctx.beginPath()
            ctx.moveTo(x0, u.bbox.top)
            ctx.lineTo(x0, u.bbox.top + u.bbox.height)
            ctx.stroke()
            ctx.restore()
          }
        },
      ],
    },
  }
}

watch(selectedEvent, () => { channelPage.value = 0; buildChart() })
watch(channelPage, () => buildChart())

onMounted(() => { fetchERP() })
</script>

<template>
  <div class="space-y-3">
    <span class="text-xs font-semibold text-text-label uppercase tracking-wider">ERP</span>

    <!-- Event selector + stats -->
    <div v-if="eventNames.length > 0" class="flex items-center gap-2 flex-wrap">
      <button
        v-for="name in eventNames"
        :key="name"
        @click="selectedEvent = name"
        class="px-2.5 py-1 text-xs rounded transition-colors"
        :class="selectedEvent === name
          ? 'bg-accent text-white'
          : 'bg-bg-elevated text-text-muted hover:text-text-main'"
      >
        {{ name }}
        <span class="ml-1 opacity-60">({{ erp?.event_counts[name] ?? 0 }})</span>
      </button>
      <span class="text-xs text-text-disabled ml-2">{{ erp?.n_epochs ?? 0 }} total epochs</span>
    </div>

    <!-- Channel pager -->
    <div v-if="totalChannels > MAX_CHANNELS" class="flex items-center gap-2">
      <button @click="channelPage = Math.max(0, channelPage - 1)" :disabled="channelPage === 0"
        class="text-xs text-text-muted hover:text-text-main disabled:opacity-30">
        <i class="pi pi-chevron-left" />
      </button>
      <span class="text-xs text-text-muted">
        Ch {{ channelPage * MAX_CHANNELS + 1 }}-{{ Math.min((channelPage + 1) * MAX_CHANNELS, totalChannels) }}
        / {{ totalChannels }}
      </span>
      <button @click="channelPage = Math.min(maxPage, channelPage + 1)" :disabled="channelPage >= maxPage"
        class="text-xs text-text-muted hover:text-text-main disabled:opacity-30">
        <i class="pi pi-chevron-right" />
      </button>
    </div>

    <!-- Chart -->
    <div ref="chartEl" class="w-full" />

    <!-- Empty states -->
    <p v-if="!loading && erp && erp.n_epochs === 0" class="text-xs text-text-disabled text-center py-6">
      No events found in this recording. ERP requires event markers.
    </p>
    <p v-if="error" class="text-xs text-status-error text-center py-4">{{ error }}</p>
  </div>
</template>
