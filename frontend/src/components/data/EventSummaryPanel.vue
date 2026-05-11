<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { makeAxis, CURSOR_INTERACTIVE, LEGEND_HIDDEN, getMutedStroke } from '../../utils/chartDefaults'
import { useUPlot } from '../../composables/useUPlot'
import type { EventSummary } from '../../types/api'
import { CHART_COLORS } from '../../utils/colors'

const props = defineProps<{ summary: EventSummary }>()

const sortedTypes = computed(() =>
  Object.entries(props.summary.event_types).sort((a, b) => b[1] - a[1])
)

const maxCount = computed(() => sortedTypes.value[0]?.[1] ?? 1)

const colorMap = computed(() => {
  const m: Record<string, string> = {}
  sortedTypes.value.forEach(([type], i) => { m[type] = CHART_COLORS[i % CHART_COLORS.length]! })
  return m
})

/** Format event type label: "name (id)" when id is available */
function eventLabel(type: string): string {
  const id = props.summary.event_ids?.[type]
  return id != null ? `${type} (${id})` : type
}

// --- Timeline chart ---
const chartEl = ref<HTMLDivElement | null>(null)
const { create } = useUPlot(chartEl)

const showTable = ref(false)

const eventColumns = computed(() => {
  if (props.summary.events.length === 0) return []
  return Object.keys(props.summary.events[0]!)
})

const chartHeight = computed(() => Math.max(60, sortedTypes.value.length * 22 + 40))

function buildTimeline() {
  const events = props.summary.events
  if (events.length === 0) return

  const types = sortedTypes.value.map(([t]) => t)
  const typeIdx: Record<string, number> = {}
  types.forEach((t, i) => { typeIdx[t] = i })

  const typeTimes: Map<string, Set<number>> = new Map()
  for (const t of types) typeTimes.set(t, new Set())

  for (const evt of events) {
    const t = String(evt['event_type'] ?? '')
    const ts = Number(evt['timestamp'] ?? 0)
    if (!typeTimes.has(t) || isNaN(ts)) continue
    typeTimes.get(t)!.add(ts)
  }

  const allX = events.map(e => Number(e['timestamp'] ?? 0)).filter(v => !isNaN(v)).sort((a, b) => a - b)
  if (allX.length === 0) return

  const uSeries: uPlot.Series[] = [{}]
  const data: (number | null)[][] = [allX]

  for (const t of types) {
    const timeSet = typeTimes.get(t)!
    const yLevel = typeIdx[t]!
    data.push(allX.map(x => timeSet.has(x) ? yLevel : null))
    uSeries.push({
      label: t,
      stroke: colorMap.value[t],
      fill: colorMap.value[t],
      width: 0,
      paths: () => null,
      points: {
        show: true,
        size: 6,
        fill: colorMap.value[t],
        stroke: colorMap.value[t],
      },
    })
  }

  create(({ width }) => ({
    width,
    height: chartHeight.value,
    cursor: CURSOR_INTERACTIVE,
    legend: LEGEND_HIDDEN,
    series: uSeries,
    axes: [
      makeAxis({ label: 'Time (s)', size: 28 }),
      { show: false },
    ],
    scales: {
      x: { time: false },
      y: { range: [-0.5, types.length - 0.5] },
    },
    hooks: {
      draw: [
        (u: uPlot) => {
          const ctx = u.ctx
          ctx.save()
          ctx.font = '10px Inter, sans-serif'
          ctx.textAlign = 'right'
          ctx.textBaseline = 'middle'
          for (let i = 0; i < types.length; i++) {
            const yPos = u.valToPos(i, 'y')
            ctx.fillStyle = colorMap.value[types[i]!] ?? getMutedStroke()
            ctx.fillText(eventLabel(types[i]!), u.bbox.left / devicePixelRatio - 4, yPos)
          }
          ctx.restore()
        },
      ],
    },
  }), data as uPlot.AlignedData)
}

onMounted(() => nextTick(buildTimeline))
watch(() => props.summary, () => nextTick(buildTimeline))
</script>

<template>
  <div class="space-y-3">
    <!-- Full-width timeline chart (aligned with signal plots) -->
    <div
      ref="chartEl"
      class="bg-bg-elevated rounded-lg"
      :style="{ minHeight: chartHeight + 'px' }"
    />

    <!-- Compact distribution summary -->
    <div class="flex flex-wrap gap-x-4 gap-y-1 px-1">
      <span
        v-for="([type, count], i) in sortedTypes"
        :key="type"
        class="inline-flex items-center gap-1.5 text-xs"
      >
        <span
          class="w-2 h-2 rounded-full shrink-0"
          :style="{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }"
        />
        <span class="text-text-main">{{ eventLabel(type) }}</span>
        <span class="text-text-disabled tabular-nums">{{ count }}</span>
        <span class="w-12 h-2 bg-bg-input rounded-sm overflow-hidden shrink-0">
          <span
            class="block h-full rounded-sm"
            :style="{ width: `${(count / maxCount) * 100}%`, backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }"
          />
        </span>
      </span>
    </div>

    <!-- Expandable table -->
    <div v-if="summary.events.length > 0">
      <button
        @click="showTable = !showTable"
        class="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-main transition-colors"
      >
        <i class="pi text-xs" :class="showTable ? 'pi-chevron-down' : 'pi-chevron-right'" />
        Event log ({{ summary.total_count }})
      </button>
      <div v-if="showTable" class="max-h-48 overflow-auto mt-2 rounded border border-border">
        <table class="w-full text-xs">
          <thead>
            <tr class="text-text-muted border-b border-border">
              <th
                v-for="key in eventColumns"
                :key="key"
                class="text-left px-2 py-1 font-medium sticky top-0 bg-bg-elevated"
              >{{ key }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(evt, i) in summary.events" :key="i" class="border-b border-border/30">
              <td v-for="key in eventColumns" :key="key" class="px-2 py-1 text-text-main">
                {{ evt[key] }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
