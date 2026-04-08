<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  tmin: number
  tmax: number
}>()

const windowMs = computed(() => Math.round((props.tmax - props.tmin) * 1000))

// Layout: we show a timeline with the event marker at center,
// and the epoch window highlighted relative to it.
// Range spans from min(tmin, -0.5) to max(tmax, tmax + 0.5) for context.
const viewMin = computed(() => Math.min(props.tmin, -0.5) - 0.2)
const viewMax = computed(() => Math.max(props.tmax, 0.5) + 0.2)
const viewSpan = computed(() => viewMax.value - viewMin.value)

function toPercent(t: number): number {
  return ((t - viewMin.value) / viewSpan.value) * 100
}

const eventPos = computed(() => toPercent(0))
const windowLeft = computed(() => toPercent(props.tmin))
const windowRight = computed(() => toPercent(props.tmax))
const windowWidth = computed(() => windowRight.value - windowLeft.value)

// Tick marks at nice intervals
const ticks = computed(() => {
  const step = viewSpan.value > 4 ? 1 : viewSpan.value > 2 ? 0.5 : 0.25
  const result: number[] = []
  const start = Math.ceil(viewMin.value / step) * step
  for (let t = start; t <= viewMax.value; t += step) {
    result.push(Math.round(t * 100) / 100)
  }
  return result
})
</script>

<template>
  <div class="mt-3">
    <div class="flex items-center justify-between mb-1.5">
      <span class="text-xs text-text-disabled">Epoch window relative to event</span>
      <span class="text-xs text-text-muted font-mono">{{ windowMs }}ms</span>
    </div>
    <div class="relative h-10 bg-bg-input rounded border border-border overflow-hidden">
      <!-- Epoch window highlight -->
      <div
        class="absolute top-0 bottom-0 bg-accent/12 border-l border-r border-accent/40"
        :style="{ left: windowLeft + '%', width: windowWidth + '%' }"
      />

      <!-- Event marker (t=0) -->
      <div
        class="absolute top-0 bottom-0 w-px bg-status-warn"
        :style="{ left: eventPos + '%' }"
      />
      <div
        class="absolute top-0.5 text-[10px] font-semibold text-status-warn -translate-x-1/2"
        :style="{ left: eventPos + '%' }"
      >E</div>

      <!-- tmin label -->
      <div
        class="absolute bottom-0.5 text-[10px] text-accent font-mono -translate-x-1/2"
        :style="{ left: windowLeft + '%' }"
      >{{ tmin }}s</div>

      <!-- tmax label -->
      <div
        class="absolute bottom-0.5 text-[10px] text-accent font-mono -translate-x-1/2"
        :style="{ left: windowRight + '%' }"
      >{{ tmax }}s</div>

      <!-- Tick marks -->
      <div
        v-for="t in ticks" :key="t"
        class="absolute bottom-0 w-px h-1.5 bg-border"
        :style="{ left: toPercent(t) + '%' }"
      />
    </div>
  </div>
</template>
