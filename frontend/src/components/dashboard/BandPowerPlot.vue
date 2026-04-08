<script setup lang="ts">
import { computed } from 'vue'
import { useVisualizationStore } from '../../stores/visualization'

const props = defineProps<{ modeName: string }>()
const viz = useVisualizationStore()

const BAND_COLORS: Record<string, string> = {
  delta: '#8B5CF6', theta: '#3B82F6', alpha: '#10B981',
  smr: '#14B8A6', beta: '#F59E0B', gamma: '#EF4444',
  default: '#6B7280',
}

const data = computed(() => viz.modeBandPowers[props.modeName])
const hasData = computed(() => !!data.value?.channelPowers)
const iaf = computed(() => viz.modeIAF[props.modeName])

const channels = computed(() => {
  if (!data.value) return []
  return Object.entries(data.value.channelPowers).map(([ch, bands]) => ({
    name: ch,
    bands: Object.entries(bands as Record<string, number>).map(([band, power]) => ({
      band, power: power as number,
      color: BAND_COLORS[band.toLowerCase()] || BAND_COLORS.default,
    })),
  }))
})

// Scale: find max power across all channels/bands for normalization
const maxPower = computed(() => {
  if (!channels.value.length) return 1
  let max = 0
  for (const ch of channels.value) {
    for (const b of ch.bands) {
      if (b.power > max) max = b.power
    }
  }
  return max || 1
})
</script>

<template>
  <div v-if="hasData">
    <div class="flex items-center gap-2 mb-1">
      <span class="text-xs text-text-muted uppercase tracking-wide">Band Power</span>
      <span v-if="iaf" class="text-[10px] text-accent font-mono"
        :title="`IAF: ${iaf.iafHz} Hz — bands shifted by ${iaf.offsetHz > 0 ? '+' : ''}${iaf.offsetHz} Hz`">
        IAF {{ iaf.iafHz.toFixed(1) }} Hz
      </span>
    </div>
    <div class="space-y-1 max-h-[200px] overflow-y-auto">
      <div v-for="ch in channels" :key="ch.name" class="flex items-center gap-2">
        <span class="text-xs text-text-disabled font-mono w-16 truncate shrink-0">{{ ch.name }}</span>
        <div class="flex-1 flex gap-0.5 h-3">
          <div
            v-for="b in ch.bands" :key="b.band"
            class="rounded-sm"
            :style="{
              width: Math.max(2, (b.power / maxPower) * 100) + '%',
              backgroundColor: b.color,
            }"
            :title="`${b.band}: ${b.power.toFixed(4)}`"
          />
        </div>
      </div>
    </div>
    <!-- Band legend -->
    <div class="flex gap-2 mt-1.5">
      <span v-for="b in (channels[0]?.bands || [])" :key="b.band"
        class="flex items-center gap-1 text-xs text-text-disabled">
        <div class="w-2 h-2 rounded-sm" :style="{ backgroundColor: b.color }" />
        {{ b.band }}
      </span>
    </div>
  </div>
</template>
