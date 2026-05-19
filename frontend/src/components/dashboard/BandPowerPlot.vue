<script setup lang="ts">
import { computed } from 'vue'
import { useVisualizationStore } from '../../stores/visualization'
import { getBandColor } from '../../utils/colors'
import BandPowerChannelPlot from './BandPowerChannelPlot.vue'

const props = defineProps<{ modeName: string }>()
const viz = useVisualizationStore()

const history = computed(() => viz.modeBandPowerHistory[props.modeName])
const iaf = computed(() => viz.modeIAF[props.modeName])

const channelNames = computed(() => history.value ? Object.keys(history.value.channels) : [])
const bandNames = computed(() => history.value?.bandNames ?? [])
const hasData = computed(() => channelNames.value.length > 0 && bandNames.value.length > 0)

const windowSec = computed(() => {
  const h = history.value
  if (!h) return 30
  // Buffer holds 120 samples (BAND_POWER_BUFFER_SAMPLES); window = 120 * stepSec.
  return Math.round(120 * h.stepSec)
})
</script>

<template>
  <div v-if="hasData">
    <div class="flex items-center justify-between mb-1 gap-2">
      <div class="flex items-center gap-2">
        <span class="text-xs text-text-muted uppercase tracking-wide">Band Power</span>
        <span v-if="iaf" class="text-[10px] text-accent font-mono"
          :title="`IAF: ${iaf.iafHz} Hz — bands shifted by ${iaf.offsetHz > 0 ? '+' : ''}${iaf.offsetHz} Hz`">
          IAF {{ iaf.iafHz.toFixed(1) }} Hz
        </span>
      </div>
      <div class="flex gap-2">
        <span v-for="b in bandNames" :key="b"
          class="flex items-center gap-1 text-[10px] text-text-disabled">
          <span class="inline-block w-2.5 h-0.5 rounded-full" :style="{ backgroundColor: getBandColor(b) }" />
          {{ b }}
        </span>
      </div>
    </div>
    <div class="space-y-0.5">
      <BandPowerChannelPlot
        v-for="ch in channelNames" :key="ch"
        :mode-name="modeName" :channel-name="ch"
        :band-names="bandNames"
      />
    </div>
    <div class="mt-1 pr-12 text-[10px] text-text-disabled font-mono text-right">
      ← {{ windowSec }}s
    </div>
  </div>
</template>
