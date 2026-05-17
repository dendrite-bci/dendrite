<script setup lang="ts">
import { computed } from 'vue'
import type { FileInfo } from '../../stores/streamManager'

const props = defineProps<{
  info: FileInfo
  filePath?: string
  enableEvents: boolean
  loading: boolean
  startLabel?: string
  modality?: string | null
}>()

const emit = defineEmits<{
  start: []
  'start-all': []
  'update:enableEvents': [value: boolean]
  'update:modality': [value: string | null]
}>()

const modalities = computed(() => props.info.available_modalities ?? [])
const hasMulti = computed(() => modalities.value.length > 1)
</script>

<template>
  <div class="space-y-3">
    <!-- File path -->
    <p v-if="filePath" class="text-xs text-text-muted font-mono truncate">{{ filePath }}</p>

    <!-- Metadata + events card -->
    <div class="rounded border border-border bg-bg-elevated">
      <!-- Stats row -->
      <div class="flex items-center gap-3 px-3 py-2 text-xs">
        <span class="text-text-muted">Duration <span class="font-mono text-text-main">{{ info.duration_s.toFixed(1) }}s</span></span>
        <span class="text-text-muted">Rate <span class="font-mono text-text-main">{{ info.sample_rate }} Hz</span></span>
        <span class="text-text-muted">Ch <span class="font-mono text-text-main">{{ info.n_channels }}</span></span>
        <span v-if="info.n_events > 0" class="text-text-muted">Events <span class="font-mono text-text-main">{{ info.n_events }}</span></span>
      </div>

      <!-- Modalities row -->
      <div v-if="modalities.length > 0" class="flex items-center gap-1.5 px-3 py-2 border-t border-border/50">
        <span class="text-xs text-text-muted">Modalities</span>
        <span
          v-for="m in modalities"
          :key="m"
          class="px-1.5 py-0.5 rounded bg-bg-input text-xs font-mono uppercase tracking-wide text-text-main"
        >{{ m }}</span>
      </div>

      <!-- Event mapping -->
      <div v-if="info.event_id && Object.keys(info.event_id).length > 0"
           class="px-3 py-2 border-t border-border/50 space-y-1">
        <div
          v-for="[name, code] in Object.entries(info.event_id).sort((a, b) => a[1] - b[1])"
          :key="name"
          class="flex items-center gap-2 text-xs"
        >
          <span class="font-mono text-text-disabled w-4 text-right">{{ code }}</span>
          <span class="text-text-muted">{{ name }}</span>
        </div>
      </div>
    </div>

    <!-- Modality selector (multi-modality files) -->
    <label v-if="hasMulti" class="flex items-center gap-3">
      <span class="text-xs text-text-main shrink-0">Replay</span>
      <select
        :value="modality ?? modalities[0]"
        @change="emit('update:modality', ($event.target as HTMLSelectElement).value)"
        class="flex-1 text-xs"
      >
        <option v-for="m in modalities" :key="m" :value="m">{{ m.toUpperCase() }}</option>
      </select>
    </label>

    <!-- Separate events stream toggle -->
    <label v-if="info.n_events > 0" class="flex items-center gap-3 cursor-pointer">
      <div
        class="relative w-9 h-5 rounded-full transition-colors"
        :class="enableEvents ? 'bg-accent' : 'bg-bg-elevated'"
        @click="emit('update:enableEvents', !enableEvents)"
      >
        <div
          class="absolute top-0.5 w-4 h-4 rounded-full bg-text-main shadow transition-transform"
          :class="enableEvents ? 'translate-x-4' : 'translate-x-0.5'"
        />
      </div>
      <span class="text-xs text-text-main">Create separate events stream</span>
    </label>

    <!-- Start buttons -->
    <div class="flex gap-2">
      <button
        @click="emit('start')"
        :disabled="loading"
        class="flex-1 py-2 rounded text-xs font-medium text-white transition-colors
               bg-accent hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed"
      >
        <i v-if="loading" class="pi pi-spin pi-spinner mr-1" />
        <i v-else class="pi pi-play mr-1" />
        {{ startLabel ?? 'Start Replay' }}
      </button>
      <button
        v-if="hasMulti"
        @click="emit('start-all')"
        :disabled="loading"
        class="px-3 py-2 rounded text-xs font-medium transition-colors
               bg-bg-elevated border border-border text-text-main
               hover:border-accent disabled:opacity-30 disabled:cursor-not-allowed"
        :title="`Start all ${modalities.length} modalities in parallel`"
      >
        <i class="pi pi-th-large mr-1" />
        Start All
      </button>
    </div>
  </div>
</template>
