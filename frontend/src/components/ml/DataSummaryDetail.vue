<script setup lang="ts">
import { computed } from 'vue'
import { useMLStore } from '../../stores/ml'
import { predClassColor } from '../../utils/colors'

const ml = useMLStore()

const sourceName = computed(() => {
  if (!ml.loadedData) return ''
  const m = ml.loadedData.metadata
  if (ml.loadedData.source === 'moabb') return m?.dataset_code ?? 'MOABB'
  if (m?.n_recordings && m.n_recordings > 1) return `${m.n_recordings} recordings`
  if (m?.recording_name) return m.recording_name
  return 'Recording'
})

const windowSec = computed(() => {
  if (!ml.loadedData || !ml.loadedData.sample_rate) return null
  return (ml.loadedData.n_times / ml.loadedData.sample_rate).toFixed(1)
})

const classEntries = computed(() => {
  const names: string[] = (ml.loadedData?.metadata?.class_names ?? []).map(String)
  const counts: Record<string, number> = ml.loadedData?.metadata?.class_counts ?? {}
  const total = names.reduce((sum, name) => sum + (counts[name] ?? 0), 0)
  return names.map(name => {
    const count = counts[name] ?? 0
    return { name, count, pct: total > 0 ? (count / total * 100) : 0 }
  })
})

const channelTypes = computed(() => {
  const types = ml.loadedData?.channel_types ?? []
  const groups: Record<string, number> = {}
  for (const t of types) {
    const key = (t || 'unknown').toUpperCase()
    groups[key] = (groups[key] ?? 0) + 1
  }
  return Object.entries(groups)
})
</script>

<template>
  <div v-if="ml.loadedData" class="mb-4 text-xs border border-border/40 rounded-lg px-4 py-3">
    <div class="flex items-center gap-2 flex-wrap">
      <span class="text-sm font-semibold text-text-main">{{ sourceName }}</span>
      <span v-if="ml.loadedData.metadata?.paradigm" class="text-text-disabled">{{ ml.loadedData.metadata.paradigm }}</span>
      <span class="text-border">&middot;</span>
      <span class="text-text-muted font-mono">{{ ml.loadedData.n_samples }} samples</span>
      <span class="text-border">&middot;</span>
      <span class="text-text-muted font-mono">{{ ml.loadedData.n_channels }}ch</span>
      <span v-for="[type, count] in channelTypes" :key="type" class="text-text-disabled font-mono">({{ count }} {{ type }})</span>
      <span class="text-border">&middot;</span>
      <span class="text-text-muted font-mono">{{ ml.loadedData.sample_rate }}Hz</span>
      <span v-if="windowSec" class="text-text-disabled font-mono">{{ windowSec }}s</span>
      <span class="text-border">&middot;</span>
      <span class="text-data-train font-medium">{{ ml.loadedData.n_samples }} train</span>
      <template v-if="ml.evalData">
        <span class="text-border">&middot;</span>
        <span class="text-data-eval font-medium">{{ ml.evalData.n_samples }} eval{{ ml.evalData.metadata?.auto_split ? ' (auto)' : '' }}</span>
      </template>
    </div>

    <!-- Class distribution -->
    <div v-if="classEntries.length > 0" class="flex items-center gap-3 mt-2">
      <div class="flex h-2 rounded-full overflow-hidden gap-px w-28 shrink-0">
        <div
          v-for="cls in classEntries" :key="cls.name"
          class="first:rounded-l-full last:rounded-r-full"
          :style="{ flex: cls.count || 0.1, backgroundColor: predClassColor(cls.name), opacity: cls.count > 0 ? 1 : 0.2 }"
          :title="`${cls.name}: ${cls.count} (${cls.pct.toFixed(0)}%)`"
        />
      </div>
      <div class="flex items-center gap-3 flex-wrap">
        <span v-for="cls in classEntries" :key="cls.name" class="flex items-center gap-1 text-text-muted">
          <span class="w-2 h-2 rounded-full shrink-0" :style="{ backgroundColor: predClassColor(cls.name) }" />
          {{ /^\d+$/.test(cls.name) ? `Class ${cls.name}` : cls.name }}
          <span class="text-text-disabled font-mono">{{ cls.count }}</span>
        </span>
      </div>
    </div>
  </div>
</template>
