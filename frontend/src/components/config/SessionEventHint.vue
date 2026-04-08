<script setup lang="ts">
defineProps<{
  events: number[]
  eventMapping?: Record<number, string>
}>()
defineEmits<{
  apply: []
}>()
</script>

<template>
  <div v-if="events.length > 0" class="mb-3 bg-accent/5 rounded p-2.5">
    <div class="flex items-center justify-between mb-1.5">
      <span class="text-xs text-text-muted">Session events detected</span>
      <button @click="$emit('apply')"
        class="text-xs text-accent hover:text-accent-hover transition-colors">
        <i class="pi pi-plus mr-0.5" />Use
      </button>
    </div>
    <div class="flex flex-wrap gap-1">
      <span v-for="code in events" :key="code"
        class="px-1.5 py-0.5 bg-bg-input rounded text-xs font-mono text-accent border border-accent/20">
        {{ code }}<template v-if="eventMapping?.[code]">: {{ eventMapping[code] }}</template>
      </span>
    </div>
  </div>
</template>
