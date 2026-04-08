<script setup lang="ts">
defineProps<{
  events: Array<{ id: number; label: string }>
  decoderMappingCount?: number
  lockedIds?: Set<number>
}>()

defineEmits<{
  add: []
  remove: [index: number]
  importDecoder: []
}>()
</script>

<template>
  <div class="grid grid-cols-[60px_1fr_28px] gap-2 mb-2 px-1">
    <span class="text-xs text-text-disabled font-medium">ID</span>
    <span class="text-xs text-text-disabled font-medium">Label</span>
    <span></span>
  </div>
  <div class="space-y-1.5 mb-3">
    <div v-for="(evt, i) in events" :key="i" class="grid grid-cols-[60px_1fr_28px] gap-2 items-center"
      :class="lockedIds?.has(evt.id) ? 'opacity-50' : ''">
      <input v-model.number="evt.id" type="number"
        class="font-mono text-center" placeholder="ID"
        :disabled="lockedIds?.has(evt.id)" />
      <input v-model="evt.label"
        placeholder="Event label"
        :disabled="lockedIds?.has(evt.id)" />
      <button v-if="!lockedIds?.has(evt.id)" @click="$emit('remove', i)" class="w-6 h-6 rounded flex items-center justify-center text-text-disabled hover:text-status-error hover:bg-status-error/10 transition-colors justify-self-center">
        <i class="pi pi-times text-xs" />
      </button>
      <span v-else></span>
    </div>
  </div>
  <div class="flex items-center gap-3">
    <button @click="$emit('add')"
      class="px-3 py-1.5 text-xs rounded border border-dashed border-border text-text-muted hover:border-accent hover:text-accent transition-colors">
      <i class="pi pi-plus mr-1" />Add Event
    </button>
    <button v-if="decoderMappingCount"
      @click="$emit('importDecoder')"
      class="px-3 py-1.5 text-xs rounded border border-accent/30 text-accent hover:bg-accent/10 transition-colors">
      <i class="pi pi-download mr-1" />Import from decoder ({{ decoderMappingCount }})
    </button>
  </div>
</template>
