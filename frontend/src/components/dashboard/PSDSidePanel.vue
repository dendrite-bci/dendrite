<script setup lang="ts">
import InlinePSD from './InlinePSD.vue'

defineProps<{
  modality: string
  width: number
}>()

defineEmits<{
  dragStart: [e: MouseEvent]
}>()
</script>

<template>
  <div
    @mousedown="$emit('dragStart', $event)"
    class="w-1.5 shrink-0 cursor-col-resize group flex items-center justify-center mx-0.5"
  >
    <div class="w-px h-8 bg-border group-hover:bg-accent/50 transition-colors rounded-full" />
  </div>
  <div class="shrink-0" :style="{ width: width + 'px' }">
    <InlinePSD :modality="modality" />
  </div>
</template>
