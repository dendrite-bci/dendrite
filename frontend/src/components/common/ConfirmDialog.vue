<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'

withDefaults(defineProps<{
  title?: string
  message: string
  confirmLabel?: string
}>(), {
  title: 'Confirm',
  confirmLabel: 'Delete',
})

const emit = defineEmits<{ confirm: []; cancel: [] }>()

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('cancel')
}
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('cancel')">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl p-8 w-[400px]">
        <h3 class="text-base font-semibold text-text-main mb-4">{{ title }}</h3>
        <p class="text-xs text-text-muted mb-8">{{ message }}</p>
        <div class="flex justify-end gap-3">
          <button
            @click="emit('cancel')"
            class="px-5 py-2.5 text-xs rounded border border-border text-text-muted
                   hover:text-text-main hover:border-text-muted transition-colors"
          >Cancel</button>
          <button
            @click="emit('confirm')"
            class="px-5 py-2.5 text-xs rounded bg-status-error text-white
                   hover:bg-status-error/80 transition-colors"
          >{{ confirmLabel }}</button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
