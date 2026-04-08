<script setup lang="ts">
import { useToast } from '../../composables/useToast'

const { toasts, dismiss } = useToast()

function icon(type: string) {
  switch (type) {
    case 'success': return 'pi-check-circle'
    case 'error': return 'pi-exclamation-circle'
    case 'info': return 'pi-info-circle'
    default: return 'pi-info-circle'
  }
}

function colorClass(type: string) {
  switch (type) {
    case 'success': return 'border-status-ok/30 bg-status-ok/15 text-status-ok'
    case 'error': return 'border-status-error/30 bg-status-error/15 text-status-error'
    case 'info': return 'border-accent/30 bg-accent/15 text-accent'
    default: return ''
  }
}
</script>

<template>
  <Teleport to="body">
    <div class="fixed top-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      <TransitionGroup name="toast">
        <div
          v-for="t in toasts"
          :key="t.id"
          class="pointer-events-auto flex items-center gap-2.5 px-4 py-2.5 rounded-lg border
                 shadow-lg backdrop-blur-sm min-w-[260px] max-w-[400px]"
          :class="colorClass(t.type)"
        >
          <i class="pi text-sm shrink-0" :class="icon(t.type)" />
          <span class="text-xs text-text-main flex-1">{{ t.message }}</span>
          <button
            @click="dismiss(t.id)"
            class="text-text-disabled hover:text-text-main transition-colors shrink-0 p-0.5"
          >
            <i class="pi pi-times text-xs" />
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<style scoped>
.toast-enter-active {
  transition: all 0.3s ease;
}
.toast-leave-active {
  transition: all 0.2s ease;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateX(80px);
}
</style>
