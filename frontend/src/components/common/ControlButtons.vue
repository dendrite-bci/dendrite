<script setup lang="ts">
import { ref, computed } from 'vue'
import { usePipelineStore } from '../../stores/pipeline'
import { useToast } from '../../composables/useToast'
import ConfirmDialog from './ConfirmDialog.vue'
import PreflightChecklist from './PreflightChecklist.vue'

const pipeline = usePipelineStore()
const toast = useToast()
const showStopConfirm = ref(false)

async function handleStart() {
  await pipeline.start()
  if (pipeline.status.recording) {
    toast.success('Pipeline started')
  } else if (pipeline.error) {
    toast.error(pipeline.error)
  }
}

async function confirmStop() {
  showStopConfirm.value = false
  await pipeline.stop()
  if (!pipeline.status.recording) {
    toast.info('Pipeline stopped')
  }
}

const elapsedFormatted = computed(() => {
  const s = Math.floor(pipeline.status.elapsed_seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
})

</script>

<template>
  <div class="grid grid-cols-3 items-center px-6 py-3 bg-bg-panel border-t border-border shrink-0 h-[72px]">
    <!-- Left: Preflight checks -->
    <div class="flex items-center min-w-0">
      <span v-if="pipeline.status.recording" class="text-xs text-text-muted">Recording</span>
      <PreflightChecklist v-else />
    </div>

    <!-- Center: Play/Stop -->
    <div class="flex items-center justify-center">
      <button
        v-if="!pipeline.status.recording"
        @click="handleStart"
        :disabled="!pipeline.canStart"
        class="w-12 h-12 rounded-full flex items-center justify-center text-white
               bg-status-ok hover:bg-status-ok/80 shadow-lg shadow-status-ok/25
               hover:scale-105 active:scale-95 transition-all duration-150
               disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:shadow-none"
        :title="pipeline.canStart ? 'Start recording' : 'Pre-flight checks not passed'"
      >
        <i v-if="pipeline.loading" class="pi pi-spin pi-spinner text-lg" />
        <i v-else class="pi pi-play text-lg ml-0.5" />
      </button>
      <button
        v-else
        @click="showStopConfirm = true"
        :disabled="pipeline.loading"
        class="w-12 h-12 rounded-full flex items-center justify-center text-white
               bg-status-error hover:bg-status-error/80 shadow-lg shadow-status-error/25
               hover:scale-105 active:scale-95 transition-all duration-150
               disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
      >
        <i v-if="pipeline.loading" class="pi pi-spin pi-spinner text-lg" />
        <i v-else class="pi pi-stop text-lg" />
      </button>
    </div>

    <!-- Right: Timer + Error -->
    <div class="flex items-center justify-end gap-3 min-w-0">
      <span v-if="pipeline.status.recording" class="text-sm font-mono text-text-muted tabular-nums">
        {{ elapsedFormatted }}
      </span>
      <span v-if="pipeline.error" class="text-xs text-status-error truncate max-w-[240px]" :title="pipeline.error">
        {{ pipeline.error }}
      </span>
    </div>

    <ConfirmDialog
      v-if="showStopConfirm"
      title="Stop Recording"
      message="Stop the current recording session? Data recorded so far will be saved."
      confirm-label="Stop"
      @confirm="confirmStop"
      @cancel="showStopConfirm = false"
    />
  </div>
</template>
