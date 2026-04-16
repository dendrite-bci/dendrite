<script setup lang="ts">
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { RouterView } from 'vue-router'
import TopNavBar from './components/layout/TopNavBar.vue'
import ControlButtons from './components/common/ControlButtons.vue'
import StreamManagerPanel from './components/stream-manager/StreamManagerPanel.vue'
import ToastContainer from './components/common/ToastContainer.vue'
import { useVisualizationStore } from './stores/visualization'
import { useTelemetryStore } from './stores/telemetry'
import { usePipelineStore } from './stores/pipeline'
import { useTheme } from './composables/useTheme'

useTheme() // initialize theme (applies .dark class to <html>)
const viz = useVisualizationStore()
const telemetry = useTelemetryStore()
const pipeline = usePipelineStore()
const disconnected = computed(() => !viz.connected || !telemetry.connected)

const streamManagerOpen = ref(false)

function toggleStreamManager() {
  streamManagerOpen.value = !streamManagerOpen.value
}

// Warn before closing tab during recording
function onBeforeUnload(e: BeforeUnloadEvent) {
  e.preventDefault()
}

watch(() => pipeline.status.recording, (recording) => {
  if (recording) {
    window.addEventListener('beforeunload', onBeforeUnload)
  } else {
    window.removeEventListener('beforeunload', onBeforeUnload)
  }
}, { immediate: true })

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', onBeforeUnload)
})
</script>

<template>
  <div class="flex flex-col h-screen bg-bg-main">
    <TopNavBar @toggle-stream-manager="toggleStreamManager" />
    <!-- Connection lost banner -->
    <div v-if="disconnected"
         class="flex items-center justify-center gap-2 px-4 py-1.5 bg-amber-900/80 border-b border-amber-700/50 text-amber-200 text-xs shrink-0">
      <span class="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
      Server connection lost — reconnecting…
    </div>
    <main class="flex-1 overflow-hidden">
      <RouterView />
    </main>
    <ControlButtons />
    <StreamManagerPanel v-model="streamManagerOpen" />
    <ToastContainer />
  </div>
</template>
