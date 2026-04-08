<script setup lang="ts">
import { ref } from 'vue'
import GeneralParams from '../components/config/GeneralParams.vue'
import StreamConfig from '../components/config/StreamConfig.vue'
import ModeInstanceList from '../components/config/ModeInstanceList.vue'
import OutputConfig from '../components/config/OutputConfig.vue'
import ConfigLoaderDialog from '../components/config/ConfigLoaderDialog.vue'
import ConfigSaveDialog from '../components/config/ConfigSaveDialog.vue'
import DashboardView from './DashboardView.vue'
import { usePipelineStore } from '../stores/pipeline'

const pipeline = usePipelineStore()
const activeTab = ref(0)
const showLoadDialog = ref(false)
const showSaveDialog = ref(false)

const configTabs = [
  { label: 'General', icon: 'pi pi-cog' },
  { label: 'Modes', icon: 'pi pi-th-large' },
  { label: 'Output', icon: 'pi pi-send' },
]

function isTabLocked(index: number): boolean {
  return pipeline.status.recording && index !== 1
}

</script>

<template>
  <div class="flex h-full relative">
    <!-- Centered empty state when not recording -->
    <div v-if="!pipeline.status.recording" class="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <div class="text-center">
        <i class="pi pi-chart-line text-4xl text-text-disabled mb-4 block" />
        <p class="text-text-muted">Start a recording to view the dashboard</p>
      </div>
    </div>

    <!-- Left Panel: Configuration -->
    <div class="w-[560px] max-w-[560px] flex flex-col border-r border-border bg-bg-panel">
      <!-- Header with tabs + load/save -->
      <div class="flex items-stretch gap-2 px-3 border-b border-border shrink-0 h-[42px]">
        <div class="flex flex-1 items-stretch bg-bg-main/50 rounded-md my-1 gap-0.5">
          <button
            v-for="(tab, i) in configTabs"
            :key="tab.label"
            @click="activeTab = i"
            class="flex-1 flex items-center justify-center gap-1.5 px-3 rounded
                   text-sm font-medium transition-colors"
            :class="activeTab === i
              ? 'bg-bg-elevated text-text-main'
              : 'text-text-muted hover:text-text-main'"
          >
            <i v-if="isTabLocked(i)" class="pi pi-lock text-sm text-text-disabled" />
            <i v-else :class="tab.icon" class="text-sm" />
            <span>{{ tab.label }}</span>
          </button>
        </div>
      </div>

      <!-- Config content -->
      <div class="flex-1 overflow-y-auto p-4">
        <div v-if="isTabLocked(activeTab)" class="flex items-center gap-2 text-xs text-status-warn bg-status-warn/10 rounded px-3 py-2 mb-3">
          <i class="pi pi-lock text-sm" />
          <span>Configuration locked during recording</span>
        </div>
        <GeneralParams v-if="activeTab === 0" @load="showLoadDialog = true" @save="showSaveDialog = true" />
        <div v-if="activeTab === 0" class="border-t border-border mt-8 pt-6" />
        <StreamConfig v-if="activeTab === 0" />
        <ModeInstanceList v-if="activeTab === 1" class="mt-1" />
        <OutputConfig v-if="activeTab === 2" />
      </div>
    </div>

    <!-- Right Panel: Dashboard -->
    <div class="flex-1 flex flex-col min-h-0 min-w-0">
      <DashboardView />
    </div>

    <ConfigLoaderDialog v-if="showLoadDialog" @close="showLoadDialog = false" />
    <ConfigSaveDialog v-if="showSaveDialog" @close="showSaveDialog = false" />
  </div>
</template>
