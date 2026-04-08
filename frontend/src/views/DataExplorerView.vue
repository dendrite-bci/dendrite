<script setup lang="ts">
import { useDataStore } from '../stores/data'
import DataSidebar from '../components/data/DataSidebar.vue'
import StudyDetail from '../components/data/StudyDetail.vue'
import RecordingDetail from '../components/data/RecordingDetail.vue'
import DecoderDetail from '../components/data/DecoderDetail.vue'

const data = useDataStore()
</script>

<template>
  <div class="flex h-full relative">
    <!-- Centered empty state when nothing selected -->
    <div v-if="!data.selectedRecording && !data.selectedDecoder && !data.selectedStudyDetail"
         class="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <div class="text-center">
        <i class="pi pi-database text-4xl text-text-disabled mb-3 block" />
        <p class="text-text-muted text-sm">Select a study or recording</p>
      </div>
    </div>

    <!-- Left Panel: Hierarchical sidebar -->
    <div class="w-[400px] max-w-[400px] flex flex-col bg-bg-panel border-r border-border shrink-0">
      <DataSidebar />
    </div>

    <!-- Right Panel: Detail -->
    <div class="flex-1 overflow-y-auto p-5 bg-bg-main">
      <RecordingDetail v-if="data.selectedRecording" />
      <DecoderDetail v-else-if="data.selectedDecoder" />
      <StudyDetail v-else-if="data.selectedStudyDetail" />
    </div>
  </div>
</template>
