<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useDataStore } from '../../stores/data'
import { usePipelineStore } from '../../stores/pipeline'
import { useConfigStore } from '../../stores/config'
import { formatDate } from '../../utils/format'
import CreateStudyDialog from './CreateStudyDialog.vue'

const data = useDataStore()
const pipeline = usePipelineStore()
const config = useConfigStore()

function isRecording(studyName: string) {
  return pipeline.status.recording && config.general.study_name === studyName
}

function isActiveRec(recordingId: number) {
  return pipeline.status.recording && pipeline.status.recording_id === recordingId
}

const showCreate = ref(false)

const filteredStudies = computed(() => {
  if (!data.searchQuery) return data.studies
  const q = data.searchQuery.toLowerCase()
  return data.studies.filter(s =>
    s.study_name.toLowerCase().includes(q) ||
    (s.description ?? '').toLowerCase().includes(q)
  )
})

onMounted(() => data.fetchStudies())
</script>

<template>
  <div class="flex flex-col h-full">
    <!-- Header -->
    <div class="flex items-center justify-between px-4 pt-4 pb-2 shrink-0">
      <h2 class="text-sm font-semibold text-text-label">Data</h2>
      <button
        @click="showCreate = true"
        class="px-3 py-1.5 text-xs border border-border rounded text-text-muted
               hover:text-text-main hover:border-text-muted transition-colors"
      >
        <i class="pi pi-plus mr-1" />New Study
      </button>
    </div>

    <!-- Search -->
    <div class="px-4 pb-3 shrink-0">
      <input
        v-model="data.searchQuery"
        placeholder="Search studies..."
        class="w-full"
      />
    </div>

    <!-- Study tree -->
    <div class="flex-1 overflow-y-auto px-4 pb-4">
      <div v-if="filteredStudies.length > 0" class="space-y-1">
        <div v-for="study in filteredStudies" :key="study.study_id">
          <!-- Study row -->
          <div
            class="flex items-center gap-2 px-2 py-2 rounded-lg transition-colors cursor-pointer group"
            :class="[
              data.selectedStudyDetail?.study_id === study.study_id && !data.selectedRecording
                ? 'bg-accent/10 ring-1 ring-inset ring-accent/25'
                : 'hover:bg-bg-hover',
              isRecording(study.study_name) ? 'bg-status-ok/5' : ''
            ]"
            @click="data.selectStudy(study.study_id)"
          >
            <!-- Expand chevron -->
            <button
              @click.stop="data.expandStudy(study.study_id)"
              class="w-5 h-5 flex items-center justify-center text-text-disabled hover:text-text-main transition-colors shrink-0"
            >
              <i class="pi text-xs" :class="data.expandedStudyId === study.study_id ? 'pi-chevron-down' : 'pi-chevron-right'" />
            </button>

            <!-- Recording indicator -->
            <span v-if="isRecording(study.study_name)" class="relative flex h-2 w-2 shrink-0">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-ok opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-status-ok"></span>
            </span>

            <!-- Study name + counts -->
            <span class="text-sm font-semibold text-text-main truncate flex-1">{{ study.study_name }}</span>
            <span v-if="study.recording_count" class="text-[10px] text-text-disabled tabular-nums shrink-0">
              {{ study.recording_count }}
            </span>

          </div>

          <!-- Expanded children -->
          <div v-if="data.expandedStudyId === study.study_id" class="ml-5 mt-1 space-y-1">
            <!-- Recordings -->
            <div
              v-for="rec in data.recordings"
              :key="rec.recording_id"
              @click="data.selectRecording(rec.recording_id)"
              class="flex items-center gap-2 px-2.5 py-2 rounded transition-colors cursor-pointer group/rec"
              :class="[
                isActiveRec(rec.recording_id)
                  ? 'bg-status-ok/5 backdrop-blur-sm'
                  : data.selectedRecording?.recording_id === rec.recording_id
                    ? 'bg-accent/10 ring-1 ring-inset ring-accent/25 text-accent'
                    : 'hover:bg-bg-hover text-text-muted hover:text-text-main'
              ]"
            >
              <span v-if="isActiveRec(rec.recording_id)" class="relative flex h-1.5 w-1.5 shrink-0">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-ok opacity-75"></span>
                <span class="relative inline-flex rounded-full h-1.5 w-1.5 bg-status-ok"></span>
              </span>
              <i v-else class="pi pi-file text-xs shrink-0" />
              <div class="flex-1 min-w-0">
                <span class="text-sm block truncate">{{ rec.recording_name }}</span>
                <span class="text-xs text-text-disabled block mt-0.5">sub-{{ rec.subject_id }} &middot; ses-{{ rec.session_id }} &middot; run-{{ rec.run_number }} &middot; {{ formatDate(rec.session_timestamp) }}</span>
              </div>
            </div>
            <p v-if="data.recordings.length === 0" class="text-xs text-text-disabled px-2.5 py-1">
              No recordings
            </p>

            <!-- Decoders -->
            <template v-if="data.decoders.length > 0">
              <div class="border-t border-border/30 mt-1 pt-1">
                <div
                  v-for="dec in data.decoders"
                  :key="'d' + dec.decoder_id"
                  @click.stop="data.selectStudy(study.study_id); data.selectDecoder(dec.decoder_id)"
                  class="flex items-center gap-2 px-2.5 py-1.5 rounded transition-colors cursor-pointer text-text-muted hover:text-text-main hover:bg-bg-hover"
                >
                  <i class="pi pi-box text-xs shrink-0" />
                  <span class="text-sm truncate flex-1">{{ dec.decoder_name }}</span>
                  <span class="text-[10px] text-accent font-semibold uppercase shrink-0">{{ dec.model_type }}</span>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
      <p v-else class="text-xs text-text-disabled">No studies found.</p>
    </div>

    <CreateStudyDialog v-if="showCreate" @close="showCreate = false" />

  </div>
</template>
