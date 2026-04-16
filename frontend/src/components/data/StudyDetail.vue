<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useDataStore } from '../../stores/data'
import { usePipelineStore } from '../../stores/pipeline'
import { useConfigStore } from '../../stores/config'
import ConfirmDialog from '../common/ConfirmDialog.vue'
import { formatDate, formatPercent } from '../../utils/format'

const data = useDataStore()
const pipeline = usePipelineStore()
const config = useConfigStore()

const isActiveStudy = computed(() =>
  pipeline.status.recording && config.general.study_name === data.selectedStudyDetail?.study_name
)

const showDeleteConfirm = ref(false)
async function confirmDelete() {
  if (!data.selectedStudyDetail) return
  await data.deleteStudy(data.selectedStudyDetail.study_id)
  showDeleteConfirm.value = false
}

const editDesc = ref('')
const editing = ref(false)

watch(() => data.selectedStudyDetail, (s) => {
  if (s) {
    editDesc.value = s.description || ''
    data.fetchDecoders(s.study_id)
  }
  editing.value = false
})

async function saveDescription() {
  if (!data.selectedStudyDetail) return
  await data.updateStudy(data.selectedStudyDetail.study_id, editDesc.value)
  editing.value = false
}

function bestAccuracy(dec: any): number | null {
  return dec.validation_accuracy ?? dec.training_accuracy
}
</script>

<template>
  <div v-if="data.selectedStudyDetail" class="space-y-5">
    <!-- Header card -->
    <div class="bg-bg-elevated rounded-lg px-4 py-3">
      <div class="flex items-start justify-between mb-2">
        <h2 class="text-sm font-semibold text-text-main">{{ data.selectedStudyDetail.study_name }}</h2>
        <div class="flex items-center gap-2 shrink-0 ml-3">
          <span class="text-xs text-text-disabled">{{ formatDate(data.selectedStudyDetail.created_at) }}</span>
          <button
            v-if="!isActiveStudy"
            @click="showDeleteConfirm = true"
            class="w-5 h-5 flex items-center justify-center text-text-disabled hover:text-status-error transition-colors rounded"
            title="Delete study"
          >
            <i class="pi pi-trash text-xs" />
          </button>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-x-5 gap-y-1 mb-2">
        <span class="text-xs">
          <span class="text-text-disabled">Recordings</span>
          <span class="text-text-main font-mono ml-1">{{ data.selectedStudyDetail.recording_count }}</span>
        </span>
        <span class="text-xs">
          <span class="text-text-disabled">Decoders</span>
          <span class="text-text-main font-mono ml-1">{{ data.selectedStudyDetail.decoder_count }}</span>
        </span>
      </div>
      <!-- Description -->
      <div v-if="!editing" class="flex items-start justify-between gap-3">
        <p class="text-xs text-text-muted">{{ data.selectedStudyDetail.description || 'No description' }}</p>
        <button
          @click="editing = true; editDesc = data.selectedStudyDetail!.description || ''"
          class="text-[11px] text-text-disabled hover:text-text-main transition-colors shrink-0"
        >Edit</button>
      </div>
      <div v-else class="space-y-2">
        <textarea v-model="editDesc" class="w-full text-sm resize-none h-20" />
        <div class="flex gap-2">
          <button
            @click="saveDescription"
            class="px-3 py-1 text-xs bg-accent text-white rounded hover:bg-accent/80 transition-colors"
          >Save</button>
          <button
            @click="editing = false"
            class="px-3 py-1 text-xs text-text-muted hover:text-text-main transition-colors"
          >Cancel</button>
        </div>
      </div>
    </div>

    <!-- Recordings & Decoders side by side -->
    <div class="grid grid-cols-2 gap-5" v-if="data.recordings.length > 0 || data.decoders.length > 0">
      <!-- Recordings column -->
      <div>
        <span class="text-[11px] text-text-muted block mb-2">
          Recordings ({{ data.recordings.length }})
        </span>
        <div v-if="data.recordings.length > 0" class="border border-border rounded-lg overflow-hidden">
          <div
            v-for="(rec, i) in data.recordings"
            :key="rec.recording_id"
            @click="data.selectRecording(rec.recording_id)"
            class="flex items-center gap-2.5 px-3 py-2 cursor-pointer hover:bg-bg-hover transition-colors"
            :class="{ 'border-t border-border': i > 0 }"
          >
            <i class="pi pi-wave-pulse text-[11px] text-accent/60 shrink-0" />
            <div class="flex-1 min-w-0">
              <span class="text-xs font-medium text-text-main block truncate">{{ rec.recording_name }}</span>
              <span class="text-[11px] text-text-disabled">sub-{{ rec.subject_id }} &middot; ses-{{ rec.session_id }}</span>
            </div>
            <span class="text-[11px] text-text-disabled shrink-0">{{ formatDate(rec.session_timestamp) }}</span>
          </div>
        </div>
        <p v-else class="text-xs text-text-disabled">None yet</p>
      </div>

      <!-- Decoders column -->
      <div>
        <span class="text-[11px] text-text-muted block mb-2">
          Decoders ({{ data.decoders.length }})
        </span>
        <div v-if="data.decoders.length > 0" class="border border-border rounded-lg overflow-hidden">
          <div
            v-for="(dec, i) in data.decoders"
            :key="dec.decoder_id"
            @click="data.selectDecoder(dec.decoder_id)"
            class="flex items-center gap-2.5 px-3 py-2 cursor-pointer hover:bg-bg-hover transition-colors"
            :class="{ 'border-t border-border': i > 0 }"
          >
            <i class="pi pi-microchip text-[11px] text-status-ok/60 shrink-0" />
            <div class="flex-1 min-w-0">
              <span class="text-xs font-medium text-text-main block truncate">{{ dec.decoder_name }}</span>
              <span class="text-[11px] text-text-disabled">
                {{ dec.model_type }}
                <template v-if="dec.num_classes"> &middot; {{ dec.num_classes }}cls</template>
              </span>
            </div>
            <span v-if="bestAccuracy(dec) != null" class="text-xs font-mono text-text-main shrink-0">{{ formatPercent(bestAccuracy(dec)) }}</span>
          </div>
        </div>
        <p v-else class="text-xs text-text-disabled">None yet</p>
      </div>
    </div>

    <!-- Empty state (no recordings AND no decoders) -->
    <div v-else class="text-center py-8">
      <p class="text-sm text-text-muted">No recordings or decoders yet</p>
    </div>
    <ConfirmDialog
      v-if="showDeleteConfirm"
      title="Delete Study"
      :message="`Delete study &quot;${data.selectedStudyDetail.study_name}&quot;? All recordings and decoders belonging to this study will be permanently deleted.`"
      @confirm="confirmDelete"
      @cancel="showDeleteConfirm = false"
    />
  </div>
</template>
