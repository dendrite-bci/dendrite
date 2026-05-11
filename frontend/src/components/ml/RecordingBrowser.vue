<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useMLStore } from '../../stores/ml'
import { useDataStore } from '../../stores/data'
import type { Recording } from '../../types/api'
import { formatDate } from '../../utils/format'

const ml = useMLStore()
const data = useDataStore()
const expandedId = ref<number | null>(null)

onMounted(() => {
  data.fetchStudies()
  ml.fetchRecordings()
})

watch(() => ml.selectedStudyId, (id) => {
  ml.fetchRecordings(id ?? undefined)
  ml.recordingRoles = {}
  ml.dataPreproc.selected_events = null
  if (id != null) ml.restoreStudyState(id)
})

watch([() => ml.recordingRoles, () => ml.dataPreproc], () => {
  ml.saveStudyState()
}, { deep: true })

function toggleSelect(recordingId: number) {
  if (ml.recordingRoles[recordingId]) {
    delete ml.recordingRoles[recordingId]
  } else {
    ml.recordingRoles[recordingId] = 'train'
    ml.fetchRecordingEvents([recordingId])
  }
}

function isSelected(recordingId: number): boolean {
  return !!ml.recordingRoles[recordingId]
}

function toggleExpand(recordingId: number) {
  expandedId.value = expandedId.value === recordingId ? null : recordingId
  if (expandedId.value != null) ml.fetchRecordingEvents([recordingId])
}

const groupedBySubject = computed(() => {
  const groups: Record<string, Recording[]> = {}
  for (const rec of ml.recordings) {
    const key = rec.subject_id
    if (!groups[key]) groups[key] = []
    groups[key].push(rec)
  }
  return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b))
})

function refreshData() {
  data.fetchStudies()
  ml.fetchRecordings(ml.selectedStudyId ?? undefined)
}

function handleStudyChange(event: Event) {
  const value = (event.target as HTMLSelectElement).value
  ml.selectedStudyId = value ? Number(value) : null
}
</script>

<template>
  <div class="mt-2">
    <!-- Study filter + refresh -->
    <div class="flex items-center gap-2 mb-2">
      <select
        :value="ml.selectedStudyId ?? ''"
        @change="handleStudyChange"
        class="flex-1 text-xs"
      >
        <option value="">All Studies</option>
        <option v-for="s in data.studies" :key="s.study_id" :value="s.study_id">
          {{ s.study_name }}
        </option>
      </select>
      <button @click="refreshData" class="text-xs text-accent hover:text-accent/80 shrink-0">
        <i class="pi pi-refresh text-xs" />
      </button>
    </div>

    <!-- Recording list -->
    <div v-if="ml.recordings.length > 0" class="max-h-[350px] overflow-y-auto">
      <template v-for="[subjectId, recs] in groupedBySubject" :key="subjectId">
        <div class="text-[11px] text-text-disabled uppercase tracking-wider px-2 pt-2.5 pb-1 sticky top-0 bg-bg-elevated/30 backdrop-blur-sm">
          {{ subjectId }}
        </div>
        <div v-for="rec in recs" :key="rec.recording_id">
          <div
            @click="toggleSelect(rec.recording_id)"
            class="flex items-center gap-2 px-2 py-2 cursor-pointer rounded transition-colors"
            :class="isSelected(rec.recording_id) ? 'bg-accent/10 ring-1 ring-inset ring-accent/25' : 'hover:bg-text-main/[0.04]'"
          >
            <!-- Checkbox -->
            <span class="w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0 transition-colors text-[8px]"
              :class="isSelected(rec.recording_id)
                ? 'border-accent/50 bg-accent/20 text-accent'
                : 'border-border/50'"
            >
              <i v-if="isSelected(rec.recording_id)" class="pi pi-check" />
            </span>
            <span class="text-xs text-text-main truncate flex-1">{{ rec.recording_name }}</span>
            <span class="text-[10px] text-text-disabled shrink-0">{{ formatDate(rec.session_timestamp) }}</span>
            <!-- Expand chevron -->
            <i @click.stop="toggleExpand(rec.recording_id)"
              class="pi text-[8px] text-text-disabled hover:text-text-muted shrink-0"
              :class="expandedId === rec.recording_id ? 'pi-chevron-down' : 'pi-chevron-right'" />
          </div>

          <!-- Expanded detail -->
          <div v-if="expandedId === rec.recording_id" class="ml-7 mr-2 mb-1 px-2 py-2 rounded bg-text-main/[0.04] border-l border-border/30 text-xs space-y-1.5">
            <div v-if="ml.recordingEventSummaries[rec.recording_id]" class="flex items-center gap-1.5 flex-wrap">
              <span class="text-text-disabled">Events:</span>
              <span
                v-for="(count, name) in ml.recordingEventSummaries[rec.recording_id]"
                :key="name"
                class="px-1.5 py-0.5 rounded bg-text-main/[0.04] text-text-muted"
              >{{ name }} <span class="text-text-disabled font-mono">{{ count }}</span></span>
            </div>
            <div class="text-text-disabled">
              ses-{{ rec.session_id }} · {{ rec.study_name }}
            </div>
          </div>
        </div>
      </template>
    </div>

    <p v-else class="text-xs text-text-disabled text-center py-4">
      No recordings found.
    </p>
  </div>
</template>
