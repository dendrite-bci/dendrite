<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useConfigStore } from '../../stores/config'
import { usePipelineStore } from '../../stores/pipeline'

const config = useConfigStore()
const pipeline = usePipelineStore()

const emit = defineEmits<{
  load: []
  save: []
}>()

const showStudyDropdown = ref(false)

const actionBtnClass = computed(() =>
  pipeline.status.recording
    ? 'text-text-disabled cursor-not-allowed'
    : 'text-text-muted hover:text-text-main hover:border-text-muted'
)

const studyNames = computed(() => {
  const names = new Set([
    ...config.availableConfigs.map(c => c.study_name),
    ...config.knownStudyNames,
  ])
  return [...names].sort()
})

function update(field: string, event: Event) {
  const value = (event.target as HTMLInputElement).value
  config.updateGeneral({ [field]: value })
}

function incrementSession() {
  const cur = config.general.session_id
  const match = cur.match(/^(\d+)$/)
  if (match) {
    const next = String(parseInt(match[1]!, 10) + 1).padStart(cur.length, '0')
    config.updateGeneral({ session_id: next })
  }
}

function selectStudy(name: string) {
  config.updateGeneral({ study_name: name })
  showStudyDropdown.value = false
}

onMounted(() => config.listConfigs())
</script>

<template>
  <div>
    <div class="space-y-3">
      <!-- Study Name -->
      <div>
        <label class="block text-[11px] text-text-muted mb-1">Study Name</label>
        <div class="relative">
          <div class="flex gap-1.5">
            <div class="flex flex-1">
              <input
                type="text"
                :value="config.general.study_name"
                placeholder="default_study"
                @input="update('study_name', $event)"
                @focus="showStudyDropdown = true"
                class="flex-1 text-sm rounded-r-none"
                :class="config.validationErrors.study_name ? 'border-status-error' : ''"
              />
              <button
                @click="showStudyDropdown = !showStudyDropdown"
                class="px-2 border border-l-0 border-border rounded-r bg-bg-elevated text-text-muted hover:text-text-main transition-colors"
              ><i class="pi pi-chevron-down text-[10px]" /></button>
            </div>
            <button
              @click="emit('load')"
              :disabled="pipeline.status.recording"
              class="px-2 py-1 text-xs rounded border border-border transition-colors"
              :class="actionBtnClass"
              title="Load config"
            ><i class="pi pi-folder-open" /></button>
            <button
              @click="emit('save')"
              :disabled="pipeline.status.recording"
              class="px-2 py-1 text-xs rounded border border-border transition-colors"
              :class="actionBtnClass"
              title="Save config"
            ><i class="pi pi-save" /></button>
          </div>
          <!-- Dropdown -->
          <div v-if="showStudyDropdown && studyNames.length > 0" class="absolute top-full left-0 right-0 mt-1 z-50 bg-bg-elevated border border-border rounded-lg shadow-xl py-1 max-h-[180px] overflow-y-auto">
            <button
              v-for="name in studyNames" :key="name"
              @click="selectStudy(name)"
              class="w-full text-left px-3 py-1.5 text-xs transition-colors"
              :class="name === config.general.study_name
                ? 'bg-accent/15 text-accent font-semibold'
                : 'text-text-muted hover:text-text-main hover:bg-bg-hover'"
            >{{ name }}</button>
          </div>
          <!-- Backdrop -->
          <Teleport to="body">
            <div v-if="showStudyDropdown" class="fixed inset-0 z-40" @click="showStudyDropdown = false" />
          </Teleport>
        </div>
        <p v-if="config.validationErrors.study_name" class="text-xs text-status-error mt-1">
          {{ config.validationErrors.study_name }}
        </p>

      </div>

      <!-- Subject / Session / Run row -->
      <div class="grid grid-cols-3 gap-3">
        <div>
          <label class="block text-[11px] text-text-muted mb-1">Subject ID</label>
          <input
            type="text"
            :value="config.general.subject_id"
            placeholder="01"
            @input="update('subject_id', $event)"
            class="w-full text-sm"
            :class="config.validationErrors.subject_id ? 'border-status-error' : ''"
          />
          <p v-if="config.validationErrors.subject_id" class="text-xs text-status-error mt-1">
            {{ config.validationErrors.subject_id }}
          </p>
        </div>
        <div>
          <label class="block text-[11px] text-text-muted mb-1">Session ID</label>
          <div class="flex">
            <input
              type="text"
              :value="config.general.session_id"
              placeholder="01"
              @input="update('session_id', $event)"
              class="w-full text-sm rounded-r-none"
              :class="config.validationErrors.session_id ? 'border-status-error' : ''"
            />
            <button
              @click="incrementSession"
              :disabled="pipeline.status.recording"
              class="px-2 border border-l-0 border-border rounded-r bg-bg-elevated text-text-muted
                     hover:text-text-main transition-colors disabled:opacity-30"
              title="Increment session"
            ><i class="pi pi-plus text-[10px]" /></button>
          </div>
          <p v-if="config.validationErrors.session_id" class="text-xs text-status-error mt-1">
            {{ config.validationErrors.session_id }}
          </p>
        </div>
        <div>
          <label class="block text-[11px] text-text-muted mb-1">Run</label>
          <input
            type="text"
            :value="config.nextRun !== null ? String(config.nextRun).padStart(2, '0') : '—'"
            disabled
            class="w-full text-sm text-text-disabled"
          />
        </div>
      </div>

      <!-- Recording Name -->
      <div>
        <label class="block text-[11px] text-text-muted mb-1">Recording Name</label>
        <input
          type="text"
          :value="config.general.recording_name"
          placeholder="task"
          @input="update('recording_name', $event)"
          class="w-full text-sm"
          :class="config.validationErrors.recording_name ? 'border-status-error' : ''"
        />
        <p v-if="config.validationErrors.recording_name" class="text-xs text-status-error mt-1">
          {{ config.validationErrors.recording_name }}
        </p>
      </div>
    </div>
  </div>
</template>
