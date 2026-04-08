<script setup lang="ts">
import { ref } from 'vue'
import { useMLStore } from '../../stores/ml'
import { relativeTime } from '../../utils/format'
import type { TrainingJob } from '../../types/api'

const props = withDefaults(defineProps<{
  dropdown?: boolean
}>(), {
  dropdown: false,
})

const ml = useMLStore()
const open = ref(false)

function statusClass(status: string) {
  switch (status) {
    case 'running': return 'bg-accent/20 text-accent'
    case 'completed': return 'bg-status-ok/20 text-status-ok'
    case 'failed': return 'bg-status-error/20 text-status-error'
    case 'cancelled': return 'bg-text-disabled/20 text-text-disabled'
    default: return 'bg-bg-input text-text-muted'
  }
}

function jobTypeLabel(job: TrainingJob): string {
  switch (job.job_type) {
    case 'evaluation': return 'eval'
    case 'benchmark': return 'bench'
    default: return ''
  }
}

function jobTypeClass(job: TrainingJob): string {
  switch (job.job_type) {
    case 'evaluation': return 'bg-blue-500/15 text-blue-400'
    case 'benchmark': return 'bg-purple-500/15 text-purple-400'
    default: return ''
  }
}

function progressText(job: TrainingJob): string {
  const progress = ml.trainingProgress[job.job_id]
  if (!progress?.epoch) return ''
  return `${progress.epoch}/${progress.total_epochs}`
}

function progressPercent(jobId: number): number {
  const p = ml.trainingProgress[jobId]
  if (!p?.epoch) return 0
  return (p.epoch / (p.total_epochs ?? 1)) * 100
}

function jobLabel(job: TrainingJob): string {
  if (!job.config_json) return ''
  try {
    const config = JSON.parse(job.config_json)
    const parts: string[] = []
    if (config.epochs && isNeuralJob(job)) parts.push(`${config.epochs} ep`)
    if (config.selected_events?.length)
      parts.push(config.selected_events.join(', '))
    if (config.include_background) parts.push('+ rest')
    return parts.join(' · ')
  } catch { return '' }
}

function isNeuralJob(job: TrainingJob): boolean {
  const decoder = ml.models.find(m => m.model_type === job.model_type)
  return decoder ? decoder.default_steps.includes('classifier') : true
}

function pipelineLabel(job: TrainingJob): string {
  const decoder = ml.models.find(m => m.model_type === job.model_type)
  return decoder ? decoder.default_steps.join(' → ') : ''
}

function selectJob(job: TrainingJob) {
  ml.selectJob(job)
  if (props.dropdown) open.value = false
}
</script>

<template>
  <!-- Dropdown mode: button + popover -->
  <div v-if="dropdown" class="relative">
    <button
      @click="open = !open"
      class="flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium transition-colors
             text-text-muted hover:text-text-main hover:bg-bg-elevated"
    >
      <i class="pi pi-history text-sm" />
      <span>Jobs</span>
      <span v-if="ml.jobs.length" class="text-xs font-mono text-text-disabled">({{ ml.jobs.length }})</span>
    </button>

    <!-- Backdrop -->
    <div v-if="open" class="fixed inset-0 z-30" @click="open = false" />

    <!-- Dropdown panel -->
    <Transition name="fade">
      <div
        v-if="open"
        class="absolute right-0 top-full mt-1 z-40 w-[340px] bg-bg-panel border border-border
               rounded-lg shadow-2xl overflow-hidden"
      >
        <div class="flex items-center justify-between px-3 py-2 border-b border-border">
          <span class="text-xs font-semibold text-text-label">Job History</span>
          <span v-if="ml.jobs.length" class="text-xs text-text-disabled">{{ ml.jobs.length }}</span>
        </div>
        <div v-if="ml.jobs.length > 0" class="space-y-1 p-2 max-h-[360px] overflow-y-auto">
          <div
            v-for="job in ml.jobs"
            :key="job.job_id"
            @click="selectJob(job)"
            class="group px-3 py-1.5 rounded-lg
                   hover:bg-bg-elevated transition-colors cursor-pointer"
            :class="ml.selectedJob?.job_id === job.job_id ? 'bg-bg-elevated' : ''"
          >
            <div class="flex items-center justify-between">
              <div>
                <span class="text-[12px] font-semibold text-text-main">{{ job.model_type }}</span>
                <span v-if="pipelineLabel(job)" class="text-[10px] text-text-disabled ml-1">{{ pipelineLabel(job) }}</span>
                <span v-if="jobLabel(job)" class="text-xs text-text-disabled block">{{ jobLabel(job) }}</span>
              </div>
              <div class="flex items-center gap-1.5">
                <span v-if="jobTypeLabel(job)" class="text-[10px] px-1.5 py-px rounded-full font-medium" :class="jobTypeClass(job)">{{ jobTypeLabel(job) }}</span>
                <span
                  class="text-xs px-1.5 py-px rounded-full font-medium"
                  :class="statusClass(job.status)"
                >{{ job.status }}</span>
                <button
                  @click.stop="ml.deleteJob(job.job_id)"
                  class="text-text-disabled hover:text-status-error transition-colors opacity-0 group-hover:opacity-100"
                  title="Delete job"
                ><i class="pi pi-trash text-xs" /></button>
              </div>
            </div>
            <div class="flex items-center gap-2 mt-0.5">
              <span class="text-xs text-text-muted">Job #{{ job.job_id }}</span>
              <span v-if="progressText(job)" class="text-xs text-text-muted">
                &middot; Epoch {{ progressText(job) }}
              </span>
              <span v-if="job.started_at" class="text-xs text-text-disabled ml-auto">
                {{ relativeTime(job.started_at) }}
              </span>
            </div>
            <div v-if="job.status === 'running' && ml.trainingProgress[job.job_id]?.epoch" class="mt-1">
              <div class="w-full h-1 bg-bg-input rounded-full overflow-hidden">
                <div
                  class="h-full bg-accent rounded-full transition-all duration-300"
                  :style="{ width: `${progressPercent(job.job_id)}%` }"
                />
              </div>
            </div>
          </div>
        </div>
        <p v-else class="text-xs text-text-disabled px-3 py-4 text-center">No training jobs yet.</p>
      </div>
    </Transition>
  </div>

  <!-- Inline mode (original) -->
  <div v-else>
    <div class="flex items-center justify-between px-3 py-2">
      <h4 class="text-xs text-text-muted font-semibold">
        Job History
      </h4>
      <span v-if="ml.jobs.length" class="text-xs text-text-disabled">{{ ml.jobs.length }}</span>
    </div>

    <div v-if="ml.jobs.length > 0" class="space-y-0.5 px-3 pb-3 max-h-[320px] overflow-y-auto">
      <div
        v-for="job in ml.jobs"
        :key="job.job_id"
        @click="selectJob(job)"
        class="group px-3 py-1.5 rounded-lg
               hover:bg-bg-elevated transition-colors cursor-pointer"
        :class="ml.selectedJob?.job_id === job.job_id ? 'bg-bg-elevated' : ''"
      >
        <div class="flex items-center justify-between">
          <div>
            <span class="text-[12px] font-semibold text-text-main">{{ job.model_type }}</span>
            <span v-if="jobLabel(job)" class="text-xs text-text-disabled block">{{ jobLabel(job) }}</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span
              class="text-xs px-1.5 py-px rounded-full font-medium"
              :class="statusClass(job.status)"
            >{{ job.status }}</span>
            <button
              @click.stop="ml.deleteJob(job.job_id)"
              class="text-text-disabled hover:text-status-error transition-colors opacity-0 group-hover:opacity-100"
              title="Delete job"
            ><i class="pi pi-trash text-xs" /></button>
          </div>
        </div>
        <div class="flex items-center gap-2 mt-0.5">
          <span class="text-xs text-text-muted">Job #{{ job.job_id }}</span>
          <span v-if="progressText(job)" class="text-xs text-text-muted">
            &middot; {{ isNeuralJob(job) ? 'Epoch ' : '' }}{{ progressText(job) }}
          </span>
          <span v-if="job.started_at" class="text-xs text-text-disabled ml-auto">
            {{ relativeTime(job.started_at) }}
          </span>
        </div>
        <div v-if="job.status === 'running' && ml.trainingProgress[job.job_id]?.epoch" class="mt-1.5">
          <div class="w-full h-1 bg-bg-input rounded-full overflow-hidden">
            <div
              class="h-full bg-accent rounded-full transition-all duration-300"
              :style="{ width: `${progressPercent(job.job_id)}%` }"
            />
          </div>
        </div>
      </div>
    </div>

    <p v-else class="text-xs text-text-disabled px-3 pb-3">No training jobs yet.</p>
  </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
