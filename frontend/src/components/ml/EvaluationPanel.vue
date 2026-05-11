<script setup lang="ts">
import { computed } from 'vue'
import { useMLStore } from '../../stores/ml'
import NumberInput from '../common/NumberInput.vue'

const ml = useMLStore()

const completedJobs = computed(() =>
  ml.jobs.filter(j => j.status === 'completed' && j.job_type === 'training')
)
</script>

<template>
  <div>
    <div class="flex items-end gap-2 flex-wrap">
      <div class="w-[180px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Decoder</label>
        <select v-model.number="ml.evalConfig.job_id" class="w-full text-xs">
          <option :value="null" disabled>Select job...</option>
          <option v-for="job in completedJobs" :key="job.job_id" :value="job.job_id">
            #{{ job.job_id }} — {{ job.model_type }}
          </option>
        </select>
      </div>
      <div class="w-[110px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Mode</label>
        <select v-model="ml.evalConfig.mode" class="w-full text-xs">
          <option value="sliding_window">Sliding Window</option>
          <option value="epoch">Epoch</option>
        </select>
      </div>
      <div v-if="ml.evalConfig.mode === 'sliding_window'" class="w-[72px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Step (ms)</label>
        <NumberInput v-model="ml.evalConfig.step_size_ms" :min="10" :max="1000" :step="10" class="w-full" />
      </div>

      <button
        @click="ml.startEvaluation()"
        :disabled="!ml.evalConfig.job_id || !ml.loadedData || ml.evalRunning"
        class="px-3.5 py-1.5 text-xs font-semibold rounded transition-colors"
        :class="ml.evalConfig.job_id && ml.loadedData && !ml.evalRunning
          ? 'bg-accent text-white hover:bg-accent/80'
          : 'bg-bg-input text-text-disabled cursor-not-allowed'"
      >
        <i v-if="ml.evalRunning" class="pi pi-spin pi-spinner mr-1" />
        {{ ml.evalRunning ? 'Evaluating...' : 'Evaluate' }}
      </button>

      <button
        v-if="ml.evalRunning && ml.evalJobId"
        @click="ml.cancelJob(ml.evalJobId)"
        class="px-3 py-1.5 text-xs font-medium rounded bg-status-error/10 text-status-error hover:bg-status-error/20 transition-colors"
      >Cancel</button>
    </div>

    <!-- Progress -->
    <div v-if="ml.evalRunning && ml.liveEval && ml.liveEval.total > 0" class="mt-2">
      <div class="flex items-center gap-2">
        <div class="flex-1 h-1 bg-bg-input rounded-full overflow-hidden">
          <div class="h-full bg-accent rounded-full transition-all duration-300"
            :style="{ width: `${(ml.liveEval.step / ml.liveEval.total) * 100}%` }" />
        </div>
        <span class="text-[11px] text-text-disabled shrink-0">{{ ml.liveEval.step }}/{{ ml.liveEval.total }}</span>
      </div>
    </div>
  </div>
</template>
