<script setup lang="ts">
import { useMLStore } from '../../stores/ml'
import NumberInput from '../common/NumberInput.vue'

const ml = useMLStore()

function toggleModel(modelType: string) {
  const idx = ml.benchConfig.model_types.indexOf(modelType)
  if (idx >= 0) {
    ml.benchConfig.model_types.splice(idx, 1)
  } else {
    ml.benchConfig.model_types.push(modelType)
  }
}
</script>

<template>
  <div>
    <!-- Row 1: Model toggles -->
    <div class="flex items-center gap-2 mb-2">
      <label class="text-[11px] text-text-muted shrink-0">Models</label>
      <div class="flex flex-wrap gap-1">
        <button
          v-for="m in ml.models" :key="m.model_type"
          @click="toggleModel(m.model_type)"
          class="px-2 py-0.5 text-xs font-medium rounded transition-colors"
          :class="ml.benchConfig.model_types.includes(m.model_type)
            ? 'bg-accent/15 text-accent'
            : 'bg-bg-input text-text-disabled hover:text-text-muted'"
        >{{ m.model_type }}</button>
      </div>
      <div class="flex gap-1 ml-1">
        <button @click="ml.benchConfig.model_types = ml.models.map(m => m.model_type)"
          class="text-[10px] px-1.5 py-0.5 rounded bg-bg-input text-text-muted hover:text-text-main transition-colors">All</button>
        <button @click="ml.benchConfig.model_types = []"
          class="text-[10px] px-1.5 py-0.5 rounded bg-bg-input text-text-muted hover:text-text-main transition-colors">None</button>
      </div>
    </div>

    <!-- Row 2: Folds + run -->
    <div class="flex items-end gap-2.5 flex-wrap">
      <div class="w-[72px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Folds</label>
        <NumberInput v-model="ml.benchConfig.n_folds" :min="2" :max="20" class="w-full" />
      </div>

      <button
        @click="ml.startBenchmark()"
        :disabled="ml.benchConfig.model_types.length === 0 || !ml.loadedData || ml.benchRunning"
        class="px-3.5 py-1.5 text-xs font-semibold rounded transition-colors"
        :class="ml.benchConfig.model_types.length > 0 && ml.loadedData && !ml.benchRunning
          ? 'bg-accent text-white hover:bg-accent/80'
          : 'bg-bg-input text-text-disabled cursor-not-allowed'"
      >
        <i v-if="ml.benchRunning" class="pi pi-spin pi-spinner mr-1" />
        {{ ml.benchRunning ? 'Running...' : `Benchmark (${ml.benchConfig.model_types.length})` }}
      </button>
    </div>
  </div>
</template>
