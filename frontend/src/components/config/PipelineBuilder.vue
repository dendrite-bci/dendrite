<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ModelInfo } from '../../types/api'

const props = withDefaults(defineProps<{
  modelType: string
  models: ModelInfo[]
  pipelineSteps?: string[] | null
  readonly?: boolean
}>(), {
  pipelineSteps: null,
  readonly: false,
})

const emit = defineEmits<{
  'update:modelType': [value: string]
  'update:pipelineSteps': [value: string[] | null]
}>()

const showPicker = ref(false)

const currentModel = computed(() =>
  props.models.find(m => m.model_type === props.modelType)
)

const stepTypes = computed(() => currentModel.value?.step_types ?? {})

const displaySteps = computed(() =>
  currentModel.value?.default_steps ?? ['scaler', 'classifier']
)

const scalerEnabled = computed(() =>
  props.pipelineSteps ? props.pipelineSteps.includes('scaler') : true
)

function toggleScaler() {
  if (props.readonly) return
  if (scalerEnabled.value) {
    emit('update:pipelineSteps', displaySteps.value.filter(s => s !== 'scaler'))
  } else {
    emit('update:pipelineSteps', null)
  }
}

function stepRole(step: string): string {
  return stepTypes.value[step] ?? 'unknown'
}

function stepLabel(step: string): string {
  if (stepRole(step) === 'classifier') return props.modelType || 'Model'
  const labels: Record<string, string> = { scaler: 'Scaler', csp: 'CSP', lda: 'LDA', svm: 'SVM' }
  return labels[step] ?? step.toUpperCase()
}

function stepDesc(step: string): string {
  const descs: Record<string, string> = {
    scaler: 'z-score', classifier: 'classify', csp: 'spatial filter', lda: 'linear', svm: 'nonlinear',
  }
  return descs[step] ?? ''
}

function selectModel(type: string) {
  emit('update:modelType', type)
  showPicker.value = false
}
</script>

<template>
  <div class="space-y-3">
    <!-- Model selector -->
    <div v-if="!readonly" class="relative">
      <button
        @click.stop="showPicker = !showPicker"
        class="w-full flex items-center justify-between gap-2 px-3 py-1.5 text-sm font-semibold rounded-lg border border-accent/40 bg-bg-elevated text-text-main hover:border-accent transition-colors"
      >
        {{ modelType }}
        <i class="pi pi-chevron-down text-[9px] text-text-disabled" />
      </button>
      <Teleport to="body">
        <div v-if="showPicker" class="fixed inset-0 z-40" @click="showPicker = false" />
      </Teleport>
      <div
        v-if="showPicker"
        class="absolute top-full left-0 right-0 mt-1 z-50 bg-bg-elevated border border-border rounded-lg shadow-xl py-1 max-h-[240px] overflow-y-auto"
      >
        <button
          v-for="m in models" :key="m.model_type"
          @click.stop="selectModel(m.model_type)"
          class="w-full text-left px-3 py-2 text-xs transition-colors flex items-center justify-between gap-3"
          :class="m.model_type === modelType
            ? 'bg-accent/15 text-accent font-semibold'
            : 'text-text-muted hover:text-text-main hover:bg-bg-hover'"
        >
          <div>
            <div class="font-medium">{{ m.model_type }}</div>
            <div v-if="m.description" class="text-[10px] text-text-disabled mt-0.5">{{ m.description }}</div>
          </div>
          <span class="text-[10px] text-text-disabled uppercase shrink-0">{{ m.modalities.join(', ') }}</span>
        </button>
      </div>
    </div>

    <!-- Pipeline diagram -->
    <div class="flex items-center py-1 w-full">
      <!-- Input node -->
      <div class="flex flex-col items-center shrink-0">
        <div class="w-2.5 h-2.5 rounded-full bg-text-disabled/30 border border-text-disabled/40" />
        <span class="text-[10px] text-text-disabled mt-1">Input</span>
      </div>

      <template v-for="step in displaySteps" :key="step">
        <!-- Connector -->
        <div class="flex-1 h-px bg-border min-w-3" />

        <!-- Preprocessing pill (toggleable) -->
        <button
          v-if="stepRole(step) === 'preprocessing'"
          @click.stop="toggleScaler()"
          :disabled="readonly"
          class="shrink-0 flex flex-col items-center px-3 py-1.5 rounded-lg border transition-all"
          :class="[
            scalerEnabled
              ? 'bg-accent/10 border-accent/30 text-accent'
              : 'bg-bg-input/50 border-border/50 text-text-disabled',
            readonly ? 'cursor-default' : 'cursor-pointer hover:border-accent/50',
          ]"
          :title="readonly ? stepDesc(step) : (scalerEnabled ? 'Click to disable' : 'Click to enable')"
        >
          <span class="text-xs font-medium leading-none" :class="{ 'opacity-40': !scalerEnabled }">{{ stepLabel(step) }}</span>
          <span v-if="!readonly" class="text-[9px] leading-none mt-1" :class="scalerEnabled ? 'text-accent/50' : 'text-text-disabled/50'">
            {{ scalerEnabled ? 'on' : 'off' }}
          </span>
        </button>

        <!-- Feature extraction pill -->
        <div
          v-else-if="stepRole(step) === 'features'"
          class="shrink-0 flex flex-col items-center px-3 py-1.5 rounded-lg border bg-accent/8 border-accent/25"
        >
          <span class="text-xs font-medium text-accent leading-none">{{ stepLabel(step) }}</span>
          <span class="text-[9px] text-accent/40 leading-none mt-1">{{ stepDesc(step) }}</span>
        </div>

        <!-- Classifier pill -->
        <div
          v-else-if="stepRole(step) === 'classifier'"
          class="shrink-0 flex flex-col items-center px-3 py-1.5 rounded-lg border bg-bg-elevated border-accent/40"
        >
          <span class="text-xs font-semibold text-text-main leading-none">{{ stepLabel(step) }}</span>
          <span class="text-[9px] text-text-disabled leading-none mt-1">{{ stepDesc(step) }}</span>
        </div>

        <!-- Fallback -->
        <div v-else class="shrink-0 px-3 py-1.5 rounded-lg border border-border text-xs text-text-muted">
          {{ stepLabel(step) }}
        </div>
      </template>

      <!-- Connector to output -->
      <div class="flex-1 h-px bg-border min-w-3" />

      <!-- Output node -->
      <div class="flex flex-col items-center shrink-0">
        <div class="w-2.5 h-2.5 rounded-full bg-level-ok/40 border border-level-ok/50" />
        <span class="text-[10px] text-text-disabled mt-1">Output</span>
      </div>
    </div>

    <!-- Readonly description -->
    <div v-if="readonly && currentModel?.description" class="text-[11px] text-text-disabled">
      {{ currentModel.description }}
    </div>
  </div>
</template>
