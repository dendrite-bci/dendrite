<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useMLStore } from '../../stores/ml'
import { useToast } from '../../composables/useToast'
import NumberInput from '../common/NumberInput.vue'
import ToggleSwitch from '../common/ToggleSwitch.vue'

const ml = useMLStore()
const toast = useToast()

const showAdvanced = ref(false)

// --- Decoder ---
const currentDecoder = computed(() =>
  ml.models.find(m => m.model_type === ml.trainingConfig.model_type)
)
const pipelineSteps = computed(() =>
  currentDecoder.value?.default_steps ?? []
)
const stepTypes = computed(() =>
  currentDecoder.value?.step_types ?? {}
)
const isNeural = computed(() => pipelineSteps.value.includes('classifier'))

// --- Model schema for architecture params ---
interface SchemaProperty {
  key: string
  title: string
  type: string
  default: any
  minimum?: number
  maximum?: number
  step?: number
  choices?: any[]
  description?: string
}

const modelArchParams = computed((): SchemaProperty[] => {
  const schema = ml.modelSchema
  if (!schema?.properties) return []
  const props: SchemaProperty[] = []
  for (const [key, def] of Object.entries(schema.properties) as [string, any][]) {
    const hpo = def.json_schema_extra?.hpo ?? def.hpo
    if (!hpo) continue
    if (def.description?.toLowerCase().includes('derived')) continue
    const prop: SchemaProperty = {
      key,
      title: def.title || key,
      type: hpo.type || def.type || 'float',
      default: def.default,
      description: def.description,
    }
    if (hpo.type === 'categorical' && hpo.choices) {
      prop.choices = hpo.choices
    } else {
      prop.minimum = hpo.low ?? def.ge ?? def.minimum
      prop.maximum = hpo.high ?? def.le ?? def.maximum
      prop.step = hpo.step ?? (hpo.type === 'int' ? 1 : 0.01)
    }
    props.push(prop)
  }
  return props
})

function getModelParam(key: string, defaultVal: any): any {
  return ml.trainingConfig.model_params[key] ?? defaultVal
}

function setModelParam(key: string, value: any) {
  ml.trainingConfig.model_params = { ...ml.trainingConfig.model_params, [key]: value }
}

const availableCategories = computed(() => Object.keys(ml.searchCategories))

const searchParamCount = computed(() => {
  let count = 0
  for (const cat of ml.trainingConfig.search_categories) {
    count += (ml.searchCategories[cat]?.params.length ?? 0)
  }
  return count
})

function toggleCategory(cat: string) {
  const cats = ml.trainingConfig.search_categories
  const idx = cats.indexOf(cat)
  if (idx >= 0) {
    if (cats.length > 1) cats.splice(idx, 1)
  } else {
    cats.push(cat)
  }
}


watch(() => ml.trainingConfig.model_type, (newType) => {
  ml.trainingConfig.model_params = {}
  ml.fetchModelSchema(newType)
  ml.fetchSearchCategories(newType)
})

async function handleStartTraining() {
  ml.trainingConfig.use_loaded_data = true
  const result = await ml.startTraining()
  if (result) toast.success('Training started')
}

const schedulerTypes = [
  { value: 'OneCycleLR', label: 'OneCycle' },
  { value: 'ReduceLROnPlateau', label: 'ReduceOnPlateau' },
  { value: 'CosineAnnealingLR', label: 'CosineAnnealing' },
  { value: 'StepLR', label: 'StepLR' },
]

onMounted(() => {
  ml.fetchModelSchema(ml.trainingConfig.model_type)
  ml.fetchSearchCategories(ml.trainingConfig.model_type)
})
</script>

<template>
  <div>
    <!-- No data warning -->
    <div v-if="!ml.loadedData" class="mb-2 bg-status-error/10 rounded px-3 py-1.5">
      <p class="text-xs text-status-error">No data loaded — load from the left panel first.</p>
    </div>

    <!-- Row 1: Decoder + pipeline + core params -->
    <div class="flex items-end gap-2.5 flex-wrap">
      <div>
        <label class="text-[11px] text-text-muted block mb-0.5">Decoder</label>
        <div class="flex items-center gap-2">
          <select v-model="ml.trainingConfig.model_type" class="w-[140px] text-xs">
            <option v-for="m in ml.models" :key="m.model_type" :value="m.model_type">{{ m.model_type }}</option>
          </select>
          <div class="flex items-center gap-1 text-[11px]">
            <template v-for="(step, i) in pipelineSteps" :key="step">
              <span v-if="i > 0" class="text-text-disabled">→</span>
              <span
                class="px-1.5 py-0.5 rounded"
                :class="stepTypes[step] === 'classifier' ? 'bg-accent/15 text-accent font-semibold' : 'bg-text-main/[0.04] text-text-muted'"
              >{{ step }}</span>
            </template>
          </div>
        </div>
      </div>
      <template v-if="isNeural">
        <div class="w-[80px]">
          <label class="text-[11px] text-text-muted block mb-0.5">Epochs</label>
          <NumberInput v-model="ml.trainingConfig.epochs" :min="1" :max="1000" class="w-full" />
        </div>
        <div class="w-[80px]">
          <label class="text-[11px] text-text-muted block mb-0.5">Batch</label>
          <NumberInput v-model="ml.trainingConfig.batch_size" :min="1" :max="512" class="w-full" />
        </div>
        <div class="w-[96px]">
          <label class="text-[11px] text-text-muted block mb-0.5">Learning Rate</label>
          <NumberInput v-model="ml.trainingConfig.learning_rate" :min="0.0001" :max="1" :step="0.0001" class="w-full" />
        </div>
        <div class="w-[72px]">
          <label class="text-[11px] text-text-muted block mb-0.5">Val Split</label>
          <NumberInput v-model="ml.trainingConfig.validation_split" :min="0" :max="0.5" :step="0.05" class="w-full" />
        </div>
      </template>

      <div class="flex items-center gap-1.5 self-center mt-3">
        <template v-if="isNeural">
          <button
            @click="ml.trainingConfig.optuna_enabled = !ml.trainingConfig.optuna_enabled"
            class="px-2.5 py-1.5 text-xs font-medium rounded border transition-colors"
            :class="ml.trainingConfig.optuna_enabled
              ? 'bg-accent/15 text-accent border-accent/40'
              : 'bg-bg-input text-text-disabled border-border hover:text-text-muted'"
          >Optuna</button>
          <button
            @click="showAdvanced = !showAdvanced"
            class="px-2.5 py-1.5 text-xs text-text-muted hover:text-text-main transition-colors rounded border border-border/50 hover:border-border"
          >
            <i class="pi text-[10px] mr-0.5" :class="showAdvanced ? 'pi-chevron-up' : 'pi-chevron-down'" />
            More
          </button>
        </template>
      </div>
    </div>

    <!-- Row 2: Optuna config (conditional, neural only) -->
    <div v-if="isNeural && ml.trainingConfig.optuna_enabled" class="flex items-center gap-2 mt-2 flex-wrap">
      <!-- Trial count -->
      <div class="flex items-center gap-1.5">
        <label class="text-[11px] text-text-muted">Trials</label>
        <NumberInput v-model="ml.trainingConfig.optuna_n_trials" :min="5" :max="200" :step="5" class="w-14" />
      </div>

      <!-- Category toggles -->
      <div class="flex items-center gap-1">
        <button
          v-for="cat in availableCategories" :key="cat"
          @click="toggleCategory(cat)"
          class="text-[11px] px-2 py-0.5 rounded transition-colors capitalize"
          :class="ml.trainingConfig.search_categories.includes(cat)
            ? 'bg-accent/20 text-accent'
            : 'bg-bg-input text-text-disabled hover:text-text-muted'"
        >{{ ml.searchCategories[cat]?.label ?? cat }}</button>
      </div>

      <span class="text-[11px] text-text-disabled ml-auto">{{ searchParamCount }} params</span>
    </div>

    <!-- Advanced panel (neural only) -->
    <div v-if="isNeural && showAdvanced" class="mt-2 pt-2 border-t border-border/30">
      <div class="grid grid-cols-3 gap-x-5 gap-y-2">
        <!-- Col 1: Training -->
        <div class="space-y-2">
          <h5 class="text-[11px] text-text-disabled font-medium uppercase tracking-wide">Training</h5>
          <div>
            <label class="text-[11px] text-text-muted block mb-0.5">Optimizer</label>
            <select v-model="ml.trainingConfig.optimizer_type" class="w-full text-xs">
              <option value="Adam">Adam</option>
              <option value="AdamW">AdamW</option>
            </select>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-xs text-text-muted">LR Scheduler</span>
            <ToggleSwitch v-model="ml.trainingConfig.use_lr_scheduler" compact />
          </div>
          <div v-if="ml.trainingConfig.use_lr_scheduler">
            <select v-model="ml.trainingConfig.lr_scheduler_type" class="w-full text-xs">
              <option v-for="s in schedulerTypes" :key="s.value" :value="s.value">{{ s.label }}</option>
            </select>
          </div>
        </div>

        <!-- Col 2: Regularization -->
        <div class="space-y-2">
          <h5 class="text-[11px] text-text-disabled font-medium uppercase tracking-wide">Regularization</h5>
          <div class="flex items-center justify-between">
            <span class="text-xs text-text-muted">Early Stopping</span>
            <ToggleSwitch v-model="ml.trainingConfig.use_early_stopping" compact />
          </div>
          <div v-if="ml.trainingConfig.use_early_stopping" class="flex items-center gap-2">
            <label class="text-[11px] text-text-muted shrink-0">Patience</label>
            <NumberInput v-model="ml.trainingConfig.early_stopping_patience" :min="1" :max="100" compact class="flex-1" />
          </div>
          <div class="flex items-center gap-2">
            <label class="text-[11px] text-text-muted shrink-0">Weight Decay</label>
            <NumberInput v-model="ml.trainingConfig.weight_decay" :min="0" :max="0.1" :step="0.001" compact class="flex-1" />
          </div>
          <div class="flex items-center gap-2">
            <label class="text-[11px] text-text-muted shrink-0">Label Smooth</label>
            <NumberInput v-model="ml.trainingConfig.label_smoothing_factor" :min="0" :max="0.3" :step="0.01" compact class="flex-1" />
          </div>
        </div>

        <!-- Col 3: Augmentation & Loss -->
        <div class="space-y-2">
          <h5 class="text-[11px] text-text-disabled font-medium uppercase tracking-wide">Aug & Loss</h5>
          <div>
            <label class="text-[11px] text-text-muted block mb-0.5">Loss</label>
            <select v-model="ml.trainingConfig.loss_type" class="w-full text-xs">
              <option value="cross_entropy">Cross Entropy</option>
              <option value="focal">Focal</option>
            </select>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-xs text-text-muted">Augmentation</span>
            <ToggleSwitch v-model="ml.trainingConfig.use_augmentation" compact />
          </div>
          <div v-if="ml.trainingConfig.use_augmentation">
            <select v-model="ml.trainingConfig.aug_strategy" class="w-full text-xs">
              <option value="light">Light</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-xs text-text-muted">Class Weights</span>
            <ToggleSwitch v-model="ml.trainingConfig.use_class_weights" compact />
          </div>
          <div class="flex items-center gap-2">
            <label class="text-[11px] text-text-muted shrink-0">Mixup α</label>
            <NumberInput v-model="ml.trainingConfig.mixup_alpha" :min="0" :max="1" :step="0.1" compact class="flex-1" />
          </div>
        </div>
      </div>

      <!-- Model Architecture Params -->
      <div v-if="modelArchParams.length > 0" class="mt-3 pt-2 border-t border-border/30">
        <h5 class="text-[11px] text-text-disabled font-medium uppercase tracking-wide mb-1.5">
          {{ ml.trainingConfig.model_type }} Architecture
        </h5>
        <div class="grid grid-cols-4 gap-x-3 gap-y-1.5">
          <div v-for="param in modelArchParams" :key="param.key">
            <label class="text-[11px] text-text-muted block mb-0.5" :title="param.description">
              {{ param.title }}
            </label>
            <select
              v-if="param.choices"
              :value="getModelParam(param.key, param.default)"
              @change="setModelParam(param.key, ($event.target as HTMLSelectElement).value)"
              class="w-full text-xs"
            >
              <option v-for="c in param.choices" :key="c" :value="c">{{ c }}</option>
            </select>
            <NumberInput
              v-else
              :modelValue="getModelParam(param.key, param.default)"
              @update:modelValue="setModelParam(param.key, $event)"
              :min="param.minimum"
              :max="param.maximum"
              :step="param.step"
              compact
              class="w-full"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Action -->
    <div class="mt-2.5 pt-2.5 border-t border-border/30 flex items-center gap-2">
      <button
        @click="handleStartTraining"
        :disabled="!ml.loadedData || ml.loading"
        class="px-3.5 py-1.5 text-xs font-semibold rounded transition-colors"
        :class="ml.loadedData && !ml.loading
          ? 'bg-accent text-white hover:bg-accent/80'
          : 'bg-bg-input text-text-disabled cursor-not-allowed'"
      >
        <i v-if="ml.loading" class="pi pi-spin pi-spinner mr-1" />
        {{ ml.loading ? 'Training...' : 'Train' }}
      </button>
      <span v-if="!ml.loadedData" class="text-[11px] text-text-disabled">Load data first</span>
    </div>
  </div>
</template>
