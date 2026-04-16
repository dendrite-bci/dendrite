<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useMLStore } from '../../stores/ml'
import { useToast } from '../../composables/useToast'
import { useUPlot } from '../../composables/useUPlot'
import { formatPercent } from '../../utils/format'
import { makeAxis, CURSOR_INTERACTIVE, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import ConfusionMatrix from './ConfusionMatrix.vue'
import MetricBadge from './MetricBadge.vue'
import PerClassTable from './PerClassTable.vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

const ml = useMLStore()
const toast = useToast()

function applyBestParams() {
  if (!optunaBestParams.value) return
  ml.applySearchParams(optunaBestParams.value)
  ml.trainingConfig.optuna_enabled = false
  toast.success('Best params applied to training config')
}

const lossChartEl = ref<HTMLDivElement | null>(null)
const accChartEl = ref<HTMLDivElement | null>(null)
const { create: createLossChart, setData: setLossData } = useUPlot(lossChartEl)
const { create: createAccChart, setData: setAccData } = useUPlot(accChartEl)

const currentDecoder = computed(() =>
  ml.models.find(m => m.model_type === ml.selectedJob?.model_type)
)
const pipelineSteps = computed(() => currentDecoder.value?.default_steps ?? [])
const currentStepTypes = computed(() => currentDecoder.value?.step_types ?? {})

const decoderName = ref('')
const saving = ref(false)
const saved = ref(false)

const lossLegend = [
  { label: 'Train', color: '#818cf8' },
  { label: 'Val', color: '#fbbf24' },
]
const accLegend = [
  { label: 'Train', color: '#34d399' },
  { label: 'Val', color: '#f87171' },
]

const epochHistory = ref<{
  epochs: number[]
  trainLoss: number[]
  valLoss: number[]
  trainAcc: number[]
  valAcc: number[]
}>({ epochs: [], trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] })

// Optuna trial tracking
interface OptunaTrialRow {
  trial: number
  model_type: string
  val_accuracy: number
  elapsed: number
  error?: string
}

const optunaTrials = ref<OptunaTrialRow[]>([])

const parsedResult = computed(() => {
  if (!ml.selectedJob?.result_json) return null
  try {
    return JSON.parse(ml.selectedJob.result_json)
  } catch {
    return null
  }
})

const isOptuna = computed(() =>
  parsedResult.value?.search_type === 'optuna' || progress.value?.type === 'optuna_trial'
)

const isNeural = computed(() => {
  if (!ml.selectedJob) return false
  // Check decoder registry first (most reliable)
  if (currentDecoder.value) return currentDecoder.value.default_steps.includes('classifier')
  // Fallback: check if we have epoch data (for jobs with unknown decoder)
  return epochHistory.value.epochs.length > 0 || metrics.value?.epochs_completed != null
})

const optunaResult = computed(() =>
  parsedResult.value?.search_type === 'optuna' ? parsedResult.value : null
)

const optunaBestAcc = computed(() => {
  if (optunaResult.value) return optunaResult.value.best_accuracy
  if (optunaTrials.value.length === 0) return 0
  return Math.max(...optunaTrials.value.map(t => t.val_accuracy))
})

const optunaTotalTrials = computed(() => {
  if (optunaResult.value) return optunaResult.value.n_trials
  return progress.value?.total_trials ?? 0
})

const optunaDisplayTrials = computed(() => {
  if (optunaResult.value?.trial_results) {
    return (optunaResult.value.trial_results as OptunaTrialRow[]).filter(t => !t.error)
  }
  return optunaTrials.value
})

const progress = computed(() => {
  if (!ml.selectedJob) return null
  return ml.trainingProgress[ml.selectedJob.job_id] ?? ml.selectedJob.progress ?? null
})

const metrics = computed(() => {
  const tm = parsedResult.value?.training_metrics
  if (!tm || typeof tm !== 'object') return null
  const entries = Object.values(tm) as Record<string, any>[]
  return entries[0] ?? null
})

const parsedConfig = computed(() => {
  if (!ml.selectedJob?.config_json) return null
  try {
    return JSON.parse(ml.selectedJob.config_json)
  } catch {
    return null
  }
})

const preprocSummary = computed(() => {
  const c = parsedConfig.value
  if (!c) return null
  const parts: string[] = []
  if (c.lowcut || c.highcut) {
    const lo = c.lowcut ?? '\u2014'
    const hi = c.highcut ?? '\u2014'
    parts.push(`${lo}\u2013${hi} Hz`)
  }
  if (c.apply_rereferencing) parts.push('CAR')
  if (c.epoch_tmin != null || c.epoch_tmax != null)
    parts.push(`epoch ${c.epoch_tmin ?? 0}\u2013${c.epoch_tmax ?? 1}s`)
  if (c.use_epoch_qc === false) parts.push('QC off')
  if (c.include_background) parts.push('+ rest')
  if (c.selected_events?.length)
    parts.push(c.selected_events.join(', '))
  return parts.length > 0 ? parts.join(' \u00b7 ') : null
})

// --- Eval metrics (confusion matrix, per-class metrics) ---
const evalData = computed(() => parsedResult.value?.eval_metrics ?? null)

const confusionMatrix = computed((): number[][] | null =>
  evalData.value?.confusion_matrix ?? null
)

const classLabels = computed((): string[] =>
  evalData.value?.class_labels ?? []
)


interface PerClassRow { name: string; precision: number; recall: number; f1: number; support: number }

const perClassMetrics = computed((): PerClassRow[] => {
  const report = evalData.value?.classification_report
  if (!report) return []
  return classLabels.value.map((label, i) => {
    const entry = report[String(i)] ?? {}
    return {
      name: label,
      precision: entry.precision ?? 0,
      recall: entry.recall ?? 0,
      f1: entry['f1-score'] ?? 0,
      support: entry.support ?? 0,
    }
  })
})

const metricCards = computed(() => {
  const cards: { label: string; value: string }[] = []
  const r = parsedResult.value
  if (!r) return cards

  if (metrics.value?.final_train_acc != null)
    cards.push({ label: 'Train Acc', value: formatPercent(metrics.value.final_train_acc) })
  if (metrics.value?.final_val_acc != null)
    cards.push({ label: 'Val Acc', value: formatPercent(metrics.value.final_val_acc) })
  else if (evalData.value?.classification_report?.accuracy != null)
    cards.push({ label: 'Val Acc', value: formatPercent(evalData.value.classification_report.accuracy) })

  const macroF1 = evalData.value?.classification_report?.['macro avg']?.['f1-score']
  if (macroF1 != null)
    cards.push({ label: 'Macro F1', value: formatPercent(macroF1) })

  if (metrics.value?.epochs_completed != null)
    cards.push({ label: 'Epochs', value: String(metrics.value.epochs_completed) })
  if (r.n_epochs != null)
    cards.push({ label: 'Samples', value: String(r.n_epochs) })
  if (r.elapsed != null)
    cards.push({ label: 'Time', value: `${r.elapsed.toFixed(1)}s` })

  // Fallback for classical models without standard metrics
  if (cards.length === 0 && r.training_metrics && typeof r.training_metrics === 'object') {
    for (const [key, val] of Object.entries(r.training_metrics)) {
      if (typeof val === 'number')
        cards.push({ label: key.replace(/_/g, ' '), value: val <= 1 ? formatPercent(val) : String(val) })
    }
  }

  return cards
})

// --- Early stopping tracking (computed from epoch history) ---
const earlyStopping = computed(() => {
  const h = epochHistory.value
  if (h.valLoss.length === 0) return null
  const bestIdx = h.valLoss.reduce((minIdx, v, i, arr) =>
    v < arr[minIdx]! ? i : minIdx, 0)
  return {
    bestEpoch: h.epochs[bestIdx]!,
    bestValLoss: h.valLoss[bestIdx]!,
    sinceImprovement: h.epochs.length - 1 - bestIdx,
    patience: parsedConfig.value?.early_stopping_patience ?? 10,
    enabled: parsedConfig.value?.use_early_stopping !== false,
  }
})

// --- Optuna convergence data (running best accuracy) ---
const optunaConvergence = computed(() => {
  const trials = optunaDisplayTrials.value
  if (trials.length === 0) return { x: [] as number[], y: [] as number[] }
  let best = 0
  const x: number[] = []
  const y: number[] = []
  for (const t of trials) {
    best = Math.max(best, t.val_accuracy)
    x.push(t.trial)
    y.push(best)
  }
  return { x, y }
})

// --- Optuna best params ---
const optunaBestParams = computed(() => {
  if (!optunaResult.value?.best_params) return null
  return optunaResult.value.best_params as Record<string, any>
})

const optunaChartEl = ref<HTMLDivElement | null>(null)
const { create: createOptunaChart_, setData: setOptunaData, destroy: destroyOptunaChart } = useUPlot(optunaChartEl)
const optunaChartActive = ref(false)

function createOptunaChart() {
  optunaChartActive.value = true
  createOptunaChart_(({ width }) => ({
    width, height: 180,
    cursor: CURSOR_INTERACTIVE,
    legend: LEGEND_HIDDEN,
    scales: { x: { time: false }, y: { range: [0, 1.05] } },
    axes: [
      makeAxis({ label: 'Trial', size: 28 }),
      makeAxis({ label: 'Best Acc', size: 40 }),
    ],
    series: [
      { label: 'Trial' },
      { label: 'Best Acc', stroke: '#34d399', width: 2, fill: 'rgba(52, 211, 153, 0.15)',
        paths: uPlot.paths.stepped!({ align: 1 }) },
    ],
  } as uPlot.Options), [[], []])
}

function updateOptunaChart() {
  const c = optunaConvergence.value
  if (c.x.length > 0) setOptunaData([c.x, c.y])
}

function createCharts() {
  const baseOpts = (width: number): Partial<uPlot.Options> => ({
    width,
    height: 250,
    cursor: CURSOR_INTERACTIVE,
    legend: LEGEND_HIDDEN,
    scales: { x: { time: false } },
    axes: [
      makeAxis({ label: 'Epoch', size: 28 }),
      makeAxis({ size: 45 }),
    ],
  })

  const areaFill = (color: string) => `${color}18`

  createLossChart(({ width }) => ({
    ...baseOpts(width),
    series: [
      { label: 'Epoch' },
      { label: 'Train', stroke: '#818cf8', width: 2, fill: areaFill('#818cf8') },
      { label: 'Val', stroke: '#fbbf24', width: 2, dash: [5, 3], fill: areaFill('#fbbf24') },
    ],
  } as uPlot.Options), [[], [], []])

  createAccChart(({ width }) => ({
    ...baseOpts(width),
    series: [
      { label: 'Epoch' },
      { label: 'Train', stroke: '#34d399', width: 2, fill: areaFill('#34d399') },
      { label: 'Val', stroke: '#f87171', width: 2, dash: [5, 3], fill: areaFill('#f87171') },
    ],
  } as uPlot.Options), [[], [], []])
}

function updateCharts() {
  const h = epochHistory.value
  setLossData([h.epochs, h.trainLoss, h.valLoss])
  setAccData([h.epochs, h.trainAcc, h.valAcc])
}

watch(progress, (p) => {
  if (!p) return

  if (p.type === 'epoch' && p.epoch) {
    const h = epochHistory.value
    if (h.epochs.length === 0 || p.epoch > h.epochs[h.epochs.length - 1]!) {
      h.epochs.push(p.epoch)
      h.trainLoss.push(p.train_loss ?? 0)
      h.valLoss.push(p.val_loss ?? 0)
      h.trainAcc.push(p.train_acc ?? 0)
      h.valAcc.push(p.val_acc ?? 0)
      updateCharts()
    }
  }

  if (p.type === 'optuna_trial' && p.trial) {
    const existing = optunaTrials.value.find(t => t.trial === p.trial)
    if (!existing) {
      optunaTrials.value.push({
        trial: p.trial,
        model_type: p.model_type ?? '?',
        val_accuracy: p.val_accuracy ?? 0,
        elapsed: p.elapsed_seconds ?? 0,
      })
      nextTick(() => {
        if (!optunaChartActive.value) createOptunaChart()
        updateOptunaChart()
      })
    }
  }
}, { deep: true })

watch(() => ml.selectedJob?.job_id, async () => {
  epochHistory.value = { epochs: [], trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] }
  optunaTrials.value = []
  saved.value = false
  destroyOptunaChart()
  optunaChartActive.value = false

  // Restore epoch history from saved result_json for completed jobs
  if (ml.selectedJob?.status === 'completed' && parsedResult.value) {
    const tm = parsedResult.value.training_metrics
    if (tm && typeof tm === 'object') {
      const comp = (Object.values(tm) as any[]).find(c => c?.history)
      if (comp) {
        const h = comp.history
        const n = h.loss?.length ?? h.val_loss?.length ?? 0
        if (n > 0) {
          epochHistory.value = {
            epochs: Array.from({ length: n }, (_, i) => i + 1),
            trainLoss: h.loss ?? [],
            valLoss: h.val_loss ?? [],
            trainAcc: h.accuracy ?? [],
            valAcc: h.val_accuracy ?? [],
          }
        }
      }
    }
  }

  await nextTick()
  if (isOptuna.value) {
    createOptunaChart()
    updateOptunaChart()
  } else {
    createCharts()
    if (epochHistory.value.epochs.length > 0) updateCharts()
  }
}, { immediate: true })

async function handleSave() {
  if (!ml.selectedJob || !decoderName.value.trim()) return
  saving.value = true
  const res = await ml.saveDecoder(ml.selectedJob.job_id, decoderName.value.trim())
  saving.value = false
  if (res) saved.value = true
}

</script>

<template>
  <div v-if="ml.selectedJob">

    <!-- Header -->
    <div class="flex items-center gap-3 mb-3">
      <h2 class="text-lg font-semibold text-text-main">{{ ml.selectedJob.model_type }}</h2>
      <span v-if="pipelineSteps.length" class="flex items-center gap-1 text-xs">
        <template v-for="(step, i) in pipelineSteps" :key="step">
          <span v-if="i > 0" class="text-text-disabled">&rarr;</span>
          <span class="px-1.5 py-0.5 rounded" :class="currentStepTypes[step] === 'classifier'
            ? 'bg-accent/15 text-accent font-semibold'
            : 'bg-white/[0.04] text-text-muted'">{{ step }}</span>
        </template>
      </span>
      <span class="text-sm text-text-disabled">#{{ ml.selectedJob.job_id }}</span>
      <span v-if="preprocSummary" class="text-xs text-text-disabled">{{ preprocSummary }}</span>
      <span class="flex-1" />
      <!-- Running: progress + cancel -->
      <template v-if="ml.selectedJob.status === 'running'">
        <span v-if="progress?.epoch" class="text-xs text-text-muted font-mono">
          {{ progress.epoch }}/{{ progress.total_epochs }}
        </span>
        <span v-if="progress?.val_acc != null" class="text-xs text-accent font-mono">
          {{ (progress.val_acc * 100).toFixed(1) }}%
        </span>
        <button @click="ml.cancelTraining(ml.selectedJob!.job_id)"
          class="px-3 py-1 text-xs font-medium rounded-lg bg-status-error/10 text-status-error hover:bg-status-error/20 transition-colors"
        >Cancel</button>
      </template>
      <!-- Completed: save -->
      <template v-if="ml.selectedJob.status === 'completed' && ml.selectedJob.job_type === 'training'">
        <template v-if="!ml.selectedJob.decoder_id && !saved">
          <input v-model="decoderName" placeholder="Decoder name" class="w-36 text-xs px-2 py-1 rounded" />
          <button @click="handleSave" :disabled="!decoderName.trim() || saving"
            class="px-3 py-1 text-xs font-medium rounded-md transition-colors shrink-0"
            :class="decoderName.trim() && !saving ? 'bg-accent text-white hover:bg-accent/80' : 'bg-bg-input text-text-disabled cursor-not-allowed'"
          >{{ saving ? 'Saving...' : 'Save' }}</button>
        </template>
        <span v-else-if="saved || ml.selectedJob.decoder_id" class="text-xs text-status-ok font-semibold">
          <i class="pi pi-check-circle mr-1" />Saved
        </span>
      </template>
    </div>

    <!-- Progress (running) -->
    <div v-if="ml.selectedJob.status === 'running'" class="mb-3">
      <template v-if="!isOptuna && progress?.epoch">
        <div class="w-full h-2 bg-bg-input rounded-full overflow-hidden">
          <div class="h-full bg-accent rounded-full transition-all duration-300"
            :style="{ width: `${((progress.epoch ?? 0) / (progress.total_epochs ?? 1)) * 100}%` }" />
        </div>
        <div v-if="earlyStopping?.enabled && earlyStopping.sinceImprovement > 0" class="mt-1 text-xs"
          :class="earlyStopping.sinceImprovement >= earlyStopping.patience * 0.7 ? 'text-status-warn' : 'text-text-disabled'">
          Best at epoch {{ earlyStopping.bestEpoch }} ({{ earlyStopping.sinceImprovement }}/{{ earlyStopping.patience }} patience)
        </div>
      </template>
      <template v-else-if="isOptuna && progress?.trial">
        <div class="flex items-center justify-between text-xs text-text-muted mb-1">
          <span>Trial {{ progress.trial }} / {{ optunaTotalTrials }}</span>
          <span v-if="progress.best_accuracy">Best: {{ (progress.best_accuracy * 100).toFixed(1) }}%</span>
        </div>
        <div class="w-full h-2 bg-bg-input rounded-full overflow-hidden">
          <div class="h-full bg-accent rounded-full transition-all duration-300"
            :style="{ width: `${((progress.trial ?? 0) / (optunaTotalTrials || 1)) * 100}%` }" />
        </div>
      </template>
      <div v-else class="flex items-center gap-2 text-xs text-text-muted">
        <i class="pi pi-spin pi-spinner text-accent" />
        <span>Training in progress...</span>
      </div>
    </div>

    <!-- Error -->
    <div v-if="ml.selectedJob.status === 'failed' && ml.selectedJob.error_message"
      class="mb-3 bg-status-error/10 border border-status-error/30 rounded-lg px-4 py-3">
      <h3 class="text-sm font-semibold text-status-error mb-1">Training Failed</h3>
      <p class="text-xs text-text-muted font-mono">{{ ml.selectedJob.error_message }}</p>
    </div>

    <!-- Results panel -->
    <div class="rounded-lg border border-border overflow-hidden">

      <!-- Metrics (completed, non-optuna) -->
      <div v-if="ml.selectedJob.status === 'completed' && !isOptuna && metricCards.length > 0"
        class="px-5 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
        <template v-for="(card, i) in metricCards" :key="card.label">
          <span v-if="i > 0" class="text-border">·</span>
          <MetricBadge :label="card.label" :value="card.value" />
        </template>
      </div>

      <!-- Optuna summary (completed) -->
      <div v-if="ml.selectedJob.status === 'completed' && isOptuna && optunaResult"
        class="px-5 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span class="text-xs"><span class="text-text-disabled">Best Accuracy</span> <span class="ml-1 font-semibold text-accent">{{ (optunaResult.best_accuracy * 100).toFixed(1) }}%</span></span>
        <span class="text-border">·</span>
        <span class="text-xs"><span class="text-text-disabled">Best Model</span> <span class="ml-1 font-semibold text-text-main">{{ optunaResult.best_model_type ?? 'N/A' }}</span></span>
        <span class="text-border">·</span>
        <span class="text-xs"><span class="text-text-disabled">Trials</span> <span class="ml-1 font-semibold text-text-main">{{ optunaResult.n_trials }}</span></span>
      </div>

      <!-- Neural charts (side-by-side) -->
      <div v-if="!isOptuna && isNeural" class="px-5 py-4 border-t border-border/30">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <div class="flex items-center gap-3 mb-1.5">
              <h3 class="text-xs font-semibold text-text-muted">Loss</h3>
              <div class="flex gap-1.5">
                <span v-for="s in lossLegend" :key="s.label"
                  class="flex items-center gap-1 px-1.5 py-0.5 rounded-full text-xs text-text-muted"
                  :style="{ backgroundColor: s.color + '25' }">
                  <span class="w-1.5 h-1.5 rounded-full" :style="{ backgroundColor: s.color }" />
                  {{ s.label }}
                </span>
              </div>
            </div>
            <div ref="lossChartEl" class="bg-bg-elevated rounded-lg p-2 border border-border/50 min-h-[200px]" />
          </div>
          <div>
            <div class="flex items-center gap-3 mb-1.5">
              <h3 class="text-xs font-semibold text-text-muted">Accuracy</h3>
              <div class="flex gap-1.5">
                <span v-for="s in accLegend" :key="s.label"
                  class="flex items-center gap-1 px-1.5 py-0.5 rounded-full text-xs text-text-muted"
                  :style="{ backgroundColor: s.color + '25' }">
                  <span class="w-1.5 h-1.5 rounded-full" :style="{ backgroundColor: s.color }" />
                  {{ s.label }}
                </span>
              </div>
            </div>
            <div ref="accChartEl" class="bg-bg-elevated rounded-lg p-2 border border-border/50 min-h-[200px]" />
          </div>
        </div>
      </div>

      <!-- Eval results (completed, non-optuna — unified for neural + classical) -->
      <div v-if="ml.selectedJob.status === 'completed' && !isOptuna && (confusionMatrix || perClassMetrics.length > 0)"
        class="px-5 py-4" :class="isNeural ? 'border-t border-border/30' : ''">
        <div class="flex gap-4 flex-wrap">
          <div v-if="confusionMatrix" class="min-w-[200px]">
            <h3 class="text-xs font-semibold text-text-muted mb-1.5">Confusion Matrix</h3>
            <ConfusionMatrix :matrix="confusionMatrix" :class-labels="classLabels" />
          </div>
          <div v-if="perClassMetrics.length > 0" class="flex-1 min-w-[280px]">
            <h3 class="text-xs font-semibold text-text-muted mb-1.5">Per-Class Metrics</h3>
            <PerClassTable :metrics="perClassMetrics" />
          </div>
        </div>
      </div>

      <!-- Optuna convergence chart -->
      <div v-if="isOptuna && optunaDisplayTrials.length > 1" class="px-5 py-4 border-t border-border/30">
        <h3 class="text-xs font-semibold text-text-muted mb-1.5">Convergence</h3>
        <div ref="optunaChartEl" class="bg-bg-elevated rounded-lg p-2 border border-border/50 min-h-[180px]" />
      </div>

      <!-- Optuna trial table -->
      <div v-if="isOptuna && optunaDisplayTrials.length > 0"
        class="px-5 py-4" :class="optunaDisplayTrials.length > 1 ? 'border-t border-border/30' : ''">
        <h3 class="text-xs font-semibold text-text-muted mb-2">Trial Results</h3>
        <div class="bg-bg-elevated rounded-lg border border-border/50 overflow-hidden">
          <div class="max-h-[320px] overflow-y-auto">
            <table class="w-full text-xs">
              <thead class="sticky top-0 bg-bg-elevated">
                <tr class="text-text-disabled text-xs uppercase">
                  <th class="text-left px-3 py-2 font-medium">#</th>
                  <th class="text-left px-3 py-2 font-medium">Model</th>
                  <th class="text-right px-3 py-2 font-medium">Val Accuracy</th>
                  <th class="text-right px-3 py-2 font-medium">Elapsed</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="t in optunaDisplayTrials"
                  :key="t.trial"
                  class="border-t border-border/30"
                  :class="t.val_accuracy === optunaBestAcc ? 'bg-accent/5' : ''"
                >
                  <td class="px-3 py-1.5 font-mono text-text-muted">{{ t.trial }}</td>
                  <td class="px-3 py-1.5 text-text-main">{{ t.model_type }}</td>
                  <td class="px-3 py-1.5 text-right font-mono" :class="t.val_accuracy === optunaBestAcc ? 'text-accent font-semibold' : 'text-text-main'">
                    {{ (t.val_accuracy * 100).toFixed(1) }}%
                  </td>
                  <td class="px-3 py-1.5 text-right text-text-disabled font-mono">{{ t.elapsed.toFixed(0) }}s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Optuna best trial params -->
      <div v-if="isOptuna && optunaBestParams && ml.selectedJob?.status === 'completed'"
        class="px-5 py-4 border-t border-border/30">
        <div class="flex items-center gap-2 mb-2">
          <h3 class="text-xs font-semibold text-text-muted">Best Trial Config</h3>
          <button @click="applyBestParams" class="text-[11px] text-accent hover:underline">Apply to config</button>
        </div>
        <div class="bg-bg-elevated rounded-lg border border-border/50 px-3 py-2">
          <div class="flex flex-wrap gap-x-4 gap-y-1">
            <span v-for="(val, key) in optunaBestParams" :key="key" class="text-xs">
              <span class="text-text-disabled">{{ key }}:</span>
              <span class="text-text-main font-mono ml-1">{{ typeof val === 'number' ? (val < 0.01 ? val.toExponential(1) : Number(val.toFixed(4))) : val }}</span>
            </span>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>
