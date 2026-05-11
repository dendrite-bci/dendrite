<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDataStore } from '../../stores/data'
import { formatDate, formatPercent } from '../../utils/format'
import { predClassColor } from '../../utils/colors'

const data = useDataStore()
const showChannels = ref(false)

const dec = computed(() => data.selectedDecoder)
const meta = computed(() => data.decoderMetadata)

// --- Derived ---

const pipelineSteps = computed(() => meta.value?.pipeline_steps ?? [])

const inputSpec = computed(() => {
  const m = meta.value
  if (!m) return null
  const shapes = m.input_shapes
  const firstShape = shapes ? Object.values(shapes)[0] : null
  const nCh = firstShape?.[0]
  const nPts = firstShape?.[1]
  const windowSec = (nPts && m.sample_rate) ? (nPts / m.sample_rate).toFixed(2) : null
  return { nCh, nPts, windowSec, sampleRate: m.sample_rate, modality: m.modality }
})

const epochWindow = computed(() => {
  const t0 = meta.value?.epoch_tmin
  const t1 = meta.value?.epoch_tmax
  if (t0 == null && t1 == null) return null
  return `${t0 ?? 0} – ${t1 ?? '?'}s`
})

const classEntries = computed(() => {
  const em = meta.value?.event_mapping
  const lm = meta.value?.label_mapping
  if (!em) return []
  return Object.entries(em)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([code, name]) => ({ code: Number(code), name, index: lm?.[name] ?? null }))
})

const channelLabels = computed(() => {
  const cl = meta.value?.channel_labels
  if (!cl) return []
  return Object.values(cl).flat()
})

const preprocParts = computed(() => {
  const pp = meta.value?.preprocessing_config?.modality_preprocessing
  if (!pp) return []
  const first = Object.values(pp)[0]
  if (!first) return []
  const parts: string[] = []
  if (first.lowcut || first.highcut) parts.push(`${first.lowcut ?? '—'}–${first.highcut ?? '—'} Hz`)
  if (first.apply_rereferencing) parts.push('CAR')
  return parts
})

// Training config — grouped
const coreParams = computed(() => {
  const m = meta.value
  if (!m) return []
  const p: [string, string][] = []
  if (m.optimizer_type) p.push(['Optimizer', m.optimizer_type])
  if (m.learning_rate) p.push(['Learning rate', String(m.learning_rate)])
  if (m.batch_size) p.push(['Batch size', String(m.batch_size)])
  if (m.epochs) p.push(['Epochs', String(m.epochs)])
  if (m.validation_split) p.push(['Val split', `${(m.validation_split * 100).toFixed(0)}%`])
  if (m.loss_type && m.loss_type !== 'cross_entropy') p.push(['Loss', m.loss_type])
  if (m.seed != null) p.push(['Seed', String(m.seed)])
  return p
})

const regularizationParams = computed(() => {
  const m = meta.value
  if (!m) return []
  const p: [string, string][] = []
  if (m.use_early_stopping) p.push(['Early stopping', `patience ${m.early_stopping_patience ?? 10}`])
  if (m.weight_decay) p.push(['Weight decay', String(m.weight_decay)])
  if (m.label_smoothing_factor) p.push(['Label smoothing', String(m.label_smoothing_factor)])
  if (m.use_class_weights) p.push(['Class weights', 'balanced'])
  if (m.use_augmentation) p.push(['Augmentation', m.aug_strategy ?? 'moderate'])
  if (m.mixup_alpha && m.mixup_alpha > 0) p.push(['Mixup α', String(m.mixup_alpha)])
  if (m.use_lr_scheduler) p.push(['LR scheduler', m.lr_scheduler_type ?? 'OneCycleLR'])
  if (m.use_swa) p.push(['SWA', `from ${((m.swa_start_epoch ?? 0.75) * 100).toFixed(0)}%`])
  return p
})

const modelParams = computed(() => {
  const mp = meta.value?.model_params
  if (!mp || Object.keys(mp).length === 0) return []
  return Object.entries(mp).map(([k, v]) => [
    k,
    typeof v === 'number' ? (v < 0.01 && v > 0 ? v.toExponential(1) : String(v)) : String(v),
  ] as [string, string])
})

const trainingRecordingNames = computed(() => {
  const ids = meta.value?.training_recording_ids
  if (!ids?.length) return []
  return data.recordings
    .filter(r => ids.includes(r.recording_id))
    .map(r => ({ id: r.recording_id, name: r.recording_name }))
})

const hasTrainingConfig = computed(() =>
  coreParams.value.length > 0 || regularizationParams.value.length > 0
)
</script>

<template>
  <div v-if="dec" class="space-y-4">
    <!-- ════ Header card ════ -->
    <div class="rounded-lg border border-border/30 bg-bg-elevated overflow-hidden">
      <!-- Top: name + model badge + date -->
      <div class="px-4 pt-3 pb-2">
        <div class="flex items-start justify-between">
          <div>
            <h2 class="text-base font-semibold text-text-main mb-1">{{ dec.decoder_name }}</h2>
            <div class="flex items-center gap-2 flex-wrap">
              <span class="text-[10px] font-bold text-accent uppercase px-2 py-0.5 bg-accent/10 rounded-full tracking-wide">{{ dec.model_type }}</span>
              <template v-if="pipelineSteps.length > 0">
                <template v-for="(step, i) in pipelineSteps" :key="step">
                  <span v-if="i > 0" class="text-text-disabled text-[10px]">→</span>
                  <span class="text-[10px] px-1.5 py-0.5 rounded bg-text-main/[0.04] text-text-muted">{{ step }}</span>
                </template>
              </template>
            </div>
          </div>
          <div class="text-right shrink-0 ml-4">
            <div class="text-xs text-text-muted">{{ dec.study_name }}</div>
            <div class="text-[11px] text-text-disabled">{{ formatDate(dec.created_at) }}</div>
          </div>
        </div>
      </div>

      <!-- Input spec bar -->
      <div class="flex items-center gap-3 px-4 py-2 bg-text-main/[0.02] border-t border-border/20 text-xs flex-wrap">
        <span v-if="inputSpec?.modality" class="uppercase text-[10px] font-bold text-text-muted tracking-wider">{{ inputSpec.modality }}</span>
        <span v-if="inputSpec?.nCh" class="font-mono text-text-muted">{{ inputSpec.nCh }}ch × {{ inputSpec.nPts }}pts</span>
        <span v-if="inputSpec?.windowSec" class="font-mono text-text-disabled">({{ inputSpec.windowSec }}s)</span>
        <span v-if="inputSpec?.sampleRate" class="font-mono text-text-muted">{{ inputSpec.sampleRate }} Hz</span>
        <span v-if="epochWindow" class="font-mono text-text-muted">epoch {{ epochWindow }}</span>
        <template v-for="part in preprocParts" :key="part">
          <span class="text-text-disabled">·</span>
          <span class="text-text-muted">{{ part }}</span>
        </template>
      </div>

      <!-- Accuracy bar -->
      <div v-if="dec.training_accuracy != null || dec.validation_accuracy != null"
        class="flex border-t border-border/20"
      >
        <div v-if="dec.training_accuracy != null" class="flex-1 px-4 py-2.5 text-center border-r border-border/20 last:border-r-0">
          <div class="text-[10px] text-text-disabled uppercase tracking-wider mb-0.5">Train</div>
          <div class="text-lg font-bold font-mono text-text-main">{{ formatPercent(dec.training_accuracy) }}</div>
        </div>
        <div v-if="dec.validation_accuracy != null" class="flex-1 px-4 py-2.5 text-center">
          <div class="text-[10px] text-text-disabled uppercase tracking-wider mb-0.5">Validation</div>
          <div class="text-lg font-bold font-mono text-accent">{{ formatPercent(dec.validation_accuracy) }}</div>
        </div>
      </div>
    </div>

    <!-- ════ Classes ════ -->
    <div v-if="classEntries.length > 0" class="rounded-lg border border-border/30 bg-bg-elevated px-4 py-3">
      <div class="text-[10px] text-text-disabled uppercase tracking-wider font-semibold mb-2">Classes</div>
      <!-- Distribution bar -->
      <div class="flex h-2 rounded-full overflow-hidden gap-px mb-2.5">
        <div
          v-for="cls in classEntries" :key="cls.code"
          class="first:rounded-l-full last:rounded-r-full"
          :style="{ flex: 1, backgroundColor: predClassColor(cls.name) }"
          :title="cls.name"
        />
      </div>
      <div class="flex flex-wrap gap-x-4 gap-y-1">
        <span v-for="cls in classEntries" :key="cls.code" class="flex items-center gap-1.5 text-xs">
          <span class="w-2.5 h-2.5 rounded-full shrink-0" :style="{ backgroundColor: predClassColor(cls.name) }" />
          <span class="text-text-main">{{ cls.name }}</span>
          <span class="text-text-disabled font-mono text-[10px]">[{{ cls.code }}]</span>
        </span>
      </div>
    </div>

    <!-- ════ Training config ════ -->
    <div v-if="hasTrainingConfig" class="rounded-lg border border-border/30 bg-bg-elevated px-4 py-3">
      <div class="text-[10px] text-text-disabled uppercase tracking-wider font-semibold mb-2">Training</div>
      <div class="grid grid-cols-2 gap-x-6">
        <!-- Core -->
        <div class="space-y-1">
          <div v-for="[label, value] in coreParams" :key="label" class="flex items-center justify-between text-xs">
            <span class="text-text-disabled">{{ label }}</span>
            <span class="text-text-main font-mono">{{ value }}</span>
          </div>
        </div>
        <!-- Regularization -->
        <div class="space-y-1">
          <div v-for="[label, value] in regularizationParams" :key="label" class="flex items-center justify-between text-xs">
            <span class="text-text-disabled">{{ label }}</span>
            <span class="text-text-main font-mono">{{ value }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ════ Architecture ════ -->
    <div v-if="modelParams.length > 0" class="rounded-lg border border-border/30 bg-bg-elevated px-4 py-3">
      <div class="text-[10px] text-text-disabled uppercase tracking-wider font-semibold mb-2">Architecture</div>
      <div class="flex flex-wrap gap-x-5 gap-y-1">
        <span v-for="[label, value] in modelParams" :key="label" class="text-xs">
          <span class="text-text-disabled">{{ label }}</span>
          <span class="text-text-main font-mono ml-1">{{ value }}</span>
        </span>
      </div>
    </div>

    <!-- ════ Provenance ════ -->
    <div v-if="meta?.training_recording_ids?.length || meta?.training_file_identifier"
      class="rounded-lg border border-border/30 bg-bg-elevated px-4 py-3"
    >
      <div class="text-[10px] text-text-disabled uppercase tracking-wider font-semibold mb-2">Trained on</div>
      <div v-if="trainingRecordingNames.length > 0" class="space-y-1">
        <div
          v-for="rec in trainingRecordingNames" :key="rec.id"
          @click="data.selectRecording(rec.id)"
          class="flex items-center gap-2 text-xs text-accent hover:underline cursor-pointer"
        >
          <i class="pi pi-file text-[10px]" />
          {{ rec.name }}
        </div>
      </div>
      <div v-else-if="meta?.training_recording_ids?.length" class="text-xs text-text-muted">
        {{ meta.training_recording_ids.length }} recording{{ meta.training_recording_ids.length > 1 ? 's' : '' }}
        <span class="text-text-disabled">(IDs: {{ meta.training_recording_ids.join(', ') }})</span>
      </div>
      <div v-if="meta?.training_file_identifier" class="text-xs text-text-disabled font-mono mt-1">
        {{ meta.training_file_identifier }}
      </div>
    </div>

    <!-- ════ Channels ════ -->
    <div v-if="channelLabels.length > 0" class="rounded-lg border border-border/30 bg-bg-elevated px-4 py-3">
      <button
        @click="showChannels = !showChannels"
        class="flex items-center gap-1.5 w-full text-[10px] text-text-disabled uppercase tracking-wider font-semibold hover:text-text-muted transition-colors"
      >
        <i class="pi text-[8px]" :class="showChannels ? 'pi-chevron-down' : 'pi-chevron-right'" />
        Channels ({{ channelLabels.length }})
      </button>
      <div v-if="showChannels" class="flex flex-wrap gap-1 mt-2">
        <span
          v-for="ch in channelLabels" :key="ch"
          class="px-1.5 py-0.5 text-[10px] bg-text-main/[0.04] rounded text-text-muted"
        >{{ ch }}</span>
      </div>
    </div>

    <!-- Description -->
    <p v-if="dec.description" class="text-xs text-text-muted px-1">{{ dec.description }}</p>
  </div>
</template>
