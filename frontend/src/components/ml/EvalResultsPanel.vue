<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useMLStore, evalDirty } from '../../stores/ml'
import { formatPercent } from '../../utils/format'
import { makeAxis, CURSOR_INTERACTIVE, LEGEND_HIDDEN } from '../../utils/chartDefaults'
import { useUPlot } from '../../composables/useUPlot'
import { predClassColor } from '../../utils/colors'
import ConfusionMatrix from './ConfusionMatrix.vue'
import MetricBadge from './MetricBadge.vue'
import NumberInput from '../common/NumberInput.vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

const ml = useMLStore()
const isSlidingWindow = computed(() => ml.evalMetrics?.mode === 'sliding_window')

// Dual source: live during eval, final after
const metrics = computed(() => {
  if (ml.evalMetrics) return ml.evalMetrics as Record<string, any>
  const le = ml.liveEval
  if (le && le.timeline.length > 0) {
    const nCorrect = le.trials.filter((t: any) => t.correct).length
    return {
      mode: 'sliding_window',
      accuracy: le.trials.length > 0 ? nCorrect / le.trials.length : 0,
      n_trials: le.trials.length,
      timeline: le.timeline,
      per_trial: le.trials,
      event_markers: [],
      confusion_matrix: null,
      ttd: null,
    }
  }
  return null
})

const isLive = computed(() => !ml.evalMetrics && ml.liveEval !== null)
const ttd = computed(() => metrics.value?.ttd ?? null)
const optimalGate = computed(() => metrics.value?.optimal_gate ?? null)

function applyOptimalGate() {
  const g = optimalGate.value?.gate
  if (!g) return
  ml.evalGate.detection_strategy = g.strategy
  ml.evalGate.dwell_n = g.dwell_n ?? 3
  ml.evalGate.confidence_threshold = g.confidence_threshold ?? 0
  ml.reaggregateEval()
}
const eventMarkers = computed(() => metrics.value?.event_markers ?? [])
const cmData = computed(() => metrics.value?.confusion_matrix ?? null)
const perTrial = computed(() => metrics.value?.per_trial ?? [])

// Direct access to timeline (non-reactive for rAF loop performance)
function getTimeline(): Array<{ time_s: number; confidence: number; correct: boolean; prediction: number; trial_idx: number }> {
  if (ml.evalMetrics) return (ml.evalMetrics as any).timeline ?? []
  return ml.liveEval?.timeline ?? []
}

const classNames = computed(() => {
  const names = new Set<string>()
  for (const t of perTrial.value) names.add(t.event_name)
  return Array.from(names).sort()
})

// --- Timeline chart (rAF loop like AsyncPredictionPlot) ---
const chartEl = ref<HTMLDivElement | null>(null)
const { create, setData } = useUPlot(chartEl)
let animFrame: number | null = null

function createChart() {
  // Draw plugin reads CURRENT data at draw time (works for both live + final)
  const drawPlugin: uPlot.Plugin = {
    hooks: {
      draw: [
        (u: uPlot) => {
          const ctx = u.ctx
          const { left, top, width: plotW, height: plotH } = u.bbox
          const OK = '52,211,153', ERR = '248,113,113'
          const correctColor = (ok: boolean, a: number) => `rgba(${ok ? OK : ERR},${a})`

          // Trial window shading + event onset lines
          const markers = eventMarkers.value as { event_s: number; trial_start_s: number; trial_end_s: number; correct: boolean }[]
          if (markers.length > 0) {
            // Trial epoch shading
            for (const m of markers) {
              const x0 = u.valToPos(m.trial_start_s, 'x', true)
              const x1 = u.valToPos(m.trial_end_s, 'x', true)
              if (x1 < left || x0 > left + plotW) continue
              ctx.fillStyle = correctColor(m.correct, 0.1)
              ctx.fillRect(Math.max(x0, left), top, Math.min(x1, left + plotW) - Math.max(x0, left), plotH)
            }
            // Event onset lines (actual stimulus time)
            ctx.setLineDash([4, 4])
            ctx.lineWidth = 1
            for (const m of markers) {
              const x = u.valToPos(m.event_s, 'x', true)
              if (x < left || x > left + plotW) continue
              ctx.strokeStyle = 'rgba(255,255,255,0.3)'
              ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke()
            }
            ctx.setLineDash([])
          } else {
            // Live: derive trial regions from streamed timeline + trials
            const tl = getTimeline()
            const liveTrials = perTrial.value
            for (const t of liveTrials) {
              const idx = t.trial - 1
              let minT = Infinity, maxT = -Infinity
              for (const e of tl) {
                if (e.trial_idx === idx) {
                  if (e.time_s < minT) minT = e.time_s
                  if (e.time_s > maxT) maxT = e.time_s
                }
              }
              if (minT === Infinity) continue
              const x0 = u.valToPos(minT, 'x', true)
              const x1 = u.valToPos(maxT, 'x', true)
              if (x1 < left || x0 > left + plotW) continue
              ctx.fillStyle = correctColor(t.correct, 0.1)
              ctx.fillRect(Math.max(x0, left), top, Math.min(x1, left + plotW) - Math.max(x0, left), plotH)
            }
          }

          // Detection point markers — solid line where dwell fired
          const trials = perTrial.value as { detection_time_s?: number | null; correct?: boolean }[]
          for (const t of trials) {
            if (t.detection_time_s == null) continue
            const x = u.valToPos(t.detection_time_s, 'x', true)
            if (x < left || x > left + plotW) continue
            ctx.strokeStyle = correctColor(!!t.correct, 0.7)
            ctx.lineWidth = 2
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke()
          }

          // Confidence threshold line
          const confThresh = (metrics.value as any)?.config?.gate?.confidence_threshold ?? 0
          if (confThresh > 0) {
            const y = u.valToPos(confThresh, 'y', true)
            if (y >= top && y <= top + plotH) {
              ctx.setLineDash([6, 3])
              ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)'
              ctx.lineWidth = 1
              ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke()
              ctx.setLineDash([])
            }
          }

          // Prediction class raster strip at bottom of plot
          const RASTER_H = 20
          const rasterTl = getTimeline()
          if (rasterTl.length > 1 && u.scales.x) {
            const xMin = u.scales.x.min ?? 0
            const xMax = u.scales.x.max ?? 1
            const names = classNames.value
            const stepSec = rasterTl[1]!.time_s - rasterTl[0]!.time_s
            const pxPerSec = plotW / (xMax - xMin)
            const barW = Math.max(1, Math.ceil(stepSec * pxPerSec))
            const rasterY = top + plotH - RASTER_H

            // Background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
            ctx.fillRect(left, rasterY, plotW, RASTER_H)

            // Clip to raster area
            ctx.save()
            ctx.beginPath()
            ctx.rect(left, rasterY, plotW, RASTER_H)
            ctx.clip()

            for (const p of rasterTl) {
              if (p.time_s < xMin - stepSec || p.time_s > xMax) continue
              const x = u.valToPos(p.time_s, 'x', true)
              ctx.fillStyle = predClassColor(names[p.prediction] ?? `class_${p.prediction}`)
              ctx.globalAlpha = 0.3 + p.confidence * 0.7
              ctx.fillRect(x, rasterY, barW, RASTER_H)
            }

            ctx.restore()
          }
        },
      ],
    },
  }

  create(({ width }) => ({
    width, height: 300,
    cursor: { ...CURSOR_INTERACTIVE, drag: { x: true, y: false } },
    legend: LEGEND_HIDDEN,
    scales: { x: { time: false }, y: { range: [0, 1.05] } },
    axes: [
      makeAxis({ label: 'Time (s)', size: 28 }),
      makeAxis({ label: 'Confidence', size: 40 }),
    ],
    series: [
      { label: 'Time' },
      { label: 'Confidence', stroke: '#818cf8', width: 2, fill: 'rgba(129, 140, 248, 0.15)', points: { show: false } },
    ],
    plugins: [drawPlugin],
  }), [[0], [0]])
}

function updateChartData() {
  const tl = getTimeline()
  if (tl.length === 0) return
  setData([tl.map(p => p.time_s), tl.map(p => p.confidence)])
}

// rAF render loop — polls dirty flag (same pattern as AsyncPredictionPlot)
function renderLoop() {
  if (evalDirty.timelineChanged) {
    evalDirty.timelineChanged = false
    updateChartData()
  }
  animFrame = requestAnimationFrame(renderLoop)
}

onMounted(() => {
  requestAnimationFrame(() => {
    if (chartEl.value) {
      createChart()
      updateChartData()
    }
  })
  animFrame = requestAnimationFrame(renderLoop)
})

// When final metrics arrive → recreate chart with trial shading
watch(() => ml.evalMetrics, (m) => {
  if (m) {
    requestAnimationFrame(() => {
      createChart()
      updateChartData()
    })
  }
})

onUnmounted(() => {
  if (animFrame) cancelAnimationFrame(animFrame)
})
</script>

<template>
  <div v-if="metrics" class="rounded-lg border border-border overflow-hidden">

    <!-- Summary metrics -->
    <div class="px-5 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
      <MetricBadge label="Accuracy" :value="formatPercent(metrics.accuracy)" />
      <template v-if="metrics.n_trials">
        <span class="text-border">&middot;</span>
        <MetricBadge label="Trials" :value="String(metrics.n_trials)" />
      </template>
      <template v-if="metrics.far?.far_per_min != null">
        <span class="text-border">&middot;</span>
        <MetricBadge label="FAR" :value="`${metrics.far.far_per_min.toFixed(1)}/min`" />
      </template>
      <template v-if="metrics.itr_bits_per_min != null">
        <span class="text-border">&middot;</span>
        <MetricBadge label="ITR" :value="`${metrics.itr_bits_per_min.toFixed(1)} bits/min`" />
      </template>
      <template v-if="ttd">
        <template v-if="ttd.n_detected != null">
          <span class="text-border">&middot;</span>
          <MetricBadge label="Detected" :value="`${ttd.n_detected}/${ttd.n_total}`" />
        </template>
        <template v-if="ttd.mean_ms != null">
          <span class="text-border">&middot;</span>
          <MetricBadge label="TTD mean" :value="`${ttd.mean_ms.toFixed(0)}ms`" />
          <span class="text-border">&middot;</span>
          <MetricBadge label="median" :value="`${ttd.median_ms.toFixed(0)}ms`" />
        </template>
      </template>
      <template v-if="isLive">
        <span class="text-border">&middot;</span>
        <MetricBadge label="" value="Live" :highlight="true" />
      </template>
    </div>

    <!-- Decision gate controls (post-hoc, sliding window only) -->
    <div v-if="isSlidingWindow && !isLive" class="px-5 py-2 border-t border-border/30 flex items-end gap-2 flex-wrap">
      <div class="w-[100px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Strategy</label>
        <select v-model="ml.evalGate.detection_strategy" class="w-full text-xs" @change="ml.reaggregateEval()">
          <option value="dwell">Dwell</option>
          <option value="majority">Majority</option>
        </select>
      </div>
      <div v-if="ml.evalGate.detection_strategy === 'dwell'" class="w-[56px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Dwell n</label>
        <NumberInput v-model="ml.evalGate.dwell_n" :min="1" :max="20" class="w-full" @update:model-value="ml.reaggregateEval()" />
      </div>
      <div class="w-[72px]">
        <label class="text-[11px] text-text-muted block mb-0.5">Conf. gate</label>
        <NumberInput v-model="ml.evalGate.confidence_threshold" :min="0" :max="1" :step="0.05" class="w-full" @update:model-value="ml.reaggregateEval()" />
      </div>
      <button v-if="optimalGate" class="text-xs text-accent hover:text-accent-hover ml-1 mb-0.5"
        :title="`${optimalGate.gate.strategy}${optimalGate.gate.strategy === 'dwell' ? ` n=${optimalGate.gate.dwell_n}` : ''}, thresh=${optimalGate.gate.confidence_threshold}`"
        @click="applyOptimalGate">
        Optimal ({{ formatPercent(optimalGate.balanced_accuracy) }})
      </button>
    </div>

    <!-- Prediction timeline chart -->
    <div class="px-5 py-4 border-t border-border/30">
      <h4 class="text-xs font-semibold text-text-muted mb-1.5">Prediction Timeline</h4>
      <div ref="chartEl" class="bg-bg-elevated rounded-lg p-2 border border-border/50 min-h-[300px]" />
      <!-- Legend -->
      <div v-if="classNames.length > 0" class="mt-1.5 flex gap-3 flex-wrap">
        <span v-for="cn in classNames" :key="cn" class="flex items-center gap-1 text-xs text-text-muted">
          <span class="w-2 h-2 rounded-full" :style="{ backgroundColor: predClassColor(cn) }" />
          {{ cn }}
        </span>
      </div>
    </div>

    <!-- Confusion matrix + per-trial table -->
    <div v-if="(cmData && cmData.length > 0) || perTrial.length > 0"
      class="px-5 py-4 border-t border-border/30">
      <div class="flex gap-4 flex-wrap">
        <div v-if="cmData && cmData.length > 0" class="min-w-[200px]">
          <h4 class="text-xs font-semibold text-text-muted mb-1.5">Confusion Matrix</h4>
          <ConfusionMatrix :matrix="cmData" :class-labels="classNames" />
        </div>
        <div v-if="perTrial.length > 0" class="flex-1 min-w-[300px]">
          <h4 class="text-xs font-semibold text-text-muted mb-1.5">Per-Trial Results</h4>
          <div class="bg-bg-elevated rounded-lg border border-border/50 overflow-hidden">
            <div class="max-h-[240px] overflow-y-auto">
              <table class="w-full text-xs">
                <thead class="sticky top-0 bg-bg-elevated">
                  <tr class="text-text-disabled text-xs uppercase">
                    <th class="text-left px-2.5 py-1.5 font-medium">#</th>
                    <th class="text-left px-2.5 py-1.5 font-medium">Event</th>
                    <th class="text-right px-2.5 py-1.5 font-medium">Steps</th>
                    <th class="text-right px-2.5 py-1.5 font-medium">Acc</th>
                    <th class="text-right px-2.5 py-1.5 font-medium">TTD</th>
                    <th class="text-center px-2.5 py-1.5 font-medium">Vote</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="t in perTrial" :key="t.trial" class="border-t border-border/30"
                    :class="t.correct ? '' : 'bg-status-error/5'">
                    <td class="px-2.5 py-1 font-mono text-text-muted">{{ t.trial }}</td>
                    <td class="px-2.5 py-1 text-text-main">{{ t.event_name }}</td>
                    <td class="px-2.5 py-1 text-right font-mono text-text-muted">{{ t.n_steps }}</td>
                    <td class="px-2.5 py-1 text-right font-mono text-text-main">{{ (t.step_accuracy * 100).toFixed(0) }}%</td>
                    <td class="px-2.5 py-1 text-right font-mono" :class="t.ttd_ms != null ? 'text-text-main' : 'text-text-disabled'">
                      {{ t.ttd_ms != null ? `${t.ttd_ms.toFixed(0)}ms` : '—' }}
                    </td>
                    <td class="px-2.5 py-1 text-center">
                      <span :class="t.correct ? 'text-status-ok' : 'text-status-error'" class="font-semibold">
                        {{ t.vote_name ?? `class ${t.vote}` }}
                        {{ t.correct ? '✓' : '✗' }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>
