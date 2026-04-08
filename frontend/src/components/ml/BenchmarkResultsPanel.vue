<script setup lang="ts">
import { computed } from 'vue'
import { useMLStore } from '../../stores/ml'

const ml = useMLStore()

const sortedResults = computed(() =>
  [...ml.benchResults].sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))
)

const maxAcc = computed(() =>
  Math.max(...ml.benchResults.map(r => r.accuracy ?? 0), 0.01)
)
</script>

<template>
  <div v-if="ml.benchResults.length > 0" class="rounded-lg border border-border overflow-hidden">
    <table class="w-full text-xs">
      <thead>
        <tr class="text-text-disabled text-xs uppercase">
          <th class="text-center px-2 py-2 font-medium w-8">#</th>
          <th class="text-left px-3 py-2 font-medium">Model</th>
          <th class="text-left px-3 py-2 font-medium w-[200px]">Accuracy</th>
          <th class="text-right px-3 py-2 font-medium">Std</th>
          <th class="text-right px-3 py-2 font-medium">Time (s)</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(r, idx) in sortedResults" :key="r.model_type"
          class="border-t border-border/30"
          :class="idx === 0 ? 'bg-accent/5' : ''"
        >
          <td class="text-center px-2 py-1.5 font-mono text-text-disabled">{{ idx + 1 }}</td>
          <td class="px-3 py-1.5 text-text-main font-medium">{{ r.model_type }}</td>
          <td class="px-3 py-1.5">
            <div class="flex items-center gap-2">
              <div class="flex-1 h-4 bg-bg-input rounded overflow-hidden relative">
                <div
                  class="h-full rounded transition-all duration-300"
                  :class="idx === 0 ? 'bg-accent/30' : 'bg-accent/15'"
                  :style="{ width: `${((r.accuracy ?? 0) / maxAcc) * 100}%` }"
                />
                <span class="absolute inset-0 flex items-center px-1.5 text-xs font-mono text-text-main">
                  {{ r.accuracy ? (r.accuracy * 100).toFixed(1) + '%' : 'N/A' }}
                </span>
              </div>
            </div>
          </td>
          <td class="px-3 py-1.5 text-right font-mono text-text-muted">
            {{ r.std ? '\u00b1' + (r.std * 100).toFixed(1) + '%' : '\u2014' }}
          </td>
          <td class="px-3 py-1.5 text-right font-mono text-text-muted">
            {{ r.elapsed_s?.toFixed(1) ?? '\u2014' }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>

  <div v-else class="flex items-center justify-center h-full">
    <div class="text-center">
      <i class="pi pi-table text-4xl text-text-disabled mb-4 block" />
      <p class="text-text-muted text-sm">Benchmark Results</p>
      <p class="text-text-disabled text-xs mt-1">
        Select models and run a benchmark to see comparison results.
      </p>
    </div>
  </div>
</template>
