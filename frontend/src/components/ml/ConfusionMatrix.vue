<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  matrix: number[][]
  classLabels?: string[]
}>()

const rowMaxes = computed(() => props.matrix.map(row => Math.max(...row, 1)))

function cellStyle(value: number, rowIdx: number, isDiag: boolean): Record<string, string> {
  const rowMax = rowMaxes.value[rowIdx]!
  const intensity = rowMax! > 0 ? value / rowMax! : 0
  const color = isDiag ? '52, 211, 153' : '248, 113, 113'
  return { backgroundColor: `rgba(${color}, ${intensity * 0.7})` }
}
</script>

<template>
  <div class="bg-bg-elevated rounded-lg border border-border/50 overflow-hidden inline-block">
    <table class="text-xs">
      <thead v-if="classLabels?.length">
        <tr class="text-text-disabled text-xs uppercase">
          <th class="text-left px-3 py-1.5.5 font-medium"></th>
          <th v-for="name in classLabels" :key="name" class="text-center px-3 py-1.5.5 font-medium">{{ name }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, i) in matrix" :key="i" :class="i > 0 || classLabels?.length ? 'border-t border-border/30' : ''">
          <td v-if="classLabels?.length" class="px-3 py-1.5 text-text-muted font-medium text-xs">{{ classLabels[i] ?? i }}</td>
          <td v-for="(val, j) in row" :key="j"
            class="text-center px-3 py-1.5 font-mono"
            :style="cellStyle(val, i, i === j)"
          >{{ val }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
