<script setup lang="ts">
defineProps<{
  metrics: { name: string; precision: number; recall: number; f1: number; support: number }[]
}>()
</script>

<template>
  <div class="bg-bg-elevated rounded-lg border border-border/50 overflow-hidden">
    <table class="w-full text-xs">
      <thead>
        <tr class="text-text-disabled text-xs uppercase">
          <th class="text-left px-2.5 py-1.5 font-medium">Class</th>
          <th class="text-right px-2.5 py-1.5 font-medium">Prec</th>
          <th class="text-right px-2.5 py-1.5 font-medium">Rec</th>
          <th class="text-right px-2.5 py-1.5 font-medium">F1</th>
          <th class="text-right px-2.5 py-1.5 font-medium">N</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in metrics" :key="row.name"
          class="border-t border-border/30"
          :class="row.recall < 0.5 ? 'bg-status-error/5' : ''">
          <td class="px-2.5 py-1 text-text-main font-medium">{{ row.name }}</td>
          <td class="px-2.5 py-1 text-right font-mono text-text-main">{{ (row.precision * 100).toFixed(1) }}%</td>
          <td class="px-2.5 py-1 text-right font-mono" :class="row.recall < 0.5 ? 'text-status-error' : 'text-text-main'">
            {{ (row.recall * 100).toFixed(1) }}%
          </td>
          <td class="px-2.5 py-1 text-right font-mono text-text-main">{{ (row.f1 * 100).toFixed(1) }}%</td>
          <td class="px-2.5 py-1 text-right font-mono text-text-disabled">{{ row.support }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
