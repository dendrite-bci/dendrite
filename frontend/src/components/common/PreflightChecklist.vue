<script setup lang="ts">
import { computed } from 'vue'
import { usePipelineStore } from '../../stores/pipeline'

const pipeline = usePipelineStore()

const failures = computed(() =>
  pipeline.preflight?.checks?.filter(c => !c.passed) ?? []
)
const allPassed = computed(() => pipeline.preflight?.ready ?? false)
</script>

<template>
  <div v-if="pipeline.preflight" class="flex items-center gap-2.5 text-xs">
    <template v-if="allPassed">
      <div class="w-2 h-2 rounded-full bg-status-ok" />
      <span class="text-text-muted">Ready</span>
    </template>
    <template v-else>
      <span
        v-for="check in failures"
        :key="check.id"
        class="flex items-center gap-1.5 text-text-muted"
        :title="check.detail || ''"
      >
        <div
          class="w-2 h-2 rounded-full shrink-0"
          :class="check.required ? 'bg-status-error' : 'bg-status-warn'"
        />
        {{ check.label }}
      </span>
    </template>
  </div>
</template>
