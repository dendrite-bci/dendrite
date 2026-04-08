<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMLStore } from '../../stores/ml'
import type { MoabbDataset } from '../../types/api'

const ml = useMLStore()
const search = ref('')
const paradigmFilter = ref<string | null>(null)

onMounted(() => {
  if (ml.moabbDatasets.length === 0) ml.discoverMoabb()
})

const paradigms = computed(() => {
  const set = new Set(ml.moabbDatasets.map(d => d.paradigm))
  return Array.from(set).sort()
})

const filtered = computed(() => {
  let list = ml.moabbDatasets
  if (paradigmFilter.value) {
    list = list.filter(d => d.paradigm === paradigmFilter.value)
  }
  if (search.value) {
    const q = search.value.toLowerCase()
    list = list.filter(d =>
      d.code.toLowerCase().includes(q) ||
      d.name.toLowerCase().includes(q) ||
      d.paradigm.toLowerCase().includes(q)
    )
  }
  return list
})

function selectDataset(ds: MoabbDataset) {
  ml.selectedMoabbDataset = ds
  ml.dataPreproc.subject = 1
}
</script>

<template>
  <div class="mt-2">
    <!-- Search + refresh -->
    <div class="flex items-center gap-2 mb-2">
      <input v-model="search" placeholder="Search..." class="flex-1 text-xs placeholder-text-disabled" />
      <div class="flex gap-1">
        <button
          @click="paradigmFilter = null"
          class="text-[10px] px-1.5 py-0.5 rounded transition-colors"
          :class="!paradigmFilter ? 'bg-accent/15 text-accent' : 'bg-bg-input text-text-muted hover:text-text-main'"
        >All</button>
        <button
          v-for="p in paradigms" :key="p"
          @click="paradigmFilter = paradigmFilter === p ? null : p"
          class="text-[10px] px-1.5 py-0.5 rounded transition-colors"
          :class="paradigmFilter === p ? 'bg-accent/15 text-accent' : 'bg-bg-input text-text-muted hover:text-text-main'"
        >{{ p }}</button>
      </div>
      <button @click="ml.discoverMoabb()" :disabled="ml.moabbLoading" class="text-xs text-accent hover:text-accent/80 shrink-0">
        <i class="pi pi-refresh text-xs" />
      </button>
    </div>

    <!-- Loading -->
    <div v-if="ml.moabbLoading && ml.moabbDatasets.length === 0" class="text-center py-4">
      <i class="pi pi-spin pi-spinner text-sm text-text-disabled" />
      <p class="text-xs text-text-muted mt-1">Discovering datasets...</p>
    </div>

    <!-- Dataset list (click to select, same row style as RecordingBrowser) -->
    <div v-else class="max-h-[350px] overflow-y-auto">
      <div
        v-for="ds in filtered" :key="ds.code"
        @click="selectDataset(ds)"
        class="flex items-center gap-2 px-2 py-1.5 cursor-pointer rounded hover:bg-white/[0.03] transition-colors border-l-2"
        :class="ml.selectedMoabbDataset?.code === ds.code ? 'border-l-accent' : 'border-l-transparent'"
      >
        <div class="flex-1 min-w-0">
          <span class="text-xs text-text-main truncate block">{{ ds.code }}</span>
        </div>
        <span class="text-[10px] text-text-disabled shrink-0">{{ ds.n_subjects }} subj</span>
        <span class="text-[10px] text-text-disabled shrink-0">{{ ds.paradigm }}</span>
      </div>
    </div>

    <p v-if="!ml.moabbLoading && filtered.length === 0" class="text-xs text-text-disabled text-center py-4">
      No datasets found.
    </p>
  </div>
</template>
