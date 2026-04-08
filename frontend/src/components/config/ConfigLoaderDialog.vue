<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useConfigStore } from '../../stores/config'
import { useToast } from '../../composables/useToast'
import type { ConfigFile } from '../../types/api'

const emit = defineEmits<{ close: [] }>()
const config = useConfigStore()
const toast = useToast()

const search = ref('')
const selectedPath = ref<string | null>(null)
const isLoading = ref(false)

const grouped = computed(() => {
  const groups: Record<string, ConfigFile[]> = {}
  const q = search.value.toLowerCase()
  for (const c of config.availableConfigs) {
    if (q && !c.file_name.toLowerCase().includes(q) && !c.study_name.toLowerCase().includes(q)) {
      continue
    }
    if (!groups[c.study_name]) groups[c.study_name] = []
    groups[c.study_name]!.push(c)
  }
  // Sort newest-first within each study
  for (const files of Object.values(groups)) {
    files.sort((a, b) => b.modified - a.modified)
  }
  return groups
})

const sortedStudies = computed(() => {
  const current = config.general.study_name
  return Object.keys(grouped.value).sort((a, b) => {
    if (a === current) return -1
    if (b === current) return 1
    return a.localeCompare(b)
  })
})

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  return `${(bytes / 1024).toFixed(1)}KB`
}

function select(path: string) {
  selectedPath.value = path
}

async function load(path?: string) {
  const target = path || selectedPath.value
  if (!target) return
  isLoading.value = true
  try {
    const ok = await config.loadConfig(target)
    if (ok) {
      toast.success('Configuration loaded')
      emit('close')
    } else {
      toast.error('Failed to load configuration')
    }
  } finally {
    isLoading.value = false
  }
}

function onKey(e: KeyboardEvent) { if (e.key === 'Escape') emit('close') }
onMounted(() => { config.listConfigs(); window.addEventListener('keydown', onKey) })
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl shadow-black/40 w-[480px] max-h-[500px] flex flex-col">

        <!-- Header -->
        <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <h2 class="text-base font-semibold text-text-main">Load Configuration</h2>
          <button @click="emit('close')" class="text-text-disabled hover:text-text-main transition-colors p-1">
            <i class="pi pi-times" />
          </button>
        </div>

        <!-- Search -->
        <div class="px-5 py-2 border-b border-border shrink-0">
          <input
            v-model="search"
            placeholder="Search configs..."
            class="w-full"
          />
        </div>

        <!-- Config list -->
        <div class="flex-1 overflow-y-auto px-5 py-2">
          <div v-if="sortedStudies.length === 0" class="py-8 text-center">
            <i class="pi pi-folder-open text-2xl text-text-disabled block mb-2" />
            <p class="text-xs text-text-muted">No saved configurations found</p>
            <p class="text-xs text-text-disabled mt-1">Save a configuration first</p>
          </div>

          <div v-for="study in sortedStudies" :key="study" class="mb-3">
            <div class="text-xs font-medium text-text-muted uppercase tracking-wider mb-1 flex items-center gap-1.5">
              {{ study }}
              <span
                v-if="study === config.general.study_name"
                class="px-1 py-0.5 rounded bg-accent/20 text-accent text-xs normal-case tracking-normal"
              >current</span>
            </div>
            <div
              v-for="c in grouped[study]"
              :key="c.file_path"
              @click="select(c.file_path)"
              @dblclick="load(c.file_path)"
              class="flex items-center gap-3 px-3 py-2 rounded cursor-pointer transition-colors"
              :class="selectedPath === c.file_path
                ? 'bg-accent/10 border border-accent/30'
                : 'hover:bg-bg-elevated border border-transparent'"
            >
              <i class="pi pi-file text-xs text-text-disabled" />
              <div class="flex-1 min-w-0">
                <div class="text-xs text-text-main truncate">{{ c.file_name }}</div>
              </div>
              <span class="text-xs text-text-disabled">{{ timeAgo(c.modified) }}</span>
              <span class="text-xs text-text-disabled w-10 text-right">{{ fileSize(c.size) }}</span>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="flex justify-end gap-2 px-5 py-3 border-t border-border shrink-0">
          <button
            @click="emit('close')"
            :disabled="isLoading"
            class="px-4 py-1.5 text-xs rounded border border-border text-text-muted
                   hover:text-text-main hover:border-text-muted transition-colors disabled:opacity-30"
          >Cancel</button>
          <button
            @click="load()"
            :disabled="!selectedPath || isLoading"
            class="px-4 py-1.5 text-xs rounded bg-accent text-white hover:bg-accent-hover
                   disabled:opacity-30 transition-colors"
          >
            <i v-if="isLoading" class="pi pi-spin pi-spinner mr-1" />
            {{ isLoading ? 'Loading...' : 'Load' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
