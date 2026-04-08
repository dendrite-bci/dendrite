<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import type { Decoder, Study } from '../../types/api'
import { relativeTime } from '../../utils/format'

const props = defineProps<{
  initialStudyFilter?: number | null
}>()

const emit = defineEmits<{
  select: [decoder: Decoder]
  close: []
}>()

const decoders = ref<Decoder[]>([])
const studies = ref<Study[]>([])
const search = ref('')
const filterStudyId = ref<number | null>(null)
const selected = ref<Decoder | null>(null)
const loading = ref(false)

async function fetchDecoderList() {
  loading.value = true
  try {
    const params = new URLSearchParams()
    if (filterStudyId.value) params.set('study_id', String(filterStudyId.value))
    if (search.value) params.set('search', search.value)
    const qs = params.toString()
    const res = await fetch(`/api/data/decoders${qs ? '?' + qs : ''}`)
    if (res.ok) decoders.value = await res.json()
  } finally {
    loading.value = false
  }
}

async function fetchStudyList() {
  const res = await fetch('/api/data/studies')
  if (res.ok) studies.value = await res.json()
}

const grouped = computed(() => {
  const groups: Record<string, Decoder[]> = {}
  for (const dec of decoders.value) {
    const key = dec.study_name || 'Unknown'
    if (!groups[key]) groups[key] = []
    groups[key].push(dec)
  }
  return groups
})

const sortedStudies = computed(() =>
  Object.keys(grouped.value).sort((a, b) => a.localeCompare(b))
)

function formatAccuracy(acc: number | null): string {
  if (acc == null) return ''
  return `${(acc * 100).toFixed(1)}%`
}

function selectDecoder(dec: Decoder) {
  selected.value = dec
}

function confirm(dec?: Decoder) {
  const target = dec || selected.value
  if (target) emit('select', target)
}

let searchTimeout: ReturnType<typeof setTimeout> | null = null
watch(search, () => {
  if (searchTimeout) clearTimeout(searchTimeout)
  searchTimeout = setTimeout(fetchDecoderList, 200)
})

watch(filterStudyId, () => fetchDecoderList())

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => {
  fetchDecoderList()
  fetchStudyList()
  window.addEventListener('keydown', onKey)
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  if (searchTimeout) clearTimeout(searchTimeout)
})
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl shadow-black/40 w-[520px] max-h-[550px] flex flex-col">

        <!-- Header -->
        <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <h2 class="text-base font-semibold text-text-main">Select Decoder</h2>
          <button @click="emit('close')" class="text-text-disabled hover:text-text-main transition-colors p-1">
            <i class="pi pi-times" />
          </button>
        </div>

        <!-- Filters -->
        <div class="flex items-center gap-2 px-5 py-2 border-b border-border shrink-0">
          <input
            v-model="search"
            placeholder="Search decoders..."
            class="flex-1"
          />
          <select
            v-model="filterStudyId"
            class="w-40"
          >
            <option :value="null">All studies</option>
            <option v-for="s in studies" :key="s.study_id" :value="s.study_id">{{ s.study_name }}</option>
          </select>
        </div>

        <!-- Decoder list -->
        <div class="flex-1 overflow-y-auto px-5 py-2">
          <!-- Loading -->
          <div v-if="loading && decoders.length === 0" class="py-8 text-center">
            <i class="pi pi-spin pi-spinner text-2xl text-text-disabled block mb-2" />
            <p class="text-xs text-text-muted">Loading decoders...</p>
          </div>

          <!-- Empty state -->
          <div v-else-if="decoders.length === 0" class="py-8 text-center">
            <i class="pi pi-box text-2xl text-text-disabled block mb-2" />
            <p class="text-xs text-text-muted">No decoders found</p>
            <p class="text-xs text-text-disabled mt-1">Train a decoder in the ML Workbench first</p>
          </div>

          <!-- Grouped list -->
          <div v-for="study in sortedStudies" :key="study" class="mb-3">
            <div class="text-xs font-medium text-text-muted uppercase tracking-wider mb-1">
              {{ study }}
            </div>
            <div
              v-for="dec in grouped[study]"
              :key="dec.decoder_id"
              @click="selectDecoder(dec)"
              @dblclick="confirm(dec)"
              class="flex items-center gap-3 px-3 py-2 rounded cursor-pointer transition-colors"
              :class="selected?.decoder_id === dec.decoder_id
                ? 'bg-accent/10 ring-1 ring-inset ring-accent/25 border border-transparent'
                : 'hover:bg-bg-elevated border border-transparent'"
            >
              <i class="pi pi-box text-xs text-text-disabled shrink-0" />
              <div class="flex-1 min-w-0">
                <div class="text-xs text-text-main truncate">{{ dec.decoder_name }}</div>
                <div class="flex items-center gap-1.5 mt-0.5">
                  <span class="text-xs font-semibold text-accent uppercase">{{ dec.model_type }}</span>
                  <template v-if="dec.num_classes">
                    <span class="text-xs text-text-disabled">&middot;</span>
                    <span class="text-xs text-text-muted">{{ dec.num_classes }} classes</span>
                  </template>
                  <template v-if="dec.training_accuracy != null">
                    <span class="text-xs text-text-disabled">&middot;</span>
                    <span class="text-xs text-status-ok">{{ formatAccuracy(dec.training_accuracy) }}</span>
                  </template>
                </div>
              </div>
              <span class="text-xs text-text-disabled shrink-0">{{ relativeTime(dec.created_at) }}</span>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="flex justify-end gap-2 px-5 py-3 border-t border-border shrink-0">
          <button
            @click="emit('close')"
            class="px-4 py-1.5 text-xs rounded border border-border text-text-muted
                   hover:text-text-main hover:border-text-muted transition-colors"
          >Cancel</button>
          <button
            @click="confirm()"
            :disabled="!selected"
            class="px-4 py-1.5 text-xs rounded bg-accent text-white hover:bg-accent-hover
                   disabled:opacity-30 transition-colors"
          >Select</button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
