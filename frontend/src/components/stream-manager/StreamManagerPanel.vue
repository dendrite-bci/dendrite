<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useStreamManagerStore } from '../../stores/streamManager'
import type { FileInfo } from '../../stores/streamManager'
import { useToast } from '../../composables/useToast'
import { apiFetch, apiFetchOrNull } from '../../utils/api'
import NumberInput from '../common/NumberInput.vue'
import StreamPreview from './StreamPreview.vue'
import type { Recording, Study } from '../../types/api'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()

const store = useStreamManagerStore()
const toast = useToast()

// --- Tab state ---
const activeTab = ref<'file' | 'recordings' | 'moabb'>('recordings')

// --- File tab state ---
const filePath = ref('')
const fileInfo = ref<FileInfo | null>(null)
const fileLoading = ref(false)
const fileError = ref('')
const fileEnableEvents = ref(false)

async function browseFile() {
  fileLoading.value = true
  fileError.value = ''
  fileInfo.value = null
  try {
    const data = await apiFetch<{ path?: string }>('/api/stream-manager/pick-file', { method: 'POST' })
    if (!data.path) {
      fileLoading.value = false
      return
    }
    filePath.value = data.path
    const info = await store.fetchFileInfo(data.path)
    if (info) {
      fileInfo.value = info
      fileEnableEvents.value = info.n_events > 0
    } else {
      fileError.value = 'Could not read file.'
    }
  } catch {
    fileError.value = 'Failed to open file picker.'
  } finally {
    fileLoading.value = false
  }
}

// --- MOABB tab state ---
const moabbLoaded = ref(false)
const selectedDataset = ref<string>('')
const moabbSubject = ref(1)

const selectedMoabbData = computed(() =>
  store.moabbDatasets.find(d => d.name === selectedDataset.value) ?? null
)

async function loadMoabb() {
  await store.fetchMoabb()
  moabbLoaded.value = true
}

// --- Recordings tab state ---
const studies = ref<Study[]>([])
const recordings = ref<Recording[]>([])
const recordingsLoading = ref(false)
const recordingsStudyId = ref<number | null>(null)
const selectedRecordingId = ref<number | null>(null)
const recordingInfo = ref<FileInfo | null>(null)
const recordingInfoLoading = ref(false)
const recordingEnableEvents = ref(false)

const selectedRecordingData = computed(() =>
  recordings.value.find(r => r.recording_id === selectedRecordingId.value) ?? null
)

async function fetchStudies() {
  const data = await apiFetchOrNull<Study[]>('/api/data/studies')
  if (data) studies.value = data
}

async function fetchRecordings() {
  recordingsLoading.value = true
  const url = recordingsStudyId.value != null
    ? `/api/data/recordings?study_id=${recordingsStudyId.value}`
    : '/api/data/recordings'
  const data = await apiFetchOrNull<Recording[]>(url)
  if (data) recordings.value = data
  recordingsLoading.value = false
}

watch(recordingsStudyId, async () => {
  selectedRecordingId.value = null
  await fetchRecordings()
})

watch(selectedRecordingId, async (id) => {
  recordingInfo.value = null
  recordingEnableEvents.value = false
  if (!id) return
  const rec = recordings.value.find(r => r.recording_id === id)
  if (!rec) return
  recordingInfoLoading.value = true
  try {
    const info = await store.fetchFileInfo(rec.hdf5_file_path)
    recordingInfo.value = info
    if (info) recordingEnableEvents.value = info.n_events > 0
  } finally {
    recordingInfoLoading.value = false
  }
})

// --- Active streams ---
const activeStreams = computed(() => store.streams.filter(s => s.running))

const SOURCE_BADGE = 'pr-2.5 mr-2.5 border-r border-border text-xs font-medium uppercase tracking-wider text-accent/70'
const SOURCE_LABELS: Record<string, string> = { file: 'FILE', moabb: 'MOABB' }

function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] ?? source.toUpperCase()
}

function progressInfo(stream: { progress: number }): { width: string; label: string } {
  if (stream.progress < 0) return { width: '100%', label: 'Continuous' }
  const pct = `${Math.min(100, Math.round(stream.progress * 100))}%`
  return { width: pct, label: pct }
}

// --- Actions ---
async function launchStream(config: Record<string, any>, label: string) {
  try {
    await store.startStream(config, label)
    toast.success('Stream started')
  } catch {
    toast.error('Failed to start stream')
  }
}

function startFile() {
  if (!fileInfo.value) return
  launchStream(
    { source: 'file', path: fileInfo.value.path, enable_events: fileEnableEvents.value },
    filePath.value.split(/[\\/]/).pop() ?? 'File',
  )
}

function startMoabb() {
  if (!selectedDataset.value) return
  launchStream(
    { source: 'moabb', dataset: selectedDataset.value, subject: moabbSubject.value, enable_events: true },
    `${selectedDataset.value} (sub-${moabbSubject.value})`,
  )
}

function startRecording() {
  const rec = selectedRecordingData.value
  if (!rec) return
  launchStream(
    { source: 'file', path: rec.hdf5_file_path, enable_events: recordingEnableEvents.value },
    rec.recording_name,
  )
}

async function stopStream(id: string) {
  try {
    await store.stopStream(id)
    toast.info('Stream stopped')
  } catch {
    toast.error('Failed to stop stream')
  }
}

function close() {
  emit('update:modelValue', false)
}

// --- Keyboard ---
function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape' && props.modelValue) close()
}

// --- Lifecycle ---
watch(() => props.modelValue, async (open) => {
  if (open) {
    await store.fetchStatus()
    if (store.streams.length > 0) {
      store.startPolling()
    }
    fetchStudies()
    fetchRecordings()
  }
})

onMounted(() => {
  window.addEventListener('keydown', onKey)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  store.stopPolling()
})
</script>

<template>
  <Teleport to="body">
    <!-- Backdrop -->
    <Transition name="fade">
      <div
        v-if="modelValue"
        class="fixed inset-0 z-40 bg-black/50"
        @click="close"
      />
    </Transition>

    <!-- Panel -->
    <div
      class="fixed top-0 right-0 z-50 h-full w-[560px] bg-bg-panel border-l border-border
             shadow-2xl flex flex-col transition-transform duration-300 ease-in-out"
      :class="modelValue ? 'translate-x-0' : 'translate-x-full'"
    >
      <!-- Header -->
      <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
        <h2 class="text-sm font-semibold text-text-main">Stream Manager</h2>
        <button @click="close" class="text-text-disabled hover:text-text-main transition-colors p-1">
          <i class="pi pi-times text-xs" />
        </button>
      </div>

      <!-- ==================== Active / Finished streams ==================== -->
      <div
        v-if="activeStreams.length > 0 || store.finishedStreams.length > 0"
        class="shrink-0 max-h-[200px] overflow-y-auto border-b border-border"
      >
        <!-- Active -->
        <div v-if="activeStreams.length > 0" class="px-4 pt-3 pb-2">
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs font-semibold text-text-label">Active</span>
            <span class="text-xs text-text-disabled">{{ activeStreams.length }} running</span>
          </div>
          <div class="space-y-1.5">
            <div
              v-for="stream in activeStreams"
              :key="stream.id"
              class="px-3 py-2 bg-bg-elevated rounded border border-border"
            >
              <div class="flex items-center gap-2 mb-1">
                <span class="shrink-0"
                      :class="SOURCE_BADGE">{{ sourceLabel(stream.source) }}</span>
                <span class="text-xs text-text-main font-medium truncate flex-1">
                  {{ store.streamLabels[stream.id] || (stream.source === 'moabb' ? 'MOABB' : 'File') }}
                </span>
                <div class="w-1.5 h-1.5 rounded-full bg-status-ok animate-pulse shrink-0" />
              </div>
              <div class="flex items-center gap-2">
                <div class="flex-1 h-1.5 bg-bg-input rounded-full overflow-hidden">
                  <div
                    class="h-full rounded-full transition-all duration-500"
                    :class="stream.progress < 0
                      ? 'progress-indeterminate'
                      : 'bg-gradient-to-r from-accent to-accent-hover'"
                    :style="stream.progress >= 0 ? { width: progressInfo(stream).width } : undefined"
                  />
                </div>
                <span class="text-xs text-text-disabled shrink-0 w-16 text-right font-mono">{{ progressInfo(stream).label }}</span>
                <button
                  @click="stopStream(stream.id)"
                  class="text-xs px-1.5 py-0.5 rounded border border-border text-text-muted
                         hover:text-status-error hover:border-status-error/50 transition-colors shrink-0"
                >Stop</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Finished -->
        <div v-if="store.finishedStreams.length > 0" class="px-4 pt-2 pb-2">
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs font-semibold text-text-label">Completed</span>
            <button
              @click="store.clearFinished()"
              class="text-xs text-text-disabled hover:text-text-muted transition-colors"
            >Clear</button>
          </div>
          <div class="space-y-1.5">
            <div
              v-for="fs in store.finishedStreams"
              :key="fs.id"
              class="flex items-center gap-2 px-3 py-2 bg-bg-elevated/50 rounded border border-border/50"
            >
              <span class="shrink-0"
                    :class="SOURCE_BADGE">{{ sourceLabel(fs.source) }}</span>
              <span class="text-xs text-text-muted truncate flex-1">{{ fs.label }}</span>
              <i class="pi pi-check-circle text-xs text-status-ok shrink-0" />
              <button
                @click="store.restartFinished(fs.id)"
                class="text-xs px-1.5 py-0.5 rounded border border-border text-text-muted
                       hover:text-accent hover:border-accent/50 transition-colors shrink-0"
              >Restart</button>
              <button
                @click="store.dismissFinished(fs.id)"
                class="text-text-disabled hover:text-text-muted transition-colors shrink-0"
                title="Dismiss"
              ><i class="pi pi-times text-xs" /></button>
            </div>
          </div>
        </div>
      </div>

      <!-- Source Tabs -->
      <div class="flex border-b border-border shrink-0">
        <button
          v-for="tab in (['file', 'recordings', 'moabb'] as const)"
          :key="tab"
          @click="activeTab = tab"
          class="flex-1 px-4 py-2.5 text-xs font-medium transition-colors border-b-2"
          :class="activeTab === tab
            ? 'border-accent text-accent'
            : 'border-transparent text-text-muted hover:text-text-main'"
        >
          {{ tab === 'moabb' ? 'MOABB' : tab.charAt(0).toUpperCase() + tab.slice(1) }}
        </button>
      </div>

      <!-- Tab Content -->
      <div class="flex-1 overflow-y-auto min-h-0">
        <!-- ==================== File Tab ==================== -->
        <div v-if="activeTab === 'file'" class="p-5 space-y-4">
          <button
            @click="browseFile"
            :disabled="fileLoading"
            class="w-full py-2 rounded text-xs font-medium transition-colors
                   bg-bg-elevated border border-border text-text-main
                   hover:border-accent disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <i v-if="fileLoading" class="pi pi-spin pi-spinner mr-1" />
            <i v-else class="pi pi-folder-open mr-1" />
            Browse File...
          </button>

          <!-- Error -->
          <div v-if="fileError" class="px-3 py-2 rounded border border-status-error/30 bg-status-error/5">
            <p class="text-xs text-status-error">
              <i class="pi pi-exclamation-circle mr-1" />{{ fileError }}
            </p>
          </div>

          <!-- File info preview -->
          <StreamPreview
            v-if="fileInfo"
            :info="fileInfo"
            :file-path="filePath"
            :enable-events="fileEnableEvents"
            :loading="store.loading"
            @start="startFile"
            @update:enable-events="fileEnableEvents = $event"
          />

          <!-- Empty state -->
          <div v-if="!fileInfo && !fileError" class="py-10 text-center">
            <p class="text-xs text-text-disabled">Supports .fif, .h5, .xdf, .edf, .set</p>
          </div>
        </div>

        <!-- ==================== Recordings Tab ==================== -->
        <div v-if="activeTab === 'recordings'" class="p-5 space-y-4">
          <select v-model="recordingsStudyId" class="w-full text-xs">
            <option :value="null">All studies</option>
            <option v-for="s in studies" :key="s.study_id" :value="s.study_id">
              {{ s.study_name }}
            </option>
          </select>

          <div v-if="recordingsLoading" class="text-center py-6">
            <i class="pi pi-spin pi-spinner text-accent" />
          </div>

          <div v-else-if="recordings.length === 0" class="py-8 text-center">
            <p class="text-xs text-text-disabled">No recordings found</p>
          </div>

          <div v-else class="space-y-1 max-h-[280px] overflow-y-auto">
            <div
              v-for="rec in recordings"
              :key="rec.recording_id"
              @click="selectedRecordingId = rec.recording_id"
              class="px-3 py-2 rounded cursor-pointer transition-colors"
              :class="selectedRecordingId === rec.recording_id
                ? 'bg-accent/10 ring-1 ring-inset ring-accent/25'
                : 'hover:bg-bg-elevated'"
            >
              <div class="text-sm truncate" :class="selectedRecordingId === rec.recording_id ? 'text-accent' : 'text-text-main'">{{ rec.recording_name }}</div>
              <div class="text-xs text-text-disabled mt-1">
                sub-{{ rec.subject_id }} · ses-{{ rec.session_id }}
                <template v-if="rec.study_name"> · {{ rec.study_name }}</template>
              </div>
            </div>
          </div>

          <!-- Selected recording info (loading) -->
          <div v-if="selectedRecordingId && recordingInfoLoading" class="text-center py-4">
            <i class="pi pi-spin pi-spinner text-accent" />
          </div>

          <!-- Selected recording info preview -->
          <div v-if="selectedRecordingData && recordingInfo" class="pt-3 border-t border-border">
            <StreamPreview
              :info="recordingInfo"
              :file-path="selectedRecordingData.hdf5_file_path"
              :enable-events="recordingEnableEvents"
              :loading="store.loading"
              @start="startRecording"
              @update:enable-events="recordingEnableEvents = $event"
            />
          </div>
        </div>

        <!-- ==================== MOABB Tab ==================== -->
        <div v-if="activeTab === 'moabb'" class="p-5 space-y-4">
          <!-- Load button (first time) -->
          <div v-if="!moabbLoaded" class="text-center py-10">
            <p class="text-xs text-text-muted mb-3">Load public BCI datasets from MOABB</p>
            <button
              @click="loadMoabb"
              :disabled="store.loading"
              class="px-4 py-2 text-xs rounded bg-accent text-white hover:bg-accent-hover
                     disabled:opacity-30 transition-colors"
            >
              <i v-if="store.loading" class="pi pi-spin pi-spinner mr-1" />
              <i v-else class="pi pi-download mr-1" />
              Load Datasets
            </button>
          </div>

          <!-- Dataset list -->
          <template v-if="moabbLoaded">
            <div v-if="store.moabbDatasets.length === 0" class="py-8 text-center">
              <p class="text-xs text-text-disabled">No MOABB datasets available</p>
            </div>

            <div v-else class="space-y-1 max-h-[280px] overflow-y-auto">
              <div
                v-for="ds in store.moabbDatasets"
                :key="ds.name"
                @click="selectedDataset = ds.name"
                class="px-3 py-2 rounded cursor-pointer transition-colors"
                :class="selectedDataset === ds.name
                  ? 'bg-accent/10 ring-1 ring-inset ring-accent/25'
                  : 'hover:bg-bg-elevated'"
              >
                <div class="text-sm truncate" :class="selectedDataset === ds.name ? 'text-accent' : 'text-text-main'">{{ ds.name }}</div>
                <div class="text-xs text-text-disabled mt-1">
                  {{ ds.paradigm }} · {{ ds.n_subjects }} subjects
                </div>
              </div>
            </div>

            <!-- Selected dataset config -->
            <div v-if="selectedMoabbData" class="space-y-3 pt-3 border-t border-border">
              <div class="flex items-center gap-3 text-xs text-text-muted">
                <span>{{ selectedMoabbData.paradigm }}</span>
                <span class="text-border">·</span>
                <span><span class="font-mono text-text-main">{{ selectedMoabbData.n_subjects }}</span> subjects</span>
              </div>

              <!-- Events -->
              <div v-if="Object.keys(selectedMoabbData.events).length > 0" class="flex flex-wrap gap-1.5">
                <span
                  v-for="[name, count] in Object.entries(selectedMoabbData.events).sort((a, b) => a[0].localeCompare(b[0]))"
                  :key="name"
                  class="px-1.5 py-0.5 rounded bg-bg-elevated text-xs text-text-muted"
                >
                  {{ name }}: {{ count }}
                </span>
              </div>

              <!-- Subject input -->
              <label class="block">
                <span class="text-xs text-text-muted block mb-1">Subject</span>
                <NumberInput
                  v-model="moabbSubject"
                  :min="1"
                  :max="selectedMoabbData.n_subjects"
                  class="w-full text-xs"
                />
              </label>

              <button
                @click="startMoabb"
                :disabled="store.loading"
                class="w-full py-2 rounded text-xs font-medium text-white transition-colors
                       bg-accent hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <i v-if="store.loading" class="pi pi-spin pi-spinner mr-1" />
                <i v-else class="pi pi-play mr-1" />
                Start Stream
              </button>
            </div>
          </template>
        </div>
      </div>

    </div>
  </Teleport>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.progress-indeterminate {
  width: 100%;
  background: linear-gradient(
    90deg,
    var(--color-bg-input) 0%,
    var(--color-accent) 40%,
    var(--color-accent-hover) 60%,
    var(--color-bg-input) 100%
  );
  background-size: 200% 100%;
  animation: indeterminate 1.5s ease-in-out infinite;
}

@keyframes indeterminate {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}
</style>
