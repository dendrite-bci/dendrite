<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, reactive } from 'vue'
import { useStreamsStore } from '../../stores/streams'
import { usePipelineStore } from '../../stores/pipeline'
import { useToast } from '../../composables/useToast'
import { typeBadgeClass, typeBadgeBase } from '../../utils/streamBadge'
import type { StreamMetadata } from '../../types/api'

const emit = defineEmits<{
  close: []
  applied: []
}>()

const streams = useStreamsStore()
const pipeline = usePipelineStore()
const toast = useToast()

const CHANNEL_TYPES = ['EEG', 'EOG', 'VEOG', 'HEOG', 'EMG', 'ECG', 'AUX', 'Markers', 'Other']
const CHANNEL_UNITS = ['µV', 'mV', 'V', 'n/a']

// Local state: all discovered streams with include flag + editable channel data
interface LocalStream {
  stream: StreamMetadata
  included: boolean
  labels: string[]
  channelTypes: string[]
  channelUnits: string[]
}

const localStreams = reactive<Record<string, LocalStream>>({})
const selectedUid = ref<string | null>(null)
const isDiscovering = ref(false)
const isApplying = ref(false)
const selectedRows = ref(new Set<number>())
const bulkType = ref('EEG')

// Constraint: max 1 Events stream
const hasEvents = computed(() =>
  Object.values(localStreams).some(s => s.included && s.stream.type.toUpperCase() === 'MARKERS')
)

const selectedStream = computed(() =>
  selectedUid.value ? localStreams[selectedUid.value] : null
)

const includedCount = computed(() =>
  Object.values(localStreams).filter(s => s.included).length
)

const totalCount = computed(() => Object.keys(localStreams).length)

const globalSummary = computed(() => {
  const counts: Record<string, number> = {}
  for (const s of Object.values(localStreams)) {
    if (!s.included) continue
    for (const t of s.channelTypes) {
      counts[t] = (counts[t] || 0) + 1
    }
  }
  return Object.entries(counts).map(([t, c]) => `${c} ${t}`).join(', ')
})

function createLocalStream(stream: StreamMetadata, included: boolean): LocalStream {
  return {
    stream,
    included,
    labels: [...stream.labels],
    channelTypes: [...stream.channel_types],
    channelUnits: stream.channel_units?.length
      ? [...stream.channel_units]
      : stream.labels.map(() => 'µV'),
  }
}

function populateFromDiscovery(discovered: Record<string, StreamMetadata>) {
  for (const [uid, stream] of Object.entries(discovered)) {
    if (!localStreams[uid]) {
      localStreams[uid] = createLocalStream(stream, false)
    }
  }
}

async function discover() {
  isDiscovering.value = true
  try {
    await streams.discover()
    // Clear stale local state and rebuild from fresh discovery
    for (const key of Object.keys(localStreams)) delete localStreams[key]
    populateFromDiscovery(streams.discoveredStreams)
  } finally {
    isDiscovering.value = false
  }
}

function toggleInclude(uid: string) {
  const s = localStreams[uid]
  if (!s) return

  if (s.included) {
    s.included = false
    return
  }

  // Enforce constraint: max 1 Events stream
  const type = s.stream.type.toUpperCase()
  if (type === 'MARKERS' && hasEvents.value) return

  s.included = true
}

function selectStream(uid: string) {
  selectedUid.value = uid
  selectedRows.value = new Set()
}

function canInclude(uid: string): boolean {
  const s = localStreams[uid]
  if (!s || s.included) return true
  const type = s.stream.type.toUpperCase()
  if (type === 'MARKERS' && hasEvents.value) return false
  return true
}

// Row selection for bulk ops
function toggleRow(index: number) {
  const s = new Set(selectedRows.value)
  if (s.has(index)) s.delete(index)
  else s.add(index)
  selectedRows.value = s
}

function selectAllRows() {
  if (!selectedStream.value) return
  const s = new Set<number>()
  for (let i = 0; i < selectedStream.value.stream.channel_count; i++) s.add(i)
  selectedRows.value = s
}

function selectNoRows() {
  selectedRows.value = new Set()
}

function applyBulkType() {
  if (!selectedStream.value || selectedRows.value.size === 0) return
  for (const i of selectedRows.value) {
    selectedStream.value.channelTypes[i] = bulkType.value
  }
}

async function apply() {
  const includedUids = Object.entries(localStreams)
    .filter(([, s]) => s.included)
    .map(([uid]) => uid)

  if (includedUids.length === 0) return

  const overrides: Record<string, any> = {}
  for (const uid of includedUids) {
    const ls = localStreams[uid]
    if (!ls) continue
    overrides[uid] = {
      labels: ls.labels,
      channel_types: ls.channelTypes,
      channel_units: ls.channelUnits,
    }
  }

  isApplying.value = true
  try {
    await streams.configure(includedUids, overrides)
    emit('applied')
    emit('close')
  } catch (err: any) {
    const detail = err?.detail || err?.message || 'Stream configuration failed'
    toast.error(detail)
    isApplying.value = false
  }
}

function typeSelectColor(t: string): string {
  switch (t.toUpperCase()) {
    case 'EEG': return 'bg-accent/20'
    case 'EMG': return 'bg-status-ok/20'
    case 'MARKERS': return 'bg-status-error/20'
    default: return 'bg-bg-elevated'
  }
}

function onKey(e: KeyboardEvent) { if (e.key === 'Escape') emit('close') }
onUnmounted(() => window.removeEventListener('keydown', onKey))

onMounted(() => {
  window.addEventListener('keydown', onKey)
  // Pre-populate with already discovered streams
  if (Object.keys(streams.discoveredStreams).length > 0) {
    populateFromDiscovery(streams.discoveredStreams)
  }
  // Pre-populate with already configured streams
  for (const [uid, stream] of Object.entries(streams.configuredStreams)) {
    if (!localStreams[uid]) {
      localStreams[uid] = createLocalStream(stream, true)
    } else {
      localStreams[uid].included = true
      // Apply config's channel_type overrides if channel count matches
      if (stream.channel_types.length === localStreams[uid].channelTypes.length) {
        localStreams[uid].channelTypes = [...stream.channel_types]
      }
    }
  }
  // Auto-discover if no streams
  if (Object.keys(localStreams).length === 0) {
    discover()
  }
})
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl shadow-black/40
                  w-[1100px] max-w-[92vw] h-[80vh] flex flex-col">

        <!-- Header -->
        <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <h2 class="text-base font-semibold text-text-main">Stream Setup</h2>
          <div class="flex items-center gap-2">
            <button
              @click="discover"
              :disabled="isDiscovering || isApplying || pipeline.status.recording"
              class="px-3 py-1 text-xs rounded bg-bg-input border border-border text-text-muted
                     hover:text-text-main hover:border-accent disabled:opacity-30 transition-colors"
            >
              <i v-if="isDiscovering" class="pi pi-spin pi-spinner mr-1" />
              <i v-else class="pi pi-refresh mr-1" />
              Rescan
            </button>
            <button @click="emit('close')" class="text-text-disabled hover:text-text-main transition-colors p-1">
              <i class="pi pi-times" />
            </button>
          </div>
        </div>

        <!-- Split panel -->
        <div class="flex flex-1 min-h-0">
          <!-- Left panel: stream list -->
          <div class="w-[280px] border-r border-border flex flex-col shrink-0">
            <div class="flex-1 overflow-y-auto">
              <div v-if="totalCount === 0 && !isDiscovering" class="p-4 text-center">
                <i class="pi pi-wifi text-2xl text-text-disabled block mb-2" />
                <p class="text-xs text-text-muted">No streams found</p>
                <p class="text-xs text-text-disabled mt-1">Click Rescan to discover LSL streams</p>
              </div>

              <div v-if="isDiscovering && totalCount === 0" class="p-4 text-center">
                <i class="pi pi-spin pi-spinner text-xl text-accent block mb-2" />
                <p class="text-xs text-text-muted">Discovering streams...</p>
              </div>

              <div
                v-for="(ls, uid) in localStreams"
                :key="uid"
                @click="selectStream(uid as string)"
                class="flex items-center gap-2 px-3 py-2.5 border-b border-border/50 cursor-pointer transition-colors"
                :class="[
                  selectedUid === uid ? 'bg-accent/10 ring-1 ring-inset ring-accent/25' : 'hover:bg-bg-elevated/50',
                  ls.included ? 'border-l-2 border-l-accent' : 'border-l-2 border-l-transparent'
                ]"
              >
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-1.5">
                    <span class="text-sm text-text-main truncate">{{ ls.stream.name }}</span>
                    <span :class="[typeBadgeBase, typeBadgeClass(ls.stream.type)]">
                      {{ ls.stream.type }}
                    </span>
                  </div>
                  <div class="text-xs text-text-muted">
                    {{ ls.stream.channel_count }}ch · {{ ls.stream.sample_rate }}Hz
                  </div>
                </div>
                <i v-if="ls.stream.has_metadata_issues"
                   class="pi pi-exclamation-triangle text-xs text-status-warn mr-1" />
                <button
                  @click.stop="toggleInclude(uid as string)"
                  :disabled="!canInclude(uid as string) && !ls.included"
                  :title="!canInclude(uid as string) && !ls.included
                    ? 'Only one Events stream allowed'
                    : undefined"
                  class="relative w-8 h-[18px] rounded-full shrink-0 transition-colors disabled:opacity-30"
                  :class="ls.included ? 'bg-accent' : 'bg-bg-input border border-border'"
                >
                  <span class="absolute top-[2px] w-[14px] h-[14px] rounded-full bg-text-main shadow transition-all"
                        :class="ls.included ? 'left-[15px]' : 'left-[1px]'" />
                </button>
              </div>
            </div>
          </div>

          <!-- Right panel: stream config -->
          <div class="flex-1 flex flex-col min-w-0">
            <!-- No selection state -->
            <div v-if="!selectedStream" class="flex-1 flex items-center justify-center">
              <div class="text-center">
                <i class="pi pi-arrow-left text-2xl text-text-disabled block mb-2" />
                <p class="text-sm text-text-muted">Select a stream to configure</p>
              </div>
            </div>

            <!-- Stream config -->
            <template v-else>
              <!-- Stream name header -->
              <div class="px-6 py-3 border-b border-border shrink-0">
                <div class="text-sm font-semibold text-text-main">{{ selectedStream.stream.name }}</div>
                <div class="text-xs text-text-muted mt-0.5">{{ selectedStream.stream.type }} stream</div>
              </div>

              <!-- Merged content -->
              <div class="flex-1 overflow-y-auto p-6 space-y-5">

                <!-- Stream Info -->
                <section class="text-[11px]">
                  <span class="text-text-muted">Rate:</span>
                  <span class="text-text-label font-mono ml-1">{{ selectedStream.stream.sample_rate }} Hz</span>
                  <span class="text-text-disabled mx-2">·</span>
                  <span class="text-text-muted">Channels:</span>
                  <span class="text-text-label font-mono ml-1">{{ selectedStream.stream.channel_count }}</span>
                  <span class="text-text-disabled mx-2">·</span>
                  <span class="text-text-muted">Format:</span>
                  <span class="text-text-label font-mono ml-1">{{ selectedStream.stream.channel_format }}</span>
                  <template v-if="selectedStream.stream.source_id">
                    <span class="text-text-disabled mx-2">·</span>
                    <span class="text-text-muted">ID:</span>
                    <span class="text-text-label font-mono ml-1">{{ selectedStream.stream.source_id }}</span>
                  </template>

                  <!-- Metadata issues -->
                  <div v-if="selectedStream.stream.has_metadata_issues"
                       class="mt-3 px-3 py-2 rounded border border-status-warn/30 bg-status-warn/5 font-sans text-xs">
                    <div class="flex items-center gap-2 text-status-warn mb-1">
                      <i class="pi pi-exclamation-triangle" />
                      <span class="font-medium">Metadata Issues</span>
                    </div>
                    <ul class="text-text-muted space-y-0.5 ml-5">
                      <li v-for="(detail, key) in selectedStream.stream.metadata_issues" :key="key">
                        {{ key }}: {{ detail }}
                      </li>
                    </ul>
                  </div>
                </section>

                <!-- Divider -->
                <div class="border-t border-border" />

                <!-- Channels -->
                <section>
                  <div class="flex items-center justify-between mb-3">
                    <h4 class="text-xs font-semibold text-text-label uppercase tracking-wide">
                      Channels
                      <span v-if="selectedRows.size > 0" class="text-text-muted font-normal normal-case tracking-normal ml-1">
                        ({{ selectedRows.size }} selected)
                      </span>
                    </h4>
                    <div v-if="selectedRows.size > 0" class="flex items-center gap-2">
                      <select
                        v-model="bulkType"
                        class="px-2 py-0.5 text-xs cursor-pointer"
                        :class="typeSelectColor(bulkType)"
                      >
                        <option v-for="ct in CHANNEL_TYPES" :key="ct" :value="ct">{{ ct }}</option>
                      </select>
                      <button
                        @click="applyBulkType"
                        class="px-2 py-0.5 text-xs rounded bg-accent text-white hover:bg-accent-hover transition-colors"
                      >
                        Set Type
                      </button>
                    </div>
                  </div>

                  <div class="rounded border border-border bg-bg-elevated/30 overflow-hidden">
                  <table class="w-full text-xs">
                    <thead class="bg-bg-panel">
                      <tr class="text-text-label border-b border-border">
                        <th class="py-1.5 pl-1 w-7">
                          <span
                            @click="selectedRows.size === selectedStream.stream.channel_count ? selectNoRows() : selectAllRows()"
                            class="w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0 cursor-pointer transition-colors text-[8px]"
                            :class="selectedRows.size === selectedStream.stream.channel_count
                              ? 'border-accent/50 bg-accent/20 text-accent'
                              : 'border-border/50'"
                          >
                            <i v-if="selectedRows.size === selectedStream.stream.channel_count" class="pi pi-check" />
                          </span>
                        </th>
                        <th class="py-1.5 text-left w-8">#</th>
                        <th class="py-1.5 text-left w-40">Label</th>
                        <th class="py-1.5 text-left w-28">Type</th>
                        <th class="py-1.5 text-left w-20">Unit</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr
                        v-for="i in selectedStream.stream.channel_count"
                        :key="i - 1"
                        @click="toggleRow(i - 1)"
                        class="border-b border-border/30 cursor-pointer"
                        :class="selectedRows.has(i - 1) ? 'bg-accent/10' : 'hover:bg-bg-elevated/30'"
                      >
                        <td class="py-1 pl-1">
                          <span
                            class="w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0 transition-colors text-[8px]"
                            :class="selectedRows.has(i - 1)
                              ? 'border-accent/50 bg-accent/20 text-accent'
                              : 'border-border/50'"
                          >
                            <i v-if="selectedRows.has(i - 1)" class="pi pi-check" />
                          </span>
                        </td>
                        <td class="py-1 text-text-disabled font-mono">{{ i - 1 }}</td>
                        <td class="py-1 px-1">
                          <input
                            v-model="selectedStream.labels[i - 1]"
                            @click.stop
                            class="w-full bg-transparent px-1 py-0.5"
                          />
                        </td>
                        <td class="py-1 px-1">
                          <select
                            v-model="selectedStream.channelTypes[i - 1]"
                            @click.stop
                            class="w-full px-1.5 py-0.5 text-xs cursor-pointer"
                            :class="typeSelectColor(selectedStream.channelTypes[i - 1] ?? '')"
                          >
                            <option v-for="ct in CHANNEL_TYPES" :key="ct" :value="ct">{{ ct }}</option>
                          </select>
                        </td>
                        <td class="py-1 px-1">
                          <select
                            v-model="selectedStream.channelUnits[i - 1]"
                            @click.stop
                            class="w-full px-1.5 py-0.5 text-xs bg-bg-elevated cursor-pointer"
                          >
                            <option v-for="cu in CHANNEL_UNITS" :key="cu" :value="cu">{{ cu }}</option>
                          </select>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  </div>
                </section>

              </div>
            </template>
          </div>
        </div>

        <!-- Footer -->
        <div class="flex items-center justify-between px-5 py-3 border-t border-border shrink-0">
          <div class="text-xs text-text-muted">
            <template v-if="includedCount > 0">
              {{ includedCount }}/{{ totalCount }} streams · {{ globalSummary }}
            </template>
            <template v-else>
              No streams selected
            </template>
          </div>
          <div class="flex gap-2">
            <button
              @click="emit('close')"
              :disabled="isApplying"
              class="px-4 py-1.5 text-xs rounded border border-border text-text-muted
                     hover:text-text-main hover:border-text-muted transition-colors
                     disabled:opacity-30"
            >
              Cancel
            </button>
            <button
              @click="apply"
              :disabled="includedCount === 0 || isApplying"
              class="px-4 py-1.5 text-xs rounded bg-accent text-white hover:bg-accent-hover
                     disabled:opacity-30 transition-colors"
            >
              <i v-if="isApplying" class="pi pi-spin pi-spinner mr-1" />
              {{ isApplying ? 'Configuring...' : `Apply (${includedCount})` }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
