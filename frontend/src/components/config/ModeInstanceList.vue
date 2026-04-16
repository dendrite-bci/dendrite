<script setup lang="ts">
import { ref } from 'vue'
import { useModesStore } from '../../stores/modes'
import { usePipelineStore } from '../../stores/pipeline'
import { useToast } from '../../composables/useToast'
import type { ModeInstance } from '../../types/api'
import ModeConfigDialog from './ModeConfigDialog.vue'

const modes = useModesStore()
const pipeline = usePipelineStore()
const toast = useToast()

const editingMode = ref<{ name: string; instance: ModeInstance } | null>(null)

function openModeDialog(name: string, instance: ModeInstance) {
  editingMode.value = { name, instance }
}

function closeDialog() {
  editingMode.value = null
}

const modeTypes = [
  { value: 'synchronous', label: 'Synchronous', icon: 'pi pi-sync' },
  { value: 'asynchronous', label: 'Asynchronous', icon: 'pi pi-bolt' },
  { value: 'neurofeedback', label: 'Neurofeedback', icon: 'pi pi-wave-pulse' },
]

function badgeDetail(instance: ModeInstance): string {
  if (instance.mode === 'neurofeedback') {
    const bands = instance.feature_config?.target_bands
    if (bands) {
      const keys = Object.keys(bands)
      if (keys.length === 1) {
        const [low, high] = bands[keys[0]!]
        return `Band: ${low}-${high} Hz`
      }
      return `${keys.length} target bands`
    }
    return 'Default: 8-12 Hz'
  }
  const modelType = instance.decoder_config?.model_config?.model_type
  return modelType || 'Not configured'
}

function badgeEventNames(instance: ModeInstance): string[] {
  const mapping = instance.event_mapping
  if (!mapping || Object.keys(mapping).length === 0) return []
  return Object.entries(mapping)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([, name]) => name)
}

const MODE_ICON: Record<string, string> = Object.fromEntries(
  modeTypes.map(mt => [mt.value, mt.icon])
)

function modeState(name: string): string {
  return modes.modeStates[name] ?? 'idle'
}

interface StateStyle { dot: string; bg: string; text: string; label: string }

const STATE_MAP: Record<string, StateStyle> = {
  running:  { dot: 'bg-status-ok',    bg: 'bg-status-ok/15',    text: 'text-status-ok',    label: 'Running' },
  starting: { dot: 'bg-status-warn',  bg: 'bg-status-warn/15',  text: 'text-status-warn',  label: 'Starting...' },
  stopping: { dot: 'bg-status-warn',  bg: 'bg-status-warn/15',  text: 'text-status-warn',  label: 'Stopping...' },
  error:    { dot: 'bg-status-error', bg: 'bg-status-error/15', text: 'text-status-error', label: 'Error' },
  idle:     { dot: 'bg-text-disabled', bg: 'bg-white/[0.04]',   text: 'text-text-disabled', label: 'Idle' },
  stopped:  { dot: 'bg-status-warn',  bg: 'bg-status-warn/10',  text: 'text-status-warn',  label: 'Stopped' },
}
const IDLE_STYLE = STATE_MAP.idle!

function stateStyle(name: string): StateStyle {
  return STATE_MAP[modeState(name)] ?? IDLE_STYLE
}

function isLoading(name: string): boolean {
  if (pipeline.loading) return true
  const s = modeState(name)
  return !!modes.modeActionLoading[name] || s === 'starting' || s === 'stopping'
}

function canStartMode(name: string, instance: ModeInstance): boolean {
  const s = modeState(name)
  return (s === 'idle' || s === 'stopped') && instance.enabled !== false && !isLoading(name)
}

function canStopMode(name: string): boolean {
  return modeState(name) === 'running' && !isLoading(name)
}

function isModeRunning(name: string): boolean {
  return modeState(name) === 'running'
}

async function spawnAsyncFromSync(name: string, instance: ModeInstance) {
  const asyncName = `${name}_async`
  const tmin = instance.epoch_tmin ?? 0
  const tmax = instance.epoch_tmax ?? 2.0
  const config: Record<string, any> = {
    channel_selection: instance.channel_selection,
    event_mapping: instance.event_mapping,
    mode_preprocessing: instance.mode_preprocessing,
    decoder_source: 'online',
    decoder_config: {
      decoder_type: 'Decoder',
      model_type: instance.decoder_config?.model_config?.model_type ?? 'EEGNet',
      num_classes: Object.keys(instance.event_mapping ?? {}).length,
    },
    source_mode: name,
    window_length_sec: tmax - tmin,
    step_size_ms: 100,
  }
  const ok = await modes.addInstance('asynchronous', config, asyncName)
  if (!ok) return
  toast.success(`Async mode "${asyncName}" created`)
  const newInstance = modes.instances[asyncName]
  if (newInstance) openModeDialog(asyncName, newInstance)
}

</script>

<template>
  <div>
    <!-- Add mode buttons -->
    <div class="flex gap-1.5 mb-4">
      <button
        v-for="mt in modeTypes"
        :key="mt.value"
        @click="modes.addInstance(mt.value)"
        class="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs
               border border-dashed border-border text-text-disabled
               hover:border-accent/50 hover:text-accent transition-colors"
      >
        <i :class="mt.icon" class="text-[11px]" />
        {{ mt.label }}
      </button>
    </div>

    <div v-if="modes.instanceCount > 0" class="border-t border-border pt-4 space-y-[9px]">
      <div
        v-for="(instance, name) in modes.instances"
        :key="name"
        class="group/card flex rounded-lg overflow-hidden border border-border
               hover:border-white/[0.12] transition-colors cursor-pointer"
        @click="openModeDialog(name as string, instance)"
      >
        <!-- Mode type icon + state dot -->
        <div
          class="w-12 shrink-0 flex flex-col items-center justify-center gap-2 bg-white/[0.02]"
        >
          <i :class="MODE_ICON[instance.mode]" class="text-base text-text-disabled group-hover/card:text-text-muted transition-colors" />
          <span
            v-if="pipeline.status.recording"
            class="w-2 h-2 rounded-full shrink-0"
            :class="[stateStyle(name as string).dot, isModeRunning(name as string) ? 'animate-pulse' : '']"
          />
        </div>

        <!-- Card content -->
        <div class="flex-1 min-w-0 bg-bg-elevated px-3 py-3">
          <!-- Row 1: Name + state + actions -->
          <div class="flex items-center gap-2">
            <span class="text-[13px] font-semibold text-text-main truncate">{{ name }}</span>
            <!-- Linked source mode -->
            <span
              v-if="instance.source_mode"
              class="inline-flex items-center gap-1 text-[10px] text-text-disabled"
            >
              <i class="pi pi-link text-[9px]" />
              {{ instance.source_mode }}
            </span>
            <span class="flex-1" />
            <button
              v-if="instance.mode === 'synchronous'"
              @click.stop="spawnAsyncFromSync(name as string, instance)"
              class="w-6 h-6 flex items-center justify-center text-text-disabled hover:text-accent transition-colors rounded hover:bg-accent/10"
              title="Create async mode from this"
            >
              <i class="pi pi-forward text-xs" />
            </button>
            <button
              @click.stop="modes.cloneInstance(name as string)"
              class="w-6 h-6 flex items-center justify-center text-text-disabled hover:text-text-main transition-colors rounded hover:bg-bg-input"
              title="Clone"
            >
              <i class="pi pi-copy text-xs" />
            </button>
            <button
              v-if="!pipeline.status.recording || !isModeRunning(name as string)"
              @click.stop="modes.removeInstance(name as string)"
              class="w-6 h-6 flex items-center justify-center text-text-disabled hover:text-status-error transition-colors rounded hover:bg-status-error/10"
              title="Remove"
            >
              <i class="pi pi-times text-xs" />
            </button>
          </div>

          <!-- Row 2: detail + events -->
          <div class="flex items-center gap-1.5 mt-1 flex-wrap">
            <span class="text-[11px] text-text-muted">{{ badgeDetail(instance) }}</span>
            <span
              v-for="ev in badgeEventNames(instance)" :key="ev"
              class="text-[10px] text-text-disabled bg-white/[0.06] rounded px-1.5 py-0.5"
            >{{ ev }}</span>
          </div>
        </div>

        <!-- Right: Play/Stop button (vertically centered, during recording) -->
        <div v-if="pipeline.status.recording || pipeline.loading" class="flex items-center px-3 bg-bg-elevated border-l border-white/[0.06]">
          <!-- Loading/Starting spinner -->
          <div
            v-if="isLoading(name as string)"
            class="w-10 h-10 flex items-center justify-center"
          >
            <i class="pi pi-spin pi-spinner text-lg text-text-muted" />
          </div>
          <!-- Start -->
          <button
            v-else-if="canStartMode(name as string, instance)"
            @click.stop="modes.startMode(name as string)"
            class="w-10 h-10 flex items-center justify-center rounded-full
                   border border-text-muted/40 text-text-label
                   hover:border-text-main hover:text-text-main hover:bg-white/5
                   active:scale-95 transition-all"
            title="Start mode"
          >
            <i class="pi pi-play text-sm" />
          </button>
          <!-- Stop -->
          <button
            v-else-if="canStopMode(name as string)"
            @click.stop="modes.stopMode(name as string)"
            class="w-10 h-10 flex items-center justify-center rounded-full
                   border border-text-muted/40 text-text-label
                   hover:border-status-error hover:text-status-error hover:bg-status-error/5
                   active:scale-95 transition-all"
            title="Stop mode"
          >
            <i class="pi pi-stop text-sm" />
          </button>
          <!-- Running (green pulse) -->
          <div v-else class="w-10 h-10 flex items-center justify-center">
            <div class="w-3 h-3 rounded-full bg-status-ok animate-pulse" />
          </div>
        </div>
      </div>
    </div>
    <p v-else class="text-xs text-text-disabled">No mode instances configured.</p>

    <ModeConfigDialog
      v-if="editingMode"
      :instance-name="editingMode.name"
      :instance="editingMode.instance"
      @close="closeDialog"
    />
  </div>
</template>
