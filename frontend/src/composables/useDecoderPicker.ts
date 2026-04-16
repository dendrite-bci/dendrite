import { ref, onMounted, type Ref, type ComputedRef } from 'vue'
import type { Decoder, ModalityChannel } from '../types/api'
import { apiFetchOrNull } from '../utils/api'

interface EventEntry { id: number; label: string }

interface DecoderPickerDeps {
  eventMapping: Ref<EventEntry[]>
  epochTmin: Ref<number>
  epochTmax: Ref<number>
  modelType: Ref<string>
  channelSelection: Ref<Record<string, number[]>>
  selectedModality: Ref<string>
  modePreproc: Ref<Record<string, Record<string, any>>>
  selectedChannels: ComputedRef<ModalityChannel[]>
  switchModality?: (key: string) => void
}

export function useDecoderPicker(
  initial: { decoder_config?: Record<string, any>; decoder_source?: string; source_mode?: string | null },
  deps: DecoderPickerDeps,
) {
  const decoderCfg = initial.decoder_config ?? {}
  const modelCfg = decoderCfg.model_config ?? {}

  const source = ref(initial.decoder_source ?? 'database')
  const path = ref<string>(decoderCfg.decoder_path ?? '')
  const id = ref<number | null>(decoderCfg.decoder_id ?? null)
  const showPicker = ref(false)
  const selectedInfo = ref<Decoder | null>(null)
  const decoderEventMapping = ref<Record<string, string> | null>(null)
  const sourceMode = ref<string | null>(initial.source_mode ?? null)
  const numClasses = ref(modelCfg.num_classes ?? 2)

  function applyMappings() {
    if (!decoderEventMapping.value) return
    deps.eventMapping.value = Object.entries(decoderEventMapping.value)
      .map(([k, label]) => ({ id: Number(k), label: label as string }))
  }

  function applyChannels(channelLabels: Record<string, string[]>) {
    for (const [mod, labels] of Object.entries(channelLabels)) {
      // Switch modality if needed
      if (mod !== deps.selectedModality.value) {
        if (deps.switchModality) deps.switchModality(mod)
        else deps.selectedModality.value = mod
      }
      const available = deps.selectedChannels.value
      if (!available.length) continue
      const indices = labels
        .map(lbl => available.findIndex(ch => ch.label === lbl))
        .filter(i => i >= 0)
      if (indices.length > 0) {
        deps.channelSelection.value[mod] = indices
      }
    }
  }

  async function onSelected(decoder: Decoder) {
    selectedInfo.value = decoder
    id.value = decoder.decoder_id
    path.value = decoder.decoder_path
    deps.modelType.value = decoder.model_type
    numClasses.value = decoder.num_classes ?? numClasses.value
    showPicker.value = false
    decoderEventMapping.value = null

    const meta = await apiFetchOrNull<any>(`/api/data/decoders/${decoder.decoder_id}/metadata`)
    if (!meta) {
      console.warn('[DecoderPicker] metadata fetch failed for', decoder.decoder_id)
      return
    }

    decoderEventMapping.value = meta.event_mapping ?? null
    if (decoderEventMapping.value) applyMappings()

    const modPreproc = meta.preprocessing_config?.modality_preprocessing
    if (modPreproc) {
      for (const [mod, cfg] of Object.entries(modPreproc as Record<string, any>)) {
        deps.modePreproc.value[mod] = {
          lowcut: cfg.lowcut ?? null,
          highcut: cfg.highcut ?? null,
          apply_rereferencing: cfg.apply_rereferencing ?? false,
          ...(cfg.line_freq != null ? { line_freq: cfg.line_freq } : {}),
        }
      }
    }

    if (meta.epoch_tmin != null) deps.epochTmin.value = meta.epoch_tmin
    if (meta.epoch_tmax != null) deps.epochTmax.value = meta.epoch_tmax
    if (meta.channel_labels) applyChannels(meta.channel_labels)
  }

  function clear() {
    selectedInfo.value = null
    id.value = null
    path.value = ''
    decoderEventMapping.value = null
  }

  // Auto-restore decoder state when re-opening a mode with an existing database decoder
  onMounted(async () => {
    if (source.value !== 'database' || !id.value) return
    const [decoder, meta] = await Promise.all([
      apiFetchOrNull<Decoder>(`/api/data/decoders/${id.value}`),
      apiFetchOrNull<any>(`/api/data/decoders/${id.value}/metadata`),
    ])

    if (!decoder && !meta) {
      console.warn('[DecoderPicker] restore failed for decoder', id.value)
      return
    }

    if (decoder) selectedInfo.value = decoder

    if (meta) {
      decoderEventMapping.value = meta.event_mapping ?? null
      if (decoderEventMapping.value && deps.eventMapping.value.length === 0) {
        applyMappings()
      }
      if (meta.channel_labels && Object.values(deps.channelSelection.value).every(a => a.length === 0)) {
        applyChannels(meta.channel_labels)
      }
    }
  })

  return {
    source, path, id, showPicker, selectedInfo,
    decoderEventMapping, sourceMode, numClasses,
    onSelected, applyMappings, clear,
  }
}
